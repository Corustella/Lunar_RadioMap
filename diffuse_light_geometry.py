"""Batched fixed-step intersections with a finite bilinear DEM.

Lightweight geometric approximation. Half-pixel marching can miss grazing contacts or
sub-step enter/exit pairs; it is not exact DDA. A detected first crossing is
refined by bisection and a final secant step. Returned points are projected
vertically onto the reconstructed surface to remove float32 root residuals;
the returned distance is the pre-projection ray parameter.
"""

import math

import torch


def _check(height, resolution):
    if height.ndim != 2 or min(height.shape) < 2:
        raise ValueError("height must have shape [H,W], H,W >= 2")
    if height.dtype not in (torch.float32, torch.float64):
        raise ValueError("height must be float32 or float64")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive and finite")


def _height_only(height, x, y, resolution):
    """Internal gather; callers must enforce the finite XY domain."""
    h, w = height.shape
    gx = (x / resolution).clamp(0, w - 1)
    gy = (y / resolution).clamp(0, h - 1)
    col = gx.floor().long().clamp(0, w - 2)
    row = gy.floor().long().clamp(0, h - 2)
    u, v = gx - col, gy - row
    flat, base = height.reshape(-1), row * w + col
    z00, z10 = flat[base], flat[base + 1]
    z01, z11 = flat[base + w], flat[base + w + 1]
    return (z00 * (1 - u) + z10 * u) * (1 - v) + (z01 * (1 - u) + z11 * u) * v


@torch.no_grad()
def sample_surface(height, xy, resolution):
    """Bilinear height and up normal; grid-line derivatives are symmetric."""
    _check(height, resolution)
    xy = torch.as_tensor(xy, dtype=height.dtype, device=height.device)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape [M,2]")
    h, w = height.shape
    x, y = xy[:, 0] / resolution, xy[:, 1] / resolution
    right = x.floor().long().clamp(0, w - 2)
    top = y.floor().long().clamp(0, h - 2)
    left = (x.ceil().long() - 1).clamp(0, w - 2)
    bottom = (y.ceil().long() - 1).clamp(0, h - 2)
    z = _height_only(height, xy[:, 0], xy[:, 1], resolution)
    dx, dy = torch.zeros_like(z), torch.zeros_like(z)
    flat = height.reshape(-1)
    for row, col in ((top, right), (top, left), (bottom, right), (bottom, left)):
        base = row * w + col
        a = flat[base]
        b, c = flat[base + 1] - a, flat[base + w] - a
        d = flat[base + w + 1] - a - b - c
        dx += (b + d * (y - row)) / (4 * resolution)
        dy += (c + d * (x - col)) / (4 * resolution)
    normal = torch.stack((-dx, -dy, torch.ones_like(z)), 1)
    normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    return z, normal


def _rays(height, origins, directions):
    origins = torch.as_tensor(origins, device=height.device, dtype=height.dtype)
    directions = torch.as_tensor(directions, device=height.device, dtype=height.dtype)
    if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
        raise ValueError("origins and directions must have shape [M,3]")
    return origins, directions


@torch.no_grad()
def first_hit(height, origins, dirs, res, ray_chunk=8192, step_px=.5, refine_steps=6):
    """First sampled positive-to-nonpositive crossing, then root refinement.

    Directions must be unit vectors; origins must be inside the XY domain and
    on/above its surface. The global terrain Z bounds shorten the interval.
    Misses return False, zero point/normal and infinite ray distance. Sampling
    and bisection are float32 when height is float32. No DEM border extension.
    """
    _check(height, res)
    if ray_chunk < 1 or step_px <= 0 or refine_steps < 0:
        raise ValueError("invalid marching configuration")
    origins, dirs = _rays(height, origins, dirs)
    count, (h, w) = len(origins), height.shape
    hit = torch.zeros(count, dtype=torch.bool, device=height.device)
    points, normals = torch.zeros_like(origins), torch.zeros_like(origins)
    distances = torch.full((count,), float("inf"), dtype=height.dtype, device=height.device)
    if count == 0:
        return hit, points, normals, distances
    hmin, hmax = height.amin(), height.amax()
    # Expand only the acceleration box, not the terrain or visibility rule.
    # A zero-width Z box on a flat DEM can otherwise lose the root to one ULP.
    box_margin = 8 * torch.finfo(height.dtype).eps * (hmin.abs() + hmax.abs() + 1)
    hmin, hmax = hmin - box_margin, hmax + box_margin
    bounds = height.new_tensor(((w - 1) * res, (h - 1) * res))
    eps = 1e-4
    for start in range(0, count, ray_chunk):
        stop = min(start + ray_chunk, count)
        o, d = origins[start:stop], dirs[start:stop]
        dsafe = torch.where(d != 0, d, torch.ones_like(d))
        txy = torch.where(d[:, :2] > 0, (bounds - o[:, :2]) / dsafe[:, :2],
                          -o[:, :2] / dsafe[:, :2])
        txy = txy.masked_fill(d[:, :2] == 0, float("inf")).amin(1)
        tz0 = (hmin - o[:, 2]) / dsafe[:, 2]
        tz1 = (hmax - o[:, 2]) / dsafe[:, 2]
        zparallel = d[:, 2] == 0
        tin = torch.minimum(tz0, tz1).masked_fill(zparallel, -float("inf"))
        tout = torch.maximum(tz0, tz1).masked_fill(zparallel, float("inf"))
        valid_z = ~zparallel | ((o[:, 2] >= hmin) & (o[:, 2] <= hmax))
        lo = tin.clamp_min(eps)
        hi = torch.minimum(txy, tout)
        valid = valid_z & (hi >= lo) & torch.isfinite(hi)
        active = valid.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            continue
        oa, da, ta, tb = o[active], d[active], lo[active], hi[active]
        horizontal = torch.linalg.vector_norm(da[:, :2], dim=1)
        steps = ((tb - ta) * horizontal / (step_px * res)).ceil().long().clamp_min(1)
        ns = int(steps.max().item())
        j = torch.arange(ns + 1, device=height.device)
        times = ta[:, None] + (tb - ta)[:, None] * (j[None, :] / steps[:, None]).clamp_max(1)
        x = oa[:, 0, None] + da[:, 0, None] * times
        y = oa[:, 1, None] + da[:, 1, None] * times
        clear = oa[:, 2, None] + da[:, 2, None] * times - _height_only(height, x, y, res)
        candidates = (j[None, :] <= steps[:, None]) & (clear <= 0)
        first = torch.where(candidates, j[None, :], ns + 1).amin(1)
        found = first <= ns
        ids = found.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            continue
        f = first[ids]
        low = times[ids, (f - 1).clamp_min(0)]
        high = times[ids, f]
        of, df = oa[ids], da[ids]
        for _ in range(refine_steps):
            middle = (low + high) * .5
            xyz = of + df * middle[:, None]
            c = xyz[:, 2] - _height_only(height, xyz[:, 0], xyz[:, 1], res)
            high = torch.where(c <= 0, middle, high)
            low = torch.where(c > 0, middle, low)
        xyzlo, xyzhi = of + df * low[:, None], of + df * high[:, None]
        clo = xyzlo[:, 2] - _height_only(height, xyzlo[:, 0], xyzlo[:, 1], res)
        chi = xyzhi[:, 2] - _height_only(height, xyzhi[:, 0], xyzhi[:, 1], res)
        denom = clo - chi
        fraction = torch.where(denom.abs() > torch.finfo(height.dtype).tiny,
                               clo / denom, torch.full_like(clo, .5)).clamp(0, 1)
        root = low + fraction * (high - low)
        q = of + df * root[:, None]
        q[:, :2] = torch.minimum(q[:, :2].clamp_min(0), bounds)
        surface, normal = sample_surface(height, q[:, :2], res)
        q[:, 2] = surface
        outids = start + active[ids]
        hit[outids], points[outids], normals[outids], distances[outids] = True, q, normal, root
    return hit, points, normals, distances


@torch.no_grad()
def visible_segment(height, a, b, res, ray_chunk=8192, step_px=.5, tolerance_m=.001):
    """Sample strictly inside segments; endpoints on the surface are excluded.

    Every interval is <= step_px in horizontal Euclidean pixel distance.
    The 1 mm tolerance accommodates float32 cancellation; it is not a soft
    visibility model. Sub-step/tangent intersections can still be missed.
    """
    _check(height, res)
    if ray_chunk < 1 or step_px <= 0 or tolerance_m < 0:
        raise ValueError("invalid visibility configuration")
    a, b = _rays(height, a, b)
    visible = torch.ones(len(a), device=height.device, dtype=torch.bool)
    for start in range(0, len(a), ray_chunk):
        stop = min(start + ray_chunk, len(a))
        aa, delta = a[start:stop], b[start:stop] - a[start:stop]
        steps = (torch.linalg.vector_norm(delta[:, :2], dim=1) / (step_px * res)).ceil().long().clamp_min(1)
        ns = int(steps.max().item())
        if ns <= 1:
            continue
        j = torch.arange(1, ns, device=height.device)
        fraction = (j[None, :] / steps[:, None]).clamp_max(1)
        x = aa[:, 0, None] + delta[:, 0, None] * fraction
        y = aa[:, 1, None] + delta[:, 1, None] * fraction
        clear = aa[:, 2, None] + delta[:, 2, None] * fraction - _height_only(height, x, y, res)
        obstruction = (j[None, :] < steps[:, None]) & (clear < -tolerance_m)
        visible[start:stop] = ~obstruction.any(1)
    return visible
