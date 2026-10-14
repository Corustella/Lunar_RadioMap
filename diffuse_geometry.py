"""Deterministic ray intersections with the input bilinear height field.

This reconstructs a surface from the raster; it is not the original Sionna
triangle mesh. Coordinates are metres, with x along columns and y along rows.
No target, validity mask, or simulator-generated component is used.
"""

import math

import torch


def _check_height(height, resolution):
    if height.ndim != 2 or min(height.shape) < 2:
        raise ValueError("height must have shape [H,W], with H,W >= 2")
    if height.dtype not in (torch.float32, torch.float64):
        raise ValueError("height must be float32 or float64")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive and finite")
    if not torch.isfinite(height).all():
        raise ValueError("height contains non-finite values")


def _cell_coefficients(height, row, col):
    a = height[row, col]
    b = height[row, col + 1] - a
    c = height[row + 1, col] - a
    d = height[row + 1, col + 1] - a - b - c
    return a, b, c, d


@torch.no_grad()
def terrain_height_normal(height, xy, resolution):
    """Evaluate the bilinear DEM and its upward unit normal, without extension.

    At internal grid lines the derivative is not unique. The mean of adjacent
    one-sided derivatives defines a symmetric convention, including at vertices;
    domain edges use the only available side. This convention respects D4.
    """
    _check_height(height, resolution)
    xy = torch.as_tensor(xy, dtype=height.dtype, device=height.device)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape [M,2]")
    h, w = height.shape
    if (not torch.isfinite(xy).all() or (xy < 0).any() or
            (xy[:, 0] > (w - 1) * resolution).any() or
            (xy[:, 1] > (h - 1) * resolution).any()):
        raise ValueError("xy must lie inside the finite DEM domain")
    x, y = xy[:, 0] / resolution, xy[:, 1] / resolution
    right = x.floor().long().clamp(0, w - 2)
    top = y.floor().long().clamp(0, h - 2)
    left = (x.ceil().long() - 1).clamp(0, w - 2)
    bottom = (y.ceil().long() - 1).clamp(0, h - 2)
    a, b, c, d = _cell_coefficients(height, top, right)
    u, v = x - right, y - top
    z = a + b * u + c * v + d * u * v
    dx, dy = torch.zeros_like(z), torch.zeros_like(z)
    for row, col in ((top, right), (top, left), (bottom, right), (bottom, left)):
        _, b, c, d = _cell_coefficients(height, row, col)
        dx += (b + d * (y - row)) / (4 * resolution)
        dy += (c + d * (x - col)) / (4 * resolution)
    normal = torch.stack((-dx, -dy, torch.ones_like(z)), dim=1)
    normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    return z, normal


@torch.no_grad()
def trace_terrain(height, origins, directions, resolution, max_distance=None,
                  epsilon_m=1e-4):
    """Return the first terrain contact at distance >= epsilon_m for each ray.

    Rays have unit directions, start in the XY domain on/above the surface, and
    stop at the finite domain boundary or max_distance. Within each crossed
    cell, ray/surface contacts solve the exact bilinear quadratic; there is no
    fixed-step marching. Tangencies and coplanar contacts count as blocked.
    A starting contact is excluded only by epsilon_m, never by a pixel-sized
    origin shift. Use float64 height/origins for reliable grazing/self-contact
    calculations at large terrain elevations.

    Misses have finite zero points/normals and infinite distance. Worst-case
    time is O(M(H+W)); additional working memory is O(M), apart from the DEM.
    Chunking is the caller's responsibility.
    """
    _check_height(height, resolution)
    if not math.isfinite(epsilon_m) or epsilon_m <= 0:
        raise ValueError("epsilon_m must be positive and finite")
    origins = torch.as_tensor(origins, dtype=height.dtype, device=height.device)
    directions = torch.as_tensor(directions, dtype=height.dtype, device=height.device)
    if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
        raise ValueError("origins and directions must both have shape [M,3]")
    if not torch.isfinite(origins).all() or not torch.isfinite(directions).all():
        raise ValueError("ray coordinates must be finite")
    if not torch.allclose(torch.linalg.vector_norm(directions, dim=1),
                          torch.ones(len(origins), dtype=height.dtype, device=height.device),
                          rtol=1e-5, atol=1e-6):
        raise ValueError("directions must have unit norm")
    h, w = height.shape
    floor, _ = terrain_height_normal(height, origins[:, :2], resolution)
    if (origins[:, 2] < floor - epsilon_m).any():
        raise ValueError("ray origins must be on or above the DEM")
    size = len(origins)
    hit = torch.zeros(size, dtype=torch.bool, device=height.device)
    points, normals = torch.zeros_like(origins), torch.zeros_like(origins)
    distance = torch.full((size,), float("inf"), dtype=height.dtype, device=height.device)
    if not size:
        return hit, points, normals, distance
    ox, oy = origins[:, 0], origins[:, 1]
    dx, dy = directions[:, 0], directions[:, 1]
    safe_dx = torch.where(dx != 0, dx, torch.ones_like(dx))
    safe_dy = torch.where(dy != 0, dy, torch.ones_like(dy))
    exit_x = torch.where(dx > 0, ((w - 1) * resolution - ox) / safe_dx,
                         -ox / safe_dx)
    exit_y = torch.where(dy > 0, ((h - 1) * resolution - oy) / safe_dy,
                         -oy / safe_dy)
    exit_x = exit_x.masked_fill(dx == 0, float("inf"))
    exit_y = exit_y.masked_fill(dy == 0, float("inf"))
    limit = torch.minimum(exit_x, exit_y)
    if max_distance is not None:
        maximum = torch.as_tensor(max_distance, dtype=height.dtype, device=height.device)
        if maximum.ndim == 0:
            maximum = maximum.expand(size)
        if maximum.shape != (size,) or torch.isnan(maximum).any() or (maximum < 0).any():
            raise ValueError("max_distance must be non-negative scalar or [M]")
        limit = torch.minimum(limit, maximum)
    current = torch.full_like(limit, epsilon_m)
    xy = origins[:, :2] + current[:, None] * directions[:, :2]
    col = (xy[:, 0] / resolution).floor().long().clamp(0, w - 2)
    row = (xy[:, 1] / resolution).floor().long().clamp(0, h - 2)
    active = limit >= current
    machine_epsilon = torch.finfo(height.dtype).eps
    for _ in range(h + w + 4):
        ids = active.nonzero(as_tuple=True)[0]
        if not len(ids):
            break
        start = current[ids]
        origin, direction = origins[ids], directions[ids]
        cx, cy = col[ids], row[ids]
        vx, vy, vz = direction.unbind(1)
        end_x = torch.where(vx > 0, (cx + 1) * resolution, cx * resolution)
        end_y = torch.where(vy > 0, (cy + 1) * resolution, cy * resolution)
        tx = (end_x - origin[:, 0]) / torch.where(vx != 0, vx, torch.ones_like(vx))
        ty = (end_y - origin[:, 1]) / torch.where(vy != 0, vy, torch.ones_like(vy))
        tx = tx.masked_fill(vx == 0, float("inf"))
        ty = ty.masked_fill(vy == 0, float("inf"))
        end = torch.minimum(torch.minimum(tx, ty), limit[ids]).clamp_min(start)
        extent = end - start
        p0 = origin + start[:, None] * direction
        u, v = p0[:, 0] / resolution - cx, p0[:, 1] / resolution - cy
        a, b, c, d = _cell_coefficients(height, cy, cx)
        ux, vy_scaled = vx / resolution, vy / resolution
        qa = -d * ux * vy_scaled
        qb = vz - b * ux - c * vy_scaled - d * (u * vy_scaled + v * ux)
        qc = p0[:, 2] - a - b * u - c * v - d * u * v
        discriminant = qb.square() - 4 * qa * qc
        disc_tolerance = 16 * machine_epsilon * (qb.square() + (4 * qa * qc).abs())
        real = discriminant >= -disc_tolerance
        square_root = discriminant.clamp_min(0).sqrt()
        q = -0.5 * (qb + torch.copysign(square_root, qb))
        safe_a = torch.where(qa != 0, qa, torch.ones_like(qa))
        safe_b = torch.where(qb != 0, qb, torch.ones_like(qb))
        safe_q = torch.where(q != 0, q, torch.ones_like(q))
        root1 = q / safe_a
        root2 = qc / safe_q
        root1 = torch.where(qa == 0, -qc / safe_b, root1)
        root2 = torch.where(qa == 0, torch.full_like(root2, float("inf")), root2)
        root1 = torch.where((qa != 0) & (q == 0), -qb / (2 * safe_a), root1)
        root2 = torch.where((qa != 0) & (q == 0), root1, root2)
        permitted = real & ((qa != 0) | (qb != 0))
        time_tolerance = 16 * machine_epsilon * (start.abs() + 1)
        roots = torch.stack((root1, root2), dim=1)
        valid = (permitted[:, None] & (roots >= -time_tolerance[:, None]) &
                 (roots <= extent[:, None] + time_tolerance[:, None]))
        roots = roots.masked_fill(~valid, float("inf"))
        first = roots.min(1).values
        z_tolerance = 16 * machine_epsilon * (p0[:, 2].abs() + a.abs() + 1)
        below = qc < -z_tolerance
        coplanar = (qa == 0) & (qb == 0) & (qc.abs() <= z_tolerance)
        first = torch.where(below | coplanar, torch.zeros_like(first), first)
        found = torch.isfinite(first)
        found_ids = ids[found]
        distance[found_ids] = start[found] + first[found].clamp_min(0)
        hit[found_ids] = True
        active[found_ids] = False
        proceed = ~found & (end < limit[ids])
        remaining = ids[proceed]
        active[ids[~found & ~proceed]] = False
        cross_x = tx[proceed] <= end[proceed] + time_tolerance[proceed]
        cross_y = ty[proceed] <= end[proceed] + time_tolerance[proceed]
        col[remaining] += cross_x.long() * vx[proceed].sign().long()
        row[remaining] += cross_y.long() * vy[proceed].sign().long()
        current[remaining] = end[proceed]
        inside = ((col[remaining] >= 0) & (col[remaining] < w - 1) &
                  (row[remaining] >= 0) & (row[remaining] < h - 1))
        active[remaining[~inside]] = False
    if active.any():
        raise RuntimeError("terrain DDA failed to leave a cell within H+W steps")
    points[hit] = origins[hit] + distance[hit, None] * directions[hit]
    points[hit, 0] = points[hit, 0].clamp(0, (w - 1) * resolution)
    points[hit, 1] = points[hit, 1].clamp(0, (h - 1) * resolution)
    if hit.any():
        _, normals[hit] = terrain_height_normal(height, points[hit, :2], resolution)
    return hit, points, normals, distance
