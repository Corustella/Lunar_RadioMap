"""Input-only geometry on explicit metre-valued Cartesian and polar lattices."""

import math

import torch
from torch.nn import functional as F


def sample_lattice(values, xy, stride=1.0):
    """Bilinear sampling with origin 0 and the supplied physical feature stride."""
    height, width = values.shape[-2:]
    scale = xy.new_tensor(((width - 1) * stride, (height - 1) * stride))
    grid = 2 * xy / scale - 1
    return F.grid_sample(values, grid, align_corners=True, padding_mode="border")


@torch.no_grad()
def terrain_geometry(input, config):
    with torch.autocast(device_type=input.device.type, enabled=False):
        lo, hi = config["heightmap_range_m"]
        height = input[:, :1].float() * (hi - lo) + lo
        spacing = config["resolution_m_per_px"]
        padded = F.pad(height, (1, 1, 1, 1), mode="replicate")
        dx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / (2 * spacing)
        dy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / (2 * spacing)
        normal = F.normalize(torch.cat((-dx, -dy, torch.ones_like(dx)), dim=1), dim=1)
        low = F.avg_pool2d(F.pad(height, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        relief = torch.asinh(height - low)
        shape = torch.cat((input[:, :1].float(), normal[:, :2], relief), dim=1)
        tx_index = input[:, 1].flatten(1).argmax(1)
        tx_xy = torch.stack(((tx_index % 256).float(), (tx_index // 256).float()), dim=-1) * spacing
        tx_z = height.flatten(1).gather(1, tx_index[:, None]) + config["tx_height_m_agl"]
        tx_xyz = torch.cat((tx_xy, tx_z), dim=-1)
    return height, normal, relief, shape, tx_xyz


def surface_lattice(height, normal, relief, origin, stride, size, spacing, agl=0.0):
    axis = (origin + stride * torch.arange(size, device=height.device, dtype=height.dtype)) * spacing
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1)[None].expand(height.shape[0], -1, -1, -1)
    z = sample_lattice(height, xy, spacing)[:, 0]
    xyz = torch.cat((xy, (z + agl)[..., None]), dim=-1).flatten(1, 2)
    normals = sample_lattice(normal, xy, spacing).flatten(2).transpose(1, 2)
    roughness = sample_lattice(relief, xy, spacing).flatten(2).transpose(1, 2)
    return xyz, normals, roughness, xy


@torch.no_grad()
def source_geometry(height, normal, relief, tx_xyz, config):
    """Ground anchors at 4+16k px, with half-pixel source-segment sampling."""
    spacing = config["resolution_m_per_px"]
    anchors, normals, roughness, xy = surface_lattice(
        height, normal, relief, 4, 16, 16, spacing)
    incoming = anchors - tx_xyz[:, None]
    distance = incoming.norm(dim=-1).clamp_min(1e-6)
    direction = incoming / distance[..., None]
    horizontal = incoming[..., :2].norm(dim=-1)
    intervals = 2 * torch.ceil(horizontal / spacing).clamp_min(1).long()
    step = torch.arange(int(intervals.max().item()) + 1, device=height.device)
    alpha = (step[None, None] / intervals[..., None]).clamp_max(1)
    points = tx_xyz[:, None, None] + alpha[..., None] * incoming[:, :, None]
    ground = sample_lattice(height, points[..., :2], spacing)[:, 0]
    active = (step[None, None] > 0) & (step[None, None] < intervals[..., None])
    clearance = (points[..., 2] - ground).masked_fill(~active, float("inf"))
    minimum, obstruction = clearance.min(dim=-1)
    blocked = ((clearance < 0) & active).sum(-1) / active.sum(-1).clamp_min(1)
    log_scale = math.log1p(config["distance"]["max_distance_m"])
    height_span = config["heightmap_range_m"][1] - config["heightmap_range_m"][0]
    descriptors = torch.cat((
        normals,
        incoming[..., :2] / math.hypot(255 * spacing, 255 * spacing),
        incoming[..., 2:3] / height_span,
        torch.log1p(distance)[..., None] / log_scale,
        -(normals * direction).sum(-1, keepdim=True),
        torch.asinh(minimum)[..., None] / math.asinh(height_span),
        blocked[..., None],
        (obstruction / intervals)[..., None],
        roughness,
    ), dim=-1)
    reflection = direction - 2 * (direction * normals).sum(-1, keepdim=True) * normals
    return anchors, normals, reflection, distance, descriptors, xy


@torch.no_grad()
def interaction_geometry(height, anchors, normals, reflection, source_distance,
                         receivers, tx_xyz, config, samples=8):
    """Soft, coarse second-leg descriptors, never a hard visibility mask."""
    outgoing = receivers[:, :, None] - anchors[:, None]
    distance = outgoing.norm(dim=-1).clamp_min(1e-6)
    direction = outgoing / distance[..., None]
    direct = (receivers - tx_xyz[:, None]).norm(dim=-1).clamp_min(1e-6)
    alpha = torch.arange(1, samples + 1, device=height.device, dtype=height.dtype) / (samples + 1)
    xy = anchors[:, None, :, None, :2] + alpha[None, None, None, :, None] * outgoing[..., None, :2]
    batch, queries, count = distance.shape
    ground = sample_lattice(height, xy.reshape(batch, queries, count * samples, 2),
                            config["resolution_m_per_px"])[:, 0].reshape(batch, queries, count, samples)
    line_z = anchors[:, None, :, None, 2] + alpha * outgoing[..., None, 2]
    clearance = (line_z - ground).amin(-1)
    scale = math.log1p(config["distance"]["max_distance_m"])
    span = config["heightmap_range_m"][1] - config["heightmap_range_m"][0]
    excess = (source_distance[:, None] + distance - direct[..., None]).clamp_min(0)
    return torch.stack((
        torch.log1p(distance) / scale,
        (normals[:, None] * direction).sum(-1),
        (direction - reflection[:, None]).square().sum(-1) / 4,
        torch.log1p(excess) / scale,
        direction[..., 2],
        torch.asinh(clearance) / math.asinh(span),
    ), dim=-1)


class PolarLattice(torch.nn.Module):
    """Fixed 512-angle, 2 m radial grid, translated by each input TX."""

    def __init__(self, config, angles=512, step_m=2.0):
        super().__init__()
        if angles < 4 or angles % 4 or not math.isfinite(step_m) or step_m <= 0:
            raise ValueError("Polar lattice needs a positive step and an angle count divisible by four")
        spacing = config["resolution_m_per_px"]
        self.spacing = spacing
        self.diagonal = math.hypot(255 * spacing, 255 * spacing)
        self.angles, self.step_m = angles, step_m
        radius = torch.arange(math.ceil(self.diagonal / step_m) + 1).float() * step_m
        angle = torch.arange(angles).float() * (2 * math.pi / angles)
        directions = torch.stack((angle.cos(), angle.sin()), dim=-1)
        # sin(pi) / cos(pi/2) round-off must not discard boundary-aligned rays.
        directions[torch.arange(4) * (angles // 4)] = torch.tensor(
            [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        xy = directions[:, None] * radius[None, :, None]
        self.register_buffer("radius", radius, persistent=False)
        self.register_buffer("offset_xy", xy, persistent=False)
        self.rx_agl = config["rx_height_m_agl"]
        self.height_span = config["heightmap_range_m"][1] - config["heightmap_range_m"][0]

    @torch.no_grad()
    def forward(self, height, clearance, tx_xyz):
        xy = tx_xyz[:, None, None, :2] + self.offset_xy
        valid = ((xy >= 0) & (xy <= 255 * self.spacing)).all(-1)
        ground = sample_lattice(height, xy, self.spacing)[:, 0]
        f = sample_lattice(clearance.float(), xy, self.spacing)[:, 0]
        radius = self.radius[None, None]
        ground_angle = torch.atan2(ground - tx_xyz[:, None, None, 2], radius)
        active = valid & (radius > 0)
        angle = ground_angle.masked_fill(~active, -math.pi / 2)
        envelope, index = angle.cummax(dim=-1)
        horizon = F.pad(envelope[..., :-1], (1, 0), value=-math.pi / 2)
        peak_index = F.pad(index[..., :-1], (1, 0), value=0)
        peak_radius = self.radius[peak_index]
        rx_angle = torch.atan2(ground + self.rx_agl - tx_xyz[:, None, None, 2], radius)
        gap = (horizon - rx_angle).clamp_min(0) * valid
        # A viewshed descriptor in angle*metres, not an attenuation integral.
        integral = gap.cumsum(-1) * self.step_m / (math.pi * radius.clamp_min(self.step_m))
        descriptors = torch.stack((
            (ground - tx_xyz[:, None, None, 2]) / self.height_span,
            rx_angle / math.pi,
            (rx_angle - horizon) / math.pi,
            (radius / self.diagonal).expand_as(ground),
            peak_radius / radius.clamp_min(self.step_m),
            integral,
            f,
            valid.float(),
        ), dim=1)
        return xy, valid[:, None], descriptors.masked_fill(~valid[:, None], 0)

    def to_cartesian(self, values, valid, tx_xyz):
        axis = torch.arange(64, device=values.device, dtype=values.dtype) * (4 * self.spacing)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        delta = torch.stack((xx, yy), dim=-1)[None] - tx_xyz[:, None, None, :2]
        radius = delta.norm(dim=-1)
        angle_index = torch.atan2(delta[..., 1], delta[..., 0]).remainder(2 * math.pi)
        angle_index = angle_index * (self.angles / (2 * math.pi))
        grid = torch.stack((2 * radius / self.radius[-1] - 1,
                            2 * (angle_index + 1) / (self.angles + 1) - 1), dim=-1)
        # One halo row on each side makes angular bilinear sampling periodic.
        weights = valid.float()
        numerator = values * weights
        numerator = torch.cat((numerator[:, :, -1:], numerator, numerator[:, :, :1]), dim=2)
        weights = torch.cat((weights[:, :, -1:], weights, weights[:, :, :1]), dim=2)
        result = F.grid_sample(numerator, grid, align_corners=True)
        support = F.grid_sample(weights, grid, align_corners=True)
        return result / support.clamp_min(1e-6)
