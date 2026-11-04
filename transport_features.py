"""Input-only physical coordinates for source-conditioned latent transport."""

import torch
from torch.nn import functional as F

from los_features import log_distance_feature


def transport_feature_config(polar=False):
    config = {
        "version": 1, "external_channels": ["H", "TX", "F"],
        "physical_channels": ["H", "TX", "F", "rho"],
        "geometry_channels": ["dx/diagonal", "dy/diagonal", "dz/height_range",
                              "nx", "ny", "nz", "normal_dot_RX_to_TX", "F", "rho"],
        "normal": "central_difference_metre_height_replicate_border_upward",
        "coordinates": "x=column,y=row; state/RX height=H+rx_agl; TX=H(TX)+tx_agl",
        "state_grid": "16x16 at input centres 0,16,...,240; no endpoint resizing",
        "query_grid": "64x64 at input centres 0,4,...,252",
        "computation": "FP32_no_autograd_after_joint_augmentation; no_RT_target_or_mask",
    }
    if polar:
        config.update(version=2, coordinate_frame="TX_cylindrical_polar",
                      geometry_channels=["r/diagonal", "sin_theta", "cos_theta", "dz/height_range",
                                         "normal_r", "normal_theta", "nz", "normal_dot_RX_to_TX", "F", "rho"],
                      polar_origin="r=0: sin_theta=0, cos_theta=1; basis er=+x, etheta=+y",
                      polar_basis="er=(cos_theta,sin_theta,0); etheta=(-sin_theta,cos_theta,0)",
                      sampling="unchanged Cartesian state/query/input/output grids; no polar image resampling")
    return config


def polar_direction_codes(directions, geometry):
    """Encode the original world-XY slots in each site's cylindrical basis."""
    sine, cosine = geometry[..., 1:2], geometry[..., 2:3]
    x, y = directions[..., 0], directions[..., 1]
    return torch.stack((cosine * x + sine * y, -sine * x + cosine * y), dim=-1)


@torch.no_grad()
def transport_features(input, config, polar=False):
    """O(BHW) coordinates/geometry, recomputed after the loader's joint D4."""
    with torch.autocast(device_type=input.device.type, enabled=False):
        height_lo, height_hi = config["heightmap_range_m"]
        height_range = height_hi - height_lo
        height = input[:, :1].float() * height_range + height_lo
        batch, _, h, w = height.shape
        resolution = config["resolution_m_per_px"]
        tx_index = input[:, 1].flatten(1).argmax(1)
        tx_col = (tx_index % w).float()[:, None, None, None]
        tx_row = (tx_index // w).float()[:, None, None, None]
        tx_z = height.flatten(2).gather(2, tx_index[:, None, None])[:, :, :, None]
        tx_z = tx_z + config["tx_height_m_agl"]
        cols = torch.arange(w, device=input.device, dtype=torch.float32)[None, None, None, :]
        rows = torch.arange(h, device=input.device, dtype=torch.float32)[None, None, :, None]
        dx = ((cols - tx_col) * resolution).expand(batch, 1, h, w)
        dy = ((rows - tx_row) * resolution).expand(batch, 1, h, w)
        dz = height + config["rx_height_m_agl"] - tx_z
        relative = torch.cat((dx, dy, dz), dim=1)
        unit_to_tx = -relative / relative.square().sum(1, keepdim=True).clamp_min(1e-12).sqrt()
        padded = F.pad(height, (1, 1, 1, 1), mode="replicate")
        gx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / (2 * resolution)
        gy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / (2 * resolution)
        normal = F.normalize(torch.cat((-gx, -gy, torch.ones_like(gx)), dim=1), dim=1)
        rho = log_distance_feature(input, config).float()
        diagonal = (((h - 1) * resolution) ** 2 + ((w - 1) * resolution) ** 2) ** .5
        incidence = (normal * unit_to_tx).sum(1, keepdim=True)
        if polar:
            radius = (dx.square() + dy.square()).sqrt()
            cosine = torch.where(radius > 0, dx / radius.clamp_min(1e-12), torch.ones_like(radius))
            sine = torch.where(radius > 0, dy / radius.clamp_min(1e-12), torch.zeros_like(radius))
            nx, ny, nz = normal.split(1, dim=1)
            radial = nx * cosine + ny * sine
            tangent = -nx * sine + ny * cosine
            geometry = torch.cat((radius / diagonal, sine, cosine, dz / height_range,
                                  radial, tangent, nz, incidence, input[:, 2:3].float(), rho), dim=1)
        else:
            geometry = torch.cat((dx / diagonal, dy / diagonal, dz / height_range, normal,
                                  incidence, input[:, 2:3].float(), rho), dim=1)
        physical = torch.cat((input.float(), rho), dim=1)
    return physical.to(input.dtype), geometry, tx_index
