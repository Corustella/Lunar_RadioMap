"""Input-only physical coordinates for source-conditioned latent transport."""

import torch
from torch.nn import functional as F

from los_features import log_distance_feature


def transport_feature_config():
    return {
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


@torch.no_grad()
def transport_features(input, config):
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
        geometry = torch.cat((dx / diagonal, dy / diagonal, dz / height_range, normal,
                              (normal * unit_to_tx).sum(1, keepdim=True), input[:, 2:3].float(), rho), dim=1)
        physical = torch.cat((input.float(), rho), dim=1)
    return physical.to(input.dtype), geometry, tx_index
