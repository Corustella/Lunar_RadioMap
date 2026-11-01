"""Input-only region priors computed after joint H/TX/F augmentation.

No RT, targets, validity masks, SciPy, extra cache or CPU transfer. The signed
distance is a TRUNCATED CHEBYSHEV grid distance, not Euclidean distance or an
official visibility label. Its finite band is deliberate and versioned.
"""

import torch
from torch.nn import functional as F


def region_feature_config():
    return {
        "version": 1,
        "visible": "F>=0; sign_of_existing_minimum_clearance_proxy",
        "distance": "signed_Chebyshev_to_opposite_class_centres_minus_half_pixel",
        "distance_band_px": 16,
        "distance_clip_px": 15.5,
        "image_exterior": "ignored; never_treated_as_opposite_class",
        "uniform_class": "constant_signed_clip_distance; no_fake_boundary",
        "distance_units": "metres; multiplied_by_resolution_m_per_px",
        "normal": "height_central_difference_replicate_border_upward_unit_normal",
        "incidence": "signed_dot_RX_to_TX_unit_direction_and_grid_surface_normal",
        "channels": ["signed_distance_over_clip", "normal_x", "normal_y",
                     "normal_z", "normal_dot_RX_to_TX"],
        "computation": "FP32_pure_torch_after_joint_D4; input_only_no_autograd",
    }


@torch.no_grad()
def signed_grid_distance(visible, band_px=16):
    """Half-cell signed chessboard distance, saturated at band_px-0.5.

    Repeated 3x3 erosion counts distance to the other class in the L-infinity
    metric, exactly inside the band. Max-pool's -inf padding ignores the exterior,
    so a uniform map stays uniform and image edges do not become LOS boundaries.
    Time O(band_px*BHW), working memory O(BHW). No device/host synchronization.
    """
    if visible.ndim != 4 or visible.shape[1] != 1 or visible.dtype != torch.bool:
        raise ValueError("Expected boolean Bx1xHxW visibility proxy")
    if not isinstance(band_px, int) or band_px < 1:
        raise ValueError("band_px must be a positive integer")
    regions = torch.cat((visible, ~visible), dim=1).float()
    distance = torch.zeros_like(regions)
    for step in range(band_px):
        distance = distance + regions
        if step + 1 < band_px:
            regions = -F.max_pool2d(-regions, 3, stride=1, padding=1)
    return distance[:, :1] - distance[:, 1:] - (visible.float() - .5)


@torch.no_grad()
def region_features(input, config):
    """Return signed distance in metres and five normalized geometric channels.

    F has the original odd asinh(clearance) encoding, so its zero is the zero
    clearance threshold. Normal/angle channels are local geometry proxies, not
    normals at unknown scattering points. Only the first three input channels are
    accepted, preventing target/mask/RT arrays being routed in accidentally.
    """
    if input.ndim != 4 or tuple(input.shape[1:]) != (3, 256, 256):
        raise ValueError("Region features require Bx3x256x256 [H,TX,F]")
    if not input.is_floating_point():
        raise ValueError("Region features require floating-point input")
    settings = region_feature_config()
    with torch.autocast(device_type=input.device.type, enabled=False):
        work = input.float()
        resolution = config["resolution_m_per_px"]
        distance = signed_grid_distance(work[:, 2:3] >= 0,
                                        settings["distance_band_px"]) * resolution
        lo, hi = config["heightmap_range_m"]
        height = work[:, :1] * (hi - lo) + lo
        padded = F.pad(height, (1, 1, 1, 1), mode="replicate")
        dx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / (2 * resolution)
        dy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / (2 * resolution)
        normal = torch.cat((-dx, -dy, torch.ones_like(dx)), dim=1)
        normal = F.normalize(normal, dim=1)
        tx = work[:, 1:2].flatten(1).argmax(1)
        tx_row = torch.div(tx, 256, rounding_mode="floor").float()[:, None, None, None]
        tx_col = (tx % 256).float()[:, None, None, None]
        z_tx = height.flatten(1).gather(1, tx[:, None])[:, :, None, None]
        z_tx = z_tx + config["tx_height_m_agl"]
        rows = torch.arange(256, dtype=torch.float32, device=input.device)[None, None, :, None]
        cols = torch.arange(256, dtype=torch.float32, device=input.device)[None, None, None, :]
        toward_tx = torch.cat(((tx_col - cols).expand_as(height) * resolution,
                               (tx_row - rows).expand_as(height) * resolution,
                               z_tx - height - config["rx_height_m_agl"]), dim=1)
        toward_tx = F.normalize(toward_tx, dim=1)
        incidence = (normal * toward_tx).sum(dim=1, keepdim=True)
        encoded = torch.cat((distance / (settings["distance_clip_px"] * resolution),
                             normal, incidence), dim=1)
    return distance, encoded
