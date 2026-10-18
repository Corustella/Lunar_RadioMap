"""One TX-oriented sampling residual in the LOS+distance RadioWNet refinement.

This is a research operator, not a ray tracer or a reproduction of DCN.
The original convolution and all other upstream layers remain intact.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from radiounet import load_module


def direction_model_config():
    return {
        "arch": "radiownet_tx_direction", "version": 1,
        "base": "RadioUNet.modules.RadioWNet", "phase": "secondU",
        "first_features": "los", "second_features": "los_distance",
        "operator": {
            "layer": "Wlayer20.0", "resolution": [32, 32],
            "in_channels": 60, "out_channels": 70,
            "base_convolution": "unchanged_3x3_padding1",
            "feature_stride_px": 8, "feature_origin_px": 3.5,
            "tx": "jointly_augmented_one_hot_argmax_subpixel_feature_coordinates",
            "sampling": "3_radial_positions_by_3_tangential_positions",
            "radii": "2+1.5*tanh(raw);two_global_learned_feature_pixel_scales",
            "initial_radii": [2.0, 2.0], "radius_bounds": [0.5, 3.5],
            "tangential_pair": "average_plus_minus_then_shared_depthwise_weight",
            "depthwise_weights": [3, 2, 60], "projection": "1x1_60_to_70_no_bias",
            "fusion": "add_before_original_ReLU",
            "interpolation": "bilinear_align_corners_border",
            "coincident_tx": "zero_radial_and_tangent_directions",
            "direction_epsilon": 1e-6,
            "computation_dtype": "float32_under_amp_update_cast_to_base_dtype",
            "initialization": "normal_std_1_over_sqrt6_depthwise_zero_projection_zero_raw_radii",
            "rng": "fork_cpu_after_shared_initialization",
            "geometry_scope": "sampling_branch_D4_covariant_not_full_network_equivariant",
        },
        "training": {"firstU_frozen": True, "secondU_trainable": True,
                     "output": "unchanged_direct_ReLU_prediction"},
    }


class TXDirectionalConv(nn.Conv2d):
    """Preserve a base Conv2d and add depthwise TX-frame sampling + projection.

    Time O(BHW(K*Cin + Cin*Cout)), activation space O(BHW*K*Cin), K=9.
    The two tangential samples in each radial row share a coefficient through
    averaging. Reflection flips their order without changing the branch.
    """
    def __init__(self, base):
        super().__init__(base.in_channels, base.out_channels, base.kernel_size,
                         stride=base.stride, padding=base.padding,
                         dilation=base.dilation, groups=base.groups,
                         bias=base.bias is not None, device="meta",
                         dtype=base.weight.dtype)
        # Preserve legacy parameter names and objects for exact initialization.
        self.weight = base.weight
        self.bias = base.bias
        self.direction_weights = nn.Parameter(torch.empty(3, 2, base.in_channels))
        nn.init.normal_(self.direction_weights, std=1 / math.sqrt(6))
        self.direction_radii = nn.Parameter(torch.zeros(2))
        self.direction_projection = nn.Conv2d(base.in_channels, base.out_channels, 1, bias=False)
        nn.init.zeros_(self.direction_projection.weight)
        self._tx_xy = None

    def sample_grid(self, features, tx_xy):
        """tx_xy is in feature coordinates; preserve fractional TX positions."""
        batch, _, height, width = features.shape
        if (height, width) != (32, 32) or tuple(tx_xy.shape) != (batch, 2):
            raise ValueError("TX sampling requires 32x32 features and Bx2 source coordinates")
        axis = torch.arange(32, device=features.device, dtype=features.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        centers = torch.stack((xx, yy), -1)[None]
        delta = centers - tx_xy.to(features)[:, None, None, :]
        radial = delta / delta.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
        tangent = torch.stack((-radial[..., 1], radial[..., 0]), -1)
        radii = 2 + 1.5 * self.direction_radii.tanh()
        offsets = features.new_tensor([-1, 0, 1])
        points = (centers[:, :, :, None, None, :]
                  + offsets[None, None, None, :, None, None] * radii[0]
                  * radial[:, :, :, None, None, :]
                  + offsets[None, None, None, None, :, None] * radii[1]
                  * tangent[:, :, :, None, None, :])
        return (2 * points / 31 - 1).reshape(batch, 32, 32 * 9, 2)

    def direction_update(self, features, tx_xy):
        with torch.autocast(device_type=features.device.type, enabled=False):
            work = features.float() if features.dtype != torch.float64 else features
            grid = self.sample_grid(work, tx_xy)
            sampled = F.grid_sample(work, grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)
            sampled = sampled.reshape(work.shape[0], work.shape[1], 32, 32, 3, 3)
            center = sampled[..., 1]
            pair = (sampled[..., 0] + sampled[..., 2]) * 0.5
            values = torch.stack((center, pair), -1)
            coefficients = self.direction_weights.permute(2, 0, 1)[None, :, None, None]
            aggregated = (values * coefficients).sum((-1, -2))
            return self.direction_projection(aggregated)

    def forward(self, features):
        if self._tx_xy is None:
            raise RuntimeError("TXDirectionalConv must run inside TXDirectionalRadioWNet")
        base = super().forward(features)
        update = self.direction_update(features, self._tx_xy)
        return base + update.to(base.dtype)


class TXDirectionalRadioWNet(load_module().RadioWNet):
    """Reuse the upstream forward and scope source coordinates to one call."""
    def __init__(self, inputs=2, phase="secondU"):
        if inputs != 2 or phase != "secondU":
            raise ValueError("TX direction supports a single-band secondU only")
        super().__init__(inputs=inputs, phase=phase)
        with torch.random.fork_rng(devices=[]):
            self.Wlayer20[0] = TXDirectionalConv(self.Wlayer20[0])

    def forward(self, input):
        if self.phase != "secondU":
            raise ValueError("TX direction requires the existing secondU training phase")
        if input.ndim != 4 or tuple(input.shape[1:]) != (5, 256, 256):
            raise ValueError("Expected expanded [H, TX, F, rho, 0] input")
        source = input[:, 1].flatten(1).argmax(1)
        # Three stride-2 max pools: centers 3.5, 11.5, ..., 251.5 in input pixels.
        tx_xy = torch.stack((source % 256, torch.div(source, 256, rounding_mode="floor")), -1)
        operator = self.Wlayer20[0]
        operator._tx_xy = (tx_xy.float() - 3.5) / 8
        try:
            return super().forward(input)
        finally:
            # No source tensor is stored in state_dict or retained across batches.
            operator._tx_xy = None
