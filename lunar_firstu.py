"""Standalone upstream firstU with jointly encoded H/TX/F/log-distance.

The layer layout and forward connections follow RadioUNet/modules.py; see
RadioUNet/LICENSE for the upstream MIT license. The second U-Net is neither
registered nor executed. All retained layers are trained from scratch.
"""

import torch
from torch import nn

from los_features import log_distance_feature
from radiounet import load_module


def firstu_model_config():
    return {
        "arch": "lunar_firstu", "version": 1,
        "implementation": "lunar_firstu.LunarFirstU",
        "input_size": [256, 256],
        "external_input_channels": ["H", "TX", "F"],
        "network_input_channels": ["H", "TX", "F", "rho"],
        "distance": "los_features.log_distance_feature_after_joint_augmentation",
        "backbone": {
            "source": "RadioUNet/modules.py:RadioWNet.firstU",
            "hidden_channels": [6, 40, 50, 60, 100, 100, 150, 300, 500],
            "feature_resolutions": [256, 128, 64, 64, 32, 32, 16, 8, 4],
            "input_convolutions": ["layer00", "conv_up00", "conv_up000"],
            "activation": "ReLU", "normalization": None,
            "secondU": False,
        },
        "output": {"quantity": "normalized_pathloss", "activation": "ReLU",
                   "parameterization": "direct_prediction"},
        "initialization": "upstream_scratch_then_reset_three_four_input_convolutions",
        "training": {"phase": "single", "frozen_parameters": False,
                     "uses_old_firstU": False},
    }


class LunarFirstU(nn.Module):
    """Keep the old firstU widths/skips, adding F/rho at all raw-input links.

    Do not construct RadioWNet(inputs=4): upstream would also change the first
    hidden width from 6 to 10. Construct its two-input layout, retain firstU,
    then rebuild only the three input-facing convolutions with four inputs.
    New columns use the same Conv2d scratch initialization as every old column.

    Time is O(B*sum(H_l*W_l*C_in_l*C_out_l*k_l^2)); saved training activations
    are O(B*sum(H_l*W_l*C_l)), plus O(P) parameters/optimizer state.
    """
    def __init__(self, feature_config):
        super().__init__()
        if (not isinstance(feature_config, dict) or
                feature_config.get("grid") != [256, 256] or
                "los" not in feature_config or "distance" not in feature_config):
            raise ValueError("LunarFirstU requires 256x256 LOS/distance feature_config")
        self.arch = "lunar_firstu"
        self.phase = "single"
        self.base_inputs = 2
        self.feature_config = feature_config
        self.model_config = firstu_model_config()
        upstream = load_module().RadioWNet(inputs=2, phase="firstU")
        for name, module in upstream.named_children():
            if not name.startswith("W"):
                self.add_module(name, module)
        for name in ("layer00", "conv_up00", "conv_up000"):
            sequence = getattr(self, name)
            old = sequence[0]
            sequence[0] = nn.Conv2d(
                old.in_channels + 2, old.out_channels, old.kernel_size,
                old.stride, old.padding, old.dilation, old.groups,
                old.bias is not None, old.padding_mode)

    def forward(self, input):
        if input.ndim != 4 or tuple(input.shape[1:]) != (3, 256, 256):
            raise ValueError("LunarFirstU input must be Bx3x256x256 [H, TX, F]")
        if not input.is_floating_point():
            raise ValueError("LunarFirstU requires normalized floating-point input")
        physical = torch.cat((input, log_distance_feature(input, self.feature_config)), dim=1)
        layer00 = self.layer00(physical)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer2 = self.layer2(layer10)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        layer4u = torch.cat((self.conv_up5(layer5), layer4), dim=1)
        layer3u = torch.cat((self.conv_up4(layer4u), layer3), dim=1)
        layer20u = torch.cat((self.conv_up3(layer3u), layer20), dim=1)
        layer2u = torch.cat((self.conv_up20(layer20u), layer2), dim=1)
        layer10u = torch.cat((self.conv_up2(layer2u), layer10), dim=1)
        layer1u = torch.cat((self.conv_up10(layer10u), layer1), dim=1)
        layer0u = torch.cat((self.conv_up1(layer1u), layer0), dim=1)
        layer00u = torch.cat((self.conv_up0(layer0u), layer00, physical), dim=1)
        layer000u = torch.cat((self.conv_up00(layer00u), physical), dim=1)
        return [self.conv_up000(layer000u)]
