"""Single-stage MaxViT encoder and convolution decoder for lunar radio maps.

This is a lunar-input adaptation inspired by RMTransformer, not a reproduction
of its wider encoder or its urban-data training recipe. The torchvision MaxViT
blocks retain their official normalization, activation and attention mechanisms.
Only the RGB stem, partition size and classification head are adapted; local
full-resolution skips preserve terrain and TX locations on the 256x256 grid.
"""

import torch
from torch import nn
from torchvision.models.maxvit import MaxVit

from los_features import log_distance_feature


def lunar_model_config():
    """Complete fixed architecture description, also saved in checkpoints."""
    return {
        "arch": "lunar_maxvit", "version": 1,
        "input_size": [256, 256],
        "external_input_channels": ["H", "TX", "F"],
        "network_input_channels": ["H", "TX", "F", "rho"],
        "distance": "los_features.log_distance_feature_after_joint_augmentation",
        "encoder": {
            "implementation": "torchvision.models.maxvit.MaxVit",
            "stem_channels": 64, "stem_input_channels": 4,
            "stem_weight_initialization": "normal_std_0.02_bias_zero",
            "block_channels": [64, 128, 256, 512],
            "block_layers": [2, 2, 5, 2], "head_dim": 32,
            "partition_size": 8, "stochastic_depth_prob": 0.2,
            "squeeze_ratio": 0.25, "expansion_ratio": 4,
            "mlp_ratio": 4, "mlp_dropout": 0.0,
            "attention_dropout": 0.0,
            "conv_norm": {"name": "BatchNorm2d", "eps": 0.001,
                          "momentum": 0.01},
            "attention_norm": "LayerNorm_torchvision_default",
            "activation": "GELU", "classifier": "Identity",
            "feature_resolutions": [128, 64, 32, 16, 8],
            "pretrained": False,
        },
        "full_resolution_branch": {
            "channels": [4, 16, 16], "kernel_size": 3, "padding": 1,
            "activation": "ReLU", "normalization": None,
        },
        "decoder": {
            "output_channels": [256, 128, 64, 32, 16],
            "skip_channels": [256, 128, 64, 64, 16],
            "output_resolutions": [16, 32, 64, 128, 256],
            "upsample": {"name": "ConvTranspose2d", "kernel_size": 4,
                         "stride": 2, "padding": 1,
                         "activation": "ReLU"},
            "fusion": "concat_then_two_3x3_convs_padding1",
            "activation": "ReLU", "normalization": None,
            "initialization": "torch_default",
        },
        "output": {"raw_input_skip": True, "kernel_size": 3,
                   "padding": 1, "channels": 1, "activation": "ReLU",
                   "quantity": "normalized_pathloss",
                   "parameterization": "direct_prediction",
                   "initialization": {"weight": "normal_std_0.001",
                                      "bias": 0.5}},
        "training": {"phase": "single", "frozen_parameters": False,
                     "uses_old_firstU": False},
    }


def _double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fuse = _double_conv(out_channels + skip_channels, out_channels)

    def forward(self, input, skip):
        return self.fuse(torch.cat((self.up(input), skip), dim=1))


class LunarMaxViT(nn.Module):
    """Jointly trained, full-map predictor using only test-available inputs.

    F is the existing cached clearance proxy, transformed jointly with H/TX.
    The inexpensive rho proxy is recomputed from those transformed H/TX inputs;
    neither a target, validity mask nor a previous network prediction is used.

    For fixed stage widths and depth, local/grid attention has O(B*N*P) token
    interactions rather than O(B*N*N), with P=8*8 tokens per partition. Channel
    projections/convolutions add O(B*sum(N_s*C_s^2)); attention working memory
    is O(B*sum(N_s*heads_s*P)). Training also retains layer/decoder activations.
    """

    def __init__(self, feature_config):
        super().__init__()
        if not isinstance(feature_config, dict):
            raise ValueError("LunarMaxViT requires LOS/distance feature_config")
        if feature_config.get("grid") != [256, 256]:
            raise ValueError("LunarMaxViT requires the challenge's 256x256 grid")
        if "los" not in feature_config or "distance" not in feature_config:
            raise ValueError("LunarMaxViT requires both LOS and distance configuration")

        self.arch = "lunar_maxvit"
        self.phase = "single"
        self.base_inputs = 2
        self.feature_config = feature_config
        self.model_config = lunar_model_config()
        self.encoder = MaxVit(
            input_size=(256, 256), stem_channels=64, partition_size=8,
            block_channels=[64, 128, 256, 512], block_layers=[2, 2, 5, 2],
            head_dim=32, stochastic_depth_prob=0.2,
            squeeze_ratio=0.25, expansion_ratio=4, mlp_ratio=4,
            mlp_dropout=0.0, attention_dropout=0.0,
        )
        # Do not resize the physical grid to a classifier's 224x224 default.
        # All four channels learn together from scratch; no RGB weights are used.
        old = self.encoder.stem[0][0]
        stem = nn.Conv2d(4, old.out_channels, old.kernel_size,
                         stride=old.stride, padding=old.padding, bias=False)
        nn.init.normal_(stem.weight, std=0.02)
        self.encoder.stem[0][0] = stem
        self.encoder.classifier = nn.Identity()

        self.full_resolution = _double_conv(4, 16)
        self.decoder = nn.ModuleList([
            _DecoderBlock(512, 256, 256),
            _DecoderBlock(256, 128, 128),
            _DecoderBlock(128, 64, 64),
            _DecoderBlock(64, 64, 32),
            _DecoderBlock(32, 16, 16),
        ])
        self.output = nn.Sequential(nn.Conv2d(16 + 4, 1, 3, padding=1),
                                    nn.ReLU(inplace=True))
        # Keep the baseline ReLU output active at initialization. Small nonzero
        # weights also let gradients reach the encoder on the first update.
        nn.init.normal_(self.output[0].weight, std=0.001)
        nn.init.constant_(self.output[0].bias, 0.5)

    def forward(self, input):
        if input.ndim != 4 or tuple(input.shape[1:]) != (3, 256, 256):
            raise ValueError("LunarMaxViT input must be Bx3x256x256 [H, TX, F]")
        if not input.is_floating_point():
            raise ValueError("LunarMaxViT requires normalized floating-point input")
        distance = log_distance_feature(input, self.feature_config)
        physical = torch.cat((input, distance), dim=1)
        features = [self.full_resolution(physical), self.encoder.stem(physical)]
        encoded = features[-1]
        for block in self.encoder.blocks:
            encoded = block(encoded)
            features.append(encoded)
        # Skips are, in order, block3/block2/block1/stem/full-resolution.
        for decoder, skip in zip(self.decoder, reversed(features[:-1])):
            encoded = decoder(encoded, skip)
        prediction = self.output(torch.cat((encoded, physical), dim=1))
        return [prediction]
