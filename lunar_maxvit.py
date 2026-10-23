"""Single-stage MaxViT encoder and convolution decoder for lunar radio maps.

This is a lunar-input adaptation inspired by RMTransformer, not a reproduction
of its wider encoder or its urban-data training recipe. The torchvision MaxViT
blocks retain their official normalization, activation and attention mechanisms.
Only the RGB stem, partition size and classification head are adapted; local
full-resolution skips preserve terrain and TX locations on the 256x256 grid.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.maxvit import MaxVit

from los_features import log_distance_feature


def lunar_model_config(arch="lunar_maxvit"):
    """Complete fixed architecture description, also saved in checkpoints."""
    if arch not in ("lunar_maxvit", "lunar_maxvit_fade", "lunar_maxvit_c"):
        raise ValueError(f"Unsupported lunar architecture {arch!r}")
    config = {
        "arch": arch, "version": 1,
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
    if arch == "lunar_maxvit_fade":
        config["decoder"]["upsample"] = {
            "first_two_stages": {"name": "ConvTranspose2d", "kernel_size": 4,
                                 "stride": 2, "padding": 1, "activation": "ReLU"},
            "last_three_stages": {
                "name": "FADE_L2H", "output_resolutions": [64, 128, 256],
                "guidance_sources": ["encoder_block1", "encoder_stem", "full_resolution_branch"],
                "output_channels": [64, 32, 16],
                "scale": 2, "reassembly_kernel_size": 5,
                "content_kernel_size": 3, "embedding_dim": 64,
                "coarse_projection": "Conv2d_1x1_to_output_channels",
                "guide_projection": "Conv2d_1x1_to_output_channels",
                "kernel_generator": "joint_semi_shift_softmax",
                "gate": "sigmoid_of_coarse_1x1_then_nearest_upsample",
                "fusion": "gate*guide+(1-gate)*reassembled_coarse",
                "padding": "zero", "implementation": "pure_torch_bounded_memory",
                "initialization": "FADE_xavier_uniform_bias_zero",
            },
        }
        config["decoder"]["fusion"] = (
            "first_two_concat_then_double_conv;last_three_FADE_then_double_conv")
    if arch == "lunar_maxvit_c":
        config["tx_scene_attention"] = {
            "encoder_block_index": 1, "resolution": [32, 32], "channels": 128,
            "anchors": {"operation": "avg_pool2d", "kernel_size": 2,
                        "stride": 2, "resolution": [16, 16]},
            "source": {"input": "augmented_TX_one_hot_argmax",
                       "sampling": "bilinear", "feature_stride": 8,
                       "feature_origin_px": 0, "align_corners": True,
                       "padding_mode": "border", "computation_dtype": "float32"},
            "qkv": "independent_linear_concat_feature_and_TX_feature",
            "attention_dim": 64, "heads": 4, "linear_bias": True,
            "scale": "inverse_sqrt_head_dim", "dropout": 0.0,
            "geometry_bias": False, "attention_mask": None,
            "implementation": "torch_scaled_dot_product_attention",
            "computation_dtype": "float32_under_amp_output_cast_to_feature_dtype",
            "fusion": "residual_before_next_encoder_block_and_skip",
            "initialization": "torch_default_qkv_zero_output_weight_and_bias",
            "initialization_rng": "fork_cpu_rng_after_shared_model_initialization",
        }
    return config


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


class _FADEDecoderBlock(nn.Module):
    """Project coarse/context and high-resolution guidance before joint FADE.

    The gate already fuses the guide; no second concatenation duplicates it.
    Learned projections do not force height/clearance contours onto the output.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        from fade import FADE
        self.coarse_projection = nn.Conv2d(in_channels, out_channels, 1)
        self.guide_projection = nn.Conv2d(skip_channels, out_channels, 1)
        for projection in (self.coarse_projection, self.guide_projection):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.up = FADE(out_channels, kernel_size=5, embedding_dim=64)
        self.fuse = _double_conv(out_channels, out_channels)

    def forward(self, input, skip):
        coarse = self.coarse_projection(input)
        guide = self.guide_projection(skip)
        return self.fuse(self.up(guide, coarse))


class _TXSceneAttention(nn.Module):
    """Source-conditioned content attention; subclasses may add a soft bias.

    The stride-8 feature lattice is centered at input pixels 0, 8, ..., 248.
    TX sampling uses those centers and border extension, not endpoint resizing
    of a 256-pixel map to 32 pixels. Only the already augmented one-hot TX is
    consulted. Anchors cover the full scene; there is no LOS or validity gate.

    N=1024 queries, M=256 anchors, d=64: attention costs O(B*N*M*d), plus
    O(B*(N+M)*C*d) projections. A materialized attention matrix is O(B*h*N*M);
    PyTorch may select a fused kernel. Training also saves network activations.
    """
    def __init__(self):
        super().__init__()
        self.heads = 4
        self.attention_dim = 64
        self.q = nn.Linear(256, self.attention_dim)
        self.k = nn.Linear(256, self.attention_dim)
        self.v = nn.Linear(256, self.attention_dim)
        self.output = nn.Linear(self.attention_dim, 128)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def source_feature(features, tx):
        # Coordinate arithmetic and interpolation stay FP32 under AMP. The TX
        # coordinates are discrete inputs; gradients still reach feature values.
        with torch.autocast(device_type=features.device.type, enabled=False):
            index = tx.flatten(1).argmax(1)
            row = torch.div(index, 256, rounding_mode="floor").float()
            col = (index % 256).float()
            grid = torch.stack((2 * col / (8 * 31) - 1,
                                2 * row / (8 * 31) - 1), dim=-1)
            grid = grid[:, None, None, :]
            source = F.grid_sample(features.float(), grid, mode="bilinear",
                                   padding_mode="border", align_corners=True)
        return source.flatten(1).to(features.dtype)

    def attention_bias(self, tx, height):
        return None

    def forward(self, features, tx, height=None):
        if features.ndim != 4 or tuple(features.shape[1:]) != (128, 32, 32):
            raise ValueError("TX scene attention requires Bx128x32x32 features")
        if tuple(tx.shape) != (features.shape[0], 1, 256, 256):
            raise ValueError("TX scene attention requires Bx1x256x256 one-hot TX")
        # The zero residual starts with tiny task gradients. FP16 Q/K backward
        # underflows in the local bounded check, even once the output learns.
        # Keep this new branch FP32 under AMP; the shared model stays unchanged.
        with torch.autocast(device_type=features.device.type, enabled=False):
            work = features.float() if features.dtype != torch.float64 else features
            batch = features.shape[0]
            source = self.source_feature(work, tx).unsqueeze(1)
            queries = work.flatten(2).transpose(1, 2)
            anchors = F.avg_pool2d(work, 2, 2).flatten(2).transpose(1, 2)
            query_input = torch.cat((queries, source.expand(-1, queries.shape[1], -1)), dim=-1)
            anchor_input = torch.cat((anchors, source.expand(-1, anchors.shape[1], -1)), dim=-1)

            def split_heads(tensor):
                return tensor.reshape(batch, -1, self.heads,
                                      self.attention_dim // self.heads).transpose(1, 2)

            attended = F.scaled_dot_product_attention(
                split_heads(self.q(query_input)), split_heads(self.k(anchor_input)),
                split_heads(self.v(anchor_input)), attn_mask=self.attention_bias(tx, height),
                dropout_p=0.0, is_causal=False)
            attended = attended.transpose(1, 2).reshape(batch, 1024, self.attention_dim)
            update = self.output(attended).transpose(1, 2).reshape_as(features)
        return features + update.to(features.dtype)


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

    def __init__(self, feature_config, arch="lunar_maxvit"):
        super().__init__()
        if not isinstance(feature_config, dict):
            raise ValueError("LunarMaxViT requires LOS/distance feature_config")
        if feature_config.get("grid") != [256, 256]:
            raise ValueError("LunarMaxViT requires the challenge's 256x256 grid")
        if "los" not in feature_config or "distance" not in feature_config:
            raise ValueError("LunarMaxViT requires both LOS and distance configuration")

        self.model_config = lunar_model_config(arch)
        self.arch = arch
        self.phase = "single"
        self.base_inputs = 2
        self.feature_config = feature_config
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
        final_block = _FADEDecoderBlock if arch == "lunar_maxvit_fade" else _DecoderBlock
        self.decoder = nn.ModuleList([
            _DecoderBlock(512, 256, 256),
            _DecoderBlock(256, 128, 128),
            final_block(128, 64, 64),
            final_block(64, 64, 32),
            final_block(32, 16, 16),
        ])
        self.output = nn.Sequential(nn.Conv2d(16 + 4, 1, 3, padding=1),
                                    nn.ReLU(inplace=True))
        # Keep the baseline ReLU output active at initialization. Small nonzero
        # weights also let gradients reach the encoder on the first update.
        nn.init.normal_(self.output[0].weight, std=0.001)
        nn.init.constant_(self.output[0].bias, 0.5)
        self.tx_scene_attention = None
        if arch == "lunar_maxvit_c":
            # Keep shared seeded weights and subsequent CPU RNG state identical
            # to B; the new branch has its own deterministic initialization draw.
            with torch.random.fork_rng(devices=[]):
                self.tx_scene_attention = _TXSceneAttention()

    def forward_features(self, input):
        """Return the prediction, physical inputs and decoder states at 64/128/256.

        The prediction path and state-dict keys stay identical for existing
        models. A refinement model can reuse these states without a second
        encoder pass or forward hooks.
        """
        if input.ndim != 4 or tuple(input.shape[1:]) != (3, 256, 256):
            raise ValueError("LunarMaxViT input must be Bx3x256x256 [H, TX, F]")
        if not input.is_floating_point():
            raise ValueError("LunarMaxViT requires normalized floating-point input")
        distance = log_distance_feature(input, self.feature_config)
        physical = torch.cat((input, distance), dim=1)
        features = [self.full_resolution(physical), self.encoder.stem(physical)]
        encoded = features[-1]
        for index, block in enumerate(self.encoder.blocks):
            encoded = block(encoded)
            if index == 1 and self.tx_scene_attention is not None:
                encoded = self.tx_scene_attention(encoded, input[:, 1:2], input[:, :1])
            features.append(encoded)
        # Skips are, in order, block3/block2/block1/stem/full-resolution.
        decoded = []
        for decoder, skip in zip(self.decoder, reversed(features[:-1])):
            encoded = decoder(encoded, skip)
            decoded.append(encoded)
        prediction = self.output(torch.cat((encoded, physical), dim=1))
        return prediction, physical, decoded[-3:]

    def forward(self, input):
        prediction, _, _ = self.forward_features(input)
        return [prediction]
