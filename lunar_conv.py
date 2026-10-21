"""Attention-free control matched to the lunar MaxViT encoder/decoder.

Keep the stem, MBConv convolutions, two pointwise residual MLPs per layer,
normalization, stochastic depth and decoder. Remove both spatial attention
updates and squeeze-excitation. This is a single-stage, scratch-trained model,
not a migration of trained RadioWNet or MaxViT weights.
"""

from collections import OrderedDict

import torch
from torch import nn

from lunar_maxvit import LunarMaxViT, lunar_model_config


def conv_model_config(arch="lunar_conv_aligned"):
    """Fixed control definition; distinct from every existing checkpoint type."""
    if arch not in ("lunar_conv_aligned", "lunar_conv_local3", "lunar_conv_multiscale"):
        raise ValueError(f"Unsupported convolution architecture {arch!r}")
    config = lunar_model_config()
    config["arch"] = arch
    encoder = config["encoder"]
    encoder["implementation"] = "lunar_conv.LunarConvAligned"
    for key in ("head_dim", "partition_size", "squeeze_ratio",
                "attention_dropout", "attention_norm"):
        del encoder[key]
    encoder.update({
        "spatial_attention": False, "squeeze_excitation": False,
        "layer_order": ["MBConv_without_SE", "channel_MLP_1", "channel_MLP_2"],
        "channel_mlp": {
            "count_per_layer": 2, "layout": "NHWC_without_partitioning",
            "normalization": {"name": "LayerNorm", "eps": 1e-5},
            "operation": "x+stochastic_depth(Linear_GELU_Linear_Dropout(LN(x)))",
            "stochastic_depth_mode": "row",
        },
        "initialization": "construct_lunar_maxvit_then_remove_attention_without_new_random_draws",
    })
    if arch != "lunar_conv_aligned":
        encoder["spatial_operator"] = {
            "implementation": "lunar_conv._ParallelDepthwise",
            "encoder_block_index": 1, "layer_index": 1,
            "resolution": [32, 32], "expanded_channels": 512,
            "kernel_size": 3, "stride": 1, "groups": 512, "bias": False,
            "dilations": [1, 1, 1] if arch == "lunar_conv_local3" else [1, 2, 3],
            "padding": "zero_equal_to_dilation",
            "fusion": "sum_then_original_BatchNorm2d_GELU",
            "branch_normalization": None, "branch_activation": None,
            "initialization": "retain_base_kernel_zero_two_extra_kernels",
            "initialization_rng": "fork_cpu_rng_after_shared_model_initialization",
        }
    return config


class _ParallelDepthwise(nn.Module):
    """Three linear spatial branches, followed by the existing shared BN/GELU.

    Keep the original d=1 kernel; zero extra kernels preserve C's initial
    function without blocking their gradients. L repeats d=1, M uses d=2/3.
    Added work is 18*B*H*W*C MACs and 18*C parameters; activation storage stays
    O(B*H*W*C) with a larger constant. L can fold to 3x3, M to sparse 7x7:
    equal parameter counts do not imply equal effective kernel capacity.
    """
    def __init__(self, original, dilations):
        super().__init__()
        if (not isinstance(original, nn.Conv2d) or
                original.kernel_size != (3, 3) or original.stride != (1, 1) or
                original.padding != (1, 1) or original.dilation != (1, 1) or
                original.groups != original.in_channels or
                original.out_channels != original.in_channels or
                original.bias is not None or original.padding_mode != "zeros"):
            raise ValueError("Spatial controls require the original stride-1 depthwise 3x3")
        self.base = original
        # The shared model is constructed on CPU before train.py moves it.
        # Even kernels subsequently zeroed would otherwise consume RNG draws.
        with torch.random.fork_rng(devices=[]):
            self.extra = nn.ModuleList([
                nn.Conv2d(original.in_channels, original.out_channels, 3,
                          padding=dilation, dilation=dilation,
                          groups=original.groups, bias=False)
                for dilation in dilations[1:]
            ])
        for branch in self.extra:
            nn.init.zeros_(branch.weight)

    def forward(self, input):
        output = self.base(input)
        for branch in self.extra:
            output = output + branch(input)
        return output


class _ResidualChannelMLP(nn.Module):
    """Keep only the per-position MLP residual from a MaxViT attention layer.

    Partitioning is unnecessary: LayerNorm/Linear act on C and stochastic
    depth drops whole samples in both layouts. In particular, removing an
    attention update does NOT mean inserting Identity inside its residual sum.
    """
    def __init__(self, partition_layer):
        super().__init__()
        self.mlp_layer = partition_layer.mlp_layer
        self.stochastic_dropout = partition_layer.stochastic_dropout

    def forward(self, input):
        channels_last = input.permute(0, 2, 3, 1)
        output = channels_last + self.stochastic_dropout(self.mlp_layer(channels_last))
        return output.permute(0, 3, 1, 2)


class LunarConvAligned(LunarMaxViT):
    """Single U-shaped control with H/TX/F/log-distance jointly encoded.

    Construct the shared model first so retained weights and the post-init CPU
    RNG state match MaxViT for the same seed. Discarded attention/SE parameters
    are never optimized or evaluated, and no pretrained checkpoint is loaded.

    With expansion/MLP ratios fixed, encoder time is O(B*sum(L_s*N_s*C_s^2));
    retained training activations are O(B*sum(L_s*N_s*C_s)), plus the unchanged
    decoder. Parameters/optimizer state are O(sum(L_s*C_s^2)). No attention
    matrix is allocated during forward or backward.
    """
    def __init__(self, feature_config, arch="lunar_conv_aligned"):
        config = conv_model_config(arch)
        super().__init__(feature_config, arch="lunar_maxvit")
        for block in self.encoder.blocks:
            for layer in block.layers:
                original = layer.layers
                original.MBconv.layers.squeeze_excitation = nn.Identity()
                layer.layers = nn.Sequential(OrderedDict([
                    ("MBconv", original.MBconv),
                    ("channel_mlp1", _ResidualChannelMLP(original.window_attention)),
                    ("channel_mlp2", _ResidualChannelMLP(original.grid_attention)),
                ]))
        if arch != "lunar_conv_aligned":
            spatial = self.encoder.blocks[1].layers[1].layers.MBconv.layers.conv_b
            spatial[0] = _ParallelDepthwise(
                spatial[0], config["encoder"]["spatial_operator"]["dilations"])
        self.arch = arch
        self.model_config = config
