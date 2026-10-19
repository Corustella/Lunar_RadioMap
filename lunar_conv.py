"""Attention-free control matched to the lunar MaxViT encoder/decoder.

Keep the stem, MBConv convolutions, two pointwise residual MLPs per layer,
normalization, stochastic depth and decoder. Remove both spatial attention
updates and squeeze-excitation. This is a single-stage, scratch-trained model,
not a migration of trained RadioWNet or MaxViT weights.
"""

from collections import OrderedDict

from torch import nn

from lunar_maxvit import LunarMaxViT, lunar_model_config


def conv_model_config():
    """Fixed control definition; distinct from every existing checkpoint type."""
    config = lunar_model_config()
    config["arch"] = "lunar_conv_aligned"
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
    return config


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
    def __init__(self, feature_config):
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
        self.arch = "lunar_conv_aligned"
        self.model_config = conv_model_config()
