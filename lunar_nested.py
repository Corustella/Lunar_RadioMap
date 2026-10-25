"""UNet++-inspired nested skips inside T's single-stage prediction network.

Full nesting spans all six native resolutions. Local nesting retains the
8->16->32 decoder and nests only its 64/128/256 reconstruction. This is an
adaptation of nested dense skips, not a reproduction of the medical-segmentation
model: T's widths, transpose upsampling, double convolutions and output remain.
No deep supervision, auxiliary loss, second prediction stage or mask is added.
"""

import torch
from torch import nn

from los_features import log_distance_feature
from lunar_conv import LunarConvAligned, conv_model_config
from lunar_maxvit import _DecoderBlock


_NATIVE_CHANNELS = (16, 64, 64, 128, 256, 512)
_NODE_CHANNELS = (16, 32, 64, 128, 256)


def nested_model_config(arch="lunar_conv_tx_nested_full"):
    if arch not in ("lunar_conv_tx_nested_full", "lunar_conv_tx_nested_local"):
        raise ValueError(f"Unsupported nested architecture {arch!r}")
    depth = 5 if arch == "lunar_conv_tx_nested_full" else 3
    config = conv_model_config("lunar_conv_tx_content")
    config["arch"] = arch
    config["decoder"].update({
        "fusion": "upsampled_then_native_then_all_prior_same_scale_nodes_double_conv",
        "initialization": "retain_seeded_T_weights_zero_added_diagonal_input_weights",
        "nested_skip": {
            "implementation": "lunar_nested.LunarConvNested",
            "depth": depth,
            "native_resolutions": [256, 128, 64, 32, 16, 8],
            "native_channels": list(_NATIVE_CHANNELS),
            "node_channels_high_to_low": list(_NODE_CHANNELS),
            "bottom_context": "encoder_8" if depth == 5 else "T_decoder_32",
            "nodes": [[i, j] for j in range(1, depth + 1)
                      for i in range(depth - j + 1)],
            "recurrence": "X[i,j]=H([Up(X[i+1,j-1]),X[i,0],...,X[i,j-1]])",
            "independent_node_weights": True,
            "lateral_initialization": "torch_default",
            "initialization_rng": "fork_cpu_rng_common_local_nodes_before_full_only_nodes",
            "deep_supervision": False,
        },
    })
    return config


def _extend_fusion(block, extra_channels):
    """Append trainable channels without changing the seeded T contribution.

    Only the added weights start at zero; lateral nodes have ordinary random
    weights. Their task gradients start after these connections leave zero.
    All parameters train from scratch together, with no trained T checkpoint.
    """
    old = block.fuse[0]
    extended = nn.Conv2d(old.in_channels + extra_channels, old.out_channels,
                         3, padding=1)
    with torch.no_grad():
        extended.weight.zero_()
        extended.weight[:, :old.in_channels].copy_(old.weight)
        extended.bias.copy_(old.bias)
    block.fuse[0] = extended


class LunarConvNested(LunarConvAligned):
    """Dense same-scale skips with adjacent-scale learned upsampling.

    X[i,0] is the native feature at row i, except the local bottom is T's D32.
    Nodes on i+j=depth reuse T's five decoder blocks (the first two remain
    outside the local triangle). All other nodes are independent lateral blocks.

    For depth d the triangle has d(d+1)/2 nodes. A node with output c_i,
    native width e_i and coarse width b costs approximately
    B*H_i*W_i*(4*b*c_i + 9*c_i*(e_i+j*c_i) + 9*c_i*c_i) MACs:
    the stride-2 transpose convolution operates on one quarter as many pixels.
    Training retains node outputs plus concatenations/convolution activations;
    this is O(B*sum_nodes(H_i*W_i*(e_i+j*c_i+b))) at fixed kernel sizes.
    Dense concatenations grow with j; small parameter increments do not imply
    small high-resolution activation or runtime costs.
    """
    def __init__(self, feature_config, arch="lunar_conv_tx_nested_full"):
        config = nested_model_config(arch)
        super().__init__(feature_config, arch="lunar_conv_tx_content")
        self.nested_depth = config["decoder"]["nested_skip"]["depth"]
        self.nested_nodes = nn.ModuleDict()
        # Build the three common lateral nodes first in both variants. Forking
        # also preserves T's post-initialization RNG stream for data/stochastic
        # depth. Common encoder, TX attention and decoder weights remain equal.
        common = [(0, 1), (1, 1), (0, 2)]
        extra = [(i, j) for j in range(1, self.nested_depth)
                 for i in range(self.nested_depth - j) if (i, j) not in common]
        with torch.random.fork_rng(devices=[]):
            for i, j in common + extra:
                source = _NATIVE_CHANNELS[i + 1] if j == 1 else _NODE_CHANNELS[i + 1]
                skip = _NATIVE_CHANNELS[i] + (j - 1) * _NODE_CHANNELS[i]
                self.nested_nodes[f"{i}_{j}"] = _DecoderBlock(source, skip, _NODE_CHANNELS[i])
            for i in range(self.nested_depth - 1):
                extra_channels = (self.nested_depth - i - 1) * _NODE_CHANNELS[i]
                _extend_fusion(self.decoder[4 - i], extra_channels)
        self.arch = arch
        self.model_config = config

    def forward_features(self, input):
        if input.ndim != 4 or tuple(input.shape[1:]) != (3, 256, 256):
            raise ValueError("LunarConvNested expects Bx3x256x256 input [H, TX, F]")
        if not input.is_floating_point():
            raise ValueError("LunarConvNested requires normalized floating-point input")
        rho = log_distance_feature(input, self.feature_config)
        physical = torch.cat((input, rho), dim=1)
        features = [self.full_resolution(physical), self.encoder.stem(physical)]
        encoded = features[-1]
        for index, block in enumerate(self.encoder.blocks):
            encoded = block(encoded)
            if index == 1:
                encoded = self.tx_scene_attention(encoded, input[:, 1:2], input[:, :1])
            features.append(encoded)

        # Local nesting is anchored in decoded D32; full nesting starts at E8.
        for block, skip in zip(self.decoder[:5 - self.nested_depth], reversed(features[:-1])):
            encoded = block(encoded, skip)
        nodes = {(i, 0): feature for i, feature in enumerate(features[:self.nested_depth])}
        nodes[self.nested_depth, 0] = encoded
        for j in range(1, self.nested_depth + 1):
            for i in range(self.nested_depth - j + 1):
                block = (self.decoder[4 - i] if i + j == self.nested_depth
                         else self.nested_nodes[f"{i}_{j}"])
                skips = torch.cat([nodes[i, k] for k in range(j)], dim=1)
                nodes[i, j] = block(nodes[i + 1, j - 1], skips)
        decoded = [nodes[i, self.nested_depth - i] for i in (2, 1, 0)]
        prediction = self.output(torch.cat((decoded[-1], physical), dim=1))
        return prediction, physical, decoded
