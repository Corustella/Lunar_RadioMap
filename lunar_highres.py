"""T with a persistent 64x64 encoder stream and two spatial exchanges."""

import torch
from torch import nn
from torch.nn import functional as F

from lunar_conv import LunarConvAligned, conv_model_config


_FINE_SIZE = 64
_FINE_CHANNELS = 64
_CONTEXT_CHANNELS = 32
_COARSE_STAGES = ((32, 128), (16, 256))


def highres_model_config():
    config = conv_model_config("lunar_conv_tx_content")
    config["arch"] = "lunar_conv_tx_highres"
    config["encoder"]["highres_exchange"] = {
        "implementation": "lunar_highres.LunarConvHighRes",
        "fine_size": _FINE_SIZE,
        "fine_channels": _FINE_CHANNELS,
        "coarse_stages": [dict(size=size, channels=channels) for size, channels in _COARSE_STAGES],
        "context_channels": _CONTEXT_CHANNELS,
        "context_sources": ["post_TX_attention_32", "encoder_16_after_exchange_32"],
        "upsample": "origin_aligned_bilinear_align_corners_then_replicate_right_bottom",
        "fine_update": "concat_then_pointwise_GELU_depthwise3x3_GELU_pointwise_residual",
        "coarse_update": "pool_fine_delta_kernel_2f_minus1_stride_f_padding_f_minus1_exclude_pad_then_pointwise_residual",
        "decoder_skip": "updated_fine64_and_updated_coarse32_16",
        "precision": "parameter_dtype_inside_exchange_output_cast_to_each_stream_dtype",
        "initialization": "zero_final_fine_projection_other_weights_torch_default",
        "initialization_rng": "fork_cpu_rng_after_T_initialization",
    }
    return config


class _HighResExchange(nn.Module):
    def __init__(self, coarse_size, coarse_channels):
        super().__init__()
        self.scale = _FINE_SIZE // coarse_size
        self.extent = (coarse_size - 1) * self.scale + 1
        self.border = _FINE_SIZE - self.extent
        self.context = nn.Conv2d(coarse_channels, _CONTEXT_CHANNELS, 1, bias=False)
        self.fine_update = nn.Sequential(
            nn.Conv2d(_FINE_CHANNELS + _CONTEXT_CHANNELS, _FINE_CHANNELS, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(_FINE_CHANNELS, _FINE_CHANNELS, 3, padding=1,
                      groups=_FINE_CHANNELS, bias=False),
            nn.GELU(),
            nn.Conv2d(_FINE_CHANNELS, _FINE_CHANNELS, 1, bias=False),
        )
        self.coarse_update = nn.Conv2d(_FINE_CHANNELS, coarse_channels, 1, bias=False)
        nn.init.zeros_(self.fine_update[-1].weight)

    def forward(self, fine, coarse):
        # After the zero head's first update, its interior gradients can
        # underflow in FP16. Use parameter precision inside the exchange.
        with torch.autocast(device_type=fine.device.type, enabled=False):
            dtype = self.context.weight.dtype
            fine_work, coarse_work = fine.to(dtype), coarse.to(dtype)
            # Interpolate between the outermost coarse centres, then extend the
            # right/bottom borders. Both lattices share origin zero.
            context = F.interpolate(self.context(coarse_work), size=(self.extent, self.extent),
                                    mode="bilinear", align_corners=True)
            context = F.pad(context, (0, self.border, 0, self.border), mode="replicate")
            delta = self.fine_update(torch.cat((fine_work, context), dim=1))
            # Centre the feedback on the original coarse lattice.
            feedback = F.avg_pool2d(delta, kernel_size=2 * self.scale - 1, stride=self.scale,
                                    padding=self.scale - 1, count_include_pad=False)
            return (fine_work + delta).to(fine.dtype), (coarse_work + self.coarse_update(feedback)).to(coarse.dtype)


class LunarConvHighRes(LunarConvAligned):
    """Update the native 64x64 skip during encoding using TX scene context."""
    def __init__(self, feature_config):
        super().__init__(feature_config, arch="lunar_conv_tx_content")
        with torch.random.fork_rng(devices=[]):
            self.exchange32 = _HighResExchange(*_COARSE_STAGES[0])
            self.exchange16 = _HighResExchange(*_COARSE_STAGES[1])
        self.model_config = highres_model_config()
        self.arch = self.model_config["arch"]

    def _encoder_features(self, physical, input):
        full = self.full_resolution(physical)
        stem = self.encoder.stem(physical)
        stage64, stage32, stage16, stage8 = self.encoder.blocks
        fine = stage64(stem)
        coarse32 = self.tx_scene_attention(stage32(fine), input[:, 1:2], input[:, :1])
        fine, coarse32 = self.exchange32(fine, coarse32)
        fine, coarse16 = self.exchange16(fine, stage16(coarse32))
        return [full, stem, fine, coarse32, coarse16, stage8(coarse16)]
