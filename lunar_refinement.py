"""T + cross-stage refinement for full-map pathloss prediction.

The information-flow hypothesis is inspired by RadioUNet retrospective
refinement and MPRNet cross-stage feature fusion, not a reproduction of either
network. No extra loss, attention, geometry feature or validity-mask input is
introduced. Stage one is the existing T, jointly trained or kept fixed in eval.
The full second-U controls keep T fixed. Their optional mean/detail pyramid
is a spatial reconstruction hypothesis, not a propagation-mechanism split.
"""

import torch
from torch import nn
from torch.nn import functional as F

from lunar_conv import LunarConvAligned, conv_model_config
from lunar_maxvit import _DecoderBlock, _double_conv


def refinement_model_config(arch="lunar_conv_tx_refine"):
    full_arches = ("lunar_conv_tx_refine_unet", "lunar_conv_tx_refine_pyramid")
    if arch not in ("lunar_conv_tx_refine", "lunar_conv_tx_refine_frozen", *full_arches):
        raise ValueError(f"Unsupported refinement architecture {arch!r}")
    config = conv_model_config("lunar_conv_tx_content")
    config["arch"] = arch
    config["refinement"] = {
        "implementation": "lunar_refinement._CrossStageRefiner",
        "base_arch": "lunar_conv_tx_content",
        "input": ["H", "TX", "F", "rho", "stage_one_prediction"],
        "resolutions": [256, 128, 64, 128, 256],
        "channels": [16, 32, 64, 32, 16],
        "cross_stage": {
            "source": "stage_one_decoder_outputs",
            "resolutions": [256, 128, 64], "channels": [16, 32, 64],
            "projection": "1x1_conv_with_bias_same_width",
            "fusion": "concat_before_each_refinement_encoder_double_conv",
            "detach": False,
        },
        "downsample": "3x3_stride2_padding1_conv_then_ReLU",
        "upsample": "4x4_stride2_padding1_conv_transpose_then_ReLU",
        "blocks": "two_3x3_padding1_convs_with_ReLU",
        "normalization": None, "dropout": 0.0,
        "skip": "refinement_encoder_to_matching_decoder_concat",
        "correction_head": "3x3_padding1_conv_signed_linear",
        "output": "ReLU(stage_one_prediction+correction)_float32_sum",
        "initialization": "torch_default_except_zero_correction_head_weight_and_bias",
        "initialization_rng": "fork_cpu_rng_after_T_initialization",
    }
    config["stage_one_output"] = config["output"]
    config["output"] = {
        "channels": 1, "quantity": "normalized_pathloss", "activation": "ReLU",
        "parameterization": "stage_one_prediction_plus_signed_correction",
        "addition_dtype": "float32", "initialization": "identical_to_loaded_T",
    }
    config["training"].update({
        "structural_stages": 2, "frozen_parameters": False,
        "initialization_source": "matching_seed_T_checkpoint_weights_only",
        "supervision": "final_prediction_only",
    })
    if arch != "lunar_conv_tx_refine":
        config["refinement"]["cross_stage"]["detach"] = True
        config["training"].update({
            "frozen_parameters": True,
            "trainable_parameters": "refiner.*",
            "stage_one_mode": "eval_including_batchnorm_and_stochastic_depth",
            "stage_one_forward": "no_grad",
        })
    if arch in full_arches:
        config["refinement"].update({
            "implementation": "lunar_refinement._FullScaleRefiner",
            "resolutions": [256, 128, 64, 32, 16, 8, 16, 32, 64, 128, 256],
            "channels": [16, 32, 64, 96, 128, 192, 128, 96, 64, 32, 16],
            "head_resolutions": [256],
            "initialization": "kaiming_normal_fan_in_relu_refiner_convs_zero_bias_zero_output_heads",
        })
    if arch == "lunar_conv_tx_refine_pyramid":
        config["refinement"].update({
            "head_resolutions": [32, 64, 128, 256],
            "correction_head": "3x3_signed_heads_then_parent_mean_preserving_pyramid_v1",
            "pyramid": {
                "coarse_resolution": 32, "detail_resolutions": [64, 128, 256],
                "average": "2x2_stride2_no_padding",
                "upsample": "nearest_2x",
                "detail_projection": "q-up(avg(q))", "detail_head_bias": False,
                "coarse_head_bias": True, "synthesis": "r=up(r)+detail",
                "arithmetic_dtype": "float32_autocast_disabled",
                "constraint_scope": "correction_before_final_ReLU_not_mask_weighted",
            },
        })
    return config


class _CrossStageRefiner(nn.Module):
    """Local/mid-scale correction using T's already contextualized features.

    Convolution time is O(B*sum(H_s*W_s*k^2*C_in,s*C_out,s)); retained
    activations are O(B*sum(H_s*W_s*C_s)) with depth-dependent constants.
    No second global interaction is computed. The zero output head preserves
    T initially; internal task gradients start after that head becomes nonzero.
    """
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, c, 1) for c in (16, 32, 64)])
        self.encode256 = _double_conv(5 + 16, 16)
        self.down128 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                     nn.ReLU(inplace=True))
        self.encode128 = _double_conv(32 + 32, 32)
        self.down64 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                    nn.ReLU(inplace=True))
        self.encode64 = _double_conv(64 + 64, 64)
        self.decode128 = _DecoderBlock(64, 32, 32)
        self.decode256 = _DecoderBlock(32, 16, 16)
        self.output = nn.Conv2d(16, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, physical, prediction, decoded):
        d64, d128, d256 = decoded
        x256 = self.encode256(torch.cat(
            (physical, prediction, self.projections[0](d256)), dim=1))
        x128 = self.encode128(torch.cat(
            (self.down128(x256), self.projections[1](d128)), dim=1))
        x64 = self.encode64(torch.cat(
            (self.down64(x128), self.projections[2](d64)), dim=1))
        return self.output(self.decode256(self.decode128(x64, x128), x256))


def _pyramid_correction(coarse, details):
    """Synthesize a correction with zero-mean 2x2 details at each finer level.

    Average(Up(r)) == r, so Average(output) == r at each level, up to
    roundoff. This is not an orthogonal decomposition of the masked loss.
    Pooling/subtraction/addition stay FP32 under AMP; gradients are retained.
    Time and stored output activations are O(B*sum(H_s*W_s)).
    """
    with torch.autocast(device_type=coarse.device.type, enabled=False):
        result = coarse.float()
        for detail in details:
            if (detail.shape[:-2] != result.shape[:-2] or
                    detail.shape[-2:] != (2 * result.shape[-2], 2 * result.shape[-1])):
                raise ValueError("Pyramid details must double each spatial dimension")
            detail = detail.float()
            mean = F.avg_pool2d(detail, 2, stride=2)
            centered = detail - F.interpolate(mean, scale_factor=2, mode="nearest")
            result = F.interpolate(result, scale_factor=2, mode="nearest") + centered
    return result


class _FullScaleRefiner(_CrossStageRefiner):
    """Shared full second U for ordinary (U) and hierarchical (P) outputs.

    The existing 256/128/64 bridges feed a 32/16/8 encoder and full decoder.
    U/P construct every shared module in the same order. Only after that is
    P's extra head capacity allocated, inside the outer refiner RNG fork.
    All heads start at zero, preserving the initial T prediction.
    Convolution time/activation orders are those of _CrossStageRefiner;
    the pyramid adds only linear work in the number of output pixels.
    """
    def __init__(self, pyramid=False):
        super().__init__()
        self.pyramid = pyramid
        self.down32 = nn.Sequential(nn.Conv2d(64, 96, 3, stride=2, padding=1),
                                    nn.ReLU(inplace=True))
        self.encode32 = _double_conv(96, 96)
        self.down16 = nn.Sequential(nn.Conv2d(96, 128, 3, stride=2, padding=1),
                                    nn.ReLU(inplace=True))
        self.encode16 = _double_conv(128, 128)
        self.down8 = nn.Sequential(nn.Conv2d(128, 192, 3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.encode8 = _double_conv(192, 192)
        self.decode16 = _DecoderBlock(192, 128, 128)
        self.decode32 = _DecoderBlock(128, 96, 96)
        self.decode64 = _DecoderBlock(96, 64, 64)
        # Default shallow-U initialization attenuates the much longer path:
        # local AMP checks found zero deep task gradients in U after 12 steps.
        # Initialize both shared trunks identically for ReLU, then restore
        # zero heads. Do this before allocating P-only heads so RNG matches.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output.weight)
        if pyramid:
            # A constant detail bias is annihilated by q-Up(Avg(q)). Remove
            # it after constructing the shared trunk, preserving U/P RNG.
            self.output.register_parameter("bias", None)
            self.coarse_output = nn.Conv2d(96, 1, 3, padding=1)
            self.detail64 = nn.Conv2d(64, 1, 3, padding=1, bias=False)
            self.detail128 = nn.Conv2d(32, 1, 3, padding=1, bias=False)
            for head in (self.coarse_output, self.detail64, self.detail128):
                nn.init.zeros_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

    def forward(self, physical, prediction, decoded):
        d64, d128, d256 = decoded
        x256 = self.encode256(torch.cat(
            (physical, prediction, self.projections[0](d256)), dim=1))
        x128 = self.encode128(torch.cat(
            (self.down128(x256), self.projections[1](d128)), dim=1))
        x64 = self.encode64(torch.cat(
            (self.down64(x128), self.projections[2](d64)), dim=1))
        x32 = self.encode32(self.down32(x64))
        x16 = self.encode16(self.down16(x32))
        x8 = self.encode8(self.down8(x16))
        f16 = self.decode16(x8, x16)
        f32 = self.decode32(f16, x32)
        f64 = self.decode64(f32, x64)
        f128 = self.decode128(f64, x128)
        f256 = self.decode256(f128, x256)
        fine = self.output(f256)
        if not self.pyramid:
            return fine
        return _pyramid_correction(self.coarse_output(f32),
                                   (self.detail64(f64), self.detail128(f128), fine))


class LunarConvRefine(LunarConvAligned):
    """Two structural stages and one final output, with optional fixed T.

    Shared T weights retain their names for strict, explicit initialization.
    Only refiner.* is new. The refiner reads test-available inputs/features;
    ground truth and the simulation validity mask are never model inputs.
    """
    def __init__(self, feature_config, arch="lunar_conv_tx_refine"):
        config = refinement_model_config(arch)
        super().__init__(feature_config, arch="lunar_conv_tx_content")
        with torch.random.fork_rng(devices=[]):
            if arch in ("lunar_conv_tx_refine_unet", "lunar_conv_tx_refine_pyramid"):
                self.refiner = _FullScaleRefiner(pyramid=arch == "lunar_conv_tx_refine_pyramid")
            else:
                self.refiner = _CrossStageRefiner()
        self.arch = arch
        self.model_config = config
        self.frozen_stage_one = config["training"]["frozen_parameters"]
        if self.frozen_stage_one:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("refiner."))
            self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_stage_one:
            # Keep T's buffers and stochastic layers fixed even when the outer
            # training loop switches the entire model back to training mode.
            for name, module in self.named_children():
                if name != "refiner":
                    module.eval()
        return self

    def forward(self, input):
        if self.frozen_stage_one:
            with torch.no_grad():
                prediction, physical, decoded = self.forward_features(input)
        else:
            prediction, physical, decoded = self.forward_features(input)
        correction = self.refiner(physical, prediction, decoded)
        # Keep small signed corrections when the convolution path uses AMP.
        # ReLU preserves the existing normalized, nonnegative output contract.
        return [F.relu(prediction.float() + correction.float())]
