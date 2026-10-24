"""T + shallow cross-stage refinement for full-map pathloss prediction.

The information-flow hypothesis is inspired by RadioUNet retrospective
refinement and MPRNet cross-stage feature fusion, not a reproduction of either
network. No extra loss, attention, geometry feature or validity-mask input is
introduced. Stage one is the existing T, jointly trained or kept fixed in eval.
"""

import torch
from torch import nn
from torch.nn import functional as F

from lunar_conv import LunarConvAligned, conv_model_config
from lunar_maxvit import _DecoderBlock, _double_conv


def refinement_model_config(arch="lunar_conv_tx_refine"):
    if arch not in ("lunar_conv_tx_refine", "lunar_conv_tx_refine_frozen"):
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
    if arch == "lunar_conv_tx_refine_frozen":
        config["refinement"]["cross_stage"]["detach"] = True
        config["training"].update({
            "frozen_parameters": True,
            "trainable_parameters": "refiner.*",
            "stage_one_mode": "eval_including_batchnorm_and_stochastic_depth",
            "stage_one_forward": "no_grad",
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
            self.refiner = _CrossStageRefiner()
        self.arch = arch
        self.model_config = config
        self.frozen_stage_one = arch == "lunar_conv_tx_refine_frozen"
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
