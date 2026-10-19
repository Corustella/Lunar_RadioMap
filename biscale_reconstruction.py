"""Two shared fine-to-coarse-to-fine exchanges in the secondU decoder.

Keep the upstream forward and parameter names. Three local hooks capture the
128 skip, refine the 64 context, and substitute the refined 128 skip. Forward
scopes and clears their tensors; no batch-dependent state enters checkpoints.
"""

import torch
from torch import nn
from torch.nn import functional as F

from radiounet import load_module


WIDTH = 32
ROUNDS = 2


def reconstruction_model_config():
    return {
        "arch": "radiownet_biscale", "version": 1,
        "base": "RadioUNet.modules.RadioWNet", "phase": "secondU",
        "first_features": "los", "second_features": "los_distance",
        "operator": {
            "context": "concat(Wconv_up10_output,Wlayer1_output)",
            "context_shape": [80, 64, 64],
            "detail": "Wlayer0_output", "detail_shape": [30, 128, 128],
            "width": WIDTH, "rounds": ROUNDS, "weights_shared_between_rounds": True,
            "order": "C_next=C+R_C(C,down(H));H_next=H+R_H(H,up(C_next))",
            "input_projections": "1x1_with_bias;80_to_32_and_30_to_32",
            "mixers": "concat_64_to_32_3x3_padding1_with_bias_then_ReLU",
            "output_projections": "1x1_with_bias;32_to_80_and_32_to_30",
            "downsample": "avg_pool2d_kernel2_stride2_no_padding",
            "upsample": "bilinear_size128_align_corners_false",
            "normalization": "none", "dtype": "follow_backbone_autocast",
            "initialization": "zero_output_projection_weights_and_biases",
            "rng": "fork_cpu_after_shared_initialization",
            "readout": "Wconv_up0(concat(Wconv_up1(C2),H2))",
        },
        "training": {"firstU_frozen": True, "secondU_trainable": True,
                     "output": "unchanged_direct_ReLU_prediction"},
    }


class BiScaleReconstruction(nn.Module):
    """Finite shared iteration, not an electromagnetic propagation solver.

    Time O(B*K*(N64+N128)*d*(d+channels)); activation space
    O(B*K*(N64+N128)*(d+channels)). Weight sharing does not share activations.
    """
    def __init__(self):
        super().__init__()
        self.context_in = nn.Conv2d(80, WIDTH, 1)
        self.detail_in = nn.Conv2d(30, WIDTH, 1)
        self.context_mix = nn.Conv2d(2 * WIDTH, WIDTH, 3, padding=1)
        self.detail_mix = nn.Conv2d(2 * WIDTH, WIDTH, 3, padding=1)
        self.context_out = nn.Conv2d(WIDTH, 80, 1)
        self.detail_out = nn.Conv2d(WIDTH, 30, 1)
        for layer in (self.context_out, self.detail_out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, context, detail):
        if (context.ndim != 4 or detail.ndim != 4 or
                tuple(context.shape[1:]) != (80, 64, 64) or
                tuple(detail.shape[1:]) != (30, 128, 128) or
                context.shape[0] != detail.shape[0]):
            raise ValueError("BiScale requires matching Bx80x64x64 and Bx30x128x128 features")
        for _ in range(ROUNDS):
            c = self.context_in(context)
            h = self.detail_in(detail)
            coarse = torch.cat((c, F.avg_pool2d(h, 2, stride=2)), dim=1)
            context = context + self.context_out(F.relu(self.context_mix(coarse)))
            # The fine update must use the newly corrected context, not c.
            updated = self.context_in(context)
            fine = torch.cat((h, F.interpolate(updated, size=detail.shape[-2:],
                                               mode="bilinear", align_corners=False)), dim=1)
            detail = detail + self.detail_out(F.relu(self.detail_mix(fine)))
        return context, detail


class BiScaleRadioWNet(load_module().RadioWNet):
    """Refine decoder context and skip without changing the upstream forward."""
    def __init__(self, inputs=2, phase="secondU"):
        if inputs != 2 or phase != "secondU":
            raise ValueError("BiScale reconstruction supports a single-band secondU only")
        super().__init__(inputs=inputs, phase=phase)
        with torch.random.fork_rng(devices=[]):
            self.Wreconstruction = BiScaleReconstruction()
        self._reconstruction_active = False
        self._detail_input = None
        self._detail_output = None
        self.Wlayer0.register_forward_hook(self._capture_detail)
        self.Wconv_up1.register_forward_pre_hook(self._refine_context)
        self.Wconv_up0.register_forward_pre_hook(self._fuse_detail)

    def _capture_detail(self, module, args, output):
        if self._reconstruction_active:
            self._detail_input = output

    def _refine_context(self, module, args):
        if not self._reconstruction_active or self._detail_input is None:
            raise RuntimeError("BiScale context requires the current forward's 128 skip")
        (context,) = args
        context, self._detail_output = self.Wreconstruction(context, self._detail_input)
        self._detail_input = None
        return (context,)

    def _fuse_detail(self, module, args):
        if not self._reconstruction_active or self._detail_output is None:
            raise RuntimeError("BiScale readout requires the current forward's refined skip")
        (features,) = args
        if tuple(features.shape[1:]) != (60, 128, 128):
            raise ValueError("BiScale readout expects Bx60x128x128 features")
        return (torch.cat((features[:, :30], self._detail_output), dim=1),)

    def forward(self, input):
        if self.phase != "secondU":
            raise ValueError("BiScale reconstruction requires the existing secondU phase")
        if input.ndim != 4 or tuple(input.shape[1:]) != (5, 256, 256):
            raise ValueError("Expected expanded [H, TX, F, rho, 0] input")
        if self._reconstruction_active:
            raise RuntimeError("BiScale forward cannot be reentered on the same model")
        self._reconstruction_active = True
        try:
            return super().forward(input)
        finally:
            self._detail_input = None
            self._detail_output = None
            self._reconstruction_active = False
