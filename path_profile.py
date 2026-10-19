"""Input-only TX--RX clearance profiles in one LOS+distance refinement block.

The recurrent state summarizes geometry; it is not a field or accumulated dB.
The upstream RadioWNet forward, LOS inputs and direct output are unchanged.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from radiounet import load_module


SEGMENTS = 64
HIDDEN = 32
SAMPLE_CHUNK = 256
RECURRENT_CHUNK = 4096


def profile_model_config():
    return {
        "arch": "radiownet_path_profile", "version": 1,
        "base": "RadioUNet.modules.RadioWNet", "phase": "secondU",
        "first_features": "los", "second_features": "los_distance",
        "operator": {
            "layer": "Wlayer10.0", "resolution": [64, 64],
            "in_channels": 40, "out_channels": 50,
            "base_convolution": "unchanged_5x5_padding2_with_bias",
            "feature_stride_px": 4, "feature_origin_px": 1.5,
            "segments": SEGMENTS,
            "samples_per_segment": "m+1;m=2*max(1,ceil(horizontal_distance_px/64))",
            "sampling": "u=(k+j/m)/64;k=0..63;j=0..m",
            "interpolation": "original_height_bilinear_align_corners_border",
            "tokens": ["minimum_clearance", "mean_clearance", "segment_midpoint_u",
                       "log_3d_segment_length"],
            "mean": "equal_weight_including_both_endpoints_per_segment",
            "clearance_encoding": "asinh(c/1m)/asinh(training_height_range/1m)",
            "length_encoding": "log1p(3d_distance/64/1m)/log1p(distance_max/1m)",
            "sequence": "TX_to_RX_geometry_only_no_features_labels_or_masks",
            "encoder": {"type": "GRU", "input_size": 4, "hidden_size": HIDDEN,
                        "num_layers": 1, "bidirectional": False, "dropout": 0.0,
                        "initial_state": "zeros", "summary": "last_hidden"},
            "fusion": "linear_82_to_50_with_bias_before_original_ReLU",
            "initialization": "preserve_base_weights_and_bias;fusion=[I,0];fusion_bias=0",
            "computation_dtype": "FP32_geometry_GRU_fusion;cast_to_base_dtype",
            "sample_chunk": SAMPLE_CHUNK, "recurrent_chunk": RECURRENT_CHUNK,
            "rng": "fork_cpu_after_shared_initialization",
        },
        "training": {"firstU_frozen": True, "secondU_trainable": True,
                     "output": "unchanged_direct_ReLU_prediction"},
    }


def receiver_coordinates(device):
    """Centers after two 2x2 pools, in original image (x, y) pixels."""
    axis = torch.arange(64, device=device, dtype=torch.float32) * 4 + 1.5
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(-1, 2)


@torch.no_grad()
def path_profile_tokens(input, config, receivers=None, chunk=SAMPLE_CHUNK):
    """Return B x N x 64 x 4 tokens from normalized height and one-hot TX.

    Receivers use original-image (x, y) coordinates, never feature indices.
    The optional receiver list also permits direct short/zero-path checks.
    Sampling is FP32 and at most half a pixel apart, including segment ends.
    Time O(BNL), output memory O(BNK), temporary memory O(B*chunk*L).
    """
    if input.ndim != 4 or input.shape[1] < 2 or tuple(input.shape[-2:]) != (256, 256):
        raise ValueError("Path profiles require BxCx256x256 height/TX inputs")
    if chunk < 1:
        raise ValueError("Path profile chunk must be positive")
    with torch.autocast(device_type=input.device.type, enabled=False):
        lo, hi = config["heightmap_range_m"]
        height = input[:, :1].float() * (hi - lo) + lo
        batch = input.shape[0]
        source = input[:, 1].flatten(1).argmax(1)
        tx = torch.stack((source % 256, source // 256), -1).float()
        z_tx = height.flatten(1).gather(1, source[:, None]) + config["tx_height_m_agl"]
        ends = (receiver_coordinates(input.device) if receivers is None
                else receivers.to(device=input.device, dtype=torch.float32))
        if ends.ndim != 2 or ends.shape[1] != 2 or not ends.shape[0]:
            raise ValueError("Receivers must be a nonempty Nx2 list of image coordinates")
        if not torch.isfinite(ends).all() or (ends < 0).any() or (ends > 255).any():
            raise ValueError("Path profile receivers must be finite and inside the input grid")

        # Fixed upper bound avoids per-chunk GPU -> CPU synchronization.
        max_m = 2 * math.ceil(math.sqrt(2) * 255 / SEGMENTS)
        j = torch.arange(max_m + 1, device=input.device, dtype=torch.float32)
        k = torch.arange(SEGMENTS, device=input.device, dtype=torch.float32)
        midpoint = (k + 0.5) / SEGMENTS
        clearance_scale = math.asinh(hi - lo)
        length_scale = math.log1p(config["distance"]["max_distance_m"])
        outputs = []
        for start in range(0, ends.shape[0], chunk):
            end = ends[start:start + chunk]
            count = end.shape[0]
            delta = end[None] - tx[:, None]
            distance_px = delta.square().sum(-1).sqrt()
            m = 2 * torch.ceil(distance_px / SEGMENTS).clamp_min(1)
            fractions = (j[None, None, :] / m[..., None]).clamp_max(1)
            u = (k[None, None, :, None] + fractions[:, :, None, :]) / SEGMENTS
            points = tx[:, None, None, None] + u[..., None] * delta[:, :, None, None]
            grid = (2 * points / 255 - 1).reshape(batch, count, -1, 2)
            terrain = F.grid_sample(height, grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)
            terrain = terrain[:, 0].reshape(batch, count, SEGMENTS, max_m + 1)
            end_grid = (2 * end / 255 - 1)[None, :, None].expand(batch, -1, -1, -1)
            z_rx = F.grid_sample(height, end_grid, mode="bilinear",
                                 padding_mode="border", align_corners=True)[:, 0, :, 0]
            z_rx = z_rx + config["rx_height_m_agl"]
            vertical = z_rx - z_tx
            clearance = z_tx[:, :, None, None] + u * vertical[:, :, None, None] - terrain
            valid = (j[None, None, :] <= m[..., None])[:, :, None, :]
            minimum = clearance.masked_fill(~valid, float("inf")).amin(-1)
            mean = clearance.masked_fill(~valid, 0).sum(-1) / (m + 1)[..., None]
            length = ((distance_px * config["resolution_m_per_px"]).square()
                      + vertical.square()).sqrt() / SEGMENTS
            tokens = torch.stack((
                torch.asinh(minimum) / clearance_scale,
                torch.asinh(mean) / clearance_scale,
                midpoint[None, None].expand(batch, count, -1),
                (torch.log1p(length) / length_scale)[..., None].expand(-1, -1, SEGMENTS),
            ), -1)
            outputs.append(tokens)
        return torch.cat(outputs, dim=1)


class PathProfileConv(nn.Conv2d):
    """Joint local convolution and per-receiver geometry sequence encoding.

    GRU time O(BNK*C*(C+4)), training activation space O(BNKC).
    Chunking limits temporary sampling/recurrent tensors, not all autograd state.
    Legacy weight/bias names are retained for strict historical initialization.
    """
    def __init__(self, base):
        super().__init__(base.in_channels, base.out_channels, base.kernel_size,
                         stride=base.stride, padding=base.padding,
                         dilation=base.dilation, groups=base.groups,
                         bias=base.bias is not None, device="meta",
                         dtype=base.weight.dtype)
        self.weight = base.weight
        self.bias = base.bias
        self.profile_gru = nn.GRU(4, HIDDEN, batch_first=True)
        self.profile_fusion = nn.Linear(base.out_channels + HIDDEN, base.out_channels)
        with torch.no_grad():
            self.profile_fusion.weight.zero_()
            self.profile_fusion.weight[:, :base.out_channels].copy_(torch.eye(base.out_channels))
            self.profile_fusion.bias.zero_()
        self._profile_context = None

    def encode_profiles(self, tokens):
        batch, receivers, steps, channels = tokens.shape
        flat = tokens.reshape(batch * receivers, steps, channels)
        states = []
        for part in flat.split(RECURRENT_CHUNK):
            _, hidden = self.profile_gru(part)
            states.append(hidden[0])
        return torch.cat(states).reshape(batch, receivers, HIDDEN)

    def forward(self, features):
        if self._profile_context is None:
            raise RuntimeError("PathProfileConv must run inside PathProfileRadioWNet")
        if tuple(features.shape[1:]) != (40, 64, 64):
            raise ValueError("Path profile block expects Bx40x64x64 features")
        base = super().forward(features)
        source_input, config = self._profile_context
        tokens = path_profile_tokens(source_input, config)
        with torch.autocast(device_type=features.device.type, enabled=False):
            state = self.encode_profiles(tokens)
            local = base.float().flatten(2).transpose(1, 2)
            output = self.profile_fusion(torch.cat((local, state), -1))
        return output.transpose(1, 2).reshape_as(base).to(base.dtype)


class PathProfileRadioWNet(load_module().RadioWNet):
    """Reuse the upstream forward and keep input geometry scoped to one call."""
    def __init__(self, inputs=2, phase="secondU"):
        if inputs != 2 or phase != "secondU":
            raise ValueError("Path profiles support a single-band secondU only")
        super().__init__(inputs=inputs, phase=phase)
        with torch.random.fork_rng(devices=[]):
            self.Wlayer10[0] = PathProfileConv(self.Wlayer10[0])

    def forward(self, input):
        if self.phase != "secondU":
            raise ValueError("Path profiles require the existing secondU phase")
        if input.ndim != 4 or tuple(input.shape[1:]) != (5, 256, 256):
            raise ValueError("Expected expanded [H, TX, F, rho, 0] input")
        operator = self.Wlayer10[0]
        operator._profile_context = (input, self.feature_config)
        try:
            return super().forward(input)
        finally:
            operator._profile_context = None
