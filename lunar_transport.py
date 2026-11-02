"""Source-conditioned scene transport with point/structured-field controls."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from lunar_conv import LunarConvAligned, conv_model_config
from lunar_maxvit import _double_conv
from transport_features import transport_feature_config, transport_features
from transport_field import LocalFieldRenderer, field_config
from transport_operator import SceneTransport


def transport_model_config(arch="lunar_transport_field"):
    if arch not in ("lunar_transport_point", "lunar_transport_field"):
        raise ValueError(f"Unsupported transport architecture {arch!r}")
    config = conv_model_config("lunar_conv_aligned")
    config["arch"] = arch
    config["version"] = 1
    config["scene_transport"] = {
        "implementation": "lunar_transport.LunarTransport", "geometry": transport_feature_config(),
        "local_backbone": "C_encoder_and_five_decoder_blocks; no_original_T_single_TX_attention_or_output",
        "surface_states": "16x16 sites x4 direction-conditioned slots=1024, width64",
        "directions": [[1, 0], [0, 1], [-1, 0], [0, -1]],
        "interaction": "4 dense single-head SDPA rounds; source injected every round; pre-LN FFN4x; layer scales0.1",
        "geometry_relation": "learned projections of 9 input geometry scalars plus direction code into Q/K; no hard visibility graph",
        "receiver_query": "64x64 stride4 queries; 4 adjacent stride16 sites x4 slots plus16 source-conditioned global summaries; duplicate local keys masked",
        "decoder_injection": "receiver query residual at D64, followed by shared D128/D256; unscaled64-channel query context also reaches readout directly",
        "readout_features": "double3x3 Conv29->32 on D256(16)+physical(4)+geometry(9); concatenate64-channel interpolated query context ->96",
        "readout": "Conv96->32 k3,pad1,ReLU; FP32 1x1 output; output_std0.01; stride1 point/stride4 field",
        "transport_numerics": "FP32 latent/source/receiver attention under AMP; convolution backbone/readout projection retain AMP",
        "initialization": "same_seed_shared_backbone_and_post_constructor_CPU_RNG; readouts_built_in_rng_fork; fresh_joint_training",
        "meaning": "learned propagation context, not ray records, mechanism powers or PDE field",
        "inference": "H/TX/F only; rho/geometry live; no_target_mask_RT_or_iterative_test_optimization",
    }
    config["decoder"]["transport_readout"] = field_config() if arch.endswith("field") else {
        "version": 1, "output": "FP32_1x1_normalized_total_PL_ReLU", "readout_stride": 1}
    config["output"]["parameterization"] = "overlapping_local_total_PL_field" if arch.endswith("field") else "total_PL_point_prediction"
    return config


class _Readout(nn.Module):
    def __init__(self, field):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(96, 32, 3, stride=4 if field else 1, padding=1), nn.ReLU())
        self.output = nn.Conv2d(32, 9 if field else 1, 1)
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[0] = .5
            if field:
                # width=1px at initialization; all bounds are fixed model config.
                ratio = (1 - .15) / 1.85
                self.output.bias[8] = math.log(ratio / (1 - ratio))

    def forward(self, features):
        hidden = self.project(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.output(hidden.float())


class LunarTransport(nn.Module):
    def __init__(self, feature_config, arch="lunar_transport_field"):
        super().__init__()
        self.model_config = transport_model_config(arch)
        self.arch, self.phase, self.base_inputs = arch, "single", 2
        self.feature_config = feature_config
        base = LunarConvAligned(feature_config, arch="lunar_conv_aligned")
        self.encoder = base.encoder
        self.full_resolution = base.full_resolution
        self.decoder = base.decoder
        self.transport = SceneTransport()
        self.readout_features = _double_conv(16 + 4 + 9, 32)
        self.is_field = arch == "lunar_transport_field"
        # Equal-seed common weights and training RNG independent of readout type.
        with torch.random.fork_rng(devices=[]):
            self.readout = _Readout(self.is_field)
        self.renderer = LocalFieldRenderer() if self.is_field else None

    def _forward_shared(self, input):
        if input.ndim != 4 or tuple(input.shape[1:]) != (3, 256, 256):
            raise ValueError("LunarTransport expects Bx3x256x256 [H,TX,F]")
        if not input.is_floating_point():
            raise ValueError("LunarTransport requires normalized floating-point input")
        physical, geometry, tx_index = transport_features(input, self.feature_config)
        full = self.full_resolution(physical)
        stem = self.encoder.stem(physical)
        features = [full, stem]
        encoded = stem
        for block in self.encoder.blocks:
            encoded = block(encoded)
            features.append(encoded)
        states, source = self.transport.encode(features[-2], features[-1], full, geometry, tx_index)
        for index, (decoder, skip) in enumerate(zip(self.decoder, reversed(features[:-1]))):
            encoded = decoder(encoded, skip)
            if index == 2:
                encoded, propagation = self.transport.query(encoded, geometry, states, source)
        local = self.readout_features(torch.cat((encoded, physical, geometry.to(encoded.dtype)), dim=1))
        propagation = F.interpolate(propagation, size=(256, 256), mode="bilinear", align_corners=False)
        return torch.cat((local, propagation.to(local.dtype)), dim=1)

    def forward(self, input):
        raw = self.readout(self._forward_shared(input))
        prediction = self.renderer(raw)[0] if self.is_field else F.relu(raw)
        return [prediction]

    def forward_with_boundary(self, input, pl_scale):
        if not self.is_field:
            raise ValueError("Boundary supervision requires lunar_transport_field")
        raw = self.readout(self._forward_shared(input))
        prediction, boundary, _ = self.renderer(raw, boundary_scale=pl_scale)
        return prediction, boundary

    def forward_field(self, input, pl_scale=208.):
        """Input-only diagnostic maps; PL scale must match dataset normalization."""
        if not self.is_field:
            raise ValueError("Field diagnostics require lunar_transport_field")
        raw = self.readout(self._forward_shared(input))
        prediction, boundary, parameters = self.renderer(raw, boundary_scale=pl_scale, diagnostics=True)
        return prediction, {**parameters, "boundary_strength": boundary}
