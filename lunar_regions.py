"""Shared T backbone with visibility-conditioned heads and boundary correction.

All three branches predict normalized TOTAL path loss (or its boundary
correction). They are not simulated mechanism powers. Geometry is a soft,
input-only prior; there is no hard crop, mask input or forced shadow null.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from los_features import log_distance_feature
from lunar_conv import LunarConvAligned, conv_model_config
from lunar_maxvit import _double_conv
from region_features import region_feature_config, region_features


_MODES = ("uniform", "fixed", "learned")


def region_model_config(arch="lunar_conv_tx_region_learned"):
    if arch not in tuple(f"lunar_conv_tx_region_{mode}" for mode in _MODES):
        raise ValueError(f"Unsupported region architecture {arch!r}")
    mode = arch.rsplit("_", 1)[-1]
    config = conv_model_config("lunar_conv_tx_content")
    config["arch"] = arch
    config["decoder"]["region_prediction"] = {
        "implementation": "lunar_regions.LunarConvRegions", "version": 1,
        "mode": mode, "shared_decoder_resolutions": [16, 32, 64],
        "specialized_decoder_resolutions": [128, 256],
        "visible_head": "original_T_last_two_decoder_blocks_and_output",
        "shadow_head": "independent_T_last_two_blocks; signed_correction_on_shared_visible_total",
        "boundary_head": {
            "channels": 16, "context": "shared_D64_and_native_stem/full_resolution_skips",
            "upsample": "bilinear_align_corners_false_then_1x1_projection",
            "fusion": "two_3x3_convs_ReLU_at_each_of_128_and_256",
            "geometry_channels": 5, "output": "signed_3x3_conv_zero_initialized",
        },
        "geometry": region_feature_config(),
        "routing": {
            "formula": "p=sigmoid((distance_m+delta_m)/tau_m); b=4*p*(1-p)",
            "uniform": "p=0.5,b=1", "fixed": "delta=0,tau=2*resolution_m",
            "learned": "1x1_Conv([boundary_features,geometry],2)",
            "delta_m": "resolution_m*raw_delta; unconstrained_signed",
            "tau_m": "resolution_m*(0.5+softplus(raw_tau))",
            "initial_delta_px": 0.0, "initial_tau_px": 2.0,
            "computation": "FP32_under_AMP; no_detach_on_learned_router",
            "meaning": "learned_prediction_weight_not_physical_LOS_probability",
        },
        "fusion": "ReLU(Lvisible+(1-p)*delta_shadow+b*Rboundary); Lshadow=Lvisible+delta_shadow",
        "quantities": "two_total_normalized_PL_predictions_and_signed_PL_correction",
        "initialization": "exact_seeded_T_function; shadow_decoder_copy_zero_correction; zero_boundary_output; zero_router_delta",
        "initialization_rng": "fork_cpu_rng_after_shared_T_construction",
        "trainability": "joint_end_to_end_all_existing_and_new_branch_parameters",
        "inference": "all_branches_dense; soft_gate_does_not_skip_computation",
    }
    config["output"]["parameterization"] = "soft_region_total_PL_fusion_with_boundary_correction"
    return config


class _BoundaryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.context128 = nn.Conv2d(64, 16, 1)
        self.fuse128 = _double_conv(16 + 64 + 5, 16)
        self.context256 = nn.Conv2d(16, 16, 1)
        self.fuse256 = _double_conv(16 + 16 + 5, 16)
        self.output = nn.Conv2d(16 + 4, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context64, stem128, full256, physical, geometry):
        coarse = F.interpolate(context64, scale_factor=2, mode="bilinear", align_corners=False)
        geo128 = F.avg_pool2d(geometry, 2, 2)
        hidden = self.fuse128(torch.cat((self.context128(coarse), stem128, geo128.to(coarse.dtype)), dim=1))
        fine = F.interpolate(hidden, scale_factor=2, mode="bilinear", align_corners=False)
        hidden = self.fuse256(torch.cat((self.context256(fine), full256, geometry.to(fine.dtype)), dim=1))
        correction = self.output(torch.cat((hidden, physical), dim=1))
        return correction, hidden


class LunarConvRegions(LunarConvAligned):
    """Share global T context; specialize inside its final two decoder levels.

The baseline encoder, TX interaction, shallow skip and decoder weights retain
    their same-seed initialization/RNG stream. Shadow copies start identical to
    visible but use a zero signed correction output, so even uniform routing
    does not leave two identical heads permanently symmetric. The boundary
    output starts at zero. The initial total prediction is
    therefore T's, without loading any trained checkpoint or freezing T.

    New work: O(BHW*c^2) for fixed-width heads and O(R*BHW) for R=16 grid-distance
    erosions. Additional activation storage is O(BHW*c); no second encoder,
    RT inference, CPU distance transform or mechanism-feature cache is required.
    """
    def __init__(self, feature_config, arch="lunar_conv_tx_region_learned"):
        config = region_model_config(arch)
        super().__init__(feature_config, arch="lunar_conv_tx_content")
        self._t_state_keys = tuple(self.state_dict())
        self.region_mode = config["decoder"]["region_prediction"]["mode"]
        self.shadow_decoder = copy.deepcopy(self.decoder[3:])
        with torch.random.fork_rng(devices=[]):
            self.shadow_output = nn.Conv2d(16 + 4, 1, 3, padding=1)
            self.boundary_decoder = _BoundaryDecoder()
            self.router = nn.Conv2d(16 + 5, 2, 1) if self.region_mode == "learned" else None
        self.reset_boundary_routing()
        self.arch = arch
        self.model_config = config

    def reset_boundary_routing(self):
        nn.init.zeros_(self.shadow_output.weight)
        nn.init.zeros_(self.shadow_output.bias)
        nn.init.zeros_(self.boundary_decoder.output.weight)
        nn.init.zeros_(self.boundary_decoder.output.bias)
        if self.router is not None:
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
            with torch.no_grad():
                # tau/resolution=0.5+softplus(bias)=2 at initialization.
                self.router.bias[1] = math.log(math.expm1(1.5))

    def initialize_from_T(self, state):
        """Explicit weights-only migration; strict shared keys and no freezing."""
        if set(state) != set(self._t_state_keys):
            raise ValueError("T initialization requires the exact original T state keys")
        merged = self.state_dict()
        for name in self._t_state_keys:
            if state[name].shape != merged[name].shape:
                raise ValueError(f"T initialization shape differs for {name}")
            merged[name] = state[name]
        self.load_state_dict(merged, strict=True)
        self.shadow_decoder.load_state_dict(self.decoder[3:].state_dict(), strict=True)
        self.reset_boundary_routing()

    def _forward_region_features(self, input):
        if input.ndim != 4 or tuple(input.shape[1:]) != (3, 256, 256):
            raise ValueError("LunarConvRegions expects Bx3x256x256 [H,TX,F]")
        if not input.is_floating_point():
            raise ValueError("LunarConvRegions requires normalized floating-point input")
        rho = log_distance_feature(input, self.feature_config)
        physical = torch.cat((input, rho), dim=1)
        distance_m, geometry = region_features(input, self.feature_config)
        full = self.full_resolution(physical)
        stem = self.encoder.stem(physical)
        features = [full, stem]
        encoded = stem
        for index, block in enumerate(self.encoder.blocks):
            encoded = block(encoded)
            if index == 1:
                encoded = self.tx_scene_attention(encoded, input[:, 1:2], input[:, :1])
            features.append(encoded)
        for decoder, skip in zip(self.decoder[:3], reversed(features[2:-1])):
            encoded = decoder(encoded, skip)
        context64 = encoded
        visible128 = self.decoder[3](context64, stem)
        visible256 = self.decoder[4](visible128, full)
        visible = self.output(torch.cat((visible256, physical), dim=1))
        shadow = self.shadow_decoder[0](context64, stem)
        shadow = self.shadow_decoder[1](shadow, full)
        shadow_delta = self.shadow_output(torch.cat((shadow, physical), dim=1))
        correction, boundary_features = self.boundary_decoder(
            context64, stem, full, physical, geometry)
        # Tiny geometry/routing gradients should not be rounded through FP16.
        with torch.autocast(device_type=input.device.type, enabled=False):
            resolution = self.feature_config["resolution_m_per_px"]
            if self.router is None:
                delta_m = torch.zeros_like(distance_m)
                tau_m = torch.full_like(distance_m, 2 * resolution)
            else:
                raw = self.router(torch.cat((boundary_features.float(), geometry), dim=1))
                delta_m = raw[:, :1] * resolution
                tau_m = (.5 + F.softplus(raw[:, 1:])) * resolution
            if self.region_mode == "uniform":
                p = torch.full_like(distance_m, .5)
            else:
                p = torch.sigmoid((distance_m + delta_m) / tau_m)
            boundary_weight = 4 * p * (1 - p)
            # Equivalent to p*Lv+(1-p)*Ls, with Ls=Lv+delta_shadow.
            # This parametrization preserves T exactly without the permanent
            # symmetry of two copied total heads under uniform routing.
            shadow = visible.float() + shadow_delta.float()
            prediction = F.relu(visible.float() + (1 - p) * shadow_delta.float()
                                + boundary_weight * correction.float())
        diagnostics = {
            "visible_prediction": visible, "shadow_prediction": shadow,
            "shadow_correction": shadow_delta,
            "boundary_correction": correction, "visible_weight": p,
            "boundary_weight": boundary_weight, "distance_m": distance_m,
            "delta_m": delta_m, "tau_m": tau_m,
        }
        return prediction, physical, [context64, visible128, visible256], diagnostics

    def forward_features(self, input):
        """Routed prediction plus physical input and visible-path decoder states."""
        prediction, physical, decoded, _ = self._forward_region_features(input)
        return prediction, physical, decoded

    def forward_regions(self, input):
        """Complete forward plus routing diagnostics; forward() keeps [prediction]."""
        prediction, _, _, diagnostics = self._forward_region_features(input)
        return prediction, diagnostics

    def forward(self, input):
        prediction, _ = self.forward_regions(input)
        return [prediction]
