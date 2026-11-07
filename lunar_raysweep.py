"""T with input-only terrain morphology, radial memory and ground-surface context."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from lunar_conv import LunarConvAligned, conv_model_config
from ray_sweep_geometry import (PolarLattice, interaction_geometry, sample_lattice,
                                source_geometry, surface_lattice, terrain_geometry)


RAYSWEEP_ARCHITECTURES = ("lunar_raysweep_sweep", "lunar_raysweep_scatter", "lunar_raysweep_full")


def raysweep_model_config(arch):
    if arch not in RAYSWEEP_ARCHITECTURES:
        raise ValueError(f"Unsupported RaySweep architecture {arch}")
    config = deepcopy(conv_model_config("lunar_conv_tx_content"))
    config.update(arch=arch, version=1)
    config["terrain_morphology"] = {
        "input": ["H_normalized", "normal_x", "normal_y", "asinh(height_m-mean9_height_m)"],
        "normal": "metre_central_difference_replicate_border_upward",
        "relief": "9x9_mean_reflect_padding_scale_1m",
        "encoder": "Conv3_s2_16_GELU_Conv3_s2_32_GELU_DWConv3_GELU_Conv1_32",
        "lattice_px": {"origin": 0, "stride": 4, "size": 64},
        "depends_on": "H_only_recomputed_after_joint_augmentation",
    }
    config["radial_sweep"] = None
    if arch != "lunar_raysweep_scatter":
        config["radial_sweep"] = {
            "angles": 512, "radial_step_m": 2.0, "radius": "ceil(tile_diagonal/2)*2",
            "width": 16, "blocks": 2, "decay_length_m": [16.0, 256.0],
            "recurrence": "s_r=a_r*s_previous+(1-a_r)*value_r; a=exp(-dr/tau_r)",
            "evaluation": "FP32_bounded_log_prefix_product_and_weighted_cumsum",
            "source": "learned_projection_of_terrain_feature_at_actual_TX",
            "geometry": ["relative_ground_height/height_span", "RX_elevation_angle/pi",
                         "RX_angle_minus_exclusive_ground_horizon/pi", "r/tile_diagonal",
                         "dominant_upstream_ridge_radius/r", "viewshed_gap_integral/(pi*r)",
                         "F_RX", "in_tile"],
            "angular_mixing": "circular_depthwise3_then_pointwise_after_each_prefix_scan",
            "outside_tile": "freeze_recurrence_zero_features_exclude_from_horizon",
            "return_sampling": "periodic_bilinear_support_normalized_to_64_origin0_stride4",
            "fusion": "add_16to64_projection_normal_std0.001_bias0_after_encoder_block0",
        }
    config["ground_scatter"] = None
    if arch != "lunar_raysweep_sweep":
        config["tx_scene_attention"] = None
        config["ground_scatter"] = {
            "width": 128, "attention_dim": 64, "heads": 4, "query_chunk": 128,
            "anchor_lattice_px": {"origin": 4, "stride": 16, "size": 16},
            "query_lattice_px": {"origin": 0, "stride": 8, "size": 32},
            "heights": "TX=H+tx_agl; anchor=H; RX=H+rx_agl",
            "source_features": "terrain_anchor,terrain_TX,their_product,12_geometry_descriptors,optional_sweep16",
            "source_mlp": "Linear_GELU_Linear_to128",
            "source_geometry": ["nx", "ny", "nz", "dx/diagonal", "dy/diagonal",
                                "dz/height_span", "log1p(d1)/log1p(D)", "normal_dot_to_TX",
                                "asinh(ground_clearance)/asinh(height_span)", "blocked_fraction",
                                "minimum_clearance_position/segment_length", "asinh(local_relief_m)"],
            "source_clearance": "half_pixel_bilinear_TX_to_ground_anchor_interior_only",
            "surface_interactions": 1,
            "surface_context_initialization": "torch_default",
            "receiver_query": "encoder32,terrain_TX,F_RX,rho",
            "pair_bias": ["log1p(d2)/log1p(D)", "normal_dot_outgoing",
                          "specular_direction_mismatch_squared/4", "log1p(path_excess)/log1p(D)",
                          "outgoing_z", "asinh(second_leg_clearance)/asinh(height_span)"],
            "second_leg": "8_uniform_interior_samples_soft_descriptor_no_visibility_mask",
            "pair_bias_mlp": {"channels": [6, 16, 4], "last_bias": False,
                              "initialization": "torch_default"},
            "readout": "nonlinear_sources_then_ground_context_then_RX_queries_before_block2",
            "output_initialization": "normal_std0.001_bias0",
        }
    config["new_branch_precision"] = "FP32_with_autograd_output_cast_to_main_dtype"
    config["physical_interpretation"] = "learned_geometry_context_not_identified_path_powers"
    return config


def small_projection(layer):
    nn.init.normal_(layer.weight, std=0.001)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class TerrainMorphologyEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.GELU(),
            nn.Conv2d(32, 32, 1))


class RadialMemory(nn.Module):
    """Exact full-prefix recurrence in parallel, with bounded FP32 exponents."""

    def __init__(self, width=16):
        super().__init__()
        self.value = nn.Conv2d(width, width, 1)
        self.decay = nn.Conv2d(width, width, 1)
        self.angular = nn.Conv2d(width, width, (3, 1), groups=width)
        self.output = nn.Conv2d(width, width, 1)

    def forward(self, input, source, valid, radial_step):
        tau = 16.0 + 240.0 * self.decay(input).sigmoid()
        dr = input.new_full((input.shape[-1],), radial_step)
        dr[0] = 0
        log_decay = (-dr / tau).masked_fill(~valid, 0)
        product = log_decay.cumsum(-1).exp()
        increment = -torch.expm1(log_decay) * F.gelu(self.value(input))
        state = product * (source[:, :, None, None] + (increment / product).cumsum(-1))
        state = state.masked_fill(~valid, 0)
        mixed = self.angular(F.pad(state, (0, 0, 1, 1), mode="circular"))
        return (input + self.output(F.gelu(mixed))).masked_fill(~valid, 0)


class PolarCausalSweep(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lattice = PolarLattice(config)
        self.input = nn.Conv2d(32 + 8, 16, 1)
        self.source = nn.Linear(32, 16)
        self.blocks = nn.ModuleList([RadialMemory(), RadialMemory()])
        self.projection = nn.Conv2d(16, 64, 1)
        small_projection(self.projection)

    def forward(self, terrain, height, clearance, tx_xyz, source):
        xy, valid, geometry = self.lattice(height, clearance, tx_xyz)
        sampled = sample_lattice(terrain, xy, 4 * self.lattice.spacing)
        states = self.input(torch.cat((sampled, geometry), dim=1)).masked_fill(~valid, 0)
        source = self.source(source)
        for block in self.blocks:
            states = block(states, source, valid, self.lattice.step_m)
        context = self.lattice.to_cartesian(states, valid, tx_xyz)
        return self.projection(context), context


class GroundScatterAttention(nn.Module):
    def __init__(self, config, wave_channels):
        super().__init__()
        self.config = config
        self.source_mlp = nn.Sequential(nn.Linear(3 * 32 + 12 + wave_channels, 128),
                                         nn.GELU(), nn.Linear(128, 128))
        self.surface_norm = nn.LayerNorm(128)
        self.surface_qkv = nn.Linear(128, 3 * 64)
        self.surface_output = nn.Linear(64, 128)
        self.query = nn.Sequential(nn.Linear(128 + 32 + 2, 64), nn.GELU(), nn.Linear(64, 64))
        self.key_value = nn.Linear(128, 2 * 64)
        self.geometry_bias = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 4, bias=False))
        self.output = nn.Linear(64, 128)
        small_projection(self.output)

    @staticmethod
    def split_heads(value):
        return value.reshape(value.shape[0], value.shape[1], 4, 16).transpose(1, 2)

    def attend(self, query, key, value, receivers, height, source_data, tx_xyz):
        anchors, normals, reflection, source_distance = source_data[:4]
        key, value = self.split_heads(key), self.split_heads(value)
        outputs = []
        for start in range(0, query.shape[1], 128):
            stop = start + 128
            geometry = interaction_geometry(height, anchors, normals, reflection, source_distance,
                                            receivers[:, start:stop], tx_xyz, self.config)
            bias = self.geometry_bias(geometry).permute(0, 3, 1, 2).contiguous()
            attended = F.scaled_dot_product_attention(
                self.split_heads(query[:, start:stop]), key, value, attn_mask=bias)
            outputs.append(attended.transpose(1, 2).reshape(query.shape[0], -1, 64))
        return torch.cat(outputs, dim=1)

    def forward(self, features, terrain, height, normal, relief, physical, tx_xyz, source, wave):
        source_data = source_geometry(height, normal, relief, tx_xyz, self.config)
        anchors, _, _, _, descriptors, xy = source_data
        anchor_features = sample_lattice(terrain, xy, 4 * self.config["resolution_m_per_px"])
        anchor_features = anchor_features.flatten(2).transpose(1, 2)
        source_expanded = source[:, None].expand_as(anchor_features)
        parts = [anchor_features, source_expanded, anchor_features * source_expanded, descriptors]
        if wave is not None:
            parts.append(sample_lattice(wave, xy, 4 * self.config["resolution_m_per_px"])
                         .flatten(2).transpose(1, 2))
        states = self.source_mlp(torch.cat(parts, dim=-1))
        q, k, v = self.surface_qkv(self.surface_norm(states)).chunk(3, dim=-1)
        # Shadowed ground anchors remain in the graph and can receive context.
        states = states + self.surface_output(
            self.attend(q, k, v, anchors, height, source_data, tx_xyz))
        receivers, _, _, _ = surface_lattice(height, normal, relief, 0, 8, 32,
                                             self.config["resolution_m_per_px"],
                                             self.config["rx_height_m_agl"])
        query_features = features.flatten(2).transpose(1, 2)
        extra = physical[:, 2:4, ::8, ::8].flatten(2).transpose(1, 2)
        query = self.query(torch.cat((query_features,
                           source[:, None].expand(-1, 1024, -1), extra), dim=-1))
        key, value = self.key_value(states).chunk(2, dim=-1)
        context = self.attend(query, key, value, receivers, height, source_data, tx_xyz)
        update = self.output(context).transpose(1, 2).reshape_as(features)
        return features + update.to(features.dtype)


class LunarRaySweepNet(LunarConvAligned):
    def __init__(self, feature_config, arch="lunar_raysweep_full"):
        super().__init__(feature_config, arch="lunar_conv_tx_content")
        # Reuse T's shared random initialization and the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.terrain_encoder = TerrainMorphologyEncoder()
            self.sweep = PolarCausalSweep(feature_config) if arch != "lunar_raysweep_scatter" else None
            self.scatter = GroundScatterAttention(
                feature_config, 16 if self.sweep is not None else 0) if arch != "lunar_raysweep_sweep" else None
        if self.scatter is not None:
            self.tx_scene_attention = None
        self.arch = arch
        self.model_config = raysweep_model_config(arch)

    def _encoder_features(self, physical, input):
        with torch.autocast(device_type=input.device.type, enabled=False):
            height, normal, relief, shape, tx_xyz = terrain_geometry(input, self.feature_config)
            terrain = self.terrain_encoder(shape)
            source = sample_lattice(terrain, tx_xyz[:, None, None, :2],
                                    4 * self.feature_config["resolution_m_per_px"]).flatten(1)
        features = [self.full_resolution(physical), self.encoder.stem(physical)]
        encoded = features[-1]
        wave = None
        for index, block in enumerate(self.encoder.blocks):
            encoded = block(encoded)
            if index == 0 and self.sweep is not None:
                with torch.autocast(device_type=input.device.type, enabled=False):
                    update, wave = self.sweep(terrain, height, input[:, 2:3], tx_xyz, source)
                encoded = encoded + update.to(encoded.dtype)
            if index == 1:
                if self.scatter is not None:
                    with torch.autocast(device_type=input.device.type, enabled=False):
                        encoded = self.scatter(encoded.float(), terrain, height, normal, relief,
                                               physical.float(), tx_xyz, source, wave).to(encoded.dtype)
                else:
                    encoded = self.tx_scene_attention(encoded, input[:, 1:2], input[:, :1])
            features.append(encoded)
        return features
