"""Input-only two-leg terrain geometry for one source-conditioned interaction.

This is a learned feature bias, not ray tracing, visibility or power transport.
The six descriptors approximate TX -> ground anchor -> terrain-following RX.
No target, validity mask, clearance-based visibility gate or phase is used.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from lunar_maxvit import _TXSceneAttention, lunar_model_config


def terrain_attention_config(geometry):
    """Record the content control and the exact extra geometry definition."""
    config = lunar_model_config("lunar_maxvit_c")["tx_scene_attention"]
    if geometry:
        config.update({
            "implementation": "terrain_interaction.TerrainSceneAttention",
            "geometry_bias": True,
            "geometry": {
                "version": 1,
                "inputs": "augmented_normalized_H_and_one_hot_TX_only",
                "coordinates": "x=column_m,y=row_m,z=height_m",
                "query_lattice_px": {"origin": 0, "stride": 8, "size": 32},
                "anchor_lattice_px": {"origin": 4, "stride": 16, "size": 16},
                "anchor_height": "ground_at_pooling_center_not_RX_height_or_pooled_height",
                "query_height": "ground_plus_metadata_rx_height",
                "tx_height": "ground_plus_metadata_tx_height",
                "normal": "normalize([-dxH,-dyH,1])_at_anchor",
                "height_derivative": "central_difference_on_full_grid_in_metres",
                "normal_stencil": "anchor_plus_minus_one_input_pixel_each_axis",
                "features": [
                    "log1p(d_tx_anchor)/log1p(D)",
                    "log1p(d_anchor_rx)/log1p(D)",
                    "log1p(max(d_tx_anchor+d_anchor_rx-d_tx_rx,0)/max(d_tx_rx,eps))"
                    "/log1p(2*D/d_min)",
                    "dot(normal,unit_anchor_to_tx)",
                    "dot(normal,unit_anchor_to_rx)",
                    "squared_specular_direction_mismatch/4",
                ],
                "distance_unit_m": 1.0, "distance_epsilon_m": 1e-6,
                "D": "sqrt(sum(grid_xy_extents_m^2)+(height_span+max(tx_agl,rx_agl))^2)",
                "d_min": "max(eps,min(pixel_spacing,abs(tx_agl-rx_agl)))",
                "normalization": "analytic_metadata_bounds_no_fitted_statistics",
                "bias_mlp": {"channels": [6, 16, 4], "activation": "GELU",
                             "first_bias": True, "last_bias": False,
                             "initialization": "torch_default_after_shared_QKV_output"},
                "computation_dtype": "float32_under_amp",
                "input_geometry_gradients": False,
                "visibility_mask": None, "physical_power_constraint": None,
            },
        })
    return config


class TerrainSceneAttention(_TXSceneAttention):
    """T's content aggregation plus a six-input, per-head geometry bias.

    Shared Q/K/V/output are constructed first; only the output projection is
    zero initialized. Geometry MLP uses default initialization so it can receive
    gradients once the output starts learning. It adds 176 parameters.

    Time: O(B*N*K*d) attention + O(B*N*K*r*(6+h)) geometry MLP, with N=1024,
    K=256, d=64, r=16, h=4, plus shared projections. Geometry and saved MLP
    activations cost O(B*N*K*(6+r+h)); attention O(B*h*N*K) if materialized.
    These are feature interactions, not physical paths or power coefficients.
    """
    def __init__(self, feature_config):
        super().__init__()
        self.feature_config = feature_config
        self.geometry_mlp = nn.Sequential(nn.Linear(6, 16), nn.GELU(),
                                          nn.Linear(16, self.heads, bias=False))
        lo, hi = feature_config["heightmap_range_m"]
        h, w = feature_config["grid"]
        spacing = feature_config["resolution_m_per_px"]
        tx_agl, rx_agl = (feature_config["tx_height_m_agl"],
                          feature_config["rx_height_m_agl"])
        # Bounds depend only on training metadata, never on a batch or split.
        maximum = math.sqrt(((h - 1) * spacing) ** 2 + ((w - 1) * spacing) ** 2 +
                            (hi - lo + max(tx_agl, rx_agl)) ** 2)
        self.log_distance_scale = math.log1p(maximum)
        minimum = max(1e-6, min(spacing, abs(tx_agl - rx_agl)))
        self.log_excess_scale = math.log1p(2 * maximum / minimum)

    @torch.no_grad()
    def geometry_features(self, height, tx):
        """Return Bx1024x256x6 descriptors, in query/pooled-anchor order.

        The backbone's stride-8 grid is 0,8,...,248; 2x2 pooled features have
        centers 4,20,...,244. Thus all anchor normal stencils are in bounds.
        Recompute after joint augmentation; no rotated cached normals are used.
        Relative excess length is log-compressed before analytic normalization
        to keep near-TX queries from dominating the other descriptors.
        """
        if height is None or height.ndim != 4 or tuple(height.shape[1:]) != (1, 256, 256):
            raise ValueError("Terrain interaction requires Bx1x256x256 normalized height")
        if tuple(tx.shape) != tuple(height.shape):
            raise ValueError("Terrain interaction height and one-hot TX must have matching shapes")
        with torch.autocast(device_type=height.device.type, enabled=False):
            config = self.feature_config
            lo, hi = config["heightmap_range_m"]
            ground = height[:, 0].float() * (hi - lo) + lo
            batch = ground.shape[0]
            spacing = config["resolution_m_per_px"]
            index = tx.flatten(1).argmax(1)
            tx_xy = torch.stack(((index % 256).float(), (index // 256).float()), dim=-1)
            tx_z = ground.flatten(1).gather(1, index[:, None]) + config["tx_height_m_agl"]
            transmitter = torch.cat((tx_xy * spacing, tx_z), dim=-1)

            query_axis = torch.arange(0, 256, 8, device=height.device)
            anchor_axis = torch.arange(4, 256, 16, device=height.device)
            query_row, query_col = torch.meshgrid(query_axis, query_axis, indexing="ij")
            anchor_row, anchor_col = torch.meshgrid(anchor_axis, anchor_axis, indexing="ij")
            qr, qc = query_row.flatten(), query_col.flatten()
            ar, ac = anchor_row.flatten(), anchor_col.flatten()
            query_xy = torch.stack((qc, qr), dim=-1).float() * spacing
            anchor_xy = torch.stack((ac, ar), dim=-1).float() * spacing
            receiver = torch.cat((query_xy[None].expand(batch, -1, -1),
                                  (ground[:, qr, qc] + config["rx_height_m_agl"])[..., None]), dim=-1)
            anchor = torch.cat((anchor_xy[None].expand(batch, -1, -1),
                                ground[:, ar, ac, None]), dim=-1)
            dx = (ground[:, ar, ac + 1] - ground[:, ar, ac - 1]) / (2 * spacing)
            dy = (ground[:, ar + 1, ac] - ground[:, ar - 1, ac]) / (2 * spacing)
            normal = F.normalize(torch.stack((-dx, -dy, torch.ones_like(dx)), dim=-1), dim=-1)

            incident = anchor - transmitter[:, None]
            d1 = torch.linalg.vector_norm(incident, dim=-1).clamp_min(1e-6)
            incident = incident / d1[..., None]
            outgoing = receiver[:, :, None] - anchor[:, None]
            d2 = torch.linalg.vector_norm(outgoing, dim=-1).clamp_min(1e-6)
            outgoing = outgoing / d2[..., None]
            direct = torch.linalg.vector_norm(receiver - transmitter[:, None], dim=-1).clamp_min(1e-6)
            incidence = (normal * incident).sum(-1)
            reflection = incident - 2 * incidence[..., None] * normal
            extra = (d1[:, None] + d2 - direct[..., None]).clamp_min(0) / direct[..., None]
            result = torch.stack((
                torch.log1p(d1)[:, None].expand_as(d2) / self.log_distance_scale,
                torch.log1p(d2) / self.log_distance_scale,
                torch.log1p(extra) / self.log_excess_scale,
                (-incidence)[:, None].expand_as(d2),
                (normal[:, None] * outgoing).sum(-1),
                (outgoing - reflection[:, None]).square().sum(-1) / 4,
            ), dim=-1)
        return result

    def attention_bias(self, tx, height):
        # Called inside the shared attention's FP32 context. Keep the MLP in
        # autograd even though its physical inputs need no gradients.
        geometry = self.geometry_features(height, tx)
        return self.geometry_mlp(geometry).permute(0, 3, 1, 2)
