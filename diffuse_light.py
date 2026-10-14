"""Input-only lightweight single-bounce Lambertian context feature.

The material follows the Sionna RT 1.1.0 reference implementation. Numerical
shortcuts are dual-center angular quadrature, float32 half-pixel ray sampling,
and a coarse receiver grid interpolated in linear power. This is an untrained
proxy, not the simulator's diffuse component or an accurate boundary map.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from diffuse_features import (
    SPEED_OF_LIGHT, diffuse_feature_config, direction_quadrature, slab_coefficients,
)
from diffuse_light_geometry import first_hit, sample_surface, visible_segment


METHOD = "dual_center_light_v1"
QUANTITY = "lightweight_single_bounce_lambertian_gain_proxy"


def light_diffuse_config(meta, band="58", n_mu=4, n_phi=16, rx_grid=16):
    """Validated, JSON-serializable physical and approximation configuration.

    The default numerical budget is a versioned engineering choice, not a
    demonstrated accuracy optimum. No physical constant is tuned for speed.
    """
    if not isinstance(rx_grid, int) or isinstance(rx_grid, bool) or rx_grid < 2:
        raise ValueError("rx_grid must be an integer >= 2")
    config = diffuse_feature_config(meta, str(band), n_mu, n_phi)
    config["quantity"] = QUANTITY
    config["method"] = METHOD
    config["surface"] = "bilinear_reconstruction_original_resolution_not_original_mesh"
    config["quadrature"].update({
        "domain": "full_sphere_at_RX_and_TX",
        "rule": "uniform_cos_theta_azimuth_midpoints_dual_center_partition",
        "partition_RX": "r_tx_q^2/(r_tx_q^2+r_q_rx^2)",
        "partition_TX": "r_q_rx^2/(r_tx_q^2+r_q_rx^2)",
        "reflectance": "true_TX_incidence_and_V_polarization_in_both_integrals",
    })
    config["receiver_sampling"] = {
        "grid_size": rx_grid,
        "xy": "linspace_in_metres_between_both_original_raster_endpoints",
        "height": "original_resolution_bilinear_height_plus_metadata_RX_AGL",
        "output": "bilinear_upsample_linear_gain_then_encode",
        "align_corners": True,
        "role": "coarse_illumination_context_not_resolved_visibility_boundary",
    }
    config["visibility"] = {
        "method": "vectorized_fixed_half_pixel_samples_bilinear_DEM",
        "both_segments": True, "step_px": 0.5,
        "refine_steps": 6, "root_finish": "bisection_then_secant",
        "hit_point": "vertical_projection_onto_bilinear_surface_after_root_refinement",
        "distance": "ray_parameter_before_vertical_projection",
        "origin_epsilon_m": 1e-4,
        "segment_endpoints": "excluded_strictly_interior_samples",
        "segment_tolerance_m": 1e-3,
        "tolerance_role": "float32_cancellation_only_not_physical_soft_visibility",
        "domain": "raster_only_no_border_extrapolation",
        "acceleration_box": "global_height_bounds_expanded_by_float_roundoff_only",
        "limitation": "may_miss_grazing_contacts_or_sub_step_enter_exit_pairs",
    }
    config["reflectance_lut"] = {
        "count": 4097, "axis": "uniform_cos_incidence_0_to_1",
        "precompute": "float64_angles_complex128_finite_slab_coefficients",
        "storage": "float32_TE_TM_power_reflectance",
        "lookup": "linear_interpolation_then_V_polarization_weighting",
    }
    config["height_origin"] = "subtract_common_DEM_min_after_float32_transfer_no_AGL_change"
    config["integrand"] = (
        "(lambda/(4*pi))^2*S^2/pi*["
        "integral_RX(R_eff*ci/(rt^2+rr^2)*V_TX*dOmega_RX)+"
        "integral_TX(R_eff*co/(rt^2+rr^2)*V_RX*dOmega_TX)]"
    )
    config["computation_dtype"] = "float32"
    config["complexity"] = "O(N_anchor*K*L); coarse_gain_upsample_O(H*W)"
    return config


class LightDiffuse:
    """Reusable directions/material LUT; gain and feature remain on device.

    N_anchor=rx_grid^2, K=n_mu*n_phi, L is the segment sampling length.
    Construction costs O(N_anchor*K*L); chunked ray tensors bound memory.
    Working memory is O(ray_chunk*L + N_anchor*K + H*W), plus fixed material
    tables; output interpolation costs O(H*W).
    No target, mask, cached prediction, or validation statistic is consumed.
    """

    def __init__(self, config, device="cuda", ray_chunk=8192):
        if config.get("quantity") != QUANTITY or config.get("method") != METHOD:
            raise ValueError("Expected a light_diffuse_config configuration")
        if not isinstance(ray_chunk, int) or isinstance(ray_chunk, bool) or ray_chunk < 1:
            raise ValueError("ray_chunk must be a positive integer")
        self.config = copy.deepcopy(config)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for lightweight diffuse features")
        self.resolution = self.config["resolution_m_per_px"]
        self.receiver_size = self.config["receiver_sampling"]["grid_size"]
        self.ray_chunk = ray_chunk
        visibility = self.config["visibility"]
        self.step_px = visibility["step_px"]
        self.refine_steps = visibility["refine_steps"]
        self.tolerance_m = visibility["segment_tolerance_m"]
        rule = self.config["quadrature"]
        self.directions, self.weight = direction_quadrature(
            rule["n_mu"], rule["n_phi"], self.device, torch.float32,
        )
        self.lut_count = self.config["reflectance_lut"]["count"]
        c = torch.linspace(0, 1, self.lut_count, dtype=torch.float64, device=self.device)
        rte, rtm, _, _ = slab_coefficients(c, self.config)
        self.rte, self.rtm = rte.abs().square().float(), rtm.abs().square().float()
        wavelength = SPEED_OF_LIGHT / self.config["frequency_hz"]
        self.factor = (wavelength / (4 * math.pi)) ** 2
        self.factor *= self.config["material"]["scattering_coefficient"] ** 2 / math.pi * self.weight
        h, w = self.config["grid"]
        yy, xx = torch.meshgrid(
            torch.linspace(0, (h - 1) * self.resolution, self.receiver_size,
                           device=self.device, dtype=torch.float32),
            torch.linspace(0, (w - 1) * self.resolution, self.receiver_size,
                           device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        self.receiver_xy = torch.stack((xx.flatten(), yy.flatten()), dim=1)
        encoding = self.config["encoding"]
        self.encoding_scale = math.log1p(encoding["gain_at_min_db"] / encoding["gain_floor"])

    def trace(self, height, origins, rays):
        return first_hit(height, origins, rays, self.resolution,
                         ray_chunk=self.ray_chunk, step_px=self.step_px,
                         refine_steps=self.refine_steps)

    def visible(self, height, a, b):
        return visible_segment(height, a, b, self.resolution, ray_chunk=self.ray_chunk,
                               step_px=self.step_px, tolerance_m=self.tolerance_m)

    def illuminate(self, q, normal, tx):
        """TX-incidence TE/TM projection, including in the TX-center integral."""
        delta = tx - q
        rt = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-8)
        to_tx = delta / rt[:, None]
        signed = (normal * to_tx).sum(1)
        n = torch.where((signed >= 0)[:, None], normal, -normal)
        ci = signed.abs().clamp(0, 1)
        k = -to_tx
        horizontal = torch.linalg.vector_norm(k[:, :2], dim=1)
        safe = horizontal.clamp_min(1e-12)
        cos_phi = torch.where(horizontal > 1e-12, k[:, 0] / safe, torch.ones_like(safe))
        sin_phi = torch.where(horizontal > 1e-12, k[:, 1] / safe, torch.zeros_like(safe))
        electric = torch.stack((k[:, 2] * cos_phi, k[:, 2] * sin_phi, -horizontal), dim=1)
        te = torch.linalg.cross(k, n, dim=1)
        te_norm = torch.linalg.vector_norm(te, dim=1)
        te = te / te_norm.clamp_min(1e-12)[:, None]
        wte = (electric * te).sum(1).square().clamp(0, 1)
        wte = torch.where(te_norm > 1e-12, wte, torch.zeros_like(wte))
        position = ci * (self.lut_count - 1)
        lo = position.long().clamp(0, self.lut_count - 2)
        alpha = position - lo
        rte = self.rte[lo] * (1 - alpha) + self.rte[lo + 1] * alpha
        rtm = self.rtm[lo] * (1 - alpha) + self.rtm[lo + 1] * alpha
        reflectance = wte * rte + (1 - wte) * rtm
        return rt, n, ci, reflectance

    @torch.no_grad()
    def gain(self, height_m, tx):
        """Return [1,1,H,W] nonnegative float32 gain on the configured device."""
        if isinstance(height_m, np.ndarray) and not height_m.flags.writeable:
            height_m = np.array(height_m, copy=True)
        height = torch.as_tensor(height_m, device=self.device, dtype=torch.float32)
        if tuple(height.shape) != tuple(self.config["grid"]) or not torch.isfinite(height).all():
            raise ValueError("height_m must be a finite raw-metre array matching config grid")
        try:
            if len(tx) != 2 or any(isinstance(v, (bool, np.bool_)) or int(v) != v for v in tx):
                raise ValueError
            row, col = map(int, tx)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("TX must be an integer (row,col) pair") from None
        h, w = height.shape
        if not (0 <= row < h and 0 <= col < w):
            raise ValueError("TX is outside the height raster")
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self._gain(height - height.amin(), row, col)

    def _gain(self, height, row, col):
        h, w = height.shape
        r = self.resolution
        tx_point = torch.stack((height.new_tensor(col * r), height.new_tensor(row * r),
                                height[row, col] + self.config["tx_height_m_agl"]))
        size, count = self.receiver_size, len(self.directions)
        z, _ = sample_surface(height, self.receiver_xy, r)
        rx = torch.cat((self.receiver_xy, (z + self.config["rx_height_m_agl"])[:, None]), dim=1)
        n_rx = len(rx)

        # Partition w_RX=rt^2/(rt^2+rr^2). RX first-hit geometry absorbs
        # co/rr^2 through dOmega_RX; do not multiply either factor again.
        origins = rx.repeat_interleave(count, dim=0)
        rays = self.directions.repeat(n_rx, 1)
        hit, q, normals, _ = self.trace(height, origins, rays)
        value = torch.zeros(len(origins), device=self.device, dtype=torch.float32)
        ids = hit.nonzero().flatten()
        if ids.numel():
            rt, normal, ci, reflectance = self.illuminate(q[ids], normals[ids], tx_point)
            rr = torch.linalg.vector_norm(origins[ids] - q[ids], dim=1)
            co = (normal * -rays[ids]).sum(1)
            selected = (co > 0) & (ci > 0)
            ids, rt, rr, ci = ids[selected], rt[selected], rr[selected], ci[selected]
            reflectance = reflectance[selected]
            if ids.numel():
                visible = self.visible(height, q[ids], tx_point.expand(len(ids), -1))
                value[ids] = reflectance * ci / (rt.square() + rr.square()) * visible
        rx_part = value.reshape(n_rx, count).sum(1)

        # w_TX=rr^2/(rt^2+rr^2). TX first hits absorb ci/rt^2. The TX
        # polarization and first-hit geometry are shared by all receivers.
        tx_origins = tx_point.expand(count, -1)
        hit, q, normals, _ = self.trace(height, tx_origins, self.directions)
        tx_ids = hit.nonzero().flatten()
        tx_part = torch.zeros(n_rx, device=self.device, dtype=torch.float32)
        if tx_ids.numel():
            q, normal = q[tx_ids], normals[tx_ids]
            rt, normal, ci, reflectance = self.illuminate(q, normal, tx_point)
            nr = len(q)
            qa = q.repeat(n_rx, 1)
            pa = rx.repeat_interleave(nr, dim=0)
            delta = pa - qa
            rr = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-8)
            co = (normal.repeat(n_rx, 1) * (delta / rr[:, None])).sum(1)
            rt, ci = rt.repeat(n_rx), ci.repeat(n_rx)
            reflectance = reflectance.repeat(n_rx)
            valid = (co > 0) & (ci > 0)
            visible = torch.zeros_like(valid)
            ids = valid.nonzero().flatten()
            if ids.numel():
                visible[ids] = self.visible(height, qa[ids], pa[ids])
            weighted = reflectance * co.clamp_min(0) / (rt.square() + rr.square()) * visible
            tx_part = weighted.reshape(n_rx, nr).sum(1)
        anchor = ((rx_part + tx_part) * self.factor).reshape(1, 1, size, size)
        return F.interpolate(anchor, size=(h, w), mode="bilinear", align_corners=True)

    @torch.no_grad()
    def feature(self, height_m, tx):
        """Return the reference soft-log encoding, after power interpolation."""
        with torch.autocast(device_type=self.device.type, enabled=False):
            gain = self.gain(height_m, tx)
            return torch.log1p(gain / self.config["encoding"]["gain_floor"]) / self.encoding_scale

    @torch.no_grad()
    def feature_numpy(self, height_m, tx):
        """Explicit CPU transfer for input-only cache construction."""
        return self.feature(height_m, tx)[0, 0].cpu().numpy().astype(np.float32, copy=False)
