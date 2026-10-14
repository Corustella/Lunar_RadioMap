"""Input-only, single-bounce Lambertian gain proxy for Sionna RT 1.1.0.

The material, polarization and power factors follow the fixed-version model.
The released raster cannot recover the original triangle mesh: a bilinear DEM
and finite angular quadrature are explicit geometric/numerical approximations.
This is not the simulator's diffuse-path decomposition or total path loss.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from diffuse_geometry import trace_terrain


SPEED_OF_LIGHT = 299792458.0
VACUUM_PERMITTIVITY = 8.8541878128e-12
SIONNA_SOURCES = {
    "material": "https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_materials/radio_material.py",
    "slab": "https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/utils/electromagnetics.py",
    "pattern": "https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_materials/scattering_pattern.py",
    "antenna": "https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/antenna_pattern.py",
    "power": "https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_map_solvers/mesh_radio_map.py",
}


def diffuse_feature_config(meta, band, n_mu, n_phi):
    """Freeze metadata physics and the requested, not presumed-converged rule."""
    band = str(band)
    if band not in ("415", "58"):
        raise ValueError("Diffuse features require a single band: 415 or 58")
    if not isinstance(n_mu, int) or isinstance(n_mu, bool) or n_mu < 2:
        raise ValueError("n_mu must be an integer >= 2")
    if not isinstance(n_phi, int) or isinstance(n_phi, bool) or n_phi < 4 or n_phi % 4:
        raise ValueError("n_phi must be a positive multiple of 4 for D4 symmetry")
    sim = meta["simulation"]
    material = sim["material"]
    if not sim["engine"].startswith("Sionna RT 1.1.0"):
        raise ValueError("This proxy was reviewed against Sionna RT 1.1.0 only")
    if not sim["propagation_mechanisms"]["diffuse_reflection"]:
        raise ValueError("Metadata does not enable diffuse reflection")
    if material["scattering_pattern"].lower() != "lambertian":
        raise ValueError("This feature implements the Lambertian pattern only")
    if "pattern='iso'" not in sim["tx_antenna"] or "polarization='V'" not in sim["tx_antenna"]:
        raise ValueError("This feature requires the metadata's isotropic V transmitter")
    if "isotropic" not in sim["rx_antenna"].lower():
        raise ValueError("This feature requires an isotropic receiving probe")
    orientation = sim.get("tx_orientation_rad", [0.0, 0.0, 0.0])
    if list(orientation) != [0.0, 0.0, 0.0]:
        raise ValueError("Nonzero transmitter antenna orientation is not implemented")

    grid = list(meta["grid"])
    resolution = float(meta["resolution_m_per_px"])
    frequency = float(meta["bands_hz"][band])
    tx_height, rx_height = float(sim["tx_height_m_agl"]), float(sim["rx_height_m_agl"])
    eps = float(material["relative_permittivity"])
    sigma = float(material["conductivity_s_per_m"][band])
    thickness = float(material["thickness_m"])
    scattering = float(material["scattering_coefficient"])
    xpd = float(material["xpd_coefficient"])
    pl_lo, pl_hi = map(float, meta["pathloss_range_db"])
    values = (resolution, frequency, tx_height, rx_height, eps, sigma,
              thickness, scattering, xpd, pl_lo, pl_hi)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Diffuse metadata must contain finite physical parameters")
    if len(grid) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v < 2 for v in grid):
        raise ValueError("Diffuse geometry requires a two-dimensional grid >= 2x2")
    if min(resolution, frequency, tx_height, rx_height) <= 0 or eps < 1 or min(sigma, thickness) < 0:
        raise ValueError("Invalid geometry, frequency or dielectric metadata")
    if not 0 <= scattering <= 1 or not 0 <= xpd <= 1 or not 0 <= pl_lo < pl_hi <= 1000:
        raise ValueError("Invalid scattering/XPD coefficient or training pathloss bounds")

    return {
        "version": 1, "quantity": "single_bounce_lambertian_gain_proxy",
        "band": band, "frequency_hz": frequency,
        "grid": grid, "resolution_m_per_px": resolution,
        "tx_height_m_agl": tx_height, "rx_height_m_agl": rx_height,
        "height_input": "raw_metres_no_training_range_clipping",
        "surface": "bilinear_reconstruction_of_released_raster_not_original_mesh",
        "normals": "bilinear_derivatives_symmetric_at_grid_lines_incident_side",
        "material": {"relative_permittivity": eps, "conductivity_s_per_m": sigma,
                     "thickness_m": thickness, "scattering_coefficient": scattering,
                     "xpd_coefficient": xpd, "pattern": "lambertian",
                     "reflection": "finite_single_layer_slab_TE_TM",
                     "xpd_role": "norm_preserving_no_single_bounce_iso_power_factor"},
        "tx_polarization": {"pattern": "iso", "polarization": "V",
                            "orientation_rad": list(orientation),
                            "orientation_source": ("metadata" if "tx_orientation_rad" in sim else
                                                   "identity_pose_convention_not_recorded_in_archive"),
                            "field": "spherical_e_theta_in_world_identity_frame"},
        "rx": "isotropic_point_probe_at_raster_height_plus_rx_agl",
        "quadrature": {"domain": "full_sphere", "n_mu": n_mu, "n_phi": n_phi,
                       "rule": "uniform_cos_theta_and_azimuth_midpoints",
                       "weight_sr": 4 * math.pi / (n_mu * n_phi)},
        "visibility": {"method": "cellwise_bilinear_quadratic_DDA_first_hit",
                       "both_segments": True, "origin_epsilon_m": 1e-4,
                       "domain": "raster_only_no_border_extrapolation"},
        "integrand": "(lambda/(4*pi))^2*S^2*R_eff*ci/(pi*r_tx_q^2)*V_tx_q",
        "power_combination": "incoherent_sum_one_diffuse_interaction_only",
        "computation_dtype": "float64", "cache_dtype": "float32",
        "encoding": {"formula": "log1p(gain/gain_floor)/log1p(gain_at_min_db/gain_floor)",
                     "training_pathloss_range_db": [pl_lo, pl_hi],
                     "gain_floor": 10 ** (-pl_hi / 10),
                     "gain_at_min_db": 10 ** (-pl_lo / 10)},
        "sources": SIONNA_SOURCES.copy(),
    }


def slab_coefficients(cos_theta, config):
    """TE/TM complex slab reflection/transmission, matching 1.1.0 equations.

    Reconstructed from ITU single-layer slab equations, as identified by the
    fixed-version source above. This is not a first-Fresnel-zone clearance.
    """
    material = config["material"]
    frequency = config["frequency_hz"]
    wavelength = SPEED_OF_LIGHT / frequency
    dtype = torch.complex128 if cos_theta.dtype == torch.float64 else torch.complex64
    eta = torch.tensor(complex(material["relative_permittivity"],
                               -material["conductivity_s_per_m"] /
                               (2 * math.pi * frequency * VACUUM_PERMITTIVITY)),
                       dtype=dtype, device=cos_theta.device)
    c = cos_theta.clamp(0, 1).to(dtype)
    # A vacuum slab has unit transmitted power but retains its propagation
    # phase. Handle it before the interface's grazing 0/0 expression.
    vacuum = material["relative_permittivity"] == 1 and material["conductivity_s_per_m"] == 0
    if vacuum:
        zero = torch.zeros_like(c)
        phase = torch.exp(-1j * (2 * math.pi * material["thickness_m"] / wavelength) * c)
        return zero, zero, phase, phase
    if material["thickness_m"] == 0:
        zero = torch.zeros_like(c)
        return zero, zero, torch.ones_like(c), torch.ones_like(c)
    root = torch.sqrt(eta - (1 - c.square()))
    phase = torch.exp(-1j * (2 * math.pi * material["thickness_m"] / wavelength) * root)
    r_te = (c - root) / (c + root)
    r_tm = (eta * c - root) / (eta * c + root)
    phase2 = phase.square()
    rte = r_te * (1 - phase2) / (1 - r_te.square() * phase2)
    rtm = r_tm * (1 - phase2) / (1 - r_tm.square() * phase2)
    tte = (1 - r_te.square()) * phase / (1 - r_te.square() * phase2)
    ttm = (1 - r_tm.square()) * phase / (1 - r_tm.square() * phase2)
    return rte, rtm, tte, ttm


def vertical_reflectance(incident_direction, normal, cos_theta, config):
    """Reflect unit iso/V incident power in TE/TM; no unpolarized averaging."""
    k = incident_direction
    horizontal = torch.linalg.vector_norm(k[:, :2], dim=1)
    tiny = torch.finfo(k.dtype).eps
    # Sionna's spherical e_theta; at the vertical pole use phi=0.
    cos_phi = torch.where(horizontal > tiny, k[:, 0] / horizontal.clamp_min(tiny),
                          torch.ones_like(horizontal))
    sin_phi = torch.where(horizontal > tiny, k[:, 1] / horizontal.clamp_min(tiny),
                          torch.zeros_like(horizontal))
    electric = torch.stack((k[:, 2] * cos_phi, k[:, 2] * sin_phi, -horizontal), dim=1)
    te = torch.linalg.cross(k, normal, dim=1)
    norm = torch.linalg.vector_norm(te, dim=1)
    te = te / norm.clamp_min(tiny)[:, None]
    weight = (electric * te).sum(1).square().clamp(0, 1)
    # At normal incidence the two power coefficients agree.
    weight = torch.where(norm > tiny, weight, torch.zeros_like(weight))
    rte, rtm, _, _ = slab_coefficients(cos_theta, config)
    return weight * rte.abs().square() + (1 - weight) * rtm.abs().square()


def direction_quadrature(n_mu, n_phi, device, dtype=torch.float64):
    """Equal-solid-angle midpoint rule, invariant as a set under raster D4."""
    if n_mu < 2 or n_phi < 4 or n_phi % 4:
        raise ValueError("Require n_mu >= 2 and n_phi a positive multiple of 4")
    mu = -1 + (torch.arange(n_mu, dtype=dtype, device=device) + 0.5) * (2 / n_mu)
    phi = (torch.arange(n_phi, dtype=dtype, device=device) + 0.5) * (2 * math.pi / n_phi)
    mu, phi = torch.meshgrid(mu, phi, indexing="ij")
    horizontal = (1 - mu.square()).sqrt()
    directions = torch.stack((horizontal * phi.cos(), horizontal * phi.sin(), mu), dim=-1)
    return directions.reshape(-1, 3), 4 * math.pi / (n_mu * n_phi)


@torch.no_grad()
def diffuse_gain(height_m, tx, config, device="cpu", chunk=65536):
    """Return linear, dimensionless one-scatter gain, without labels/masks.

    Substitute dOmega_RX=co*dA/r_q_rx^2 into the surface scattering integral.
    First-hit rays account for RX visibility and that Jacobian, so neither co
    nor r_q_rx^-2 is multiplied a second time. TX visibility is checked anew.

    N=HW, K=n_mu*n_phi, L<=H+W cells per segment: time O(NKL), working memory
    O(max(chunk,K)+N). Complete receiver direction groups avoid atomic sums.
    """
    if not isinstance(chunk, int) or chunk <= 0:
        raise ValueError("chunk must be a positive ray-batch size")
    height_m = np.asarray(height_m)
    if height_m.shape != tuple(config["grid"]) or not np.isfinite(height_m).all():
        raise ValueError("Height must be a finite raw-metre array matching feature_config.grid")
    h, w = height_m.shape
    if len(tx) != 2 or any(isinstance(v, bool) or int(v) != v for v in tx):
        raise ValueError("TX must be an integer (row, col) pair")
    row, col = map(int, tx)
    if not (0 <= row < h and 0 <= col < w):
        raise ValueError("TX is outside the terrain raster")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for diffuse feature construction")
    with torch.autocast(device.type, enabled=False):
        height = torch.tensor(np.array(height_m, copy=True), dtype=torch.float64, device=device)
        resolution = config["resolution_m_per_px"]
        tx_point = torch.stack((height.new_tensor(col * resolution),
                                height.new_tensor(row * resolution),
                                height[row, col] + config["tx_height_m_agl"]))
        rule = config["quadrature"]
        directions, weight = direction_quadrature(rule["n_mu"], rule["n_phi"], device)
        count = directions.shape[0]
        pixels_per_chunk = max(1, chunk // count)
        epsilon = config["visibility"]["origin_epsilon_m"]
        factor = (SPEED_OF_LIGHT / config["frequency_hz"] / (4 * math.pi)) ** 2
        factor *= config["material"]["scattering_coefficient"] ** 2 * weight / math.pi
        output = torch.zeros(h * w, dtype=torch.float64, device=device)
        if factor == 0:
            return output.reshape(h, w).cpu().numpy()
        for start in range(0, h * w, pixels_per_chunk):
            index = torch.arange(start, min(start + pixels_per_chunk, h * w), device=device)
            rx = torch.stack(((index % w) * resolution, (index // w) * resolution,
                              height.flatten()[index] + config["rx_height_m_agl"]), dim=1)
            origins = rx.repeat_interleave(count, dim=0)
            rays = directions.repeat(index.numel(), 1)
            hit, points, normals, _ = trace_terrain(height, origins, rays, resolution,
                                                   epsilon_m=epsilon)
            values = torch.zeros(origins.shape[0], dtype=torch.float64, device=device)
            selected = hit.nonzero().flatten()
            if selected.numel():
                q = points[selected]
                to_tx = tx_point - q
                r_tx = torch.linalg.vector_norm(to_tx, dim=1)
                to_tx = to_tx / r_tx[:, None]
                normal = normals[selected]
                signed_incidence = (normal * to_tx).sum(1)
                normal = torch.where((signed_incidence >= 0)[:, None], normal, -normal)
                ci = signed_incidence.abs().clamp(0, 1)
                co = (normal * (-rays[selected])).sum(1)
                front = (ci > 0) & (co > 0) & (r_tx > epsilon)
                selected, q, normal = selected[front], q[front], normal[front]
                to_tx, r_tx, ci = to_tx[front], r_tx[front], ci[front]
                if selected.numel():
                    blocked, _, _, _ = trace_terrain(height, q, to_tx, resolution,
                                                    max_distance=r_tx, epsilon_m=epsilon)
                    visible = ~blocked
                    if visible.any():
                        reflectance = vertical_reflectance(-to_tx[visible], normal[visible],
                                                          ci[visible], config)
                        values[selected[visible]] = factor * reflectance * ci[visible] / r_tx[visible].square()
            output[index] = values.reshape(index.numel(), count).sum(1)
        if not torch.isfinite(output).all() or (output < 0).any():
            raise FloatingPointError("Diffuse gain contains nonfinite or negative values")
        return output.reshape(h, w).cpu().numpy()


def encode_diffuse_gain(gain, config):
    """Soft-log power; zero means no sampled one-scatter route, not PL=0 dB."""
    gain = np.asarray(gain, dtype=np.float64)
    if not np.isfinite(gain).all() or (gain < 0).any():
        raise ValueError("Diffuse encoding requires finite nonnegative linear gain")
    encoding = config["encoding"]
    floor, reference = encoding["gain_floor"], encoding["gain_at_min_db"]
    return (np.log1p(gain / floor) / math.log1p(reference / floor)).astype(np.float32)


def diffuse_feature(height_m, tx, config, device="cpu", chunk=65536):
    return encode_diffuse_gain(diffuse_gain(height_m, tx, config, device, chunk), config)


def read_diffuse_cache(cache_root, dataset, config):
    """Reject incomplete, partial-for-full, wrong-band or incompatible caches."""
    root = Path(cache_root)
    manifest = json.loads((root / f"{dataset.split}.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Diffuse cache is incomplete for {dataset.split}; use a new directory")
    if manifest.get("completed_rows") != len(dataset.rows):
        raise ValueError("Diffuse cache completion count differs from the selected dataset")
    if manifest.get("band") != config["band"] or dataset.band != config["band"]:
        raise ValueError("Diffuse cache and dataset must use the same single frequency band")
    if manifest.get("feature_config") != config or manifest.get("rows") != dataset.rows:
        raise ValueError(f"Diffuse cache metadata/index differs for {dataset.split}")
    expected = (len(dataset.rows), *config["grid"])
    path = root / f"{dataset.split}.npy"
    array = np.load(path, mmap_mode="r")
    if array.shape != expected or array.dtype != np.float32:
        raise ValueError(f"Diffuse cache requires float32 {expected}, got {array.dtype} {array.shape}")
    if manifest.get("shape") != list(expected) or manifest.get("dtype") != "float32":
        raise ValueError("Diffuse cache array and manifest disagree")
    return path
