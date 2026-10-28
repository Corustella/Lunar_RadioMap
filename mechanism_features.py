"""Versioned, input-only direct-area and single-bounce specular feature maps.

Metadata/cache/model imports do not load Numba or initialize CUDA. Only live
feature construction needs the optional geometry dependencies.
"""
import math
import json
from pathlib import Path

import numpy as np

CHANNELS = ["D_A", "S_exist", "S_gain", "S_extra"]
MODES = ("a", "d", "s", "ds")


def mechanism_cache_config(meta):
    from radiounet import geometry_config
    return {**geometry_config(meta), "mechanisms": mechanism_config(meta)}


def read_mechanism_cache(cache_root, dataset, config, required_mode="ds"):
    root = Path(cache_root)
    manifest = json.loads((root/f"{dataset.split}.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete D/S cache for {dataset.split}; use a new directory")
    if manifest.get("feature_config") != config or manifest.get("rows") != dataset.rows:
        raise ValueError(f"D/S cache metadata/index differs for {dataset.split}")
    present = channel_mask(manifest.get("computed_mode"))
    required = channel_mask(required_mode)
    if any(need > have for need,have in zip(required,present)):
        raise ValueError(f"Cache lacks enabled {required_mode} channels for {dataset.split}")
    path = root/f"{dataset.split}.npy"
    array = np.load(path,mmap_mode="r")
    expected = (len(dataset.rows),4,*config["grid"])
    if array.shape != expected or array.dtype != np.float32:
        raise ValueError(f"D/S cache requires float32 {expected}, got {array.dtype} {array.shape}")
    return path


def mechanism_config(meta):
    """Reconstruct the exact definition from metadata or a saved feature config."""
    saved = meta.get("mechanisms")
    if saved is None:
        simulation = meta["simulation"]
        material = simulation["material"]
        if not all(token in simulation["tx_antenna"] for token in ("1x1","pattern='iso'","polarization='V'")):
            raise ValueError("Specular v1 requires the metadata's isotropic V-polarized TX")
        parameters = {
            "frequency_hz": float(meta["bands_hz"]["58"]),
            "relative_permittivity": float(material["relative_permittivity"]),
            "conductivity_s_per_m": float(material["conductivity_s_per_m"]["58"]),
            "thickness_m": float(material["thickness_m"]),
            "scattering_coefficient": float(material["scattering_coefficient"])}
    else:
        parameters = {key: float(saved[key]) for key in (
            "frequency_hz", "relative_permittivity", "conductivity_s_per_m",
            "thickness_m", "scattering_coefficient")}
    if (not all(math.isfinite(v) for v in parameters.values()) or
            parameters["frequency_hz"] != 5775000000. or
            parameters["relative_permittivity"] <= 0 or
            parameters["conductivity_s_per_m"] < 0 or parameters["thickness_m"] <= 0 or
            not 0 <= parameters["scattering_coefficient"] <= 1):
        raise ValueError("Invalid material or unsupported frequency (D/S features are band 58 only)")
    return {
        "version": 2, "channels": CHANNELS.copy(), **parameters,
        "mesh": "row_col_NW_NE_SE__NW_SE_SW_flat_triangles",
        "coordinates": "stored_height_TX_grid_no_additional_flip",
        "computation_dtype": "float64", "output_dtype": "float32",
        "visibility": "exact_heightfield_grid_and_diagonal_crossings",
        "contact_epsilon_m": 1e-7,
        "direct_quadrature": [[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]],
        "direct_refinement": {"trigger":"three_point_mixed_visibility",
                              "subdivisions":8,"points":64,"weights":"equal_area_subtriangle_centroids"},
        "specular_quadrature": "one_centroid_per_receiver_triangle",
        "projection": "centroid_linear_midpoint_internal_grid_border_point_evaluation",
        "candidate_search": "conservative_reflection_cone_AABB_quadtree_no_topk",
        "shared_edge": "deduplicate_coplanar_adjacent_faces_only",
        "source_coplanar": "zero_solid_angle_faces_skipped_and_counted",
        "power": "single_bounce_incoherent_isoV_slab_Jones_norm_times_1_minus_s_squared",
        "gain_encoding": "log1p(project(sum_gain)/project(free_space_gain))",
        "extra_encoding": "log1p(project(sum_gain_times_L_over_d_minus_1)/project(sum_gain))",
        "exists_encoding": "project(fraction_of_quadrature_points_with_valid_path)",
        "augmentation": "joint_D4_after_feature_construction",
        "phase": False, "diffraction": False,
    }


def channel_mask(mode):
    if mode not in MODES:
        raise ValueError(f"Unsupported mechanism mode {mode!r}")
    return [float(mode in ("d","ds"))]+[float(mode in ("s","ds"))]*3


def construct_features(height_m, tx, config, device, mode="ds", threads=8):
    """Return [D_A,S_exist,S_gain,S_extra] without reading targets or masks.

    M faces, P receiver samples, B tested tree nodes, K candidate pairs and L
    heightfield crossings: O(M+P+B+K*L+P*L) time, O(M+P+K) GPU memory. Broad
    phase is output-sensitive, with worst-case O(M*P); no latency guarantee or
    path truncation is hidden behind a cap. A skips all four new features.
    """
    mask = channel_mask(mode)
    definition = config["mechanisms"]
    if mechanism_config(config) != definition:
        raise ValueError("Unsupported mechanism feature definition")
    if tuple(height_m.shape) != tuple(config["grid"]):
        raise ValueError("Height shape differs from feature config")
    if mode == "a":
        return np.zeros((4,*height_m.shape),np.float32), {"backend":"disabled", "truncated":False}
    from mechanism_geometry import compute
    kind = device.type
    if kind not in ("cuda","cpu"):
        raise ValueError("Mechanism features require CPU or CUDA")
    index = device.index
    if index is None and kind == "cuda":
        import torch
        index = torch.cuda.current_device()
    return compute(height_m,tx,direct=bool(mask[0]),specular=bool(mask[1]),
        backend=kind,device_index=index or 0,threads=threads,
        spacing=config["resolution_m_per_px"],tx_height=config["tx_height_m_agl"],
        rx_height=config["rx_height_m_agl"],frequency=definition["frequency_hz"],
        permittivity=definition["relative_permittivity"],
        conductivity=definition["conductivity_s_per_m"],
        thickness=definition["thickness_m"],scattering=definition["scattering_coefficient"])
