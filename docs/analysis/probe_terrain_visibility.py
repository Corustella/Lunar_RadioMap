"""Input-only approximate terrain visibility; diagnostics on eight fixed caches.

Bilinear terrain, nested <=1/0.5 m ray samples, TX/RX heights from metadata.
Not an exact intersection test on the simulator mesh or a pathloss prediction.
Targets/masks are accessed only AFTER the geometry has been computed.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_tx_xyz import read_json, write_csv
from boundary import boundary_regions, dilate
from experiment_provenance import fingerprint


def clearance(height, tx, resolution, tx_agl, rx_agl, refinement=2, chunk=512):
    """Minimum vertical clearance, inclusive endpoints. O(HWL), O(chunk*L)."""
    height = np.asarray(height, dtype=np.float64)
    assert height.ndim == 2 and np.isfinite(height).all()
    assert resolution > 0 and tx_agl > 0 and rx_agl > 0 and refinement >= 1
    h, w = height.shape
    tr, tc = tx
    rows, cols = np.indices(height.shape, dtype=np.float64)
    endpoints = np.stack((rows.ravel(), cols.ravel()), axis=1)
    z_tx = height[tr, tc]+tx_agl
    output = np.empty(h*w)
    for start in range(0, h*w, chunk):
        ends = endpoints[start:start+chunk]
        distance = np.linalg.norm(ends-np.array(tx), axis=1)*resolution
        intervals = refinement*np.maximum(1, np.ceil(distance/resolution).astype(int))
        k = np.arange(intervals.max()+1)[None, :]
        alpha = np.minimum(k/intervals[:, None], 1)
        y = tr+alpha*(ends[:, 0, None]-tr)
        x = tc+alpha*(ends[:, 1, None]-tc)
        # Round-off alone can put border endpoints just outside the grid.
        y, x = np.clip(y, 0, h-1), np.clip(x, 0, w-1)
        r0, c0 = np.floor(y).astype(int), np.floor(x).astype(int)
        r1, c1 = np.minimum(r0+1, h-1), np.minimum(c0+1, w-1)
        fy, fx = y-r0, x-c0
        surface = ((1-fy)*((1-fx)*height[r0,c0]+fx*height[r0,c1])
                   +fy*((1-fx)*height[r1,c0]+fx*height[r1,c1]))
        z_rx = height[ends[:,0].astype(int), ends[:,1].astype(int)]+rx_agl
        ray = z_tx+alpha*(z_rx[:,None]-z_tx)
        values = np.where(k <= intervals[:,None], ray-surface, np.inf)
        output[start:start+len(ends)] = values.min(1)
    return output.reshape(height.shape)


def checks():
    flat = np.zeros((5,5))
    np.testing.assert_allclose(clearance(flat, (2,0), 1, 3, 1), 1, atol=1e-12)
    ridge = flat.copy()
    ridge[:,2] = 10
    result = clearance(ridge, (2,0), 1, 3, 1)
    np.testing.assert_allclose(result[2,4], -8, atol=1e-12)
    np.testing.assert_allclose(clearance(ridge+17, (2,0), 1, 3, 1), result, atol=1e-12)
    np.testing.assert_allclose(clearance(np.rot90(ridge), (4,2), 1, 3, 1), np.rot90(result), atol=1e-12)
    rng = np.random.default_rng(20261005)
    terrain = rng.uniform(0,10,(5,5))
    reference = clearance(terrain, (1,3), 1, 3, 1)
    for turns in range(4):
        rotated = np.rot90(terrain, turns)
        marker = np.zeros_like(terrain)
        marker[1,3] = 1
        tx = tuple(np.argwhere(np.rot90(marker,turns)>0)[0])
        np.testing.assert_allclose(clearance(rotated,tx,1,3,1), np.rot90(reference,turns), atol=1e-12)
        np.testing.assert_allclose(clearance(np.fliplr(rotated),(tx[0],4-tx[1]),1,3,1),
                                   np.fliplr(np.rot90(reference,turns)), atol=1e-12)
    return "Flat terrain, blocking ridge, elevation translation and eight rotations/flips passed"


def analyze(source, metadata, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    tests = checks()
    meta = read_json(metadata)
    lo, hi = meta["heightmap_range_m"]
    resolution = meta["resolution_m_per_px"]
    tx_agl, rx_agl = (meta["simulation"][key] for key in ("tx_height_m_agl", "rx_height_m_agl"))
    paths = sorted((source / "tx_xyz_v1_seed0/seed0_zeros/boundary").glob("*.npz"))
    assert len(paths) == 8
    samples, errors, profiles = [], [], []
    for path in paths:
        with np.load(path) as cache:
            inputs = cache["input"].copy()
        # Features depend solely on test-available height/TX and training metadata.
        height = inputs[0].astype(np.float64)*(hi-lo)+lo
        tx_map = inputs[1]
        assert ((tx_map == 0)|(tx_map == 1)).all() and tx_map.sum() == 1
        tx = tuple(np.argwhere(tx_map>0)[0])
        start = time.perf_counter()
        coarse = clearance(height, tx, resolution, tx_agl, rx_agl, refinement=1)
        fine = clearance(height, tx, resolution, tx_agl, rx_agl, refinement=2)
        seconds = time.perf_counter()-start
        assert np.all(fine <= coarse+1e-10)
        visible, visible_coarse = fine > 1e-4, coarse > 1e-4
        feature = np.arcsinh(fine/1.0)/np.arcsinh((hi-lo)/1.0)
        with np.load(path) as cache:
            target = cache["target"].squeeze().astype(np.float64)*float(cache["pl_scale_db"])+float(cache["pl_min_db"])
            valid = cache["mask"].squeeze()>0
        regions = {key: value.squeeze().numpy() for key, value in boundary_regions(
            torch.from_numpy(target)[None,None], torch.from_numpy(valid)[None,None]).items()}
        transitions = np.zeros_like(visible)
        dx = visible[:,1:] != visible[:,:-1]
        dy = visible[1:] != visible[:-1]
        transitions[:,1:] |= dx
        transitions[:,:-1] |= dx
        transitions[1:] |= dy
        transitions[:-1] |= dy
        proxy_band = dilate(torch.from_numpy(transitions)[None,None], 3).squeeze().numpy() & valid
        samples.append({"sample": path.stem, "valid_pixels": int(valid.sum()),
                        "visible_valid_fraction": float(visible[valid].mean()),
                        "visible_invalid_fraction": float(visible[~valid].mean()) if (~valid).any() else None,
                        "coarse_fine_visibility_changes": int((visible!=visible_coarse).sum()),
                        "coarse_fine_valid_changes": int(((visible!=visible_coarse)&valid).sum()),
                        "proxy_band_valid_pixels": int(proxy_band.sum()),
                        "gt_edge_valid_pixels": int(regions["edge_t10_r3"].sum()),
                        "gt_edge_in_proxy_band": int((proxy_band & regions["edge_t10_r3"]).sum()),
                        "clearance_p01_m": float(np.quantile(fine[valid],.01)),
                        "clearance_median_m": float(np.median(fine[valid])),
                        "clearance_p99_m": float(np.quantile(fine[valid],.99)),
                        "feature_min": float(feature.min()), "feature_max": float(feature.max()),
                        "coarse_plus_fine_seconds_cpu": seconds})
        bins = {"proxy_visible": valid & visible, "proxy_blocked": valid & ~visible,
                "proxy_boundary_r3": proxy_band, "gt_edge_t10_r3": regions["edge_t10_r3"]}
        for mode in ("zeros", "tx_xyz"):
            predictions = []
            for seed in range(3):
                with np.load(source / f"tx_xyz_v1_seed{seed}/seed{seed}_{mode}/boundary" / path.name) as cache:
                    np.testing.assert_array_equal(cache["input"],inputs)
                    np.testing.assert_array_equal(cache["mask"].squeeze()>0,valid)
                    predictions.append(cache["prediction"].squeeze().astype(np.float64)*float(cache["pl_scale_db"])+float(cache["pl_min_db"]))
            squared = np.mean((np.array(predictions)-target)**2,axis=0)
            for name, mask in bins.items():
                count = int(mask.sum())
                errors.append({"sample": path.stem, "mode": mode, "region": name,
                               "valid_pixels": count, "mean_seed_sse": float(squared[mask].sum())})
        if path.stem == "01917_32_58":
            for col in range(80,196):
                profiles.append({"row":128,"col":col,"valid":int(valid[128,col]),
                                 "target_db":float(target[128,col]),"clearance_m":float(fine[128,col]),
                                 "visible_proxy":int(visible[128,col]),"feature":float(feature[128,col])})
        print(f"Probed {path.stem}: {seconds:.2f} s",flush=True)
    totals = {}
    for row in errors:
        key = row["mode"],row["region"]
        totals.setdefault(key,np.zeros(2))
        totals[key] += (row["valid_pixels"],row["mean_seed_sse"])
    total_valid = sum(row["valid_pixels"] for row in samples)
    summary = {"samples":8,"valid_pixels":total_valid,
               "coarse_fine_valid_changes":sum(row["coarse_fine_valid_changes"] for row in samples),
               "proxy_band_valid_share":sum(row["proxy_band_valid_pixels"] for row in samples)/total_valid,
               "gt_edge_in_proxy_band_fraction":sum(row["gt_edge_in_proxy_band"] for row in samples)/sum(row["gt_edge_valid_pixels"] for row in samples),
               "regions":[{"mode":mode,"region":region,"valid_pixels":int(n),"mean_seed_sse":sse,
                           "root_mean_seed_mse_db":float(np.sqrt(sse/n)) if n else None}
                          for (mode,region),(n,sse) in totals.items()],
               "fixed_01917_visibility_transitions_right_columns":[right["col"] for left,right in zip(profiles[:-1],profiles[1:])
                   if left["valid"] and right["valid"] and left["visible_proxy"]!=right["visible_proxy"]]}
    write_csv(out/"examples.csv",samples)
    write_csv(out/"error_regions.csv",errors)
    write_csv(out/"fixed_profile.csv",profiles)
    result = {"summary":summary,"checks":tests,"config":{"tx_height_m_agl":tx_agl,"rx_height_m_agl":rx_agl,
              "terrain_interpolation":"bilinear","ray_sampling":"nested <=1m and <=0.5m horizontal steps, endpoints included",
              "visibility_numerical_tolerance_m":1e-4,"candidate_feature":"asinh(clearance_m/1m)/asinh(training_height_range_m/1m)"},
              "source_fingerprints":[fingerprint(metadata)]+[fingerprint(path) for path in paths],"script":fingerprint(Path(__file__)),
              "limits":"Eight fixed val caches only, not full-val statistics or exact simulator LOS. Original Sionna mesh/centroid receiver geometry unavailable; sampling may miss narrow extrema. No diffraction, reflection or pathloss model in this proxy. Numerical convergence of 1m vs0.5m visibility does not validate the mesh or continuous clearance. No new training/inference or model input changes."}
    (out/"visibility_probe.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--metadata",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    analyze(args.source,args.metadata,args.out)
