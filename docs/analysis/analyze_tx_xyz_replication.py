"""Audit three paired TX-geometry seeds, without inference or training.

Same pretrained firstU and full baseline across seeds. Descriptive mean/SD only;
terrain observations and epochs are not independent training repetitions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_tx_xyz import main as audit_pair, read_csv, read_json, write_csv
from boundary import boundary_regions
from experiment_provenance import fingerprint


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"raw": values.tolist(), "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)), "n_paired_seeds": len(values)}


def analyze(source, data, out, bootstrap_repeats):
    out.mkdir(parents=True, exist_ok=False)
    audits, per_seed, per_region, terrain_tables = [], [], [], []
    train_reference = None
    for seed in range(3):
        directory = source / f"tx_xyz_v1_seed{seed}"
        result = audit_pair(directory, data, out / f"seed{seed}", bootstrap_repeats)
        assert result["manifest"]["seed"] == seed
        if audits:
            reference = audits[0]
            for key in ("baseline_settings", "baseline_checkpoint", "firstU_state_sha256", "feature_config", "data_metadata"):
                assert result["manifest"][key] == reference["manifest"][key], key
            for key in ("environment", "code", "forward_precision", "data_metadata_and_actual_loader"):
                assert result["evaluation_provenance"][key] == reference["evaluation_provenance"][key], key
            assert result["runs"]["baseline_reference"] == reference["runs"]["baseline_reference"]
        for mode in ("zeros", "tx_xyz"):
            provenance = read_json(directory / f"seed{seed}_{mode}/radiownet_58_masked_secondU_provenance.json")
            for key in ("code", "environment", "git_commit", "git_status", "data_metadata_and_actual_loader"):
                assert provenance[key] == result["evaluation_provenance"][key], (seed, mode, key)
            settings = {k: v for k, v in provenance["args"].items() if k not in ("seed", "out", "second_features")}
            if train_reference is None:
                train_reference = settings
            assert settings == train_reference
        regions = read_csv(out / f"seed{seed}/regions.csv")
        keyed = {row["region"]: row for row in regions}
        control, candidate = (result["runs"][f"seed{seed}_{mode}"] for mode in ("zeros", "tx_xyz"))
        per_seed.append({"seed": seed, "zeros_rmse": control["rmse_db_masked"],
                         "tx_xyz_rmse": candidate["rmse_db_masked"],
                         "delta_rmse": candidate["rmse_db_masked"]-control["rmse_db_masked"],
                         "delta_rmse_percent": 100*(candidate["rmse_db_masked"]/control["rmse_db_masked"]-1),
                         "zeros_best_epoch": control["best_epoch"], "tx_xyz_best_epoch": candidate["best_epoch"],
                         "zeros_last_rmse": control["last_rmse_db_masked"], "tx_xyz_last_rmse": candidate["last_rmse_db_masked"],
                         "boundary_sse_percent": float(keyed["edge_t10_r3"]["delta_sse_percent"]),
                         "near_missing_sse_percent": float(keyed["near_missing_r5"]["delta_sse_percent"]),
                         "terrains_improved": result["influence"]["terrains_improved"],
                         "samples_improved": result["influence"]["samples_improved"],
                         "epochs_candidate_better": result["influence"]["epochs_candidate_better"],
                         "seconds": sum(row["seconds"] for row in result["measured_command_times"])})
        per_region.extend({"seed": seed, **row} for row in regions)
        terrain_tables.append(read_csv(out / f"seed{seed}/paired_terrains.csv"))
        audits.append(result)

    terrain_rows = []
    for group in zip(*terrain_tables):
        assert len({row["terrain_id"] for row in group}) == 1
        assert len({row["valid_pixels"] for row in group}) == 1
        delta = [float(row["delta_sse"]) for row in group]
        row = {"terrain_id": group[0]["terrain_id"], "samples": group[0]["samples"],
               "valid_pixels": group[0]["valid_pixels"], "improved_seeds": int((np.array(delta)<0).sum()),
               "mean_delta_sse": float(np.mean(delta))}
        row.update({f"seed{s}_delta_sse": value for s, value in enumerate(delta)})
        terrain_rows.append(row)
    mean_terrain = np.array([row["mean_delta_sse"] for row in terrain_rows])
    counts = np.array([int(row["valid_pixels"]) for row in terrain_rows])
    zeros_sse = np.mean([[float(row["control_sse"]) for row in table] for table in terrain_tables], axis=0)
    candidate_sse = np.mean([[float(row["candidate_sse"]) for row in table] for table in terrain_tables], axis=0)
    n = counts.sum()
    # Root of mean model MSE, not RMSE of an averaged prediction.
    leave = np.sqrt((candidate_sse.sum()-candidate_sse)/(n-counts)) - np.sqrt((zeros_sse.sum()-zeros_sse)/(n-counts))
    influence = {"terrains_improved_in_all_three": sum(row["improved_seeds"] == 3 for row in terrain_rows),
                 "terrains_worse_in_all_three": sum(row["improved_seeds"] == 0 for row in terrain_rows),
                 "terrains_with_negative_mean_sse_delta": int((mean_terrain<0).sum()),
                 "leave_one_terrain_root_mean_model_mse_delta_range": [float(leave.min()), float(leave.max())],
                 "leave_one_terrain_reversals": int((leave>0).sum()),
                 "top5_mean_gain": sorted(terrain_rows, key=lambda row: row["mean_delta_sse"])[:5],
                 "top5_mean_harm": sorted(terrain_rows, key=lambda row: row["mean_delta_sse"], reverse=True)[:5]}

    summaries = {key: describe([row[key] for row in per_seed]) for key in
                 ("zeros_rmse", "tx_xyz_rmse", "delta_rmse", "delta_rmse_percent", "boundary_sse_percent", "near_missing_sse_percent")}
    cached = cached_consistency(source)
    audit = {"paired_seeds": per_seed, "descriptive_statistics": summaries, "terrain_influence": influence,
             "fixed_profiles": {f"seed{s}": result["fixed_01917_row128_110db_crossings"] for s, result in enumerate(audits)},
             "cached_eight_map_diagnostics": cached,
             "commits": [result["evaluation_provenance"]["git_commit"] for result in audits],
             "cross_seed_code_data_environment_initialization_checks": "passed",
             "analysis_scripts": [fingerprint(Path(__file__)), fingerprint(Path(__file__).with_name("analyze_tx_xyz.py"))],
             "limits": "Three paired training seeds, same pretrained full secondU baseline and frozen firstU, same development val used for checkpoint selection. Mean/SD are descriptive, not population inference or a p value. Per-seed terrain bootstrap in subfolders is conditional fitted-model sensitivity only. No full-val ensemble score, full-array hashing, server access, new training or inference."}
    write_csv(out / "paired_seeds.csv", per_seed)
    write_csv(out / "regions_by_seed.csv", per_region)
    write_csv(out / "terrain_seed_consistency.csv", terrain_rows)
    write_csv(out / "cached_examples.csv", cached["per_example"])
    (out / "replication_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"paired_seeds": per_seed, "descriptive_statistics": summaries,
                      "terrain_influence": influence, "cached_summary": cached["summary"]}, indent=2))


def cached_consistency(source):
    names = sorted(path.name for path in (source / "tx_xyz_v1_seed0/seed0_zeros/boundary").glob("*.npz"))
    assert len(names) == 8
    totals = {mode: {region: np.zeros(4) for region in ("all_valid", "edge_t10_r3", "near_missing_r5")}
              for mode in ("zeros", "tx_xyz")}
    per_example = []
    for name in names:
        reference = None
        for mode in totals:
            predictions = []
            for seed in range(3):
                with np.load(source / f"tx_xyz_v1_seed{seed}/seed{seed}_{mode}/boundary" / name) as arrays:
                    unchanged = {key: arrays[key].copy() for key in
                                 ("input", "target", "mask", "firstU_prediction", "pl_scale_db", "pl_min_db")}
                    if reference is None:
                        reference = unchanged
                    for key in unchanged:
                        np.testing.assert_array_equal(unchanged[key], reference[key])
                    scale, lo = float(arrays["pl_scale_db"]), float(arrays["pl_min_db"])
                    predictions.append(arrays["prediction"].squeeze().astype(np.float64)*scale+lo)
            target = reference["target"].squeeze().astype(np.float64)*scale+lo
            mask = reference["mask"].squeeze()>0
            regions = {key: value.squeeze().numpy() for key, value in boundary_regions(
                torch.from_numpy(target)[None, None], torch.from_numpy(mask)[None, None]).items()}
            prediction = np.array(predictions)
            mean_sse = np.mean((prediction-target)**2, axis=0)
            ensemble_sse = (prediction.mean(0)-target)**2
            variance = np.mean((prediction-prediction.mean(0))**2, axis=0)
            np.testing.assert_allclose(mean_sse, ensemble_sse+variance, rtol=1e-10, atol=1e-10)
            for region in totals[mode]:
                valid = regions[region]
                values = np.array([valid.sum(), mean_sse[valid].sum(), ensemble_sse[valid].sum(), variance[valid].sum()])
                totals[mode][region] += values
                per_example.append({"example": name, "mode": mode, "region": region,
                                    "valid_pixels": int(values[0]), "mean_seed_sse": values[1],
                                    "mean_prediction_sse": values[2], "seed_variation_sse": values[3]})
    summary = [{"mode": mode, "region": region, "valid_pixels": int(values[0]),
                "mean_seed_rmse": float(np.sqrt(values[1]/values[0])),
                "mean_prediction_rmse": float(np.sqrt(values[2]/values[0])),
                "shared_error_fraction": values[2]/values[1],
                "rms_seed_variation_db": float(np.sqrt(values[3]/values[0]))}
               for mode, regions in totals.items() for region, values in regions.items()]
    return {"summary": summary, "per_example": per_example,
            "limits": "Eight previously fixed val examples only; averaging normalized predictions equals averaging dB under common affine scales. Not full-val ensemble performance or population bias/variance."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("--bootstrap-repeats must be at least 100")
    analyze(args.source, args.data_root, args.out, args.bootstrap_repeats)
