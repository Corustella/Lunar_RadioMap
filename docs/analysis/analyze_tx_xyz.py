"""Audit downloaded TX-geometry artifacts; no training or inference.

Writes only a new output directory. Streams regional CSVs and checks cached
predictions, checkpoint identity, fixed firstU and paired training settings.
Terrain bootstrap is conditional on the fitted/selected models, not a seed CI.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_provenance import fingerprint, first_stage_hash
from radiounet import checkpoint_features

FEATURES = ("none", "zeros", "tx_xyz")
WEIGHTS = ("Wlayer00.0.weight", "Wconv_up00.0.weight", "Wconv_up000.0.weight")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio(sse, count):
    return float(np.sqrt(sse / count)) if count else None


def main(source, data, out, bootstrap_repeats=20000):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    manifest = read_json(source / "manifest.json")
    assert manifest["status"] == "complete" and not manifest["smoke_only"]
    seed = manifest["seed"]
    assert isinstance(seed, int) and seed >= 0 and manifest["grad_weight"] == 0
    run_names = ("baseline_reference", f"seed{seed}_zeros", f"seed{seed}_tx_xyz")
    assert all(item["status"] == "complete" for item in manifest["commands"])
    index = read_csv(data / "val_index.csv")
    ids = [row["sample_id"] for row in index]
    assert len(ids) == len(set(ids)) == 2330
    positions = {name: i for i, name in enumerate(ids)}
    terrains = sorted({row["terrain_id"] for row in index})
    assert len(terrains) == 680
    assert not set(terrains) & {row["terrain_id"] for row in read_csv(data / "train_index.csv")}
    groups = {t: np.array([i for i, row in enumerate(index) if row["terrain_id"] == t]) for t in terrains}
    runs, cached, source_hashes, code_checks = {}, {}, [], []
    run_rows, example_rows, profile_rows, weight_rows = [], [], [], []
    common_eval, common_train = None, None
    for name, mode in zip(run_names, FEATURES):
        directory = source / name
        report = read_json(directory / "boundary/summary.json")
        val = read_json(directory / "val.json")
        provenance = report["provenance"]
        assert report["samples"] == 2330 and val["summary"]["samples"] == 2330
        assert read_json(directory / "boundary/status.json")["status"] == "complete"
        assert provenance["args"]["limit"] == 0 and not provenance["args"]["no_mask"]
        assert val["summary"]["official_metric"] == "rmse_db_masked"
        assert val["summary"]["second_features"] == mode
        assert provenance["firstU_state_sha256"] == manifest["firstU_state_sha256"]
        for item in provenance["data_metadata_and_actual_loader"]:
            assert fingerprint(data / Path(item["path"]).name)["sha256"] == item["sha256"]
        if common_eval is None:
            common_eval = provenance
            local_repo = Path(__file__).resolve().parents[2]
            recorded_repo = Path(provenance["code"][0]["path"]).parent
            for entry in provenance["code"]:
                relative = Path(entry["path"]).relative_to(recorded_repo)
                raw = (local_repo / relative).read_bytes()
                blob = subprocess.run(["git", "show", f"{provenance['git_commit']}:{relative.as_posix()}"],
                                      cwd=local_repo, capture_output=True, check=True, timeout=10).stdout
                check = {"file": relative.as_posix(),
                         "local_byte_match": hashlib.sha256(raw).hexdigest() == entry["sha256"],
                         "local_lf_match": hashlib.sha256(raw.replace(b"\r\n",b"\n")).hexdigest() == entry["sha256"],
                         "git_blob_match": hashlib.sha256(blob).hexdigest() == entry["sha256"]}
                assert check["local_lf_match"] and check["git_blob_match"], relative
                code_checks.append(check)
        for key in ("git_commit", "git_status", "environment", "code", "forward_precision",
                    "data_metadata_and_actual_loader"):
            assert provenance[key] == common_eval[key], (name, key)
        assert provenance["git_status"] == ""
        assert {k: v for k, v in provenance["args"].items() if k not in ("ckpt", "out", "boundary_out")} == {
            k: v for k, v in common_eval["args"].items() if k not in ("ckpt", "out", "boundary_out")}
        definitions = ("primary_threshold_db_per_pixel", "thresholds_db_per_pixel", "radii_px",
                       "threshold_atol_db", "definitions")
        if runs:
            for key in definitions:
                assert report[key] == runs[run_names[0]]["report"][key]
        totals = {(r["output"], r["region"]): np.zeros(3) for r in report["regions"]}
        sample = {r["region"]: np.zeros((2330, 3)) for r in report["regions"] if r["output"] == "secondU"}
        seen = set()
        with (directory / "boundary/per_sample.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                assert row["band"] == "58"
                key = row["output"], row["region"]
                identity = (row["sample_id"], *key)
                assert identity not in seen
                seen.add(identity)
                values = np.array([int(row["valid_pixels"]), float(row["sse_db2"]), float(row["signed_error_sum_db"])])
                assert np.isfinite(values).all() and min(values[:2]) >= 0
                totals[key] += values
                if key[0] == "secondU":
                    sample[key[1]][positions[row["sample_id"]]] = values
        assert len(seen) == 2330 * len(totals)
        for row in report["regions"]:
            key = row["output"], row["region"]
            np.testing.assert_allclose(totals[key], [row["valid_pixels"], row["sse_db2"], row["signed_error_sum_db"]], rtol=1e-10, atol=1e-5)
            if key[1].startswith("edge_t") and "missing" not in key[1]:
                np.testing.assert_allclose(totals[key] + totals[(key[0], "non" + key[1])], totals[(key[0], "all_valid")], rtol=1e-10, atol=1e-5)
        for prefix in ("", "edge_t10_r3_", "nonedge_t10_r3_"):
            near, away = ("near_missing_r5", "away_missing_r5") if not prefix else (prefix + "near_missing", prefix + "away_missing")
            target = "all_valid" if not prefix else prefix[:-1]
            np.testing.assert_allclose(sample[near] + sample[away], sample[target], rtol=1e-10, atol=1e-5)
        if runs:
            original = runs[run_names[0]]
            for key in totals:
                if key[0] == "firstU":
                    np.testing.assert_array_equal(totals[key], original["totals"][key])
            for region in sample:
                np.testing.assert_array_equal(sample[region][:, 0], original["sample"][region][:, 0])
        for row in val["per_sample"]:
            n, sse, _ = sample["all_valid"][positions[row["sample_id"]]]
            assert abs(ratio(sse, n) - row["rmse_db_masked"]) < 1e-4
        n, sse, signed = sample["all_valid"].sum(0)
        assert abs(ratio(sse, n) - val["summary"]["rmse_db_masked"]) < 1e-5
        audit = {"rmse_db_masked": val["summary"]["rmse_db_masked"], "best_epoch": val["summary"]["epoch"],
                 "valid_pixels": int(n), "sse_db2": sse, "bias_db": signed/n, "csv_rows_verified": len(seen)}
        if mode != "none":
            history = read_csv(directory / "radiownet_58_masked_secondU_history.csv")
            assert [int(r["epoch"]) for r in history] == list(range(150))
            best = min(history, key=lambda r: float(r["val/rmse_db_masked"]))
            train_prov = read_json(directory / "radiownet_58_masked_secondU_provenance.json")
            assert train_prov["checkpoints"]["init_from"]["sha256"] == manifest["baseline_checkpoint"]["sha256"]
            assert train_prov["firstU_state_sha256"] == manifest["firstU_state_sha256"]
            assert train_prov["weight_loading"] == {"source_mode": "none", "target_mode": mode, "zero_padded_weights": list(WEIGHTS)}
            for label in ("best", "last"):
                path = directory / f"radiownet_58_masked_secondU_{label}.pt"
                ck = torch.load(path, map_location="cpu", weights_only=True)
                assert first_stage_hash(ck["model"]) == manifest["firstU_state_sha256"]
                assert checkpoint_features(ck) == (mode, manifest["feature_config"])
                assert ck["args"] == train_prov["args"]
                assert ck["args"]["seed"] == seed and ck["args"]["grad_weight"] == 0 and ck["args"]["resume"] is None
                for key, value in manifest["baseline_settings"].items():
                    assert ck["args"][key] == value, key
                if label == "best":
                    assert ck["epoch"] == int(best["epoch"])
                    assert ck["metrics"]["val/rmse_db_masked"] == val["summary"]["rmse_db_masked"]
                    assert fingerprint(path)["sha256"] == provenance["checkpoints"]["evaluated"]["sha256"]
                    for key in WEIGHTS:
                        w = ck["model"][key].float()
                        for channel, feature in enumerate(("dx", "dy", "dz")):
                            added = w[:, -3+channel]
                            weight_rows.append({"run": name, "layer": key, "feature": feature,
                                                "norm": float(added.norm()), "nonzero": int(torch.count_nonzero(added))})
                else:
                    assert ck["epoch"] == 149
                    assert ck["metrics"]["val/rmse_db_masked"] == float(history[-1]["val/rmse_db_masked"])
                comparable = {k: v for k, v in ck["args"].items() if k not in ("out", "second_features")}
                if common_train is None:
                    common_train = comparable
                assert comparable == common_train
                del ck
            components = read_csv(directory / "radiownet_58_masked_secondU_loss_components.csv")
            assert len(components) == 150
            for row in components:
                assert float(row["grad_weight"]) == 0
                assert float(row["train_mse"]) == float(row["train_total"])
            audit["last_rmse_db_masked"] = float(history[-1]["val/rmse_db_masked"])
            audit["train_loss_best"] = float(best["train_loss"])
            audit["train_loss_last"] = float(history[-1]["train_loss"])
        else:
            history = None
            assert provenance["checkpoints"]["evaluated"]["sha256"] == manifest["baseline_checkpoint"]["sha256"]
            baseline_path = source.parent / "boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt"
            assert fingerprint(baseline_path)["sha256"] == manifest["baseline_checkpoint"]["sha256"]
            baseline_ck = torch.load(baseline_path, map_location="cpu", weights_only=True)
            assert first_stage_hash(baseline_ck["model"]) == manifest["firstU_state_sha256"]
            del baseline_ck
        for example in report["examples"]:
            with np.load(directory / "boundary" / example["npz"]) as f:
                arrays = {k: f[k] for k in f.files}
            if example["name"] not in cached:
                cached[example["name"]] = arrays, example["profiles"]
            reference, profiles = cached[example["name"]]
            assert profiles == example["profiles"]
            for key in ("input", "target", "mask", "firstU_prediction", "pl_min_db", "pl_scale_db"):
                np.testing.assert_array_equal(arrays[key], reference[key])
            scale, lo = float(arrays["pl_scale_db"]), float(arrays["pl_min_db"])
            valid = arrays["mask"].squeeze() > 0
            error = (arrays["prediction"].astype(np.float64)-arrays["target"])*scale
            rmse = float(np.sqrt(np.mean(error.squeeze()[valid]**2)))
            assert abs(rmse-example["rmse_db"]) < 1e-4
            example_rows.append({"run": name, "sample_id": example["name"], "rmse_db": rmse})
            if example["name"] == "01917_32_58":
                for col in range(80, 196):
                    profile_rows.append({"run": name, "row": 128, "col": col, "valid": int(valid[128, col]),
                                         "target_db": float(arrays["target"][0, 128, col]*scale+lo),
                                         "firstU_db": float(arrays["firstU_prediction"][0, 128, col]*scale+lo),
                                         "prediction_db": float(arrays["prediction"][0, 128, col]*scale+lo)})
        runs[name] = {"audit": audit, "sample": sample, "totals": totals, "report": report, "history": history}
        run_rows.append({"run": name, **audit})
        for path in directory.rglob("*"):
            if path.suffix in (".json", ".csv", ".pt", ".npz"):
                source_hashes.append(fingerprint(path))
        print(f"Verified {name}: {audit['rmse_db_masked']:.9f} dB, epoch {audit['best_epoch']}", flush=True)

    control, candidate, baseline = (runs[name] for name in (*run_names[1:], run_names[0]))
    comparison = read_csv(source / "comparison.csv")
    assert len(comparison) == 3
    for row in comparison:
        run = runs[row["run"]]["audit"]
        assert float(row["rmse_db_masked"]) == run["rmse_db_masked"]
        assert abs(float(row["all_valid/sse_db2"])-run["sse_db2"]) < 1e-4
        assert abs(float(row["delta_rmse_vs_zeros"])-(run["rmse_db_masked"]-control["audit"]["rmse_db_masked"])) < 1e-12
    a, b = control["sample"]["all_valid"], candidate["sample"]["all_valid"]
    terrain_rows, sample_rows, region_rows, group_rows = [], [], [], []
    for t, ix in groups.items():
        n, sa = a[ix, :2].sum(0)
        sb = b[ix, 1].sum()
        terrain_rows.append({"terrain_id": t, "samples": len(ix), "valid_pixels": int(n),
                             "control_sse": sa, "candidate_sse": sb, "delta_sse": sb-sa,
                             "control_rmse": ratio(sa,n), "candidate_rmse": ratio(sb,n)})
    for i, row in enumerate(index):
        sample_rows.append({"sample_id": row["sample_id"], "terrain_id": row["terrain_id"], "group": row["group"],
                            "valid_pixels": int(a[i,0]), "control_rmse": ratio(a[i,1],a[i,0]),
                            "candidate_rmse": ratio(b[i,1],b[i,0]), "delta_sse": b[i,1]-a[i,1]})
    for group in sorted({r["group"] for r in index}):
        ix = [i for i,r in enumerate(index) if r["group"] == group]
        n,sa = a[ix,:2].sum(0)
        sb = b[ix,1].sum()
        group_rows.append({"group": group, "samples": len(ix), "terrains": len({index[i]["terrain_id"] for i in ix}),
                           "valid_pixels": int(n), "control_rmse": ratio(sa,n), "candidate_rmse": ratio(sb,n),
                           "delta_rmse": ratio(sb,n)-ratio(sa,n), "delta_sse": sb-sa})
    for region in control["sample"]:
        n,sa,sign_a = control["sample"][region].sum(0)
        _,sb,sign_b = candidate["sample"][region].sum(0)
        sbase = baseline["sample"][region][:,1].sum()
        region_rows.append({"region": region, "valid_pixels": int(n), "baseline_rmse": ratio(sbase,n),
                            "control_rmse": ratio(sa,n), "candidate_rmse": ratio(sb,n),
                            "control_sse": sa, "candidate_sse": sb, "delta_sse": sb-sa,
                            "delta_sse_percent": 100*(sb-sa)/sa if sa else None,
                            "control_bias": sign_a/n if n else None, "candidate_bias": sign_b/n if n else None})
    g = np.array([[r[k] for k in ("valid_pixels", "control_sse", "candidate_sse")] for r in terrain_rows])
    n,sa,sb = g.sum(0)
    leave = np.sqrt((sb-g[:,2])/(n-g[:,0]))-np.sqrt((sa-g[:,1])/(n-g[:,0]))
    rng = np.random.default_rng(20261004)
    bootstrap = []
    for start in range(0, bootstrap_repeats, 256):
        selection = rng.integers(0, len(g), size=(min(256, bootstrap_repeats-start),len(g)))
        totals = g[selection].sum(1)
        bootstrap.extend((np.sqrt(totals[:,2]/totals[:,0])-np.sqrt(totals[:,1]/totals[:,0])).tolist())
    delta = np.array([float(y["val/rmse_db_masked"])-float(x["val/rmse_db_masked"])
                      for x,y in zip(control["history"],candidate["history"])])
    epoch_rows = [{"epoch": i, "control_rmse": float(x["val/rmse_db_masked"]),
                   "candidate_rmse": float(y["val/rmse_db_masked"]), "delta_rmse": delta[i],
                   "control_train_mse": float(x["train_loss"]), "candidate_train_mse": float(y["train_loss"])}
                  for i,(x,y) in enumerate(zip(control["history"],candidate["history"]))]
    crossings = {}
    for name in run_names:
        rows = [r for r in profile_rows if r["run"] == name]
        crossings[name] = {}
        for key in ("target_db", "firstU_db", "prediction_db"):
            crossings[name][key] = [[left["col"],right["col"]] for left,right in zip(rows[:-1],rows[1:])
                                    if left["valid"] and right["valid"] and left[key] < 110 <= right[key]]
    influence = {"samples_improved": int((b[:,1]<a[:,1]).sum()), "samples_total": 2330,
                 "terrains_improved": int((g[:,2]<g[:,1]).sum()), "terrains_total": 680,
                 "leave_one_terrain_delta_rmse_range": [float(leave.min()),float(leave.max())],
                 "leave_one_terrain_reversals": int((leave>0).sum()),
                 "top5_gain_terrains": sorted(terrain_rows,key=lambda r:r["delta_sse"])[:5],
                 "top5_harm_terrains": sorted(terrain_rows,key=lambda r:r["delta_sse"],reverse=True)[:5],
                 "epochs_candidate_better": int((delta<0).sum()), "last30_mean_delta": float(delta[-30:].mean()),
                 "epoch0_delta": float(delta[0]), "last_delta": float(delta[-1])}
    bootstrap_note = {"unit": "terrain; all TX samples remain grouped", "repeats": bootstrap_repeats,
                      "rng_seed": 20261004, "percentile_95_delta_rmse": np.percentile(bootstrap,[2.5,97.5]).tolist(),
                      "fraction_negative": float(np.mean(np.array(bootstrap)<0)),
                      "interpretation": "Exploratory empirical terrain-resampling sensitivity conditional on these fitted/val-selected checkpoints. Does not include training seed, firstU or checkpoint-selection uncertainty; not independent-test inference or a p value. Parent-terrain correlations not modeled."}
    measured_times = [{k:item[k] for k in ("name","kind","seconds")} for item in manifest["commands"]]
    audit = {"source": str(source.resolve()), "runs": {name:r["audit"] for name,r in runs.items()},
             "influence": influence, "terrain_bootstrap": bootstrap_note,
             "fixed_01917_row128_110db_crossings": crossings, "groups": group_rows,
             "measured_command_times": measured_times, "evaluation_provenance": common_eval,
             "code_vs_git_blob_checks": code_checks, "comparison_csv_verified": True,
             "manifest": manifest, "source_fingerprints": source_hashes+[fingerprint(source/"manifest.json"),fingerprint(source/"comparison.csv")],
             "analysis_script": fingerprint(Path(__file__)),
             "environment": {"numpy":np.__version__,"torch":str(torch.__version__)},
             "limits": "One paired seed; same pretrained full secondU baseline and frozen firstU; best selected using the same official val. No model inference or full-array hashes. Eight cached predictions per run, no full prediction arrays. No causal LOS attribution, no seed-level CI or significance claim."}
    for filename,rows in (("runs.csv",run_rows),("regions.csv",region_rows),("paired_terrains.csv",terrain_rows),
                          ("paired_samples.csv",sample_rows),("groups.csv",group_rows),("epoch_deltas.csv",epoch_rows),
                          ("fixed_examples.csv",example_rows),("fixed_profile.csv",profile_rows),("added_weights.csv",weight_rows)):
        write_csv(out/filename,rows)
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"influence":influence,"terrain_bootstrap":bootstrap_note,"groups":group_rows},indent=2))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--bootstrap-repeats",type=int,default=20000)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("--bootstrap-repeats must be at least 100")
    main(args.source,args.data_root,args.out,args.bootstrap_repeats)
