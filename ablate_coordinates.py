"""Paired secondU fine-tuning from a trained grad_weight=0 baseline.

Default: inspect the checkpoint and print commands without writing files.
--execute runs zeros and tx_xyz with the same settings in a NEW directory.
"""

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from ablate_boundary import REQUIRED_SETTINGS
from experiment_provenance import fingerprint, first_stage_hash
from radiounet import checkpoint_features, geometry_config


def baseline_settings(checkpoint):
    saved = checkpoint.get("args", {})
    for key, expected in (("phase", "secondU"), ("band", "58"), ("loss", "masked")):
        if saved.get(key) != expected:
            raise ValueError(f"Expected baseline {key}={expected!r}, got {saved.get(key)!r}")
    if saved.get("grad_weight", 0.0) != 0.0 or checkpoint_features(checkpoint)[0] != "none":
        raise ValueError("Use a grad_weight=0 baseline with no added features")
    missing = [key for key in REQUIRED_SETTINGS if key not in saved]
    if missing:
        raise ValueError(f"Baseline lacks settings {missing}; no defaults guessed")
    if saved["limit_batches"] != 0 or saved["epochs"] != 150:
        raise ValueError("Expected the full 150-epoch baseline protocol")
    return {key: saved[key] for key in REQUIRED_SETTINGS}


def build_commands(args, settings):
    root = Path(__file__).resolve().parent
    out, data = Path(args.out_root).resolve(), str(Path(args.data_root).resolve())
    baseline = str(Path(args.baseline_ckpt).resolve())
    device = args.device or settings["device"]
    workers = settings["num_workers"] if args.num_workers is None else args.num_workers
    common = ["--data-root", data, "--band", "58", "--phase", "secondU",
              "--batch-size", str(settings["batch_size"]), "--num-workers", str(workers),
              "--device", device, "--amp" if settings["amp"] else "--no-amp"]

    def evaluation(checkpoint, directory):
        cmd = [sys.executable, str(root / "evaluate.py"), "--ckpt", str(checkpoint), *common,
               "--out", str(directory / "val.json"),
               "--boundary-out", str(directory / "boundary"), "--panels", "8"]
        if args.smoke:
            cmd += ["--limit", str(3 * settings["batch_size"])]
        return cmd

    commands = [{"name": "baseline_reference", "kind": "evaluate", "second_features": "none",
                 "command": evaluation(baseline, out / "baseline_reference")}]
    for mode in ("zeros", "tx_xyz"):
        name = f"seed{args.seed}_{mode}"
        directory, run_name = out / name, "radiownet_58_masked_secondU"
        checkpoint = directory / f"{run_name}_best.pt"
        cmd = [sys.executable, str(root / "train.py"), *common, "--loss", "masked",
               "--init-from", baseline, "--out", str(directory), "--run-name", run_name,
               "--seed", str(args.seed), "--grad-weight", "0", "--second-features", mode,
               "--epochs", "1" if args.smoke else str(settings["epochs"]), "--panels", "0"]
        for key in ("lr", "lr_sched", "lr_min", "lr_step", "lr_gamma", "clip"):
            cmd += ["--" + key.replace("_", "-"), str(settings[key])]
        if settings["no_augment"]:
            cmd.append("--no-augment")
        if args.smoke:
            cmd += ["--limit-batches", "3"]
        commands.append({"name": name, "kind": "train", "second_features": mode,
                         "checkpoint": str(checkpoint), "command": cmd})
        commands.append({"name": name, "kind": "evaluate", "second_features": mode,
                         "command": evaluation(checkpoint, directory)})
    return commands


def comparison(out, commands):
    rows = []
    for item in commands:
        if item["kind"] != "evaluate":
            continue
        result = json.loads((out / item["name"] / "boundary" / "summary.json").read_text(encoding="utf-8"))
        regions = {r["region"]: r for r in result["regions"] if r["output"] == "secondU" and r["band"] == "58"}
        row = {"run": item["name"], "second_features": item["second_features"],
               "rmse_db_masked": result["evaluation"]["rmse_db_masked"]}
        for name in ("all_valid", "edge_t10_r3", "nonedge_t10_r3", "near_missing_r5"):
            for metric in ("valid_pixels", "sse_db2", "rmse_db", "bias_db"):
                row[f"{name}/{metric}"] = regions[name][metric]
        rows.append(row)
    control = next(r for r in rows if r["second_features"] == "zeros")
    for row in rows:
        row["delta_rmse_vs_zeros"] = row["rmse_db_masked"] - control["rmse_db_masked"]
        for region in ("edge_t10_r3", "nonedge_t10_r3"):
            row[f"delta_{region}_sse_vs_zeros"] = row[f"{region}/sse_db2"] - control[f"{region}/sse_db2"]
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(args):
    out = Path(args.out_root).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    checkpoint = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    settings = baseline_settings(checkpoint)
    first_hash = first_stage_hash(checkpoint["model"])
    del checkpoint
    metadata_path = Path(args.data_root) / "metadata.json"
    config = geometry_config(json.loads(metadata_path.read_text(encoding="utf-8")))
    commands = build_commands(args, settings)
    manifest = {"status": "planned", "smoke_only": args.smoke, "seed": args.seed,
                "baseline_settings": settings, "grad_weight": 0.0,
                "overrides": {"device": args.device, "num_workers": args.num_workers},
                "firstU_state_sha256": first_hash, "feature_config": config,
                "baseline_checkpoint": fingerprint(args.baseline_ckpt),
                "data_metadata": fingerprint(metadata_path), "commands": commands,
                "interpretation": "Paired fine-tuning from the same trained secondU baseline, with frozen firstU. Compare tx_xyz against retrained zeros; historical baseline is context only. Official val is a development set; smoke scores are not research results."}
    print(json.dumps({k: v for k, v in manifest.items() if k != "commands"}, indent=2))
    for item in commands:
        print(f"\n[{item['name']} / {item['kind']}]\n{shlex.join(item['command'])}", flush=True)
    if not args.execute:
        print("\nRead-only plan. Add --execute to run in a NEW output directory.")
        return manifest
    out.mkdir(parents=True, exist_ok=False)
    path = out / "manifest.json"

    def save():
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    manifest["status"] = "running"
    save()
    try:
        for item in commands:
            start = time.monotonic()
            item["status"] = "running"
            save()
            subprocess.run(item["command"], cwd=Path(__file__).resolve().parent, check=True)
            if item["kind"] == "train":
                trained = torch.load(item["checkpoint"], map_location="cpu", weights_only=False)
                if first_stage_hash(trained["model"]) != first_hash:
                    raise ValueError("firstU changed during secondU training")
                del trained
            item["seconds"] = time.monotonic() - start
            item["status"] = "complete"
            save()
        manifest["comparison"] = comparison(out, commands)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        save()
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--baseline-ckpt", required=True, help="trained grad_weight=0 secondU baseline")
    parser.add_argument("--out-root", required=True, help="must not exist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="1 epoch / 3 batches per mode for throughput checks")
    parser.add_argument("--execute", action="store_true", help="otherwise print a read-only plan")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
