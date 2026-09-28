"""Inspect historical settings and run the pre-registered secondU ablation.

Default is read-only: print the verified configuration and concrete commands.
Only --execute starts experiments, always in a new output directory.
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

from experiment_provenance import fingerprint, first_stage_hash

REQUIRED_SETTINGS = ("epochs", "batch_size", "lr", "lr_sched", "lr_min",
                     "lr_step", "lr_gamma", "clip", "amp", "no_augment",
                     "num_workers", "device", "limit_batches")


def historical_settings(first, second):
    for checkpoint, phase in ((first, "firstU"), (second, "secondU")):
        saved = checkpoint.get("args", {})
        for key, expected in (("phase", phase), ("band", "58"), ("loss", "masked")):
            if saved.get(key) != expected:
                raise ValueError(f"Expected {phase} checkpoint {key}={expected!r}, got {saved.get(key)!r}")
        if saved.get("grad_weight", 0.0) != 0.0:
            raise ValueError("Historical baseline checkpoints must have grad_weight=0")
    saved = second["args"]
    missing = [key for key in REQUIRED_SETTINGS if key not in saved]
    if missing:
        raise ValueError(f"Historical checkpoint lacks settings {missing}; verify them before running, no defaults guessed")
    if saved["epochs"] != 150 or saved["limit_batches"] != 0:
        raise ValueError("Expected a full 150-epoch historical secondU run")
    if first_stage_hash(first["model"]) != first_stage_hash(second["model"]):
        raise ValueError("Historical secondU does not contain the supplied firstU weights")
    return {key: saved[key] for key in REQUIRED_SETTINGS}


def build_commands(args, settings):
    root = Path(__file__).resolve().parent
    out = Path(args.out_root).resolve()
    first = str(Path(args.first_ckpt).resolve())
    second = str(Path(args.reference_second_ckpt).resolve())
    data = str(Path(args.data_root).resolve())
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

    commands = [{"name": "historical_reference", "kind": "evaluate",
                 "command": evaluation(second, out / "historical_reference")}]
    if args.diagnose_only:
        return commands
    for weight in args.grad_weights:
        name = f"seed{args.seed}_grad{weight:g}".replace(".", "p")
        directory = out / name
        run_name = "radiownet_58_masked_secondU"
        cmd = [sys.executable, str(root / "train.py"), *common, "--loss", "masked",
               "--init-from", first, "--out", str(directory), "--run-name", run_name,
               "--seed", str(args.seed), "--grad-weight", str(weight),
               "--epochs", "1" if args.smoke else "150", "--panels", "0"]
        for key in ("lr", "lr_sched", "lr_min", "lr_step", "lr_gamma", "clip"):
            cmd += ["--" + key.replace("_", "-"), str(settings[key])]
        if settings["no_augment"]:
            cmd.append("--no-augment")
        if args.smoke:
            cmd += ["--limit-batches", "3"]
        commands.append({"name": name, "kind": "train", "grad_weight": weight, "command": cmd})
        commands.append({"name": name, "kind": "evaluate", "grad_weight": weight,
                         "command": evaluation(directory / f"{run_name}_best.pt", directory)})
    return commands


def comparison(out, commands):
    rows = []
    for item in commands:
        if item["kind"] != "evaluate":
            continue
        result = json.loads((out / item["name"] / "boundary" / "summary.json").read_text(encoding="utf-8"))
        regions = {r["region"]: r for r in result["regions"] if r["output"] == "secondU" and r["band"] == "58"}
        row = {"run": item["name"], "grad_weight": item.get("grad_weight"),
               "rmse_db_masked": result["evaluation"]["rmse_db_masked"]}
        for name in ("all_valid", "edge_t10_r3", "nonedge_t10_r3", "near_missing_r5"):
            for metric in ("valid_pixels", "sse_db2", "rmse_db", "bias_db"):
                row[f"{name}/{metric}"] = regions[name][metric]
        rows.append(row)
    control = next((r for r in rows if r["grad_weight"] == 0), None)
    if control:
        for row in rows:
            row["delta_rmse_vs_grad0"] = row["rmse_db_masked"] - control["rmse_db_masked"]
            for region in ("edge_t10_r3", "nonedge_t10_r3"):
                row[f"delta_{region}_sse_vs_grad0"] = row[f"{region}/sse_db2"] - control[f"{region}/sse_db2"]
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(args):
    out = Path(args.out_root).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    if len(set(args.grad_weights)) != len(args.grad_weights) or 0 not in args.grad_weights:
        raise ValueError("Use unique weights including the paired grad_weight=0 control")
    first = torch.load(args.first_ckpt, map_location="cpu", weights_only=False)
    second = torch.load(args.reference_second_ckpt, map_location="cpu", weights_only=False)
    settings = historical_settings(first, second)
    first_hash = first_stage_hash(first["model"])
    del first, second
    commands = build_commands(args, settings)
    manifest = {"status": "planned", "smoke_only": args.smoke, "diagnose_only": args.diagnose_only,
                "seed": args.seed, "historical_settings": settings,
                "overrides": {"device": args.device, "num_workers": args.num_workers},
                "firstU_state_sha256": first_hash,
                "first_checkpoint": fingerprint(args.first_ckpt),
                "historical_second_checkpoint": fingerprint(args.reference_second_ckpt),
                "commands": commands,
                "interpretation": "Exploratory official-val comparison conditional on one fixed firstU. Smoke results are not research results. No automatic weight search or extra seeds."}
    print(json.dumps({k: v for k, v in manifest.items() if k != "commands"}, indent=2))
    for item in commands:
        print(f"\n[{item['name']} / {item['kind']}]\n{shlex.join(item['command'])}", flush=True)
    if not args.execute:
        print("\nRead-only plan. Add --execute to run these commands in a NEW output directory.")
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
    parser.add_argument("--first-ckpt", required=True)
    parser.add_argument("--reference-second-ckpt", required=True)
    parser.add_argument("--out-root", required=True, help="must not exist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad-weights", type=float, nargs="+", default=[0.0, 0.01, 0.03], choices=[0.0, 0.01, 0.03])
    parser.add_argument("--device", default=None, help="optional explicit override, shared by every group")
    parser.add_argument("--num-workers", type=int, default=None, help="optional explicit override, shared by every group")
    parser.add_argument("--smoke", action="store_true", help="1 epoch / 3 batches for throughput checks only")
    parser.add_argument("--diagnose-only", action="store_true", help="only evaluate the historical secondU (also reports firstU boundaries)")
    parser.add_argument("--execute", action="store_true", help="otherwise print a read-only plan")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
