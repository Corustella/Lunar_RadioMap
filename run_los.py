"""Run one full LOS main experiment: cache -> 150-epoch secondU -> full val RMSE.

Uses the original grad_weight=0 baseline, freezes firstU, and inherits training
settings. No extra control training, boundary reports or hash audits are run.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from ablate_coordinates import baseline_settings
from radiounet import feature_config_for_mode


def build_commands(args, settings):
    root = Path(__file__).resolve().parent
    out = Path(args.out_root).resolve()
    data, cache = str(Path(args.data_root).resolve()), str(Path(args.cache_root).resolve())
    baseline = str(Path(args.baseline_ckpt).resolve())
    device = args.device or settings["device"]
    workers = settings["num_workers"] if args.num_workers is None else args.num_workers
    directory = out / f"seed{args.seed}_los"
    run_name = "radiownet_58_masked_secondU_los"
    checkpoint = directory / f"{run_name}_best.pt"
    common = ["--data-root", data, "--los-cache", cache, "--band", "58", "--phase", "secondU",
              "--batch-size", str(settings["batch_size"]), "--num-workers", str(workers),
              "--device", device, "--amp" if settings["amp"] else "--no-amp"]
    train = [sys.executable, str(root / "train.py"), *common, "--loss", "masked",
             "--init-from", baseline, "--out", str(directory), "--run-name", run_name,
             "--seed", str(args.seed), "--grad-weight", "0", "--second-features", "los",
             "--epochs", str(settings["epochs"]), "--panels", "0",
             "--no-fingerprints", "--no-gradient-diagnostics"]
    for key in ("lr", "lr_sched", "lr_min", "lr_step", "lr_gamma", "clip"):
        train += ["--" + key.replace("_", "-"), str(settings[key])]
    if settings["no_augment"]:
        train.append("--no-augment")
    return [
        {"kind": "cache", "command": [sys.executable, str(root / "build_los_cache.py"),
         "--data-root", data, "--cache-root", cache, "--device", device]},
        {"kind": "train", "checkpoint": str(checkpoint), "command": train},
        {"kind": "evaluate", "result": str(directory / "val.json"),
         "command": [sys.executable, str(root / "evaluate.py"), "--ckpt", str(checkpoint),
                     *common, "--split", "val", "--panels", "0", "--out", str(directory / "val.json")]},
    ]


def main(args):
    out = Path(args.out_root).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    checkpoint = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    settings = baseline_settings(checkpoint)
    reference_rmse = checkpoint.get("metrics", {}).get("val/rmse_db_masked", checkpoint.get("best"))
    reference_epoch = checkpoint.get("epoch")
    del checkpoint
    metadata = json.loads((Path(args.data_root) / "metadata.json").read_text(encoding="utf-8"))
    commands = build_commands(args, settings)
    manifest = {"status": "planned", "seed": args.seed, "baseline_settings": settings,
                "baseline_checkpoint": str(Path(args.baseline_ckpt).resolve()),
                "reference_rmse_db_masked": reference_rmse, "reference_epoch": reference_epoch,
                "feature_config": feature_config_for_mode(metadata, "los"),
                "cache_root": str(Path(args.cache_root).resolve()),
                "overrides": {"device": args.device, "num_workers": args.num_workers},
                "commands": commands,
                "interpretation": "Single LOS main run with frozen firstU and grad_weight=0. "
                "Delta against the historical baseline is descriptive; it also includes secondU fine-tuning. "
                "Full official val masked RMSE selects the checkpoint. No automatic follow-up experiments."}
    for item in commands:
        print(f"[{item['kind']}] {shlex.join(item['command'])}", flush=True)
    if args.dry_run:
        print("Read-only command preview; omit --dry-run to run the full experiment.")
        return manifest
    out.mkdir(parents=True, exist_ok=False)
    path = out / "manifest.json"

    def save():
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    start_all = time.monotonic()
    manifest["status"] = "running"
    save()
    try:
        for item in commands:
            start = time.monotonic()
            item["status"] = "running"
            save()
            subprocess.run(item["command"], cwd=Path(__file__).resolve().parent, check=True)
            item["seconds"], item["status"] = time.monotonic() - start, "complete"
            save()
        summary = json.loads(Path(commands[-1]["result"]).read_text(encoding="utf-8"))["summary"]
        result = {"reference_checkpoint": manifest["baseline_checkpoint"],
                  "reference_rmse_db_masked": reference_rmse,
                  "los_rmse_db_masked": summary["rmse_db_masked"],
                  "delta_rmse_vs_reference_db": (summary["rmse_db_masked"] - reference_rmse
                                                 if reference_rmse is not None else None),
                  "samples": summary["samples"], "selected_epoch": summary["epoch"],
                  "checkpoint": summary["checkpoint"]}
        (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        manifest["comparison"], manifest["status"] = result, "complete"
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        manifest["status"], manifest["error"] = "failed", repr(exc)
        raise
    finally:
        manifest["total_seconds"] = time.monotonic() - start_all
        save()
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--baseline-ckpt", required=True, help="original trained grad_weight=0 secondU baseline")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out-root", required=True, help="must not exist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="optional read-only command preview")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
