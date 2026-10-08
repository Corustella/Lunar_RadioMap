"""Run a batch of full two-stage LOS experiments and compare paired seeds.

Each seed starts from its historical secondU-only LOS best checkpoint, trains
firstU with LOS, then freezes firstU and trains secondU. The original optimizer,
augmentation and scoring settings are inherited; each phase starts fresh.
"""

import argparse
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

import torch

from experiment_provenance import collect_provenance
from radiounet import (checkpoint_features, checkpoint_first_features,
                       feature_config_for_mode)
from train import load_dataset_module


REQUIRED_SETTINGS = ("epochs", "batch_size", "lr", "lr_sched", "lr_min",
                     "lr_step", "lr_gamma", "clip", "amp", "no_augment",
                     "num_workers", "device", "limit_batches")


def initialization_record(path, seed, feature_config):
    """Check a historical LOS checkpoint before creating outputs or training."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    for key, expected in (("phase", "secondU"), ("band", "58"),
                          ("loss", "masked"), ("seed", seed)):
        if saved.get(key) != expected:
            raise ValueError(f"{path}: expected {key}={expected!r}, got {saved.get(key)!r}")
    if saved.get("grad_weight", 0.0) != 0.0:
        raise ValueError(f"{path}: initialization must have grad_weight=0")
    mode, config = checkpoint_features(checkpoint)
    if checkpoint_first_features(checkpoint) != "none" or mode != "los":
        raise ValueError(f"{path}: use the corresponding historical secondU-only LOS best")
    if config != feature_config:
        raise ValueError(f"{path}: LOS metadata differs from the supplied dataset")
    missing = [key for key in REQUIRED_SETTINGS if key not in saved]
    if missing:
        raise ValueError(f"{path}: missing settings {missing}; no defaults guessed")
    if saved["epochs"] != 150 or saved["limit_batches"] != 0:
        raise ValueError(f"{path}: expected the full historical 150-epoch LOS protocol")
    score = checkpoint.get("metrics", {}).get("val/rmse_db_masked")
    best = checkpoint.get("best")
    if (not isinstance(score, (int, float)) or not math.isfinite(score)
            or score < 0 or not isinstance(best, (int, float))
            or not math.isclose(score, best, rel_tol=0.0, abs_tol=1e-7)):
        raise ValueError(f"{path}: a best checkpoint with finite masked validation RMSE is required")
    state = checkpoint.get("model", {})
    expected_inputs = {"layer00.0.weight": 2, "conv_up00.0.weight": 28,
                       "conv_up000.0.weight": 22, "Wlayer00.0.weight": 6,
                       "Wconv_up00.0.weight": 46, "Wconv_up000.0.weight": 26}
    for name, inputs in expected_inputs.items():
        if name not in state or state[name].ndim != 4 or state[name].shape[1] != inputs:
            raise ValueError(f"{path}: incompatible historical LOS weight {name}")
    return {"seed": seed, "checkpoint": str(path), "epoch": checkpoint.get("epoch"),
            "rmse_db_masked": score,
            "settings": {key: saved[key] for key in REQUIRED_SETTINGS}}


def build_commands(args, references):
    root = Path(__file__).resolve().parent
    out = Path(args.out_root).resolve()
    data = str(Path(args.data_root).resolve())
    cache = str(Path(args.cache_root).resolve())
    device = args.device or references[0]["settings"]["device"]
    commands = [{"kind": "cache", "command": [sys.executable,
                 str(root / "build_los_cache.py"), "--data-root", data,
                 "--cache-root", cache, "--device", device]}]
    for reference in references:
        seed, settings = reference["seed"], reference["settings"]
        workers = settings["num_workers"] if args.num_workers is None else args.num_workers
        common = ["--data-root", data, "--los-cache", cache, "--band", "58",
                  "--batch-size", str(settings["batch_size"]),
                  "--num-workers", str(workers), "--device", device,
                  "--amp" if settings["amp"] else "--no-amp"]
        initialization = reference["checkpoint"]
        for phase, epochs in (("firstU", args.first_epochs), ("secondU", args.second_epochs)):
            directory = out / f"seed{seed}" / phase
            run_name = f"radiownet_58_masked_{phase}_los_both"
            checkpoint = directory / f"{run_name}_best.pt"
            result = directory / "val.json"
            train = [sys.executable, str(root / "train.py"), *common,
                     "--phase", phase, "--loss", "masked", "--init-from", initialization,
                     "--first-features", "los", "--second-features", "los",
                     "--out", str(directory), "--run-name", run_name,
                     "--seed", str(seed), "--epochs", str(epochs), "--panels", "0"]
            for key in ("lr", "lr_sched", "lr_min", "lr_step", "lr_gamma", "clip"):
                train += ["--" + key.replace("_", "-"), str(settings[key])]
            if settings["no_augment"]:
                train.append("--no-augment")
            commands.append({"kind": "train", "seed": seed, "phase": phase,
                             "epochs": epochs, "initialization": initialization,
                             "checkpoint": str(checkpoint), "command": train})
            commands.append({"kind": "evaluate", "seed": seed, "phase": phase,
                             "checkpoint": str(checkpoint), "result": str(result),
                             "command": [sys.executable, str(root / "evaluate.py"),
                                         "--ckpt", str(checkpoint), *common,
                                         "--phase", phase, "--split", "val", "--panels", "0",
                                         "--out", str(result)]})
            # A firstU checkpoint saves both halves; secondU starts its weights
            # from this phase's selected model, without optimizer/schedule state.
            initialization = str(checkpoint)
    return commands


def comparison(references, commands):
    rows = []
    for reference in references:
        seed = reference["seed"]
        stages = {}
        for item in commands:
            if item["kind"] != "evaluate" or item["seed"] != seed:
                continue
            summary = json.loads(Path(item["result"]).read_text(encoding="utf-8"))["summary"]
            score = summary.get("rmse_db_masked")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"Invalid full-val masked RMSE in {item['result']}")
            stages[item["phase"]] = {"rmse_db_masked": score,
                                     "samples": summary["samples"],
                                     "selected_epoch": summary["epoch"],
                                     "checkpoint": summary["checkpoint"],
                                     "result": item["result"]}
        if set(stages) != {"firstU", "secondU"}:
            raise ValueError(f"Missing completed stage evaluations for seed {seed}")
        if stages["firstU"]["samples"] != stages["secondU"]["samples"]:
            raise ValueError(f"Stage validation sample counts differ for seed {seed}")
        rows.append({"seed": seed, "reference_checkpoint": reference["checkpoint"],
                     "reference_epoch": reference["epoch"],
                     "reference_rmse_db_masked": reference["rmse_db_masked"],
                     "stages": stages,
                     "rmse_db_masked": stages["secondU"]["rmse_db_masked"],
                     "delta_rmse_vs_reference_db": (stages["secondU"]["rmse_db_masked"]
                                                     - reference["rmse_db_masked"])})
    if len({row["stages"]["secondU"]["samples"] for row in rows}) != 1:
        raise ValueError("Validation sample counts differ between seeds")
    scores = [row["rmse_db_masked"] for row in rows]
    selected = min(rows, key=lambda row: row["rmse_db_masked"])
    return {"per_seed": rows,
            "mean_rmse_db_masked": statistics.mean(scores),
            "sample_sd_rmse_db_masked": statistics.stdev(scores),
            "mean_reference_rmse_db_masked": statistics.mean(
                row["reference_rmse_db_masked"] for row in rows),
            "mean_delta_rmse_vs_reference_db": statistics.mean(
                row["delta_rmse_vs_reference_db"] for row in rows),
            "best_seed": selected["seed"],
            "best_checkpoint": selected["stages"]["secondU"]["checkpoint"],
            "aggregation": "Descriptive arithmetic mean across seed RMSEs; not an ensemble score."}


def main(args):
    if len(args.seeds) < 2 or len(args.seeds) != len(args.init_ckpts):
        raise ValueError("Supply at least two seeds and exactly one --init-ckpts path per seed")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain distinct seeds")
    if len({str(Path(path).resolve()) for path in args.init_ckpts}) != len(args.init_ckpts):
        raise ValueError("Use the corresponding distinct historical LOS checkpoint for each seed")
    if args.first_epochs <= 0 or args.second_epochs <= 0:
        raise ValueError("Both phase budgets must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    out = Path(args.out_root).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    data = Path(args.data_root).resolve()
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    feature_config = feature_config_for_mode(metadata, "los")
    references = [initialization_record(path, seed, feature_config)
                  for path, seed in zip(args.init_ckpts, args.seeds)]
    # Seeds may have different loader workers or devices in historical records;
    # the actual training protocol must otherwise match for the paired batch.
    protocol_keys = [key for key in REQUIRED_SETTINGS if key not in ("device", "num_workers")]
    protocol = {key: references[0]["settings"][key] for key in protocol_keys}
    for reference in references[1:]:
        if {key: reference["settings"][key] for key in protocol_keys} != protocol:
            raise ValueError("Initialization checkpoints use different training protocols")
    device = torch.device(args.device or references[0]["settings"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose an available device explicitly")
    loader = load_dataset_module(str(data))
    provenance_args = argparse.Namespace(**vars(args))
    provenance_args.device = str(device)
    provenance = collect_provenance(str(data), loader,
                                    {f"seed{ref['seed']}": ref["checkpoint"] for ref in references},
                                    provenance_args)
    commands = build_commands(args, references)
    manifest = {"status": "planned", "seeds": args.seeds, "references": references,
                "phase_epochs": {"firstU": args.first_epochs, "secondU": args.second_epochs},
                "first_features": "los", "second_features": "los",
                "feature_config": feature_config, "cache_root": str(Path(args.cache_root).resolve()),
                "overrides": {"device": args.device, "num_workers": args.num_workers},
                "provenance": provenance, "commands": commands,
                "interpretation": "Both U-Nets receive the same LOS feature. FirstU is trained, "
                "then its best is frozen while secondU is trained. Each phase uses fresh Adam and "
                "schedule state. Full official val masked RMSE selects each best checkpoint. "
                "Paired deltas include additional two-stage training, so they do not isolate "
                "feature placement. No automatic follow-up experiments."}
    for item in commands:
        label = item["kind"] if item["kind"] == "cache" else f"seed{item['seed']} {item['phase']} {item['kind']}"
        print(f"[{label}] {shlex.join(item['command'])}", flush=True)
    if args.dry_run:
        print("Read-only command preview; no cache, training or output directory created.")
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
            try:
                subprocess.run(item["command"], cwd=Path(__file__).resolve().parent, check=True)
            except BaseException as exc:
                item["status"], item["error"] = "failed", repr(exc)
                raise
            else:
                item["status"] = "complete"
            finally:
                item["seconds"] = time.monotonic() - start
                save()
        result = comparison(references, commands)
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
    parser.add_argument("--init-ckpts", nargs="+", required=True,
                        help="historical secondU-only LOS best paths, in --seeds order")
    parser.add_argument("--cache-root", required=True, help="existing LOS cache can be reused")
    parser.add_argument("--out-root", required=True, help="must not exist")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="at least two distinct seeds")
    parser.add_argument("--first-epochs", type=int, default=150)
    parser.add_argument("--second-epochs", type=int, default=150)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="optional read-only command preview")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
