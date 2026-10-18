"""Measure batch-one inference with freshly constructed physical inputs.

No labels, masks or precomputed LOS cache are read. Every measured row has a
different terrain/TX sample. Disk input reads are reported separately; total
latency includes normalization, FP32 clearance construction, CPU/GPU transfers,
the full network (including rho), and a CPU prediction in dB. This is a local
timing check, not an official challenge certification or a throughput test.

    python benchmark_inference.py --data-root LunarRM --ckpt runs/model_best.pt \
        --samples 16 --out results/inference_timing.json

Omit --ckpt to benchmark --arch (default lunar_maxvit) with random weights. That mode
does not establish prediction quality. This script never trains or writes a
checkpoint.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

from los_features import clearance_feature
from model_factory import (ARCHITECTURES, RADIO_ARCHITECTURES, build_model, checkpoint_architecture,
                           checkpoint_input_config, load_checkpoint,
                           model_feature_config, output_index)
from radiounet import feature_config_for_mode
from train import load_dataset_module


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_rows(count, samples, warmup):
    """Uniform measured rows, with disjoint warm-up rows; no repeated sample."""
    if samples < 1 or warmup < 0:
        raise ValueError("--samples must be positive and --warmup nonnegative")
    if samples + warmup > count:
        raise ValueError(f"Need {samples + warmup} distinct rows, but split has {count}")
    measured = np.linspace(0, count - 1, samples, dtype=np.int64).tolist()
    used = set(measured)
    available = [i for i in range(count) if i not in used]
    positions = (np.linspace(0, len(available) - 1, warmup, dtype=np.int64).tolist()
                 if warmup else [])
    warming = [available[i] for i in positions]
    if len(set(measured + warming)) != samples + warmup:
        raise ValueError("Failed to select distinct benchmark rows")
    return warming, measured


def timing_statistics(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "min": float(a.min())}


@torch.inference_mode()
def measure_row(dataset, index, model, config, los_config, device, amp, chunk,
                out_idx):
    # Use input readers directly: dataset[index] would also read hidden labels
    # on train/val, and cached feature wrappers would omit the actual F cost.
    read_start = time.perf_counter()
    raw_height = dataset._heightmap(index)
    tx_map = dataset._tx(index)
    input_read_ms = (time.perf_counter() - read_start) * 1000

    synchronize(device)
    total_start = time.perf_counter()
    lo, hi = config["heightmap_range_m"]
    height = dataset._norm(raw_height, lo, hi)
    if height.shape != tuple(config["grid"]) or tx_map.shape != height.shape:
        raise ValueError(f"Input shape differs from metadata at row {index}")
    if not np.isfinite(raw_height).all() or not np.isfinite(height).all():
        raise ValueError(f"Non-finite heightmap at row {index}")
    locations = np.argwhere(tx_map == 1)
    if locations.shape != (1, 2) or np.count_nonzero(tx_map) != 1:
        raise ValueError(f"Expected one-hot TX at row {index}")
    tx = tuple(map(int, locations[0]))
    # Match build_los_cache.py: normalize with the actual loader, reconstruct
    # metres, and evaluate the reviewed FP32 clearance proxy without a cache.
    height_m = height.astype(np.float64) * (hi - lo) + lo
    feature = clearance_feature(height_m, tx, los_config, device, chunk)
    if feature.shape != height.shape or not np.isfinite(feature).all():
        raise ValueError(f"Invalid freshly constructed clearance at row {index}")
    x = torch.from_numpy(np.stack((height, tx_map, feature))).unsqueeze(0)
    x = x.to(device)
    synchronize(device)
    feature_end = time.perf_counter()

    with torch.autocast("cuda", enabled=amp):
        outputs = model(x)
        pred = outputs[out_idx]
    synchronize(device)
    network_end = time.perf_counter()

    if pred.shape != (1, 1, *config["grid"]):
        raise ValueError(f"Expected one complete prediction, got {tuple(pred.shape)}")
    prediction_db = pred.float().cpu().numpy()
    prediction_db = prediction_db * (dataset.pl_max - dataset.pl_min) + dataset.pl_min
    if not np.isfinite(prediction_db).all():
        raise ValueError(f"Non-finite prediction at row {index}")
    synchronize(device)
    total_end = time.perf_counter()

    row = dataset.rows[index]
    total_ms = (total_end - total_start) * 1000
    return {"row": index, "sample_id": row.get("sample_id"),
            "terrain_id": row.get("terrain_id"), "tx_row": tx[0], "tx_col": tx[1],
            "input_read_ms": input_read_ms,
            "feature_ms": (feature_end - total_start) * 1000,
            "network_ms": (network_end - feature_end) * 1000,
            "output_ms": (total_end - network_end) * 1000,
            "total_ms": total_ms,
            "input_read_plus_total_ms": input_read_ms + total_ms,
            "within_500ms": total_ms <= 500.0}


def main(args):
    output_path = Path(args.out).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new --out")
    if args.chunk < 1:
        raise ValueError("--chunk must be positive")
    if args.samples < 1 or args.warmup < 0:
        raise ValueError("--samples must be positive and --warmup nonnegative")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Benchmark supports CPU or CUDA only")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    amp = args.amp and device.type == "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = None
    arch, phase = args.arch or "lunar_maxvit", "single"
    first_features, second_features = "none", "none"
    if arch in RADIO_ARCHITECTURES:
        phase, first_features, second_features = "secondU", "los", "los_distance"
    saved_config = None
    band = args.band or "58"
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        arch = checkpoint_architecture(checkpoint)
        if args.arch is not None and args.arch != arch:
            raise ValueError("--arch differs from the checkpoint architecture")
        saved = checkpoint.get("args", {})
        phase = saved.get("phase", "single")
        saved_band = saved.get("band")
        if saved_band not in ("415", "58"):
            raise ValueError("Checkpoint must record a single challenge band")
        if args.band is not None and args.band != saved_band:
            raise ValueError("--band differs from the checkpoint's trained band")
        band = saved_band
        first_features, second_features, saved_config = checkpoint_input_config(checkpoint)
    if arch in RADIO_ARCHITECTURES and (phase != "secondU" or
            first_features != "los" or second_features != "los_distance"):
        raise ValueError("RadioWNet timing requires the complete LOS+distance secondU model")
    out_idx = output_index(arch, phase)

    ld = load_dataset_module(args.data_root)
    dataset = ld.LunarRadioMapDataset(args.data_root, split=args.split, band=band,
                                     normalize=True, augment=False,
                                     return_mask=False, return_name=False)
    config = model_feature_config(dataset.meta, arch, second_features)
    if checkpoint is not None and saved_config != config:
        raise ValueError("Dataset metadata differs from checkpoint physical scales")
    los_config = feature_config_for_mode(dataset.meta, "los")
    warming, measured = select_rows(len(dataset.rows), args.samples, args.warmup)
    model = build_model(arch, inputs=2, phase=phase, feature_config=config,
                        first_features=first_features,
                        second_features=second_features)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
    model = model.to(device).eval()
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"batch=1, {arch}, {'checkpoint weights' if checkpoint is not None else 'RANDOM weights'}, "
          f"band={band}, device={device}, amp={amp}, params={parameter_count:,}", flush=True)
    print(f"loader: {Path(ld.__file__).resolve()}", flush=True)
    print("Total includes live F, input transfers, model/rho and CPU dB output; "
          "disk input reads are reported separately. No labels/masks/F cache.", flush=True)
    for index in warming:
        measure_row(dataset, index, model, config, los_config, device, amp,
                    args.chunk, out_idx)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for index in measured:
        row = measure_row(dataset, index, model, config, los_config, device, amp,
                          args.chunk, out_idx)
        rows.append(row)
        print(f"  {len(rows)}/{len(measured)} row={index}: "
              f"feature={row['feature_ms']:.2f}, network={row['network_ms']:.2f}, "
              f"total={row['total_ms']:.2f} ms", flush=True)

    keys = ("input_read_ms", "feature_ms", "network_ms", "output_ms", "total_ms",
            "input_read_plus_total_ms")
    summary = {key: timing_statistics([row[key] for row in rows]) for key in keys}
    within = sum(row["within_500ms"] for row in rows)
    summary["budget_ms"] = 500.0
    summary["samples_within_500ms"] = within
    summary["observed_all_within_500ms"] = within == len(rows)
    summary["interpretation"] = (
        "Observed batch-one latency on these distinct samples only. "
        "Not official certification, a worst-case guarantee, or an accuracy evaluation.")
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "torchvision": version("torchvision"),
                   "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                   "device": str(device), "gpu": None,
                   "torch_num_threads": torch.get_num_threads(),
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
        environment["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
        summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "batch_one_complete_physical_input_inference_latency",
              "weights": "checkpoint" if checkpoint is not None else "random",
              "checkpoint": str(Path(args.ckpt).resolve()) if args.ckpt else None,
              "checkpoint_epoch": checkpoint.get("epoch") if checkpoint is not None else None,
              "arch": arch, "phase": phase, "band": band,
              "model_config": model.model_config, "feature_config": config,
              "parameter_count": parameter_count, "batch_size": 1,
              "amp": amp, "seed": args.seed, "clearance_chunk": args.chunk,
              "data_root": str(Path(args.data_root).resolve()), "split": args.split,
              "actual_loader": str(Path(ld.__file__).resolve()),
              "metadata_path": str(Path(args.data_root).resolve() / "metadata.json"),
              "metadata": dataset.meta, "environment": environment,
              "warmup_rows": warming, "measured_rows": measured,
              "timing_scope": {
                  "input_read_ms": "Actual loader _heightmap and _tx calls; OS/mmap caches are uncontrolled",
                  "feature_ms": "Input validation/normalization, live FP32 clearance, CPU stacking and input H2D",
                  "network_ms": "Synchronized entire model forward including rho; autocast context overhead included",
                  "output_ms": "Prediction transfer to CPU, dB conversion and finite/shape checks",
                  "total_ms": "feature_ms + network_ms + output_ms; excludes disk reads and model startup",
                  "input_read_plus_total_ms": "Input readers plus the complete measured inference path",
                  "excluded": "Model/checkpoint loading, metadata/index parsing, warmup and report serialization"},
              "summary": summary, "per_sample": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"total mean={summary['total_ms']['mean']:.2f}, "
          f"p95={summary['total_ms']['p95']:.2f}, max={summary['total_ms']['max']:.2f} ms; "
          f"{within}/{len(rows)} observed samples <=500 ms. This is not official certification.",
          flush=True)
    print(f"wrote {output_path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt", default=None,
                        help="lunar_maxvit or LOS+distance RadioWNet checkpoint; omit for random-weight timing")
    parser.add_argument("--arch", choices=ARCHITECTURES, default=None,
                        help="defaults to checkpoint architecture, or lunar_maxvit for random weights")
    parser.add_argument("--band", choices=["415", "58"], default=None,
                        help="defaults to checkpoint band, or 58 for random weights")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3,
                        help="additional distinct samples, excluded from timing summary")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="new JSON path; existing files are never overwritten")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
