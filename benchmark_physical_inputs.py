"""Small local batch-one budget check; no feature cache and no training.

Diffuse checkpoints consume B in both U-Nets. Legacy LOS-distance checkpoints
remain supported as a network-cost estimate and are marked explicitly.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE
sys.path.insert(0, str(REPO))

from diffuse_light import LightDiffuse, light_diffuse_config
from evaluate import load_dataset_module
from experiment_provenance import collect_provenance
from los_features import clearance_feature
from radiounet import (RadioWNet, checkpoint_architecture, checkpoint_features,
                       checkpoint_first_features, feature_config_for_mode,
                       load_model_state)


def synchronize():
    torch.cuda.synchronize()


def main(args):
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output JSON: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU timing substitution")
    if args.repeats < 1 or not args.indices or len(set(args.indices)) != len(args.indices):
        raise ValueError("Use positive repeats and a nonempty list of distinct index rows")
    init_start = time.perf_counter()
    device = torch.device("cuda")
    args.device = str(device)
    ld = load_dataset_module(str(Path(args.data_root).resolve()))
    dataset = ld.LunarRadioMapDataset(args.data_root, split=args.split, band="58",
                                     normalize=True, augment=False)
    if any(index < 0 or index >= len(dataset.rows) for index in args.indices):
        raise ValueError("A requested row is outside the selected split")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved = checkpoint.get("args", {})
    if saved.get("band") != "58" or saved.get("phase") != "secondU":
        raise ValueError("This budget check requires the specified 58 secondU checkpoint")
    second_features, config = checkpoint_features(checkpoint)
    first_features = checkpoint_first_features(checkpoint)
    architecture = checkpoint_architecture(checkpoint)
    trained_diffuse = second_features == "los_distance_diffuse"
    if args.initialize_diffuse:
        if trained_diffuse or first_features != "los" or second_features != "los_distance":
            raise ValueError("--initialize-diffuse requires a legacy LOS-distance checkpoint")
        if config != feature_config_for_mode(dataset.meta, second_features):
            raise ValueError("Source checkpoint metadata differs from actual input metadata")
        first_features, second_features = "los_diffuse", "los_distance_diffuse"
        config = feature_config_for_mode(dataset.meta, second_features, diffuse_rx_grid=args.rx_grid)
    consume_diffuse = first_features == "los_diffuse" and second_features == "los_distance_diffuse"
    if not consume_diffuse and (second_features != "los_distance" or first_features != "los"):
        raise ValueError("Expected both-stage LOS/diffuse and second-stage distance")
    if config != feature_config_for_mode(dataset.meta, second_features, diffuse_rx_grid=args.rx_grid):
        raise ValueError("Checkpoint metadata differs from actual input metadata")
    model = RadioWNet(inputs=2, phase="secondU", second_features=second_features,
                      feature_config=config, first_features=first_features,
                      second_architecture=architecture).to(device)
    load_model_state(model, checkpoint, initialize=args.initialize_diffuse)
    model.eval()
    amp = bool(saved.get("amp", False))
    los_config = feature_config_for_mode(dataset.meta, "los")
    light_config = light_diffuse_config(dataset.meta, band="58", n_mu=args.n_mu,
                                        n_phi=args.n_phi, rx_grid=args.rx_grid)
    if consume_diffuse and light_config != config["diffuse"]:
        raise ValueError("Timing configuration differs from the trained diffuse input")
    light = LightDiffuse(light_config, device="cuda", ray_chunk=args.ray_chunk)
    synchronize()
    static_init_ms = (time.perf_counter() - init_start) * 1000

    @torch.no_grad()
    def run_row(index, repeat, warmup=False):
        # Read every sample afresh through the actual loader's input readers.
        # OS page caching is not flushed. No labels, masks or feature cache.
        synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        height = np.array(dataset._heightmap(index), dtype=np.float32, copy=True)
        tx_map = np.array(dataset._tx(index), dtype=np.float32, copy=True)
        indices = np.argwhere(tx_map == 1)
        if indices.shape != (1, 2) or np.count_nonzero(tx_map) != 1:
            raise ValueError("Expected exactly one one-hot TX")
        tx = tuple(map(int, indices[0]))
        io_end = time.perf_counter()

        los = clearance_feature(height, tx, los_config, device, chunk=4096)
        synchronize()
        los_end = time.perf_counter()

        diffuse = light.feature(height, tx)
        synchronize()
        light_end = time.perf_counter()

        normalized_height = dataset._norm(height, dataset.hm_min, dataset.hm_max)
        model_input = torch.from_numpy(np.stack((normalized_height, tx_map, los)))
        model_input = model_input.unsqueeze(0).to(device)
        if consume_diffuse:
            model_input = torch.cat((model_input, diffuse), dim=1)
        synchronize()
        encoding_end = time.perf_counter()

        # The distance pre-hook and both U-Nets match evaluation. Legacy
        # checkpoints never receive a channel they were not trained to consume.
        with torch.autocast("cuda", enabled=amp):
            prediction = model(model_input)[1]
        prediction_db = (prediction.float() * (dataset.pl_max - dataset.pl_min)
                         + dataset.pl_min).cpu().numpy().copy()
        synchronize()
        end = time.perf_counter()

        if not np.isfinite(prediction_db).all():
            raise FloatingPointError("Nonfinite model output")
        if not torch.isfinite(diffuse).all().item():
            raise FloatingPointError("Nonfinite diffuse feature")
        result = {
            "index": index, "sample_id": dataset.rows[index].get("sample_id"),
            "repeat": repeat, "warmup": warmup,
            "input_read_ms": (io_end - start) * 1000,
            "los_ms": (los_end - io_end) * 1000,
            "light_ms": (light_end - los_end) * 1000,
            "input_encoding_and_h2d_ms": (encoding_end - light_end) * 1000,
            "model_distance_db_and_d2h_ms": (end - encoding_end) * 1000,
            "total_ms_including_input_read": (end - start) * 1000,
            "total_ms_excluding_input_read": (end - io_end) * 1000,
            "peak_allocated_mb_including_static_model": torch.cuda.max_memory_allocated() / 1e6,
            "diffuse_shape": list(diffuse.shape),
            "diffuse_range": [diffuse.min().item(), diffuse.max().item()],
            "prediction_db_range": [float(prediction_db.min()), float(prediction_db.max())],
        }
        print(json.dumps(result), flush=True)
        return result

    warmup = run_row(args.indices[0], -1, warmup=True)
    rows = [run_row(index, repeat) for index in args.indices for repeat in range(args.repeats)]
    fields = ("input_read_ms", "los_ms", "light_ms", "input_encoding_and_h2d_ms",
              "model_distance_db_and_d2h_ms", "total_ms_including_input_read",
              "total_ms_excluding_input_read")
    summary = {}
    for name in fields:
        values = np.array([row[name] for row in rows])
        summary[name] = {"mean": float(values.mean()), "p95_linear": float(np.percentile(values, 95)),
                         "min": float(values.min()), "max": float(values.max())}
    result = {
        "status": "complete", "implementation": "dual_center_light_v1",
        "gpu": torch.cuda.get_device_name(0),
        "provenance": collect_provenance(args.data_root, ld, {"budget_checkpoint": args.ckpt}, args),
        "seed": None, "seed_note": "deterministic quadrature; existing trained weights; no training",
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "python": sys.executable, "loader": ld.__file__, "checkpoint": str(Path(args.ckpt).resolve()),
        "data_root": str(Path(args.data_root).resolve()), "batch_size": 1, "amp_from_checkpoint": amp,
        "split": args.split, "indices": args.indices, "repeats": args.repeats,
        "light_feature_config": light.config, "diffuse_consumed_by_checkpoint": trained_diffuse,
        "diffuse_input_in_forward": consume_diffuse,
        "diffuse_weights_zero_initialized_for_budget": args.initialize_diffuse,
        "static_init_ms": static_init_ms,
        "static_init_scope": "Actual loader/dataset setup, checkpoint load, model construction/upload, CUDA initialization and material LUT; Python/module import startup excluded.",
        "first_features": first_features, "second_features": second_features,
        "second_architecture": architecture,
        "light_config": {"n_mu": args.n_mu, "n_phi": args.n_phi,
                         "receiver_size": args.rx_grid, "ray_chunk": args.ray_chunk,
                         "step_px": light.step_px, "refine_steps": light.refine_steps},
        "scope": "Fresh actual-loader raw height/TX reads -> uncached LOS -> uncached light diffuse -> input normalization/H2D -> checkpoint's distance hook and both U-Nets -> dB numpy D2H; per-stage CUDA synchronization included.",
        "limitations": [
            ("New B input weights are zero for a pretraining architecture-cost check; no trained diffuse quality or formal challenge compliance is claimed."
             if args.initialize_diffuse else
             "Checkpoint consumes B in both U-Nets; this is a local runtime measurement, not formal challenge compliance."
             if trained_diffuse else
             "Legacy checkpoint does not consume B; this is a current-network computation budget only, not new-model quality or formal challenge compliance."),
            "Selected input rows and repeats only; P95 is a descriptive linear percentile, not a robust tail guarantee.",
            "Rows may share terrain; no feature results are reused, but OS file cache and CUDA allocator are warm.",
            "Stage synchronization can change overlap and timings; total includes synchronization overhead.",
            "Static initialization is reported separately; organizer hardware and exact timing policy remain unverified.",
        ],
        "warmup": warmup, "rows": rows, "summary": summary,
        "over_500ms_including_read": sum(row["total_ms_including_input_read"] > 500 for row in rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "static_init_ms": static_init_ms,
                      "light_config": result["light_config"],
                      "torch_num_threads": result["torch_num_threads"],
                      "torch_num_interop_threads": result["torch_num_interop_threads"],
                      "over_500ms_including_read": result["over_500ms_including_read"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt", required=True,
                        help="band58 secondU checkpoint: trained diffuse model or legacy LOS-distance cost surrogate")
    parser.add_argument("--initialize-diffuse", action="store_true",
                        help="expand legacy LOS-distance inputs with zero B weights for pretraining timing only; no checkpoint written")
    parser.add_argument("--out", required=True, help="new JSON path; existing files are never overwritten")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--rx-grid", type=int, default=16)
    parser.add_argument("--n-mu", type=int, default=4)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--ray-chunk", type=int, default=8192)
    main(parser.parse_args())
