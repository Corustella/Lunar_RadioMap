"""Build input-only single-bounce diffuse-reflection feature caches.

This is an independent physical proxy cache; it does not alter model inputs or
training. A limited-row cache is explicitly partial and needs its own directory.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiment_provenance import collect_provenance
from train import load_dataset_module


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def feature_config(meta, args):
    if args.method == "light":
        from diffuse_light import light_diffuse_config

        return light_diffuse_config(meta, band=args.band, n_mu=args.n_mu,
                                    n_phi=args.n_phi, rx_grid=args.rx_grid)
    if args.method != "reference":
        raise ValueError(f"Unsupported diffuse method: {args.method!r}")
    from diffuse_features import diffuse_feature_config

    return diffuse_feature_config(meta, args.band, n_mu=args.n_mu, n_phi=args.n_phi)


def selected_dataset(dataset, limit_rows):
    """Select an index prefix without fetching a sample, label or mask."""
    source_rows = len(dataset.rows)
    if limit_rows is not None and limit_rows > source_rows:
        raise ValueError(f"--limit-rows {limit_rows} exceeds {dataset.split}'s {source_rows} rows")
    count = source_rows if limit_rows is None else limit_rows
    selected = copy.copy(dataset)
    selected.rows = dataset.rows[:count]
    selected.samples = [(i, band) for i, band in dataset.samples if i < count]
    return selected, source_rows, list(range(count))


def existing_cache(root, dataset, config):
    from diffuse_features import read_diffuse_cache

    path = root / f"{dataset.split}.npy"
    manifest_path = root / f"{dataset.split}.json"
    if manifest_path.exists():
        read_diffuse_cache(root, dataset, config)
        return True
    if path.exists():
        raise FileExistsError(f"Cache has no completion record: {path}; use a new cache directory")
    return False


def build_split(dataset, cache_root, config, device, loader_path, provenance,
                source_rows, row_indices, chunk, log_every):
    from diffuse_features import diffuse_feature

    root = Path(cache_root)
    path = root / f"{dataset.split}.npy"
    manifest_path = root / f"{dataset.split}.json"
    if existing_cache(root, dataset, config):
        print(f"reuse complete diffuse cache: {path}", flush=True)
        return

    manifest = {
        "status": "running", "split": dataset.split, "band": dataset.band,
        "is_partial": len(dataset.rows) != source_rows,
        "source_rows": source_rows, "row_indices": row_indices,
        "data_root": str(Path(dataset.root).resolve()), "actual_loader": loader_path,
        "data_metadata": dataset.meta,
        "feature_config": config, "rows": dataset.rows,
        "shape": [len(dataset.rows), *config["grid"]], "dtype": "float32",
        "provenance": provenance, "device": str(device),
        "chunk": chunk, "chunk_unit": "rays", "completed_rows": 0,
        "input_height": "raw_metres_without_normalization_or_clipping",
        "row_seconds": [],
        "method": config.get("method", "reference"),
    }

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    start = time.monotonic()
    array = None
    try:
        light = None
        method = config.get("method", "reference")
        if method not in ("reference", "dual_center_light_v1"):
            raise ValueError(f"Unsupported diffuse construction method: {method!r}")
        if method == "dual_center_light_v1":
            from diffuse_light import LightDiffuse

            light = LightDiffuse(config, device=device, ray_chunk=chunk)
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                          shape=tuple(manifest["shape"]))
        for i in range(len(dataset.rows)):
            row_start = time.monotonic()
            # Do not call __getitem__: train/val samples also load targets and
            # masks. These private readers are the actual loader's input API.
            height_m = dataset._heightmap(i)
            tx_map = dataset._tx(i)
            locations = np.argwhere(tx_map == 1)
            if locations.shape != (1, 2) or np.count_nonzero(tx_map) != 1:
                raise ValueError(f"Expected one-hot TX at {dataset.split} row {i}")
            tx = tuple(map(int, locations[0]))
            if light is None:
                feature = np.asarray(diffuse_feature(height_m, tx, config, device, chunk=chunk))
            else:
                feature = light.feature_numpy(height_m, tx)
            if feature.shape != tuple(config["grid"]) or feature.dtype != np.float32:
                raise ValueError(f"Diffuse feature must be float32 {tuple(config['grid'])}; "
                                 f"got {feature.dtype} {feature.shape}")
            if not np.isfinite(feature).all():
                raise ValueError(f"Non-finite diffuse feature at {dataset.split} row {i}")
            array[i] = feature
            manifest["completed_rows"] = i + 1
            manifest["row_seconds"].append(time.monotonic() - row_start)
            if log_every and ((i + 1) % log_every == 0 or i + 1 == len(dataset.rows)):
                print(f"  diffuse {dataset.split}: {i + 1}/{len(dataset.rows)} "
                      f"row {manifest['row_seconds'][-1]:.2f}s; "
                      f"elapsed {time.monotonic() - start:.1f}s", flush=True)
        array.flush()
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"], manifest["error"] = "failed", repr(exc)
        raise
    finally:
        manifest["seconds"] = time.monotonic() - start
        save()
        del array
    print(f"wrote {path} ({manifest['seconds']:.1f}s including input reads and writes)", flush=True)


def main(args):
    if args.n_phi % 4:
        raise ValueError("--n-phi must be a multiple of 4")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("--splits must not contain duplicates")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for diffuse cache construction")
    args.device = str(device)
    if args.chunk is None:
        args.chunk = 8192 if args.method == "light" else 65536
    ld = load_dataset_module(args.data_root)
    datasets = [ld.LunarRadioMapDataset(args.data_root, split=split, band=args.band,
                                       normalize=False, augment=False, return_mask=False)
                for split in args.splits]
    config = feature_config(datasets[0].meta, args)
    selections = [selected_dataset(dataset, args.limit_rows) for dataset in datasets]
    root = Path(args.cache_root).resolve()
    needed = 0
    plans = []
    for dataset, source_rows, row_indices in selections:
        if feature_config(dataset.meta, args) != config:
            raise ValueError("Diffuse configuration differs between selected splits")
        reuse = existing_cache(root, dataset, config)
        if not reuse:
            needed += len(dataset.rows) * int(np.prod(config["grid"])) * 4 + 4096
        plans.append({"split": dataset.split, "rows": len(dataset.rows),
                      "source_rows": source_rows, "is_partial": len(dataset.rows) != source_rows,
                      "reuse": reuse})
    # Disk inspection also works for dry runs without creating the directory.
    existing_parent = root
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    free = shutil.disk_usage(existing_parent).free
    print(json.dumps({"cache_root": str(root), "actual_loader": str(Path(ld.__file__).resolve()),
                      "feature_config": config, "splits": plans,
                      "new_array_bytes_approx": needed, "free_bytes": free,
                      "dry_run": args.dry_run}, indent=2), flush=True)
    if needed > free:
        raise OSError(f"Diffuse arrays require about {needed / 1e9:.2f} GB; "
                      f"{free / 1e9:.2f} GB free (manifest space is additional)")
    if args.dry_run:
        return
    provenance = collect_provenance(args.data_root, ld, {}, args)
    root.mkdir(parents=True, exist_ok=True)
    for dataset, source_rows, row_indices in selections:
        build_split(dataset, root, config, device, str(Path(ld.__file__).resolve()), provenance,
                    source_rows, row_indices, args.chunk, args.log_every)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True,
                        help="new directory, or a complete cache matching all selected rows and parameters")
    parser.add_argument("--band", choices=["58", "415"], default="58")
    parser.add_argument("--method", choices=["reference", "light"], default="reference",
                        help="exact bilinear-DDA reference or coarse dual-center input proxy")
    parser.add_argument("--rx-grid", type=positive_int, default=16,
                        help="light method's receiver grid side; ignored by reference")
    parser.add_argument("--n-mu", type=positive_int, required=True,
                        help="explicit full-sphere cos(theta) quadrature order >= 2; no validated default is assumed")
    parser.add_argument("--n-phi", type=positive_int, required=True,
                        help="explicit azimuth quadrature order; must be a multiple of 4")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument("--limit-rows", type=positive_int, default=None,
                        help="build only the first N index rows per split; use a separate partial-cache directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=positive_int, default=None,
                        help="ray batch size: default 65536 for reference, 8192 for light")
    parser.add_argument("--log-every", type=positive_int, default=1,
                        help="print progress every N completed maps")
    parser.add_argument("--dry-run", action="store_true",
                        help="report configuration and approximate space without creating files")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
