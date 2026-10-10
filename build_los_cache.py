"""Build LOS or Fresnel-profile caches once, using only height and TX inputs."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from los_features import clearance_feature, fresnel_feature, read_cache
from radiounet import cache_config_for_feature
from train import load_dataset_module


def build_split(dataset, cache_root, config, device, loader_path, chunk, log_every):
    root = Path(cache_root)
    path, manifest_path = root / f"{dataset.split}.npy", root / f"{dataset.split}.json"
    feature = "fresnel" if "fresnel" in config else "los"
    compute = fresnel_feature if feature == "fresnel" else clearance_feature
    if manifest_path.exists():
        read_cache(root, dataset, config)
        print(f"reuse complete {feature} cache: {path}", flush=True)
        return
    if path.exists():
        raise FileExistsError(f"Cache has no completion record: {path}; use a new cache directory")
    manifest = {"status": "running", "split": dataset.split,
                "data_root": str(Path(dataset.root).resolve()), "actual_loader": loader_path,
                "feature_config": config, "rows": dataset.rows,
                "shape": [len(dataset.rows), *config["grid"]], "dtype": "float32",
                "device": str(device), "torch": torch.__version__, "chunk": chunk,
                "completed_rows": 0}

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    start = time.monotonic()
    array = None
    try:
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                          shape=tuple(manifest["shape"]))
        lo, hi = config["heightmap_range_m"]
        for i in range(len(dataset.rows)):
            # Use the actual loader's input normalization, without requesting
            # a sample (which would also read labels and masks).
            height = dataset._norm(dataset._heightmap(i), lo, hi)
            height_m = height.astype(np.float64) * (hi - lo) + lo
            tx_map = dataset._tx(i)
            locations = np.argwhere(tx_map == 1)
            if locations.shape != (1, 2) or np.count_nonzero(tx_map) != 1:
                raise ValueError(f"Expected one-hot TX at {dataset.split} row {i}")
            tx = tuple(map(int, locations[0]))
            array[i] = compute(height_m, tx, config, device, chunk)
            manifest["completed_rows"] = i + 1
            if log_every and ((i + 1) % log_every == 0 or i + 1 == len(dataset.rows)):
                print(f"  cache {dataset.split}: {i + 1}/{len(dataset.rows)} "
                      f"({time.monotonic() - start:.1f}s)", flush=True)
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
    if args.chunk <= 0:
        raise ValueError("--chunk must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for feature cache construction")
    ld = load_dataset_module(args.data_root)
    datasets = [ld.LunarRadioMapDataset(args.data_root, split=split, band="58", augment=False)
                for split in args.splits]
    config = cache_config_for_feature(datasets[0].meta, args.feature)
    root = Path(args.cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    needed = 0
    for dataset in datasets:
        if (root / f"{dataset.split}.json").exists():
            read_cache(root, dataset, config)
        elif (root / f"{dataset.split}.npy").exists():
            raise FileExistsError(f"Incomplete {dataset.split} cache; use a new directory")
        else:
            needed += len(dataset.rows) * int(np.prod(config["grid"])) * 4 + 1024
    free = shutil.disk_usage(root).free
    if needed > free:
        raise OSError(f"Feature cache requires about {needed / 1e9:.2f} GB; {free / 1e9:.2f} GB free")
    print(f"{args.feature} cache: {root}; new arrays {needed / 1e9:.2f} GB; loader {ld.__file__}", flush=True)
    for dataset in datasets:
        build_split(dataset, root, config, device, str(Path(ld.__file__).resolve()),
                    args.chunk, args.log_every)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True,
                        help="new directory, or a complete cache with matching metadata/index")
    parser.add_argument("--feature", choices=["los", "fresnel"], default="los",
                        help="fresnel builds only g_F; reuse the existing LOS cache separately")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=500)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
