"""Build train/val LOS clearance caches once, using only height and TX inputs."""

import argparse
from functools import partial
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from los_features import clearance_feature, read_cache
from radiounet import feature_config_for_mode
from train import load_dataset_module


def build_split(dataset, cache_root, config, device, loader_path, chunk, log_every,
                feature_set="los", geometry_threads=8, provenance=None, mechanism_mode="ds"):
    cache_reader = read_cache
    if feature_set == "mechanisms":
        from mechanism_features import construct_features, read_mechanism_cache
        cache_reader = partial(read_mechanism_cache,required_mode=mechanism_mode)
    root = Path(cache_root)
    path, manifest_path = root / f"{dataset.split}.npy", root / f"{dataset.split}.json"
    if manifest_path.exists():
        cache_reader(root, dataset, config)
        print(f"reuse complete {feature_set} cache: {path}", flush=True)
        return
    if path.exists():
        raise FileExistsError(f"Cache has no completion record: {path}; use a new cache directory")
    manifest = {"status": "running", "split": dataset.split,
                "data_root": str(Path(dataset.root).resolve()), "actual_loader": loader_path,
                "feature_config": config, "rows": dataset.rows,
                "shape": [len(dataset.rows), *([4] if feature_set == "mechanisms" else []),
                          *config["grid"]], "dtype": "float32",
                "device": str(device), "torch": torch.__version__, "chunk": chunk,
                "completed_rows": 0}
    if feature_set == "mechanisms":
        from importlib.metadata import version
        manifest.update(feature_set=feature_set,computed_mode=mechanism_mode,geometry_threads=geometry_threads,
                        numba=version("numba"),provenance=provenance,
                        valid_paths=0,max_candidates=0)

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
            if feature_set == "mechanisms":
                features, stats = construct_features(height_m,tx,config,device,
                                                      threads=geometry_threads,mode=mechanism_mode)
                array[i] = features
                manifest["valid_paths"] += stats["valid_paths"]
                manifest["max_candidates"] = max(manifest["max_candidates"],stats["point_candidates"])
            else:
                array[i] = clearance_feature(height_m, tx, config, device, chunk)
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
    if args.chunk <= 0 or args.geometry_threads <= 0:
        raise ValueError("--chunk and --geometry-threads must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for LOS cache construction")
    ld = load_dataset_module(args.data_root)
    datasets = [ld.LunarRadioMapDataset(args.data_root, split=split, band="58", augment=False)
                for split in args.splits]
    config = feature_config_for_mode(datasets[0].meta, "los")
    cache_reader, channels, provenance = read_cache, 1, None
    if args.feature_set == "mechanisms":
        from mechanism_features import mechanism_cache_config, read_mechanism_cache
        from experiment_provenance import collect_provenance
        config = mechanism_cache_config(datasets[0].meta)
        cache_reader,channels = partial(read_mechanism_cache,required_mode=args.mechanism_mode),4
        provenance = collect_provenance(args.data_root,ld,{},args)
    root = Path(args.cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    needed = 0
    for dataset in datasets:
        if (root / f"{dataset.split}.json").exists():
            cache_reader(root, dataset, config)
        elif (root / f"{dataset.split}.npy").exists():
            raise FileExistsError(f"Incomplete {dataset.split} cache; use a new directory")
        else:
            needed += len(dataset.rows) * int(np.prod(config["grid"])) * 4 * channels + 1024
    free = shutil.disk_usage(root).free
    if needed > free:
        raise OSError(f"Cache requires about {needed / 1e9:.2f} GB; {free / 1e9:.2f} GB free")
    print(f"{args.feature_set} cache: {root}; new arrays {needed / 1e9:.2f} GB; loader {ld.__file__}", flush=True)
    for dataset in datasets:
        build_split(dataset, root, config, device, str(Path(ld.__file__).resolve()),
                    args.chunk,args.log_every,args.feature_set,args.geometry_threads,provenance,args.mechanism_mode)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True,
                        help="new directory, or a complete cache with matching metadata/index")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--feature-set", choices=["los", "mechanisms"], default="los",
                        help="mechanisms builds all four D/S channels in a separate cache directory")
    parser.add_argument("--geometry-threads", type=int, default=8)
    parser.add_argument("--mechanism-mode", choices=["d", "s", "ds"], default="ds",
                        help="enabled cached channels; absent slots stay zero and cannot be loaded by a broader mode")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
