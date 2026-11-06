"""Training targets and masked reduction for T block-mean distillation."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import avg_pool2d


BLOCK_SIZE = 8
BLOCK_WEIGHT = 0.25


def block_means(values):
    return avg_pool2d(values.float(), BLOCK_SIZE)


def full_blocks(mask):
    return block_means(mask) == 1


def block_teacher_loss(pred, teacher, mask):
    valid = full_blocks(mask)
    residual = torch.where(valid, block_means(pred) - teacher, 0.0)
    sse = residual.square().sum()
    count = valid.sum()
    return sse / count.clamp_min(1), sse.detach(), count


def cache_config(dataset, feature_config):
    if (dataset.split != "train" or dataset.band != "58" or not dataset.normalize
            or not dataset.return_mask or not dataset.has_targets):
        raise ValueError("Block teacher cache requires normalized band 58 train data with official masks")
    return {
        "version": 1,
        "split": "train",
        "band": "58",
        "block_size": BLOCK_SIZE,
        "quantity": "mean_normalized_pathloss_full_valid_blocks",
        "pathloss_range_db": [float(dataset.pl_min), float(dataset.pl_max)],
        "feature_config": feature_config,
    }


def cache_shape(dataset, config):
    height, width = config["feature_config"]["grid"]
    return len(dataset.rows), height // BLOCK_SIZE, width // BLOCK_SIZE


def read_teacher_cache(root, dataset, feature_config):
    root = Path(root)
    manifest_path = root / "train.json"
    if not manifest_path.is_file():
        raise ValueError(f"Block teacher cache lacks its completion manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("cache_id"):
        raise ValueError(f"Incomplete block teacher cache: {root}")
    config = cache_config(dataset, feature_config)
    if manifest.get("config") != config or manifest.get("rows") != dataset.rows:
        raise ValueError("Block teacher cache configuration or train index order differs")
    path = root / "train.npy"
    array = np.load(path, mmap_mode="r")
    expected = cache_shape(dataset, config)
    if array.shape != expected or array.dtype != np.float32:
        raise ValueError(f"Block teacher cache requires float32 {expected}, got {array.dtype} {array.shape}")
    objective = {"cache_id": manifest["cache_id"], "block_size": BLOCK_SIZE,
                 "weight": BLOCK_WEIGHT}
    return path, objective
