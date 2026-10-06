"""Input-only terrain clearance and its memory-mapped dataset adapter.

The scalar is an approximate LOS representation, not a simulator LOS label.
Targets and validity masks are never used to construct it.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def los_feature_config(geometry):
    return {**geometry, "los": {
        "version": 1, "quantity": "minimum_vertical_clearance_m",
        "interpolation": "bilinear_align_corners_border",
        "intervals": "2*max(1,ceil(horizontal_distance_px))",
        "include_endpoints": True, "computation_dtype": "float32",
        "encoding": "asinh(C/scale_m)/asinh(height_range_m/scale_m)",
        "scale_m": 1.0, "second_channels": ["F", 0.0, 0.0]}}


@torch.no_grad()
def clearance_feature(height_m, tx, config, device, chunk=4096):
    """Compute the reviewed FP32, bilinear, <= half-pixel ray-sampling proxy.

    Time O(HW L), working memory O(chunk L + HW), where L is the maximum
    number of points on a ray. Normalization uses only training metadata.
    """
    if chunk <= 0:
        raise ValueError("--chunk must be positive")
    h, w = height_m.shape
    rows, cols = np.indices((h, w), dtype=np.float64)
    ends = np.stack((rows.ravel(), cols.ravel()), axis=1)
    distance_px = np.linalg.norm(ends - np.asarray(tx), axis=1)
    intervals = 2 * np.maximum(1, np.ceil(distance_px).astype(np.int64))
    surface = torch.as_tensor(height_m, dtype=torch.float32, device=device)[None, None]
    endpoints = torch.as_tensor(ends, dtype=torch.float32, device=device)
    steps = torch.as_tensor(intervals, device=device)
    z_tx = surface[0, 0, tx[0], tx[1]] + config["tx_height_m_agl"]
    output = []
    for start in range(0, h * w, chunk):
        end = endpoints[start:start + chunk]
        n = steps[start:start + chunk]
        k = torch.arange(int(n.max()) + 1, device=device)[None, :]
        alpha = (k / n[:, None]).clamp_max(1)
        y = tx[0] + alpha * (end[:, 0, None] - tx[0])
        x = tx[1] + alpha * (end[:, 1, None] - tx[1])
        grid = torch.stack((2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1), dim=-1)[None]
        terrain = F.grid_sample(surface, grid, mode="bilinear", padding_mode="border",
                                align_corners=True)[0, 0]
        z_rx = surface[0, 0, end[:, 0].long(), end[:, 1].long()] + config["rx_height_m_agl"]
        ray = z_tx + alpha * (z_rx[:, None] - z_tx)
        clearance = (ray - terrain).masked_fill(k > n[:, None], float("inf"))
        output.append(clearance.min(1).values)
    clearance = torch.cat(output).reshape(h, w)
    lo, hi = config["heightmap_range_m"]
    scale = config["los"]["scale_m"]
    return (torch.asinh(clearance / scale) / math.asinh((hi - lo) / scale)).cpu().numpy()


def read_cache(cache_root, dataset, config):
    root = Path(cache_root)
    manifest = json.loads((root / f"{dataset.split}.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"LOS cache is incomplete for {dataset.split}; use a new cache directory")
    if manifest.get("feature_config") != config or manifest.get("rows") != dataset.rows:
        raise ValueError(f"LOS cache metadata/index differs for {dataset.split}")
    path = root / f"{dataset.split}.npy"
    array = np.load(path, mmap_mode="r")
    expected = (len(dataset.rows), *config["grid"])
    if array.shape != expected or array.dtype != np.float32:
        raise ValueError(f"LOS cache requires float32 {expected}, got {array.dtype} {array.shape}")
    return path


class LOSDataset(Dataset):
    """Append cached F, then apply the loader's original joint augmentation.

    Cache indexing follows rows, so both bands can reuse a terrain+TX feature.
    Reads take O(HW) time/memory per sample; the whole cache stays memory-mapped.
    """

    def __init__(self, base, cache_root, config, augment=False, augmentation=None):
        if base.augment or not base.normalize:
            raise ValueError("LOS adapter needs an unaugmented, normalized base dataset")
        if augment and augmentation is None:
            raise ValueError("LOS augmentation requires the actual loader's augmentation function")
        self.base, self.augment, self.augmentation = base, augment, augmentation
        self.meta, self.rows, self.samples = base.meta, base.rows, base.samples
        self.pl_min, self.pl_max, self.split = base.pl_min, base.pl_max, base.split
        self.path = read_cache(cache_root, base, config)
        self._features = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        return state

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        items = list(item) if isinstance(item, tuple) else [item]
        if self._features is None:
            self._features = np.load(self.path, mmap_mode="r")
        row, _ = self.samples[index]
        feature = torch.from_numpy(np.array(self._features[row], copy=True)[None])
        x = torch.cat((items[0], feature), dim=0)
        if self.augment:
            positions = [i for i in range(1, len(items)) if isinstance(items[i], torch.Tensor)]
            channels, extra = self.augmentation(
                [v.numpy() for v in x], [items[i][0].numpy() for i in positions])
            x = torch.from_numpy(np.stack(channels))
            for i, value in zip(positions, extra):
                items[i] = torch.from_numpy(value).unsqueeze(0)
        items[0] = x
        return tuple(items) if isinstance(item, tuple) else x
