"""Input-only clearance, distance, Fresnel profile and cached dataset adapter.

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


def log_distance_config(geometry):
    """Fixed distance encoding bounds from training metadata, in metres."""
    h, w = geometry["grid"]
    resolution = geometry["resolution_m_per_px"]
    lo, hi = geometry["heightmap_range_m"]
    max_vertical = hi - lo + abs(geometry["rx_height_m_agl"] - geometry["tx_height_m_agl"])
    maximum = math.sqrt(((h - 1) * resolution) ** 2 +
                        ((w - 1) * resolution) ** 2 + max_vertical ** 2)
    return {"version": 1, "quantity": "tx_rx_grid_3d_distance_m",
            "encoding": "log1p(d/scale_m)/log1p(max_distance_m/scale_m)",
            "scale_m": 1.0, "max_distance_m": maximum,
            "computation_dtype": "float32"}


def fresnel_feature_config(geometry, frequency_hz):
    """Fixed, signed Fresnel-profile encoding for the 5.8 GHz experiment."""
    frequency_hz = float(frequency_hz)
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Fresnel frequency must be finite and positive")
    return {**geometry, "fresnel": {
        "version": 1, "band": "58", "frequency_hz": frequency_hz,
        "speed_of_light_m_s": 299792458.0,
        "quantity": "minimum_profile_perpendicular_clearance_over_first_fresnel_radius",
        "interpolation": "bilinear_align_corners_border",
        "intervals": "2*max(1,ceil(horizontal_distance_px))",
        "include_endpoints": False, "computation_dtype": "float32",
        "clearance": "(z_ray-z_terrain)*horizontal_distance_m/tx_rx_3d_distance_m",
        "radius": "sqrt((speed_of_light_m_s/frequency_hz)*D*alpha*(1-alpha))",
        "ray_position": "alpha*tx_rx_3d_distance_m",
        "encoding": "2/pi*atan(min(clearance/radius))",
        "zero_horizontal_distance_value": 0.0}}


@torch.no_grad()
def log_distance_feature(input, config):
    """Encode distance from normalized height and one-hot TX only, after D4.

    The RX follows the grid terrain plus its metadata antenna height. This is
    a geometric input proxy, not the length of a simulated multipath. Time and
    extra memory are O(BHW); targets, masks and LOS cache values are unused.
    """
    with torch.autocast(input.device.type, enabled=False):
        lo, hi = config["heightmap_range_m"]
        height = input[:, 0].float() * (hi - lo) + lo
        _, h, w = height.shape
        tx_index = input[:, 1].flatten(1).argmax(1)
        tx_row = (tx_index // w).float()[:, None, None]
        tx_col = (tx_index % w).float()[:, None, None]
        z_tx = height.flatten(1).gather(1, tx_index[:, None])[:, :, None]
        z_tx = z_tx + config["tx_height_m_agl"]
        rows = torch.arange(h, dtype=torch.float32, device=input.device)[None, :, None]
        cols = torch.arange(w, dtype=torch.float32, device=input.device)[None, None, :]
        resolution = config["resolution_m_per_px"]
        squared = ((rows - tx_row) * resolution).square() + ((cols - tx_col) * resolution).square()
        squared = squared + (height + config["rx_height_m_agl"] - z_tx).square()
        distance = squared.sqrt()
        encoding = config["distance"]
        feature = torch.log1p(distance / encoding["scale_m"])
        feature = feature / math.log1p(encoding["max_distance_m"] / encoding["scale_m"])
    return feature[:, None].to(dtype=input.dtype)


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


@torch.no_grad()
def fresnel_feature(height_m, tx, config, device, chunk=4096):
    """Sample a signed terrain-profile proxy, not diffraction loss or LOS labels.

    The radius uses the reference ray position alpha*D, rather than projecting
    a terrain vertex and discarding it when its foot falls beyond an endpoint.
    This retains high obstructing terrain on steep profiles. Endpoints are
    excluded; vertical same-pixel paths receive neutral zero. Time O(HW L),
    working memory O(chunk L + HW); only terrain, TX and fixed metadata enter.
    """
    if chunk <= 0:
        raise ValueError("--chunk must be positive")
    h, w = height_m.shape
    rows, cols = np.indices((h, w), dtype=np.float64)
    ends = np.stack((rows.ravel(), cols.ravel()), axis=1)
    distance_px = np.linalg.norm(ends - np.asarray(tx), axis=1)
    intervals = 2 * np.maximum(1, np.ceil(distance_px).astype(np.int64))
    with torch.autocast(torch.device(device).type, enabled=False):
        surface = torch.as_tensor(height_m, dtype=torch.float32, device=device)[None, None]
        endpoints = torch.as_tensor(ends, dtype=torch.float32, device=device)
        steps = torch.as_tensor(intervals, device=device)
        z_tx = surface[0, 0, tx[0], tx[1]] + config["tx_height_m_agl"]
        encoding = config["fresnel"]
        wavelength = encoding["speed_of_light_m_s"] / encoding["frequency_hz"]
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
            horizontal = (end - end.new_tensor(tx)).square().sum(1).sqrt()
            horizontal = horizontal * config["resolution_m_per_px"]
            distance = (horizontal.square() + (z_rx - z_tx).square()).sqrt()
            # The same-pixel ray has no horizontal profile. Substitute safe
            # denominators before marking its result neutral below.
            safe_distance = torch.where(horizontal > 0, distance, torch.ones_like(distance))
            ray = z_tx + alpha * (z_rx[:, None] - z_tx)
            clearance = (ray - terrain) * (horizontal / safe_distance)[:, None]
            interior = (k > 0) & (k < n[:, None]) & (horizontal[:, None] > 0)
            radius = (wavelength * distance[:, None] * alpha * (1 - alpha)).sqrt()
            radius = torch.where(interior, radius, torch.ones_like(radius))
            ratio = (clearance / radius).masked_fill(~interior, float("inf"))
            minimum = ratio.min(1).values
            minimum = torch.where(horizontal > 0, minimum, torch.zeros_like(minimum))
            output.append((2 / math.pi) * torch.atan(minimum))
    return torch.cat(output).reshape(h, w).cpu().numpy()


def read_cache(cache_root, dataset, config):
    root = Path(cache_root)
    manifest = json.loads((root / f"{dataset.split}.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Feature cache is incomplete for {dataset.split}; use a new cache directory")
    if manifest.get("feature_config") != config or manifest.get("rows") != dataset.rows:
        raise ValueError(f"Feature cache metadata/index differs for {dataset.split}")
    path = root / f"{dataset.split}.npy"
    array = np.load(path, mmap_mode="r")
    expected = (len(dataset.rows), *config["grid"])
    if array.shape != expected or array.dtype != np.float32:
        raise ValueError(f"Feature cache requires float32 {expected}, got {array.dtype} {array.shape}")
    return path


class LOSDataset(Dataset):
    """Append cached F and optional g_F before the original joint augmentation.

    Cache indexing follows rows, so both bands can reuse a terrain+TX feature.
    Reads take O(HW) time/memory per sample; the whole cache stays memory-mapped.
    """

    def __init__(self, base, cache_root, config, augment=False, augmentation=None,
                 fresnel_cache=None, fresnel_config=None):
        if base.augment or not base.normalize:
            raise ValueError("LOS adapter needs an unaugmented, normalized base dataset")
        if augment and augmentation is None:
            raise ValueError("LOS augmentation requires the actual loader's augmentation function")
        self.base, self.augment, self.augmentation = base, augment, augmentation
        self.meta, self.rows, self.samples = base.meta, base.rows, base.samples
        self.pl_min, self.pl_max, self.split = base.pl_min, base.pl_max, base.split
        self.path = read_cache(cache_root, base, config)
        self._features = None
        if (fresnel_cache is None) != (fresnel_config is None):
            raise ValueError("Fresnel cache and configuration must be supplied together")
        self.fresnel_path = (read_cache(fresnel_cache, base, fresnel_config)
                             if fresnel_cache is not None else None)
        self._fresnel_features = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        state["_fresnel_features"] = None
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
        if self.fresnel_path is not None:
            if self._fresnel_features is None:
                self._fresnel_features = np.load(self.fresnel_path, mmap_mode="r")
            fresnel = torch.from_numpy(np.array(self._fresnel_features[row], copy=True)[None])
            x = torch.cat((x, fresnel), dim=0)
        if self.augment:
            positions = [i for i in range(1, len(items)) if isinstance(items[i], torch.Tensor)]
            channels, extra = self.augmentation(
                [v.numpy() for v in x], [items[i][0].numpy() for i in positions])
            x = torch.from_numpy(np.stack(channels))
            for i, value in zip(positions, extra):
                items[i] = torch.from_numpy(value).unsqueeze(0)
        items[0] = x
        return tuple(items) if isinstance(item, tuple) else x
