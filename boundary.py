"""Mask-safe gradient supervision and streaming boundary diagnostics.

Thresholds are diagnostics only. Losses operate on normalized targets; reports
operate in dB. No edge/target/mask information is needed at model inference.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THRESHOLDS_DB = (5, 10, 20)
RADII_PX = (1, 3, 5)
THRESHOLD_ATOL_DB = 1e-4  # FP32 normalization/denormalization round-off only


def masked_gradient_loss(pred, target, mask):
    """Mean absolute signed-gradient error over valid horizontal/vertical pairs.

    FP32, O(BHW) time and space. Empty pair sets produce differentiable zero.
    """
    valid = mask > 0
    p = torch.where(valid, pred.float(), 0.0)
    y = torch.where(valid, target.float(), 0.0)
    mx = valid[..., :, 1:] & valid[..., :, :-1]
    my = valid[..., 1:, :] & valid[..., :-1, :]
    dx = (p[..., :, 1:] - p[..., :, :-1]) - (y[..., :, 1:] - y[..., :, :-1])
    dy = (p[..., 1:, :] - p[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])
    numerator = (dx.abs() * mx).sum() + (dy.abs() * my).sum()
    denominator = mx.sum() + my.sum()
    return numerator / denominator.clamp_min(1)


def dilate(points, radius):
    """Square (Chebyshev) neighborhood; outside-image pixels are not missing."""
    return F.max_pool2d(points.float(), 2 * radius + 1, stride=1,
                        padding=radius) > 0


def high_change_points(target_db, mask, threshold=10):
    valid = mask > 0
    dx = (target_db[..., :, 1:] - target_db[..., :, :-1]).abs()
    dy = (target_db[..., 1:, :] - target_db[..., :-1, :]).abs()
    ex = (dx >= threshold - THRESHOLD_ATOL_DB) & valid[..., :, 1:] & valid[..., :, :-1]
    ey = (dy >= threshold - THRESHOLD_ATOL_DB) & valid[..., 1:, :] & valid[..., :-1, :]
    points = torch.zeros_like(valid)
    points[..., :, :-1] |= ex
    points[..., :, 1:] |= ex
    points[..., :-1, :] |= ey
    points[..., 1:, :] |= ey
    return points


def boundary_regions(target_db, mask):
    valid = mask > 0
    near = dilate(~valid, 5) & valid
    regions = {"all_valid": valid, "near_missing_r5": near,
               "away_missing_r5": valid & ~near}
    for threshold in THRESHOLDS_DB:
        points = high_change_points(target_db, mask, threshold)
        for radius in RADII_PX:
            edge = dilate(points, radius) & valid
            regions[f"edge_t{threshold}_r{radius}"] = edge
            regions[f"nonedge_t{threshold}_r{radius}"] = valid & ~edge
            if threshold == 10 and radius == 3:
                for name, region in (("edge", edge), ("nonedge", valid & ~edge)):
                    regions[f"{name}_t10_r3_near_missing"] = region & near
                    regions[f"{name}_t10_r3_away_missing"] = region & ~near
    return regions


def stats_record(count, sse, error_sum):
    count = int(count)
    return {"valid_pixels": count, "sse_db2": float(sse),
            "signed_error_sum_db": float(error_sum),
            "rmse_db": float(np.sqrt(sse / count)) if count else None,
            "bias_db": float(error_sum / count) if count else None}


class BoundaryReport:
    """Write per-sample rows incrementally; retain only aggregate sums."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.totals = {}
        self.samples = 0
        self.csv_path = self.directory / "per_sample.csv"
        self.fields = ["sample_id", "band", "output", "region", *stats_record(0, 0, 0)]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writeheader()
        (self.directory / "status.json").write_text('{"status": "incomplete"}\n', encoding="utf-8")

    @torch.no_grad()
    def update(self, predictions, target, mask, names, bands, scale):
        valid = mask > 0
        if not torch.isfinite(target[valid]).all():
            raise ValueError("Non-finite target at valid pixels")
        regions = boundary_regions(target.double() * scale, mask)
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            for output, prediction in predictions.items():
                if not torch.isfinite(prediction[valid]).all():
                    raise ValueError("Non-finite prediction at valid pixels")
                # Same FP32 subtraction as the official accumulator, FP64 sums.
                error = torch.where(valid, prediction.float() - target.float(), 0.0).double() * scale
                squared = error.square()
                values = torch.stack([
                    torch.stack((r.sum((1, 2, 3)),
                                 (squared * r).sum((1, 2, 3)),
                                 (error * r).sum((1, 2, 3))), dim=1)
                    for r in regions.values()
                ]).cpu().numpy()
                for region, batch_values in zip(regions, values):
                    for name, band, sums in zip(names, bands, batch_values):
                        key = (band, output, region)
                        self.totals.setdefault(key, np.zeros(3, dtype=np.float64))
                        self.totals[key] += sums
                        writer.writerow({"sample_id": name.rsplit("_", 1)[0],
                                         "band": band, "output": output, "region": region,
                                         **stats_record(*sums)})
        self.samples += len(names)

    def finish(self, provenance, evaluation, examples):
        rows = []
        for (band, output, region), sums in sorted(self.totals.items()):
            total = self.totals[(band, output, "all_valid")]
            rows.append({"band": band, "output": output, "region": region,
                         **stats_record(*sums),
                         "valid_pixel_share": float(sums[0] / total[0]) if total[0] else None,
                         "sse_share": float(sums[1] / total[1]) if total[1] else None})
        payload = {
            "samples": self.samples, "primary_threshold_db_per_pixel": 10,
            "thresholds_db_per_pixel": THRESHOLDS_DB, "radii_px": RADII_PX,
            "threshold_atol_db": THRESHOLD_ATOL_DB,
            "definitions": "Valid pair endpoints, square dilation intersected with validity mask. Each edge/nonedge pair partitions all valid pixels. Different thresholds/radii overlap and MUST NOT be added together. Near-missing means Chebyshev distance <=5 to an in-image invalid pixel; image exterior is not missing.",
            "empty_regions": "RMSE, bias and shares with zero denominators are null, not zero.",
            "precision": "FP32 prediction-target subtraction, FP64 regional accumulation; tiny differences from the standard FP32 accumulator are expected.",
            "regions": rows, "evaluation": evaluation, "examples": examples,
            "provenance": provenance,
        }
        (self.directory / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        self.write_markdown(payload)
        (self.directory / "status.json").write_text('{"status": "complete"}\n', encoding="utf-8")
        return payload

    def write_markdown(self, payload):
        selected = {"all_valid", "edge_t10_r1", "edge_t10_r3", "edge_t10_r5",
                    "nonedge_t10_r3", "near_missing_r5", "away_missing_r5"}
        def number(value, percent=False):
            return "undefined" if value is None else f"{value*100:.2f}%" if percent else f"{value:.5f}"
        lines = ["# Boundary error diagnostic", "",
                 f"Processed samples: {payload['samples']}. Prediction and plots share the scoring forward pass.", "",
                 "Primary threshold: 10 dB/pixel; radii: square dilation from both valid pair endpoints. Error = prediction - target.", "",
                 "| Band | Output | Region | Valid pixels | Pixel share | SSE (dB^2) | SSE share | RMSE (dB) | Bias (dB) |",
                 "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in payload["regions"]:
            if row["region"] in selected:
                lines.append(f"| {row['band']} | {row['output']} | {row['region']} | {row['valid_pixels']} | "
                             f"{number(row['valid_pixel_share'], True)} | {number(row['sse_db2'])} | "
                             f"{number(row['sse_share'], True)} | {number(row['rmse_db'])} | {number(row['bias_db'])} |")
        rows = {(r["band"], r["output"], r["region"]): r for r in payload["regions"]}
        lines += ["", "## secondU minus firstU (paired, same forward pass)", "",
                  "Negative SSE change means improvement. Positive means degradation.", "",
                  "| Band | Region | Delta SSE (dB^2) | Delta RMSE (dB) |", "| --- | --- | ---: | ---: |"]
        for (band, output, region), row in rows.items():
            other = rows.get((band, "firstU", region))
            if output == "secondU" and other is not None and region in selected:
                delta = row["rmse_db"]-other["rmse_db"] if row["valid_pixels"] else None
                lines.append(f"| {band} | {region} | {number(row['sse_db2']-other['sse_db2'])} | {number(delta)} |")
        lines += ["", "## Interpretation limits", "",
                  "Each edge/nonedge pair partitions valid pixels. Different thresholds/radii overlap; do not sum them. Empty-region metrics are undefined. The image exterior is not missing data.", "",
                  "Boundary concentration alone does not distinguish displacement, smoothing and amplitude bias. Inspect the fixed GT-selected profiles. Sharper plots alone do not demonstrate lower overall RMSE. This report does not claim an improvement from gradient supervision.", "",
                  "Full sensitivity checks (5/10/20 dB), near-missing cross-strata, checkpoint/loader/index/metadata hashes and precision: [summary.json](summary.json). Per-sample sums: [per_sample.csv](per_sample.csv). Regional sums use FP64 and can differ slightly from the original FP32 accumulator.", "", "## Fixed examples", ""]
        for example in payload["examples"]:
            lines.append(f"- {example['name']}: [raw normalized arrays]({example['npz']})" +
                         (f", [masked map and profiles]({example['png']})" if "png" in example else "; plot unavailable"))
        (self.directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_profiles(target_db, valid, count=2, half_width=8):
    """GT-only selection: strongest valid pair, then spatial suppression.

    Profiles follow the pair's x/y axis, contain 17 valid points, and are at
    least 17 pixels apart in Chebyshev distance. Ties use axis/row/column order.
    Returns fewer profiles when insufficient valid support exists.
    """
    candidates = []
    height, width = valid.shape
    for axis in ("x", "y"):
        delta = np.diff(target_db, axis=1 if axis == "x" else 0)
        pairs = (valid[:, 1:] & valid[:, :-1]) if axis == "x" else (valid[1:] & valid[:-1])
        for row, col in np.argwhere(pairs & (np.abs(delta) >= 10 - THRESHOLD_ATOL_DB)):
            candidates.append((-float(abs(delta[row, col])), axis, int(row), int(col)))
    selected = []
    for neg_jump, axis, row, col in sorted(candidates):
        if any(max(abs(row - p["row"]), abs(col - p["col"])) <= 16 for p in selected):
            continue
        if axis == "x":
            if col < half_width or col + half_width >= width:
                continue
            support = valid[row, col - half_width:col + half_width + 1]
        else:
            if row < half_width or row + half_width >= height:
                continue
            support = valid[row - half_width:row + half_width + 1, col]
        if support.all():
            selected.append({"axis": axis, "row": row, "col": col,
                             "half_width": half_width, "target_jump_db": -neg_jump})
        if len(selected) == count:
            break
    return selected


def profile_values(array, profile):
    r, c, k = profile["row"], profile["col"], profile["half_width"]
    return array[r, c-k:c+k+1] if profile["axis"] == "x" else array[r-k:r+k+1, c]


def save_boundary_example(directory, sample, lo, scale, err_lim=10):
    """Save the exact cached forward output and its mask-aware diagnostic PNG."""
    directory = Path(directory)
    # Names originate from the dataset index, not filesystem paths.
    name = sample["name"]
    if Path(name).name != name or any(c in name for c in "/\\:"):
        raise ValueError(f"Unsafe sample name: {name!r}")
    valid = sample["mask"][0].numpy() > 0
    target = sample["target"][0].numpy().astype(np.float64) * scale + lo
    pred = sample["prediction"][0].numpy().astype(np.float64) * scale + lo
    base = sample.get("base")
    base_db = base[0].numpy().astype(np.float64) * scale + lo if base is not None else None
    profiles = select_profiles(target, valid)
    arrays = {key: sample[key].numpy() for key in ("input", "target", "prediction", "mask")}
    if base is not None:
        arrays["firstU_prediction"] = base.numpy()
    np.savez_compressed(directory / f"{name}.npz", **arrays,
                        pl_min_db=np.float64(lo), pl_scale_db=np.float64(scale))
    record = {"name": name, "npz": f"{name}.npz", "profiles": profiles,
              "arrays": "input/target/prediction are normalized; mask is binary; dB = normalized*pl_scale_db+pl_min_db"}
    error = (sample["prediction"][0] - sample["target"][0]).numpy().astype(np.float64) * scale
    record.update(stats_record(valid.sum(), np.square(error[valid]).sum(), error[valid].sum()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        record["plot_error"] = str(exc)
        return record

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("0.75")
    ec = plt.get_cmap("coolwarm").copy()
    ec.set_bad("0.75")
    low = min(target[valid].min(), pred[valid].min()) if valid.any() else 0
    high = max(target[valid].max(), pred[valid].max()) if valid.any() else 1
    for ax, value, title, cmap, v0, v1 in (
        (axes[0, 0], target, "Target (dB)", cm, low, high),
        (axes[0, 1], pred, "Prediction (dB)", cm, low, high),
        (axes[0, 2], error, f"Error (dB, colors clipped at +/-{err_lim:g})", ec, -err_lim, err_lim),
        (axes[1, 0], target, "GT boundary >=10 dB/px; profile locations", cm, low, high),
    ):
        im = ax.imshow(np.ma.masked_where(~valid, value), cmap=cmap, vmin=v0, vmax=v1)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=.8)
    points = high_change_points(torch.from_numpy(target)[None, None],
                                torch.from_numpy(valid)[None, None])[0, 0].numpy()
    overlay = np.zeros((*valid.shape, 4))
    overlay[points] = (1, 0, 0, .7)
    axes[1, 0].imshow(overlay)
    for i, ax in enumerate(axes[1, 1:]):
        if i >= len(profiles):
            ax.text(.5, .5, "No additional fully valid profile", ha="center", va="center")
            ax.set_axis_off()
            continue
        p = profiles[i]
        offsets = np.arange(-p["half_width"], p["half_width"] + 1)
        ax.plot(offsets, profile_values(target, p), "k.-", label="Target")
        ax.plot(offsets, profile_values(pred, p), ".-", color="#0072B2", label="Prediction")
        if base_db is not None:
            ax.plot(offsets, profile_values(base_db, p), "--", color="#D55E00", label="firstU")
        ax.set(title=f"{p['axis']} profile at row={p['row']}, col={p['col']}",
               xlabel="Offset (pixels)", ylabel="Path loss (dB)")
        ax.legend(fontsize=8)
        r, c, k = p["row"], p["col"], p["half_width"]
        xx, yy = ([c-k, c+k], [r, r]) if p["axis"] == "x" else ([c, c], [r-k, r+k])
        axes[1, 0].plot(xx, yy, color="cyan", linewidth=1)
        axes[1, 0].text(c, r, str(i+1), color="black", fontsize=9)
    score = f"{record['rmse_db']:.4f} dB" if valid.any() else "undefined (no valid pixels)"
    fig.suptitle(f"{name} | masked RMSE {score} | gray = invalid")
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    record["png"] = path.name
    return record
