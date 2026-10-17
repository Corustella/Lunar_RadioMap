"""Diagnostic figures from recorded metrics, without additional inference.

No smoothing or uncertainty claims: samples can share a terrain. Source CSV/JSON
remains the data artifact; each figure has a small JSON describing its encoding.
Training curves cost O(epochs), sample plots O(N log N) time and O(N) memory.
"""
import csv
import json
import os

import numpy as np


STYLE = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.alpha": 0.18, "figure.facecolor": "white",
         "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42}
BLUE, ORANGE = "#0072B2", "#D55E00"


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as error:
        print(f"(skipping diagnostic figures: {error})")
        return None


def _save(fig, path, record, plt):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stem = os.path.splitext(path)[0]
    fig.savefig(path, dpi=180)
    fig.savefig(stem + ".pdf")
    record.update({"png": os.path.abspath(path), "pdf": os.path.abspath(stem + ".pdf"),
                   "dpi": 180, "size_inches": fig.get_size_inches().tolist(),
                   "numpy": np.__version__, "matplotlib": plt.matplotlib.__version__})
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
    plt.close(fig)
    return path


def save_training_curves(history_path, path):
    plt = _pyplot()
    if plt is None:
        return None
    with open(history_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    def values(key):
        return np.array([float(r[key]) if r.get(key) else np.nan for r in rows])
    epoch = values("epoch")
    loss, val, online, lr = [values(k) for k in
                             ("train_loss", "val/rmse_db_masked",
                              "train/rmse_db_masked_online", "lr")]
    finite = np.flatnonzero(np.isfinite(val))
    best = int(finite[np.argmin(val[finite])]) if finite.size else None
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
        axes[0].plot(epoch, loss, color=BLUE, linewidth=1.5)
        axes[0].set(title="Training objective", ylabel="Normalized batch loss")
        axes[1].plot(epoch, val, color=ORANGE, linewidth=1.5, label="Validation (eval mode)")
        if np.isfinite(online).any():
            axes[1].plot(epoch, online, color=BLUE, linestyle="--", label="Training (online)")
        if best is not None:
            axes[1].scatter(epoch[best], val[best], marker="*", s=100, color="black", zorder=4)
            axes[1].set_title(f"Best validation: {val[best]:.3f} dB at epoch {epoch[best]:.0f}")
        axes[1].set(ylabel="Valid-pixel pooled RMSE (dB)")
        axes[1].legend(fontsize=8)
        axes[2].plot(epoch, lr, color="#333333", linestyle="-.")
        axes[2].set(title="Learning rate", ylabel="Learning rate")
        axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        for axis in axes:
            axis.set_xlabel("Epoch (zero-based)")
        fig.supxlabel("Unsmoothed. Online training uses augmentation, train mode and changing weights; "
                      "it is not a fixed-checkpoint train evaluation.", fontsize=8)
        return _save(fig, path, {
            "source": os.path.abspath(history_path), "kind": "training_curves",
            "epochs": len(rows), "smoothing": None,
            "missing_data": "Blank/nonfinite metrics appear as gaps; not replaced by zero.",
            "train_loss": "Existing sample-weighted mean of batch objectives (normalized units).",
            "train_online": "sqrt(accumulated masked SSE / valid pixels) * metadata PL scale; train mode.",
            "best_epoch": int(epoch[best]) if best is not None else None,
            "best_val_rmse_db": float(val[best]) if best is not None else None,
        }, plt)


def save_rmse_distribution(path, rows, summary, source):
    """Equal weight per sample in plots; pooled pixel score shown separately."""
    plt = _pyplot()
    if plt is None:
        return None
    bands = sorted({r["band"] for r in rows})
    if not bands or not any("rmse_db_masked" in r for r in rows):
        return None
    record = {"source": os.path.abspath(source), "kind": "sample_rmse_distribution",
              "checkpoint": summary.get("checkpoint"), "epoch": summary.get("epoch"),
              "split": summary.get("split"), "sample_weighting": "equal, one row per sample",
              "uncertainty": "None; TX samples sharing a terrain are not independent replicates.",
              "bands": {}}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(bands), 2, figsize=(11, 4.2*len(bands)),
                                 squeeze=False, layout="constrained")
        for i, band in enumerate(bands):
            members = [r for r in rows if r["band"] == band]
            valid = [r for r in members if np.isfinite(r.get("rmse_db_masked", np.nan))]
            if not valid:
                for axis in axes[i]:
                    axis.text(0.5, 0.5, "No scored samples", ha="center", transform=axis.transAxes)
                record["bands"][band] = {"total": len(members), "scored": 0}
                continue
            values = np.array([r["rmse_db_masked"] for r in valid])
            bins = np.arange(0, max(1, np.ceil(values.max())) + 1, dtype=float)
            axes[i, 0].hist(values, bins=bins, color=BLUE, edgecolor="white")
            axes[i, 0].set(ylabel="Samples (count)", title=f"Band {band}: {len(valid)} scored samples")
            ordered = np.sort(values)
            axes[i, 1].step(ordered, np.arange(1, len(ordered)+1)/len(ordered),
                            where="post", color=BLUE)
            pooled = summary.get("rmse_db_masked") if len(bands) == 1 else None
            if pooled is None and all("sse_db2_masked" in r and "valid_pixels" in r for r in valid):
                pooled = np.sqrt(sum(r["sse_db2_masked"] for r in valid) /
                                 sum(r["valid_pixels"] for r in valid))
            if pooled is not None:
                axes[i, 1].axvline(pooled, color=ORANGE, linestyle="--",
                                   label=f"Pixel-pooled score: {pooled:.3f} dB")
            q = np.quantile(values, [.5, .95, .99])
            axes[i, 1].set(ylabel="Fraction of scored samples", ylim=(0, 1.02),
                          title=f"Median {q[0]:.2f}; P95 {q[1]:.2f}; P99 {q[2]:.2f} dB")
            if pooled is not None:
                axes[i, 1].legend(fontsize=8)
            for axis in axes[i]:
                axis.set(xlabel="Per-sample valid-pixel RMSE (dB)", xlim=(0, bins[-1]))
            record["bands"][band] = {
                "total": len(members), "scored": len(valid),
                "zero_valid_or_nonfinite": len(members)-len(valid), "histogram_edges_db": bins.tolist(),
                "median_p95_p99_db": q.tolist(), "pixel_pooled_rmse_db": pooled,
            }
        fig.supxlabel("Each sample has equal weight in the distribution; the ranking score pools valid pixels. "
                      "Shared-terrain samples are correlated.", fontsize=8)
        return _save(fig, path, record, plt)
