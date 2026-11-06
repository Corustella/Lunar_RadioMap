"""Cache the two nested teachers' 8x8 mean pathloss targets on the train split."""

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader

from block_distillation import block_means, cache_config, cache_shape, full_blocks
from experiment_provenance import collect_provenance
from los_features import LOSDataset
from model_factory import build_model, checkpoint_architecture, load_checkpoint, model_feature_config
from radiounet import feature_config_for_mode
from train import load_dataset_module


TEACHER_ARCH = "lunar_conv_tx_nested_full"


def load_teacher(path, seed, feature_config, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    args = checkpoint["args"]
    if (checkpoint_architecture(checkpoint) != TEACHER_ARCH or args.get("seed") != seed
            or args.get("band") != "58" or args.get("phase") != "single"
            or args.get("loss") != "masked"):
        raise ValueError(f"Expected a band 58 masked nested-full teacher with seed {seed}: {path}")
    best = checkpoint["best"]
    if not math.isfinite(best) or checkpoint["metrics"]["val/rmse_db_masked"] != best:
        raise ValueError(f"Teacher must be a best checkpoint: {path}")
    model = build_model(TEACHER_ARCH, phase="single", feature_config=feature_config)
    load_checkpoint(model, checkpoint)
    record = {"path": str(Path(path).resolve()), "seed": seed, "epoch": checkpoint["epoch"],
              "val_rmse_db_masked": best}
    return model.to(device).eval(), record


def main(args):
    device = torch.device(args.device)
    ld = load_dataset_module(args.data_root)
    base = ld.LunarRadioMapDataset(root=args.data_root, split="train", band="58",
                                   augment=False, return_mask=True)
    config = cache_config(base, model_feature_config(base.meta, TEACHER_ARCH))
    dataset = LOSDataset(base, args.los_cache, feature_config_for_mode(base.meta, "los"))
    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    models, teachers = zip(*(load_teacher(path, seed, config["feature_config"], device)
                             for seed, path in enumerate(args.teacher_checkpoints)))
    provenance = collect_provenance(args.data_root, ld,
                                    {f"seed{seed}": item["path"] for seed, item in enumerate(teachers)},
                                    args)
    provenance["environment"]["torchvision"] = version("torchvision")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shape = cache_shape(base, config)
    array = np.lib.format.open_memmap(out / "train.npy", mode="w+", dtype=np.float32, shape=shape)
    written = 0
    count = torch.zeros((), dtype=torch.int64, device=device)
    amp = args.amp and device.type == "cuda"
    print(f"Generating {shape} float32 block targets on {device}, AMP={amp}", flush=True)
    with torch.inference_mode():
        for batch, (x, _, mask) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                pred0, pred1 = (model(x)[0] for model in models)
            means = block_means((pred0.float() + pred1.float()) * 0.5)
            valid = full_blocks(mask)
            count += valid.sum()
            values = means.masked_fill(~valid, 0.0)[:, 0].cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite teacher targets at train row {written}")
            array[written:written + len(values)] = values
            written += len(values)
            if batch % 100 == 0 or written == shape[0]:
                print(f"  train rows {written}/{shape[0]}", flush=True)
    if written != shape[0]:
        raise ValueError(f"Expected {shape[0]} train rows, wrote {written}")
    array.flush()
    del array
    manifest = {
        "status": "complete",
        "cache_id": uuid4().hex,
        "config": config,
        "rows": base.rows,
        "teachers": teachers,
        "ensemble": "equal_mean_in_normalized_pathloss",
        "valid_blocks": count.item(),
        "amp": amp,
        "provenance": provenance,
    }
    pending = out / "train.json.tmp"
    pending.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pending.replace(out / "train.json")
    print(f"Wrote {out} with {manifest['valid_blocks']:,} fully valid blocks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="LunarRM")
    parser.add_argument("--los-cache", required=True)
    parser.add_argument("--teacher-checkpoints", nargs=2, required=True, metavar=("SEED0", "SEED1"))
    parser.add_argument("--out", required=True, help="new output directory for the train-only cache")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
