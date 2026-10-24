"""
Baseline training for the First Lunar Pathloss Radio Map Prediction Challenge.

Trains RadioWNet to predict a 256x256 pathloss map in dB from a lunar heightmap
and a one-hot transmitter position.

``--arch lunar_maxvit`` and its attention-free control train an encoder/decoder jointly,
with phase ``single`` and input-only LOS/distance features. It reuses the LOS
cache, masked loss and evaluation path. See LUNAR_MAXVIT.md and
LUNAR_CONV_ALIGNED.md for server commands.

Two-stage training
------------------
RadioWNet is two U-Nets in series and they are trained one at a time, because
the refinement has nothing to refine until the first U-Net has converged:

    --phase firstU     trains the first U-Net (10.92M of 13.27M parameters)
    --phase secondU    freezes it and trains the refinement (2.35M)

The ``secondU`` checkpoint contains both halves and is the full model. It must
be seeded from a trained ``firstU`` run; ``--init-from auto`` (the default for
``--phase secondU``) finds ``<out>/<band>_<loss>_firstU_best.pt``.

Valid pixels only
-----------------
Training and scoring both happen on ray-traced pixels only. 

    --loss masked    (default) MSE over ray-traced pixels only
    --loss plain     MSE over every pixel, including the fill

``--loss plain`` is kept as an ablation. It is NOT the challenge objective, and
a run trained that way is still selected and reported on the masked metric, so
the two remain directly comparable.

Both metrics are always computed:

    rmse_db_masked   ray-traced pixels only -- THE RANKING METRIC
    rmse_db          every pixel, fill included -- diagnostic only

Masks are exact, ship with the train and val archives, and are complete for both
bands, so masking filters no samples -- only pixels.

Validation
----------
``val`` is a frozen split, disjoint from ``train`` at the terrain level and
shipped as its own directory. Every entrant validates on the same maps, so
numbers reported against it are comparable across entries. Do not re-split.

Examples
--------
    # stage 1
    python train.py --data-root LunarRM --band 58 --epochs 450
    # stage 2, seeded from stage 1 automatically
    python train.py --data-root LunarRM --band 58 --phase secondU --epochs 150

    # one conditional model over both bands
    python train.py --data-root LunarRM --band both --epochs 300

    # resume an interrupted run (same --out, same --run-name)
    python train.py ... --resume auto

    # seconds-long smoke test of the whole path, no GPU
    python train.py --band 58 --epochs 1 --batch-size 2 --limit-batches 3 \
        --num-workers 0 --device cpu --no-amp --out /tmp/smoke --run-name smoke
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import tempfile
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import MetricAccumulator, save_example_panels 
from radiounet import (RadioWNet, trains_in_phase, FIRST_FEATURES, SECOND_FEATURES,
                       feature_config_for_mode)
from model_factory import (ARCHITECTURES, LUNAR_ARCHITECTURES, REFINEMENT_ARCHITECTURES, build_model,
                           T_REFINEMENT_ARCHITECTURES, checkpoint_architecture, load_checkpoint,
                           model_feature_config, output_index, uses_los)
from experiment_provenance import collect_provenance
from analysis_plots import save_training_curves


def load_dataset_module(data_root):
    """Prefer the loader shipped with the data, so code and data never skew."""
    for cand in (data_root, os.path.dirname(os.path.abspath(__file__))):
        if os.path.exists(os.path.join(cand, "lunar_dataset.py")):
            sys.path.insert(0, cand)
            import lunar_dataset
            return lunar_dataset
    raise FileNotFoundError("lunar_dataset.py not found in --data-root or beside train.py")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, name, lr, weight_decay):
    """Keep legacy Adam ordering; decay only Conv/Linear weights with AdamW.

    Biases, normalization scales and relative-position tables do not decay.
    Named groups are recorded so resume cannot silently remap optimizer state.
    Construction is O(P); Adam and AdamW both retain O(P) optimizer state.
    """
    named = dict(model.named_parameters())
    if getattr(model, "frozen_stage_one", False):
        named = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
    if name == "adam":
        if weight_decay != 0:
            raise ValueError("Use --optimizer adamw for decoupled --weight-decay")
        groups = [list(named)]
        opt = torch.optim.Adam(list(named.values()), lr=lr)
        policy = "legacy_all_parameters"
    else:
        decay_ids = {id(module.weight) for module in model.modules()
                     if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))}
        groups = [[n for n, p in named.items() if id(p) in decay_ids],
                  [n for n, p in named.items() if id(p) not in decay_ids]]
        opt = torch.optim.AdamW([
            {"params": [named[n] for n in groups[0]], "weight_decay": weight_decay},
            {"params": [named[n] for n in groups[1]], "weight_decay": 0.0},
        ], lr=lr)
        policy = "conv_linear_weights_only_v1"
    if getattr(model, "frozen_stage_one", False):
        policy += "_trainable_only"
    config = {"name": name, "weight_decay": weight_decay, "policy": policy,
              "betas": [0.9, 0.999], "eps": 1e-8,
              "groups": [{"parameter_names": names,
                          "parameters": sum(named[n].numel() for n in names),
                          "weight_decay": opt.param_groups[i]["weight_decay"]}
                         for i, names in enumerate(groups)]}
    return opt, config


def check_optimizer_resume(checkpoint, config):
    """Validate before loading: load_state_dict overwrites group hyperparameters."""
    saved = checkpoint.get("optimizer_config")
    if saved is None:
        # Historical checkpoints predate optimizer selection and use plain Adam.
        previous = checkpoint.get("args", {})
        if (config["name"] != "adam" or config["weight_decay"] != 0 or
                previous.get("optimizer", "adam") != "adam" or
                previous.get("weight_decay", 0.0) != 0):
            raise ValueError("Legacy checkpoint requires Adam with weight_decay=0; start a new run")
    elif saved != config:
        raise ValueError("Resume optimizer type, decay policy or parameter groups differ; start a new run")
    actual = checkpoint["opt"]["param_groups"]
    if len(actual) != len(config["groups"]):
        raise ValueError("Resume optimizer group count differs")
    for old, wanted in zip(actual, config["groups"]):
        if (old.get("weight_decay", 0.0) != wanted["weight_decay"] or
                len(old["params"]) != len(wanted["parameter_names"]) or
                list(old["betas"]) != config["betas"] or old["eps"] != config["eps"] or
                old.get("amsgrad", False) or old.get("maximize", False) or
                (config["name"] == "adam" and old.get("decoupled_weight_decay", False)) or
                (config["name"] == "adamw" and old.get("decoupled_weight_decay") is False)):
            raise ValueError("Resume optimizer state conflicts with requested settings")


def append_history(path, row, fresh=False):
    """Extend legacy CSV columns atomically; unmeasured old values stay blank."""
    fields = list(row)
    if not fresh and os.path.exists(path):
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            previous_fields = reader.fieldnames or []
            if previous_fields != fields:
                previous = list(reader)
        if previous_fields != fields:
            fields = previous_fields + [k for k in fields if k not in previous_fields]
            with tempfile.NamedTemporaryFile(mode="w", newline="", dir=os.path.dirname(path),
                                             suffix=".csv", delete=False) as handle:
                temporary = handle.name
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(previous)
            os.replace(temporary, path)
    with open(path, "w" if fresh or not os.path.exists(path) else "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fresh or handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def resolve_init_from(args):
    """Checkpoint whose weights seed this run; 'auto' means the matching firstU run.

    Weights only, the optimizer, the LR schedule and the epoch counter all
    start fresh, because this is a new stage rather than a continuation.
    """
    if args.init_from != "auto":
        return args.init_from
    run_name = args.run_name
    stem = (run_name.replace("secondU", "firstU") if "secondU" in run_name
            else f"radiownet_{args.band}_{args.loss}_firstU")
    for suffix in ("_best.pt", "_last.pt"):
        cand = os.path.join(args.out, stem + suffix)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"--init-from auto found neither {stem}_best.pt nor {stem}_last.pt in "
        f"{args.out}. Train the firstU stage first, or pass --init-from <path> / "
        f"--init-from none.")


def make_loaders(ld, args):
    common = dict(root=args.data_root, band=args.band, return_mask=True)
    use_los = uses_los(args.arch, args.second_features)
    train_ds = ld.LunarRadioMapDataset(split="train", augment=not args.no_augment and not use_los,
                                       **common)
    val_ds = ld.LunarRadioMapDataset(split="val", augment=False, **common)
    if use_los:
        from los_features import LOSDataset
        if not args.los_cache:
            raise ValueError("LOS inputs require --los-cache")
        config = feature_config_for_mode(train_ds.meta, "los")
        train_ds = LOSDataset(train_ds, args.los_cache, config,
                              augment=not args.no_augment, augmentation=ld._augment)
        val_ds = LOSDataset(val_ds, args.los_cache, config)
    print(f"samples: {len(train_ds)} train / {len(val_ds)} val")

    dl = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=True, persistent_workers=args.num_workers > 0,
              drop_last=False)
    return (DataLoader(train_ds, shuffle=True, **dl),
            DataLoader(val_ds, shuffle=False, **dl), train_ds)


def unpack(batch, device):
    x, y, m = batch[0], batch[1], batch[2]
    return (x.to(device, non_blocking=True), y.to(device, non_blocking=True),
            m.to(device, non_blocking=True))


def compute_loss(pred, y, m, kind):
    if kind == "masked":
        se = (pred - y) ** 2 * m
        denom = m.sum()
        return se.sum() / denom if denom > 0 else se.sum() * 0.0
    return nn.functional.mse_loss(pred, y)


@torch.no_grad()
def validate(model, loader, device, scale, amp, out_idx, limit=0):
    """RMSE (dB), NMSE, PSNR and SSIM, each with its valid-pixels-only variant."""
    model.eval()
    acc = MetricAccumulator(scale)
    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        x, y, m = unpack(batch, device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = model(x)[out_idx]
        acc.update(pred.float(), y, m)
    return acc.compute(prefix="val/")


@torch.no_grad()
def write_panels(model, dataset, indices, device, out_idx, lo, scale,
                 path, err_lim, amp=False):
    """Render a FIXED set of validation samples, so the panels show the same maps
    improving rather than a different random draw each time."""
    model.eval()
    xs, ys, ms, names = [], [], [], []
    for i in indices:
        item = dataset[i]
        xs.append(item[0])
        ys.append(item[1])
        ms.append(item[2])
        # Built from .rows/.samples rather than dataset.sample_name(),
        # which older copies of the shipped loader do not have.
        i_row, band = dataset.samples[i]
        names.append(f"{dataset.rows[i_row]['sample_id']}_{band}")
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    m = torch.stack(ms).to(device) if ms else None
    with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
        out = model(x)
    # out2 is a direct prediction; the plot shows its difference from firstU.
    base = out[0].float() if out_idx == 1 else None
    return save_example_panels(path, x, out[out_idx].float(), y, m, lo, scale,
                               names, max_examples=len(indices), base=base,
                               err_lim=err_lim, mask_invalid=True)


def prepare_args(args):
    if args.stop_after_epochs is not None and not 1 <= args.stop_after_epochs <= args.epochs:
        raise ValueError("--stop-after-epochs must be between 1 and --epochs (total completed epochs)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and nonnegative")
    if args.optimizer == "adam" and args.weight_decay != 0:
        raise ValueError("Nonzero --weight-decay requires --optimizer adamw")
    if args.phase is None:
        args.phase = "single" if args.arch in LUNAR_ARCHITECTURES else "firstU"
    output_index(args.arch, args.phase)
    if args.arch in REFINEMENT_ARCHITECTURES:
        if args.band == "both" or args.first_features != "los" or args.second_features != "los_distance":
            raise ValueError(f"{args.arch} requires one band, --first-features los --second-features los_distance")
        if args.init_from in (None, "auto"):
            raise ValueError(f"{args.arch} requires an explicit historical LOS firstU --init-from path")
        if args.arch in ("radiownet_path_profile", "radiownet_biscale") and args.init_from in ("none", "scratch", ""):
            raise ValueError(f"{args.arch} training must start from the historical LOS firstU weights")
    if args.arch in LUNAR_ARCHITECTURES:
        if args.band == "both":
            raise ValueError(f"{args.arch} uses one band per model; use --band 58 or 415")
        if args.first_features != "none" or args.second_features != "none":
            raise ValueError(f"{args.arch} uses F and rho automatically; omit RadioWNet stage feature flags")
        if args.init_from == "auto":
            raise ValueError(f"{args.arch} has no firstU checkpoint; use scratch or an explicit matching checkpoint")
    if args.run_name is None:
        args.run_name = f"{args.arch}_{args.band}_{args.loss}_{args.phase}"
        if args.arch in LUNAR_ARCHITECTURES:
            args.run_name += "_los_distance"
        elif args.second_features == "los":
            args.run_name += "_los_both" if args.first_features == "los" else "_los"
        elif args.second_features == "los_distance":
            args.run_name += "_los_distance_both" if args.first_features == "los" else "_los_distance"
    if args.init_from is None and args.phase == "secondU":
        args.init_from = "auto"
    if args.init_from in ("none", "scratch", ""):
        args.init_from = None
    if args.eval_initial and not (args.init_from or args.resume):
        raise ValueError("--eval-initial requires --init-from or --resume")
    if args.arch in T_REFINEMENT_ARCHITECTURES and not args.eval_initial:
        raise ValueError("T refinement requires --eval-initial to retain its starting score")


def main(args):
    prepare_args(args)
    if args.arch == "radiownet" and args.first_features == "los" and args.second_features == "none":
        raise ValueError("--first-features los requires LOS secondU features")
    if (args.arch == "radiownet" and args.second_features != "none" and
            args.first_features == "none" and args.phase == "firstU"):
        raise ValueError("LOS firstU training requires --first-features los")
    if args.arch == "radiownet" and args.second_features == "los_distance" and args.phase != "secondU":
        raise ValueError("--second-features los_distance trains only secondU; initialize from a LOS firstU checkpoint")
    ld = load_dataset_module(args.data_root)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, ds = make_loaders(ld, args)

    # dB per unit of normalized target so RMSE is reported in real units
    lo, hi = ds.pl_min, ds.pl_max
    scale = float(hi - lo)
    in_ch = 3 if args.band == "both" else 2

    out_idx = output_index(args.arch, args.phase)
    feature_config = model_feature_config(ds.meta, args.arch, args.second_features)
    model = build_model(args.arch, inputs=in_ch, phase=args.phase,
                        second_features=args.second_features, feature_config=feature_config,
                        first_features=args.first_features).to(device)
    
    # Lunar constructors own their trainability policy, including fixed T.
    if args.arch not in LUNAR_ARCHITECTURES:
        for pname, prm in model.named_parameters():
            prm.requires_grad_(trains_in_phase(pname, args.phase))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params {n_train:,} / {n_total:,} ({args.arch}, {args.phase})")

    opt, optimizer_config = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    print(f"optimizer {args.optimizer}, weight_decay={args.weight_decay:g}, "
          f"policy={optimizer_config['policy']}")

    if args.lr_sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr_min)
    else:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step,
                                                gamma=args.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    os.makedirs(args.out, exist_ok=True)
    ckpt_last = os.path.join(args.out, f"{args.run_name}_last.pt")
    ckpt_best = os.path.join(args.out, f"{args.run_name}_best.pt")
    hist_path = os.path.join(args.out, f"{args.run_name}_history.csv")
    if not args.resume and any(os.path.exists(p) for p in (ckpt_last, ckpt_best, hist_path)):
        raise FileExistsError("Run outputs already exist; use a new --out/--run-name or explicit --resume")

    start_epoch = 0
    best = float("inf")
    resumed = False
    checkpoint_sources = {}
    weight_loading = None
    initial_validation = None
    if args.resume:
        path = ckpt_last if args.resume == "auto" else args.resume
        if os.path.exists(path):
            ck = torch.load(path, map_location=device, weights_only=False)
            if checkpoint_architecture(ck) != args.arch:
                raise ValueError("Resume requires the same architecture; train the new architecture from scratch")
            previous = ck.get("args", {})
            if (float(previous.get("grad_weight", 0.0)) != 0.0 or
                    previous.get("loss", args.loss) != args.loss or
                    previous.get("phase", args.phase) != args.phase or
                    previous.get("band", args.band) != args.band or
                    previous.get("eval_initial", False) != args.eval_initial):
                raise ValueError("Resume requires the same loss, phase, band and initial-evaluation policy")
            initial_validation = ck.get("initial_validation")
            if args.eval_initial and initial_validation is None:
                raise ValueError("Resume checkpoint lacks its initial-validation record")
            checkpoint_sources["resume"] = path
            check_optimizer_resume(ck, optimizer_config)
            weight_loading = load_checkpoint(model, ck)
            opt.load_state_dict(ck["opt"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])

            prev_sched = ck.get("lr_sched", "step")
            prev_len = ck["sched"].get("T_max") if args.lr_sched == "cosine" else None
            len_changed = prev_len is not None and prev_len != args.epochs
            if prev_sched == args.lr_sched and not len_changed:
                sched.load_state_dict(ck["sched"])
            else:
                why = (f"checkpoint used --lr-sched {prev_sched}"
                       if prev_sched != args.lr_sched else
                       f"checkpoint cosine ran to {prev_len} epochs, this run to "
                       f"{args.epochs}")
                print(f"{why}; replaying the new schedule to that epoch")

                for group, base_lr in zip(opt.param_groups, sched.base_lrs):
                    group["lr"] = base_lr

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    for _ in range(ck["epoch"] + 1):
                        sched.step()
            start_epoch = ck["epoch"] + 1
            best = ck.get("best", best)
            resumed = True
            print(f"resumed from {path} at epoch {start_epoch}")
        elif args.resume != "auto":
            raise FileNotFoundError(path)
        else:
            if any(os.path.exists(p) for p in (ckpt_best, hist_path)):
                raise FileExistsError("No last checkpoint but run artifacts exist; use a new output path")
            print("no checkpoint to resume; starting fresh")


    if args.init_from and not resumed:
        path = resolve_init_from(args)
        ck = torch.load(path, map_location=device, weights_only=False)
        checkpoint_sources["init_from"] = path
        prev = ck.get("args", {})
        if prev.get("band", args.band) != args.band:
            raise SystemExit(f"--init-from {path} was trained on band "
                             f"{prev.get('band')}, this run is band {args.band}")
        if args.arch in T_REFINEMENT_ARCHITECTURES and (
                checkpoint_architecture(ck) != "lunar_conv_tx_content" or
                prev.get("seed") != args.seed):
            raise ValueError("Initialize refinement from the matching seed's T checkpoint; use --resume for refinement")
        weight_loading = load_checkpoint(model, ck, initialize=True)
        if args.eval_initial:
            initial_validation = {"source_checkpoint": path,
                                  "source_epoch": ck.get("epoch"),
                                  "source_best": ck.get("best"),
                                  "source_seed": prev.get("seed")}
        print(f"initialized weights from {path} (phase {prev.get('phase', '?')}, "
              f"epoch {ck.get('epoch')}, best {ck.get('best', float('nan')):.4f} dB); "
              f"optimizer and LR schedule start fresh")
    elif args.phase == "secondU" and not resumed:
        print("warning: training secondU from random init -- the frozen first "
              "U-Net is untrained, so its output is noise. Pass --init-from "
              "<firstU checkpoint>.")
    if not resumed and (args.arch in T_REFINEMENT_ARCHITECTURES or args.eval_initial) and not args.init_from:
        raise ValueError("No checkpoint to resume; this experiment requires an explicit --init-from path")

    end_epoch = args.epochs if args.stop_after_epochs is None else args.stop_after_epochs
    if start_epoch >= end_epoch:
        print(f"already completed {start_epoch} epochs; requested stop is {end_epoch}; no training")
        return
    if end_epoch < args.epochs:
        print(f"stopping after {end_epoch} total epochs; LR schedule horizon remains {args.epochs}")

    def make_state(epoch, metrics):
        return {"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                "lr_sched": args.lr_sched, "epoch": epoch, "best": best,
                "args": vars(args), "metrics": metrics, "feature_config": feature_config,
                "model_config": model.model_config, "optimizer_config": optimizer_config,
                "initial_validation": initial_validation}

    if args.eval_initial and not resumed:
        t0 = time.time()
        # Validation creates DataLoader iterators. Preserve the training RNG
        # stream, including the CPU generator used for shuffled sample order.
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            metrics = validate(model, val_loader, device, scale, args.amp, out_idx,
                               limit=args.limit_batches)
        best = metrics["val/rmse_db_masked"]
        if not math.isfinite(best):
            raise ValueError("Initial validation has no finite masked RMSE")
        initial_validation.update(epoch=-1, metrics=metrics, limit_batches=args.limit_batches,
                                  seconds=time.time() - t0)
        with open(os.path.join(args.out, f"{args.run_name}_initial.json"), "w", encoding="utf-8") as handle:
            json.dump(initial_validation, handle, indent=2)
        # Epoch -1 is a resumable, untrained starting point, not a new result.
        # Keeping both files also permits recovery if interrupted before ep0.
        state = make_state(-1, metrics)
        torch.save(state, ckpt_best)
        torch.save(state, ckpt_last)
        print(f"initial validation: {best:.6f} dB; retained as epoch -1 best", flush=True)

    provenance = collect_provenance(args.data_root, ld, checkpoint_sources, args)
    provenance["first_features"] = args.first_features
    provenance["second_features"] = args.second_features
    provenance["feature_config"] = feature_config
    provenance["model_config"] = model.model_config
    provenance["optimizer_config"] = optimizer_config
    if args.arch in LUNAR_ARCHITECTURES:
        from importlib.metadata import version
        provenance["environment"]["torchvision"] = version("torchvision")
    provenance["weight_loading"] = weight_loading
    provenance["initial_validation"] = initial_validation
    # On resume retain the original initialization record as well.
    prov_path = os.path.join(args.out, f"{args.run_name}_provenance.json")
    if resumed and os.path.exists(prov_path):
        with open(prov_path, encoding="utf-8") as handle:
            previous = json.load(handle)
        provenance["initial_run"] = previous.get("initial_run", previous)
    with open(prov_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)

    n_val = len(val_loader.dataset)
    k = min(args.panels, n_val)
    panel_idx = ([int(round(i * (n_val - 1) / max(k - 1, 1))) for i in range(k)]
                 if k > 0 else [])

    print(f"training {args.run_name}: band={args.band} loss={args.loss} "
          f"arch={args.arch} phase={args.phase} in_ch={in_ch} scale={scale:.1f} dB "
          f"first_features={args.first_features} second_features={args.second_features}")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        train_sse = torch.zeros((), device=device, dtype=torch.float64)
        train_valid = torch.zeros((), device=device, dtype=torch.float64)
        for i, batch in enumerate(train_loader):
            x, y, m = unpack(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred = model(x)[out_idx]
                loss = compute_loss(pred.float(), y, m, args.loss)
            scaler.scale(loss).backward()
            if args.clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()

            # Detached online diagnostic: augmented data, train mode, changing
            # weights. It is not an eval-mode score of a fixed train checkpoint.
            with torch.no_grad():
                se = (pred.detach().float() - y.float()).square()
                train_sse += (se * m).sum(dtype=torch.float64)
                train_valid += m.sum(dtype=torch.float64)
            running += loss.item() * x.size(0)
            seen += x.size(0)
            if args.log_every and i % args.log_every == 0:
                print(f"  ep{epoch} [{i}/{len(train_loader)}] loss {loss.item():.6f}",
                      flush=True)
            if args.limit_batches and i + 1 >= args.limit_batches:
                break

        # read before stepping: get_last_lr() after the step is next epoch's LR
        epoch_lr = sched.get_last_lr()[0]
        sched.step()

        metrics = validate(model, val_loader, device, scale, args.amp, out_idx,
                           limit=args.limit_batches)
        train_loss = running / max(seen, 1)
        valid_pixels = int(train_valid.item())
        online_mse = train_sse.item() / valid_pixels if valid_pixels else float("nan")
        dt = time.time() - t0
        # the ranking metric leads, all-pixel RMSE trails as a diagnostic
        msg = (f"epoch {epoch}: train_loss {train_loss:.6f}  "
               f"rmse {metrics['val/rmse_db_masked']:.3f} dB (valid px)  "
               f"psnr {metrics['val/psnr_db']:.2f}  "
               f"ssim {metrics['val/ssim']:.4f}  "
               f"[all-px {metrics['val/rmse_db']:.3f} dB]")
        print(msg + f"  ({dt:.0f}s)", flush=True)

        row = {"epoch": epoch, "train_loss": train_loss, "lr": epoch_lr,
               "epoch_seconds": round(dt, 1),
               "train/mse_masked_online": online_mse,
               "train/rmse_db_masked_online": math.sqrt(online_mse) * scale,
               "train/valid_pixels": valid_pixels,
               **{k: v for k, v in metrics.items()}}
        append_history(hist_path, row, fresh=epoch == 0 and not resumed)

        if panel_idx and args.panels_every and epoch % args.panels_every == 0:
            write_panels(model, val_loader.dataset, panel_idx, device, out_idx, lo,
                         scale,
                         os.path.join(args.out, f"{args.run_name}_panels.png"),
                         args.panel_err_lim, args.amp)

        # Always select on the ranking metric, even for a --loss plain ablation.
        # Selecting a plain run on all-pixel RMSE would pick the checkpoint that
        # best reproduces the 5.8 GHz fill, which is not what it is scored on.
        sel = metrics["val/rmse_db_masked"]
        improved = sel < best
        if improved:
            best = sel
        state = make_state(epoch, metrics)
        torch.save(state, ckpt_last)
        if improved:
            torch.save(state, ckpt_best)
            print(f"  new best {best:.4f} dB -> {ckpt_best}")
        if args.plots_every and (epoch % args.plots_every == 0 or epoch == end_epoch - 1):
            save_training_curves(hist_path, os.path.join(args.out, f"{args.run_name}_curves.png"))

    # Separate best panels from the rolling snapshot; use validation AMP mode.
    if panel_idx and os.path.exists(ckpt_best):
        selected = torch.load(ckpt_best, map_location="cpu", weights_only=False)
        load_checkpoint(model, selected)
        best_epoch = selected["epoch"]
        del selected
        path = os.path.join(args.out, f"{args.run_name}_best_epoch{best_epoch}_panels.png")
        write_panels(model, val_loader.dataset, panel_idx, device, out_idx, lo, scale,
                     path, args.panel_err_lim, args.amp)

    with open(os.path.join(args.out, f"{args.run_name}_summary.json"), "w") as f:
        json.dump({"run": args.run_name, "band": args.band, "loss": args.loss,
                   "phase": args.phase, "arch": args.arch, "init_from": args.init_from,
                   "model_config": model.model_config,
                   "optimizer_config": optimizer_config,
                   "first_features": args.first_features,
                   "second_features": args.second_features, "feature_config": feature_config,
                   "selection_metric": "val/rmse_db_masked",
                   "initial_validation": initial_validation,
                   "best": best, "epochs": args.epochs,
                   "completed_epochs": end_epoch,
                   "stop_after_epochs": args.stop_after_epochs}, f, indent=2)
    print(f"done. best {best:.4f} dB (valid pixels only)")


def build_parser():
    p = argparse.ArgumentParser(
        description="Train RadioWNet or a single-output lunar architecture.")
    p.add_argument("--data-root", default="LunarRM")
    p.add_argument("--arch", default="radiownet", choices=ARCHITECTURES,
                   help="lunar_* use H, TX, F and rho; fixed-T refinement trains only the refiner")
    p.add_argument("--band", default="58", choices=["415", "58", "both"],
                   help="'both' trains one conditional model over both "
                        "frequencies and adds a band-flag input channel")
    p.add_argument("--loss", default="masked", choices=["masked", "plain"],
                   help="masked (default) trains on ray-traced pixels only, which "
                        "is the challenge objective. plain counts the 5.8 GHz "
                        "fill too and is kept only as an ablation; either way the "
                        "run is selected and ranked on valid pixels.")
    p.add_argument("--phase", default=None, choices=["firstU", "secondU", "single"],
                   help="default: firstU for RadioWNet, single for lunar_* architectures")
    p.add_argument("--first-features", default="none", choices=FIRST_FEATURES,
                   help="append cached LOS clearance F to firstU; requires LOS secondU features")
    p.add_argument("--second-features", default="none", choices=SECOND_FEATURES,
                   help="secondU inputs: none, los [F,0,0], or los_distance [F,rho,0]")
    p.add_argument("--los-cache", default=None, help="cache directory built by build_los_cache.py")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--stop-after-epochs", type=int, default=None,
                   help="stop after this many total completed epochs, preserving --epochs as the LR "
                        "horizon; remove this option to continue with --resume auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                   help="legacy Adam by default; AdamW enables decoupled weight decay")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW decay of Conv/Linear weights only; bias/norm/position bias excluded")
    p.add_argument("--lr-sched", default="cosine", choices=["cosine", "step"])
    p.add_argument("--lr-min", type=float, default=1e-6, help="cosine floor")
    p.add_argument("--lr-step", type=int, default=20, help="--lr-sched step only")
    p.add_argument("--lr-gamma", type=float, default=0.1, help="--lr-sched step only")
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8,
                   help="the arrays are memory-mapped, so decoding and "
                        "augmentation dominate loading -- raise this if "
                        "the GPU is starved")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="'auto' or a checkpoint path")
    p.add_argument("--init-from", default=None,
                   help="seed model weights from a checkpoint -- weights only, no "
                        "optimizer/schedule/epoch. 'auto' (the default for "
                        "--phase secondU) picks the matching firstU run in --out; "
                        "'none' forces random init. Ignored when --resume finds a "
                        "checkpoint of this run.")
    p.add_argument("--eval-initial", action="store_true",
                   help="validate initialized weights before ep0 and retain them in best selection; "
                        "optimizer/schedule still start fresh; required for T refinement")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--panels", type=int, default=6,
                   help="validation examples rendered to <run>_panels.png (0 off)")
    p.add_argument("--panels-every", type=int, default=5)
    p.add_argument("--plots-every", type=int, default=5,
                   help="save training curves every N epochs and at completion (0 disables)")
    p.add_argument("--panel-err-lim", type=float, default=10.0,
                   help="fixed +/- dB colour scale for the panel error map, so "
                        "panels stay comparable across epochs")
    p.add_argument("--limit-batches", type=int, default=0,
                   help="stop each epoch after N batches; for smoke tests")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    main(a)
