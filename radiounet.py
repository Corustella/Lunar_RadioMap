"""
Access to the RadioUNet model code.

The RadioWNet architecture is not reimplemented here. The authors' model code is
included unmodified under ``RadioUNet/modules.py`` this module is the seam between that
file and the rest of the baseline.

    from radiounet import RadioWNet
    model = RadioWNet(inputs=2, phase="firstU")

Loading is lazy, so ``--help`` and argument parsing still work in an environment
without torch; the import only happens when a model is actually built.

``RadioWNet(inputs, phase)`` returns ``[out1, out2]`` from its forward pass.
``phase="firstU"`` puts gradients on the first U-Net and detaches the
refinement; anything else detaches the first and trains the refinement.
"""

from __future__ import annotations

import importlib.util
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "RadioUNet", "modules.py")

_MISSING = """RadioUNet model code not found at
  {path}

It ships with this repository, so a missing file usually means an incomplete
clone or a stripped archive. Re-clone, or restore it from
https://github.com/RonLevie/RadioUNet"""

_cached = None
SECOND_FEATURES = ("none", "zeros", "tx_xyz", "los")
_EXPANDED_WEIGHTS = ("Wlayer00.0.weight", "Wconv_up00.0.weight",
                     "Wconv_up000.0.weight")


def geometry_config(meta):
    """Fixed geometry scales from training metadata; inputs are normalized height."""
    lo, hi = map(float, meta["heightmap_range_m"])
    simulation = meta["simulation"]
    config = {"version": 1, "grid": list(meta["grid"]),
              "resolution_m_per_px": float(meta["resolution_m_per_px"]),
              "heightmap_range_m": [lo, hi],
              "tx_height_m_agl": float(simulation["tx_height_m_agl"]),
              "rx_height_m_agl": float(simulation["rx_height_m_agl"])}
    if config["grid"] != [256, 256]:
        raise ValueError("TX geometry requires the challenge's 256x256 grid")
    values = (lo, hi, config["resolution_m_per_px"],
              config["tx_height_m_agl"], config["rx_height_m_agl"])
    if not all(math.isfinite(v) for v in values) or hi <= lo or values[2] <= 0:
        raise ValueError("Geometry metadata must have finite values and positive scales")
    return config


def feature_config_for_mode(meta, mode):
    if mode == "none":
        return None
    config = geometry_config(meta)
    if mode == "los":
        from los_features import los_feature_config
        config = los_feature_config(config)
    return config


def checkpoint_features(checkpoint):
    """Legacy checkpoints use no features; expanded checkpoints must save scales."""
    mode = checkpoint.get("args", {}).get("second_features", "none")
    if mode not in SECOND_FEATURES:
        raise ValueError(f"Unknown checkpoint second_features={mode!r}")
    config = checkpoint.get("feature_config")
    if mode != "none":
        if config is None or config.get("version") != 1:
            raise ValueError("Geometry checkpoint lacks a supported feature_config")
        expected = feature_config_for_mode({**config, "simulation": config}, mode)
        if expected != config:
            raise ValueError("Invalid checkpoint feature_config")
    elif config is not None:
        raise ValueError("Baseline checkpoint must not contain geometry feature_config")
    return mode, config


def tx_geometry(input, config):
    """Rebuild signed dx/dy/dz AFTER augmentation, using only height and TX.

    dx/dy use the image column/row axes. dz is antenna height difference divided
    by the training height range. Existing height normalization clips to [0,1],
    so out-of-range terrain would also be clipped in this input-derived proxy.
    Time and extra memory are O(BHW).
    """
    import torch

    if input.ndim != 4 or list(input.shape[-2:]) != config["grid"]:
        raise ValueError("Geometry input must be BxCx256x256")
    height, tx = input[:, 0].float(), input[:, 1].float()
    valid = ((tx == 0) | (tx == 1)).flatten(1).all(1) & (tx.flatten(1).sum(1) == 1)
    if not bool(valid.all()):
        raise ValueError("Geometry requires exactly one one-hot TX pixel per sample")
    if not bool(torch.isfinite(height).all()) or bool(((height < 0) | (height > 1)).any()):
        raise ValueError("Geometry requires finite height normalized to [0,1]")
    h, w = input.shape[-2:]
    location = tx.flatten(1).argmax(1)
    rows, cols = location // w, location % w
    x = torch.arange(w, device=input.device, dtype=torch.float32)
    y = torch.arange(h, device=input.device, dtype=torch.float32)
    dx = ((x[None, None, :] - cols[:, None, None]) / (w - 1)).expand(-1, h, -1)
    dy = ((y[None, :, None] - rows[:, None, None]) / (h - 1)).expand(-1, -1, w)
    tx_height = height.flatten(1).gather(1, location[:, None])[:, :, None]
    lo, hi = config["heightmap_range_m"]
    dz = height - tx_height + (config["rx_height_m_agl"] - config["tx_height_m_agl"]) / (hi - lo)
    return torch.stack((dx, dy, dz), dim=1).to(input.dtype)


def _append_geometry(model, args):
    import torch

    (input,) = args
    expected = model.inputs + int(model.second_features == "los")
    if input.shape[1] != expected:
        raise ValueError(f"Expected {expected} input channels, got {input.shape[1]}")
    with torch.no_grad():
        if model.second_features == "los":
            feature = input[:, model.inputs:model.inputs + 1]
            features = torch.cat((feature, torch.zeros_like(feature),
                                  torch.zeros_like(feature)), dim=1)
            input = input[:, :model.inputs]
        else:
            features = tx_geometry(input, model.feature_config)
            if model.second_features == "zeros":
                features = torch.zeros_like(features)
    return (torch.cat((input, features), dim=1),)


def load_model_state(model, checkpoint, initialize=False):
    """Strict load, allowing only baseline -> geometry padding for init-from.

    Added channels are appended in all three input skips. Old columns and all
    biases are copied exactly; the added columns start at zero. Resume never
    migrates shapes, modes or geometry scales.
    """
    source_mode, source_config = checkpoint_features(checkpoint)
    migrate = initialize and source_mode == "none" and model.second_features != "none"
    if not migrate and (source_mode != model.second_features or source_config != model.feature_config):
        raise ValueError("Checkpoint feature mode/scales differ; resume and evaluation require an exact match")
    state = checkpoint["model"]
    if migrate:
        state = state.copy()
        expected = model.state_dict()
        for name in _EXPANDED_WEIGHTS:
            old, new = state[name], expected[name]
            if old.shape != (new.shape[0], new.shape[1] - 3, *new.shape[2:]):
                raise ValueError(f"Unexpected baseline weight shape for {name}: {tuple(old.shape)}")
            padded = old.new_zeros(new.shape)
            padded[:, :old.shape[1]] = old
            state[name] = padded
    model.load_state_dict(state, strict=True)
    return {"source_mode": source_mode, "target_mode": model.second_features,
            "zero_padded_weights": list(_EXPANDED_WEIGHTS) if migrate else []}


def load_module():
    """Import ``RadioUNet/modules.py`` and return it."""
    global _cached
    if _cached is not None:
        return _cached
    if not os.path.exists(MODULE_PATH):
        raise SystemExit(_MISSING.format(path=MODULE_PATH))

    spec = importlib.util.spec_from_file_location("radiounet_modules", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _cached = module
    return module


def RadioWNet(inputs=2, phase="firstU", second_features="none", feature_config=None):  # noqa: N802
    """Build the upstream model, optionally expanding only secondU inputs.

    State keys and the upstream forward remain unchanged. firstU still sees
    only the original 2 (or 3 for band=both) input channels.
    """
    if second_features not in SECOND_FEATURES:
        raise ValueError(f"second_features must be one of {SECOND_FEATURES}")
    if second_features != "none":
        checkpoint_features({"args": {"second_features": second_features},
                             "feature_config": feature_config})
    elif feature_config is not None:
        raise ValueError("feature_config is only used by expanded models")
    model = load_module().RadioWNet(inputs=inputs, phase=phase)
    model.second_features = second_features
    model.feature_config = feature_config
    if second_features != "none":
        import torch
        from torch import nn

        # Allocate without consuming RNG: shared old weights remain identical
        # across paired modes, and every new input weight starts at zero.
        for name in ("Wlayer00", "Wconv_up00", "Wconv_up000"):
            sequence = getattr(model, name)
            old = sequence[0]
            with torch.device("meta"):
                new = nn.Conv2d(old.in_channels + 3, old.out_channels,
                                old.kernel_size, old.stride, old.padding,
                                old.dilation, old.groups, old.bias is not None,
                                old.padding_mode)
            new.to_empty(device=old.weight.device)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :old.in_channels].copy_(old.weight)
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            sequence[0] = new
        model.register_forward_pre_hook(_append_geometry)
    return model


def trains_in_phase(param_name, phase):
    """Whether a parameter belongs to the half that ``phase`` trains.

    The forward pass detaches the other half, so those parameters get no
    gradient regardless; setting ``requires_grad`` to match just makes it
    explicit and gives an honest trainable-parameter count. The optimizer is
    still handed every parameter, which keeps its ``state_dict`` layout identical
    across the two stages so a checkpoint stays resumable either way.

    Every layer of the refinement half is named with a leading ``W``.
    """
    return param_name.startswith("W") if phase == "secondU" else not param_name.startswith("W")
