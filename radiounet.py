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
FIRST_FEATURES = ("none", "los")
SECOND_FEATURES = ("none", "los", "los_distance")
SECOND_OUTPUTS = ("relu", "linear", "fspl_residual")
_FIRST_WEIGHTS = ("layer00.0.weight", "conv_up00.0.weight",
                  "conv_up000.0.weight")
_SECOND_WEIGHTS = ("Wlayer00.0.weight", "Wconv_up00.0.weight",
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
        raise ValueError("LOS geometry requires the challenge's 256x256 grid")
    values = (lo, hi, config["resolution_m_per_px"],
              config["tx_height_m_agl"], config["rx_height_m_agl"])
    if not all(math.isfinite(v) for v in values) or hi <= lo or values[2] <= 0:
        raise ValueError("Geometry metadata must have finite values and positive scales")
    return config


def feature_config_for_mode(meta, mode):
    if mode == "none":
        return None
    if mode not in ("los", "los_distance"):
        raise ValueError(f"Unsupported feature mode {mode!r}; supported modes are {SECOND_FEATURES}")
    config = geometry_config(meta)
    from los_features import los_feature_config, log_distance_config
    config = los_feature_config(config)
    if mode == "los_distance":
        config["distance"] = log_distance_config(config)
        # Keep the legacy LOS cache definition intact. This top-level layout
        # describes the refinement inputs when distance replaces a zero slot.
        config["second_channels"] = ["F", "rho", 0.0]
    return config


def second_output_config(meta, mode, band):
    """Output convention and fixed training scales for a single-band model."""
    if mode == "relu":
        return None
    if mode not in SECOND_OUTPUTS:
        raise ValueError(f"Unsupported second output {mode!r}; supported modes are {SECOND_OUTPUTS}")
    band = str(band)
    if band not in ("415", "58"):
        raise ValueError("Linear and FSPL output modes require one frequency band")
    lo, hi = map(float, meta["pathloss_range_db"])
    frequency = float(meta["bands_hz"][band])
    if not all(math.isfinite(v) for v in (lo, hi, frequency)) or hi <= lo or frequency <= 0:
        raise ValueError("Output metadata requires finite, positive frequency and normalization scales")
    return {**geometry_config(meta), "mode": mode, "band": band,
            "frequency_hz": frequency, "pathloss_range_db": [lo, hi],
            "quantity": "normalized_pathloss",
            "normalization": "(pathloss_db-lo)/(hi-lo)",
            "residual_units": "residual_db/(hi-lo)",
            "fspl": {"formula": "20*log10(4*pi*d_m*frequency_hz/c_m_s)",
                     "speed_of_light_m_s": 299792458.0,
                     "minimum_distance_m": 1e-6, "computation_dtype": "float32"}}


def checkpoint_output(checkpoint):
    """Legacy output is ReLU; new checkpoints must carry exact output scales."""
    saved = checkpoint.get("args", {})
    mode = saved.get("second_output", "relu")
    config = checkpoint.get("output_config")
    if mode not in SECOND_OUTPUTS:
        raise ValueError(f"Unknown checkpoint second_output={mode!r}")
    if mode == "relu":
        if config is not None:
            raise ValueError("ReLU checkpoint must not contain output_config")
        return mode, None
    if config is None or config.get("version") != 1:
        raise ValueError("Linear/FSPL checkpoint lacks a supported output_config")
    band = config.get("band")
    if saved.get("band", band) != band:
        raise ValueError("Checkpoint band differs from output_config")
    meta = {**config, "simulation": config,
            "bands_hz": {band: config.get("frequency_hz")}}
    if second_output_config(meta, mode, band) != config:
        raise ValueError("Invalid checkpoint output_config")
    return mode, config


def checkpoint_first_features(checkpoint):
    """Older LOS checkpoints supplied the feature only to secondU."""
    saved = checkpoint.get("args", {})
    first = saved.get("first_features", "none")
    second = saved.get("second_features", "none")
    if first not in FIRST_FEATURES:
        raise ValueError(f"Unknown checkpoint first_features={first!r}")
    if second not in SECOND_FEATURES:
        raise ValueError(f"Unknown checkpoint second_features={second!r}; this experimental mode was archived")
    if first == "los" and second == "none":
        raise ValueError("first_features='los' requires LOS secondU features")
    return first


def checkpoint_features(checkpoint):
    """Legacy checkpoints use no features; expanded checkpoints must save scales."""
    checkpoint_first_features(checkpoint)
    mode = checkpoint.get("args", {}).get("second_features", "none")
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


def _append_los(model, args):
    import torch

    (input,) = args
    expected = model.base_inputs + 1
    if input.ndim != 4:
        raise ValueError("LOS input must be BxCxHxW")
    if input.shape[1] != expected:
        raise ValueError(f"Expected {expected} input channels, got {input.shape[1]}")
    with torch.no_grad():
        feature = input[:, model.base_inputs:model.base_inputs + 1]
        distance = torch.zeros_like(feature)
        if model.second_features == "los_distance":
            from los_features import log_distance_feature
            distance = log_distance_feature(input, model.feature_config)
        # firstU slices only base channels plus F when enabled; rho occupies
        # a previously zero secondU slot and is never supplied to firstU.
        return (torch.cat((input, distance,
                           torch.zeros_like(feature)), dim=1),)


def _add_fspl(model, args, output):
    from los_features import fspl_feature

    (input,) = args
    return [output[0], output[1] + fspl_feature(input, model.output_config)]


def reset_second_output_head(model, seed):
    """Same signed head for both paired modes, independent of global RNG use."""
    import torch

    head = model.Wconv_up000[0]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # This is Conv2d's default Kaiming-uniform(a=sqrt(5)) initialization.
    bound = 1 / math.sqrt(head.in_channels * math.prod(head.kernel_size))
    with torch.no_grad():
        weight = torch.empty(head.weight.shape, dtype=head.weight.dtype, device="cpu")
        weight.uniform_(-bound, bound, generator=generator)
        head.weight.copy_(weight)
        if head.bias is not None:
            bias = torch.empty(head.bias.shape, dtype=head.bias.dtype, device="cpu")
            bias.uniform_(-bound, bound, generator=generator)
            head.bias.copy_(bias)
    return {"seed": int(seed), "weights": ["Wconv_up000.0.weight", "Wconv_up000.0.bias"],
            "initialization": "Conv2d_default_uniform_local_cpu_generator"}


def load_model_state(model, checkpoint, initialize=False, reset_second_head=False,
                     head_seed=None):
    """Strict load, with explicit input and output migrations for init-from.

    Added channels are appended in all three input skips. Old columns and all
    biases are copied exactly; the added columns start at zero. LOS to distance
    reuses the same weights and shapes. Resume never migrates feature modes.
    ReLU to a signed output requires an explicit head reset; both experimental
    modes use the same local seed while every other loaded parameter is kept.
    """
    source_first = checkpoint_first_features(checkpoint)
    source_mode, source_config = checkpoint_features(checkpoint)
    source_output, source_output_config = checkpoint_output(checkpoint)
    same_output = (source_output == model.second_output and
                   source_output_config == model.output_config)
    if reset_second_head and (not initialize or model.second_output == "relu" or head_seed is None):
        raise ValueError("Head reset requires init-from, a signed output mode and an explicit seed")
    migrate_output = (initialize and reset_second_head and source_output == "relu" and
                      model.second_output in ("linear", "fspl_residual"))
    if not same_output and not migrate_output:
        raise ValueError("Checkpoint output mode/scales differ; init-from migration requires explicit head reset")
    target_first = model.first_features
    target_mode = model.second_features
    same = (source_first == target_first and source_mode == target_mode and
            source_config == model.feature_config)
    first_added = source_first == "none" and target_first == "los"
    second_added = source_mode == "none" and target_mode in ("los", "los_distance")
    distance_added = source_mode == "los" and target_mode == "los_distance"
    source_matches_los = False
    if distance_added:
        source_matches_los = source_config == feature_config_for_mode(
            {**model.feature_config, "simulation": model.feature_config}, "los")
    compatible = ((source_first == target_first or first_added) and
                  (source_mode == target_mode or second_added or distance_added) and
                  (source_config == model.feature_config or second_added or source_matches_los))
    migrate = initialize and not same and compatible
    if not same and not migrate:
        raise ValueError("Checkpoint feature mode/scales differ; resume and evaluation require an exact match")
    state = checkpoint["model"]
    padded_names = []
    if migrate:
        state = state.copy()
        expected = model.state_dict()
        expansions = ((_FIRST_WEIGHTS, 1 if first_added else 0),
                      (_SECOND_WEIGHTS, 3 if second_added else 0))
        for names, added in expansions:
            if not added:
                continue
            for name in names:
                old, new = state[name], expected[name]
                if old.shape != (new.shape[0], new.shape[1] - added, *new.shape[2:]):
                    raise ValueError(f"Unexpected source weight shape for {name}: {tuple(old.shape)}")
                padded = old.new_zeros(new.shape)
                padded[:, :old.shape[1]] = old
                state[name] = padded
                padded_names.append(name)
    model.load_state_dict(state, strict=True)
    head_reset = reset_second_output_head(model, head_seed) if reset_second_head else None
    return {"source_mode": source_mode, "target_mode": target_mode,
            "source_first_features": source_first,
            "target_first_features": target_first,
            "zero_padded_weights": padded_names, "distance_added": distance_added,
            "source_second_output": source_output, "target_second_output": model.second_output,
            "second_head_reset": head_reset}


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


def _expand_inputs(model, names, added):
    """Append zero input weights without changing widths or consuming RNG."""
    import torch
    from torch import nn

    for name in names:
        sequence = getattr(model, name)
        old = sequence[0]
        new = nn.Conv2d(old.in_channels + added, old.out_channels,
                        old.kernel_size, old.stride, old.padding,
                        old.dilation, old.groups, old.bias is not None,
                        old.padding_mode, device="meta", dtype=old.weight.dtype)
        new.to_empty(device=old.weight.device)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_channels].copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        sequence[0] = new


def RadioWNet(inputs=2, phase="firstU", second_features="none", feature_config=None,
              first_features="none", second_output="relu", output_config=None):  # noqa: N802
    """Build the upstream model with input features and optional signed output.

    State keys and the upstream forward remain unchanged. A both-stage model
    expands firstU's three input convolutions by one channel, preserving hidden
    widths even for band=both. Distance uses an existing zero secondU slot, so
    its parameter count matches LOS. Feature preparation takes O(BHW) time and
    extra memory. At inputs=2, firstU adds only 579 parameters.
    """
    checkpoint_first_features({"args": {"first_features": first_features,
                                        "second_features": second_features}})
    if second_features != "none" and inputs < 2:
        raise ValueError("LOS requires height and TX input channels")
    if second_features != "none":
        checkpoint_features({"args": {"first_features": first_features,
                                      "second_features": second_features},
                             "feature_config": feature_config})
    elif feature_config is not None:
        raise ValueError("feature_config is only used by expanded models")
    checkpoint_output({"args": {"second_output": second_output},
                       "output_config": output_config})
    if second_output != "relu" and inputs != 2:
        raise ValueError("Linear/FSPL output modes require single-band height and TX inputs")
    model = load_module().RadioWNet(inputs=inputs, phase=phase)
    model.base_inputs = inputs
    model.first_features = first_features
    model.second_features = second_features
    model.feature_config = feature_config
    model.second_output = second_output
    model.output_config = output_config
    if first_features == "los":
        _expand_inputs(model, ("layer00", "conv_up00", "conv_up000"), 1)
        model.inputs = inputs + 1
    if second_features != "none":
        _expand_inputs(model, ("Wlayer00", "Wconv_up00", "Wconv_up000"), 3)
        model.register_forward_pre_hook(_append_los)
    if second_output != "relu":
        from torch import nn
        model.Wconv_up000[1] = nn.Identity()
    if second_output == "fspl_residual":
        model.register_forward_hook(_add_fspl)
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
