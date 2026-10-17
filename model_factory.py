"""Architecture selection and strict checkpoint loading; legacy defaults stay intact."""

from radiounet import (RadioWNet, checkpoint_features, checkpoint_first_features,
                       feature_config_for_mode, load_model_state)


LUNAR_ARCHITECTURES = ("lunar_maxvit", "lunar_maxvit_fade")
ARCHITECTURES = ("radiownet", *LUNAR_ARCHITECTURES)


def uses_los(arch, second_features="none"):
    return arch in LUNAR_ARCHITECTURES or second_features != "none"


def output_index(arch, phase):
    if arch in LUNAR_ARCHITECTURES:
        if phase != "single":
            raise ValueError(f"{arch} is a single-stage model; use --phase single")
        return 0
    if arch != "radiownet" or phase not in ("firstU", "secondU"):
        raise ValueError(f"Unsupported architecture/phase: {arch!r}/{phase!r}")
    return 0 if phase == "firstU" else 1


def model_feature_config(meta, arch, second_features="none"):
    if arch == "radiownet":
        return feature_config_for_mode(meta, second_features)
    if arch not in LUNAR_ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}")
    from los_features import log_distance_config
    config = feature_config_for_mode(meta, "los")
    config["distance"] = log_distance_config(config)
    config["input_channels"] = ["H", "TX", "F", "rho"]
    return config


def checkpoint_architecture(checkpoint):
    """Old checkpoints are RadioWNet, never guess an architecture from shapes."""
    arch = checkpoint.get("args", {}).get("arch", "radiownet")
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unsupported checkpoint architecture {arch!r}")
    config = checkpoint.get("model_config")
    if config is not None and config.get("arch") != arch:
        raise ValueError("Checkpoint args and model_config disagree on architecture")
    if arch in LUNAR_ARCHITECTURES:
        from lunar_maxvit import lunar_model_config
        if config != lunar_model_config(arch):
            raise ValueError(f"Unsupported or missing {arch} model_config")
    return arch


def checkpoint_input_config(checkpoint):
    """Return legacy stage flags plus the exact physical input definition."""
    arch = checkpoint_architecture(checkpoint)
    if arch == "radiownet":
        mode, config = checkpoint_features(checkpoint)
        return checkpoint_first_features(checkpoint), mode, config
    saved = checkpoint.get("args", {})
    if (saved.get("first_features", "none") != "none" or
            saved.get("second_features", "none") != "none"):
        raise ValueError(f"{arch} checkpoints must not use RadioWNet stage flags")
    config = checkpoint.get("feature_config")
    if config is None:
        raise ValueError(f"{arch} checkpoint lacks physical input scales")
    if model_feature_config({**config, "simulation": config}, arch) != config:
        raise ValueError(f"Invalid {arch} feature_config")
    return "none", "none", config


def build_model(arch="radiownet", inputs=2, phase="firstU", feature_config=None,
                first_features="none", second_features="none"):
    output_index(arch, phase)
    if arch == "radiownet":
        model = RadioWNet(inputs=inputs, phase=phase,
                         second_features=second_features,
                         feature_config=feature_config, first_features=first_features)
        model.arch = arch
        model.model_config = {"arch": arch, "version": 1}
        return model
    if inputs != 2:
        raise ValueError(f"{arch} currently trains one band at a time; use --band 58 or 415")
    if first_features != "none" or second_features != "none":
        raise ValueError(f"{arch} already uses F and rho; omit --first-features/--second-features")
    if feature_config is None or model_feature_config(
            {**feature_config, "simulation": feature_config}, arch) != feature_config:
        raise ValueError(f"{arch} requires its exact physical input configuration")
    from lunar_maxvit import LunarMaxViT
    return LunarMaxViT(feature_config, arch=arch)


def load_checkpoint(model, checkpoint, initialize=False):
    source = checkpoint_architecture(checkpoint)
    target = getattr(model, "arch", "radiownet")
    if source != target:
        raise ValueError(f"Cannot load {source} weights into {target}; train the new architecture from scratch")
    if target == "radiownet":
        return load_model_state(model, checkpoint, initialize=initialize)
    _, _, config = checkpoint_input_config(checkpoint)
    if config != model.feature_config or checkpoint["model_config"] != model.model_config:
        raise ValueError("Checkpoint physical scales or model configuration differ")
    model.load_state_dict(checkpoint["model"], strict=True)
    return {"source_arch": source, "target_arch": target, "strict": True}
