"""Architecture selection and strict checkpoint loading; legacy defaults stay intact."""

from radiounet import (RadioWNet, checkpoint_features, checkpoint_first_features,
                       feature_config_for_mode, load_model_state)


CONV_ARCHITECTURES = ("lunar_conv_aligned", "lunar_conv_local3", "lunar_conv_multiscale")
LUNAR_ARCHITECTURES = ("lunar_maxvit", "lunar_maxvit_fade", "lunar_maxvit_c",
                       *CONV_ARCHITECTURES)
REFINEMENT_ARCHITECTURES = ("radiownet_tx_direction", "radiownet_path_profile", "radiownet_biscale")
RADIO_ARCHITECTURES = ("radiownet", *REFINEMENT_ARCHITECTURES)
ARCHITECTURES = (*RADIO_ARCHITECTURES, *LUNAR_ARCHITECTURES)


def uses_los(arch, second_features="none"):
    return arch in LUNAR_ARCHITECTURES or second_features != "none"


def output_index(arch, phase):
    if arch in LUNAR_ARCHITECTURES:
        if phase != "single":
            raise ValueError(f"{arch} is a single-stage model; use --phase single")
        return 0
    if arch in REFINEMENT_ARCHITECTURES and phase != "secondU":
        raise ValueError(f"{arch} supports only --phase secondU")
    if arch not in RADIO_ARCHITECTURES or phase not in ("firstU", "secondU"):
        raise ValueError(f"Unsupported architecture/phase: {arch!r}/{phase!r}")
    return 0 if phase == "firstU" else 1


def model_feature_config(meta, arch, second_features="none"):
    if arch in RADIO_ARCHITECTURES:
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
    if arch in CONV_ARCHITECTURES:
        from lunar_conv import conv_model_config
        if config != conv_model_config(arch):
            raise ValueError(f"Unsupported or missing {arch} model_config")
    elif arch in LUNAR_ARCHITECTURES:
        from lunar_maxvit import lunar_model_config
        if config != lunar_model_config(arch):
            raise ValueError(f"Unsupported or missing {arch} model_config")
    if arch == "radiownet_tx_direction":
        from tx_direction import direction_model_config
        if config != direction_model_config():
            raise ValueError("Unsupported or missing TX direction model_config")
    if arch == "radiownet_path_profile":
        from path_profile import profile_model_config
        if config != profile_model_config():
            raise ValueError("Unsupported or missing path profile model_config")
    if arch == "radiownet_biscale":
        from biscale_reconstruction import reconstruction_model_config
        if config != reconstruction_model_config():
            raise ValueError("Unsupported or missing BiScale reconstruction model_config")
    return arch


def checkpoint_input_config(checkpoint):
    """Return legacy stage flags plus the exact physical input definition."""
    arch = checkpoint_architecture(checkpoint)
    if arch in RADIO_ARCHITECTURES:
        mode, config = checkpoint_features(checkpoint)
        first = checkpoint_first_features(checkpoint)
        if arch in REFINEMENT_ARCHITECTURES and (
                first != "los" or mode != "los_distance" or
                checkpoint.get("args", {}).get("phase") != "secondU"):
            raise ValueError(f"{arch} requires LOS firstU and LOS+distance secondU")
        return first, mode, config
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
    if arch in REFINEMENT_ARCHITECTURES:
        if inputs != 2 or first_features != "los" or second_features != "los_distance":
            raise ValueError(f"{arch} requires a single band, first_features=los, second_features=los_distance")
        if arch == "radiownet_tx_direction":
            from tx_direction import TXDirectionalRadioWNet as model_class
            from tx_direction import direction_model_config as config_function
        elif arch == "radiownet_path_profile":
            from path_profile import PathProfileRadioWNet as model_class
            from path_profile import profile_model_config as config_function
        else:
            from biscale_reconstruction import BiScaleRadioWNet as model_class
            from biscale_reconstruction import reconstruction_model_config as config_function
        model = RadioWNet(inputs=inputs, phase=phase, first_features=first_features,
                         second_features=second_features, feature_config=feature_config,
                         _model_class=model_class)
        model.arch = arch
        model.model_config = config_function()
        return model
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
    if arch in CONV_ARCHITECTURES:
        from lunar_conv import LunarConvAligned
        return LunarConvAligned(feature_config, arch=arch)
    from lunar_maxvit import LunarMaxViT
    return LunarMaxViT(feature_config, arch=arch)


def load_checkpoint(model, checkpoint, initialize=False):
    source = checkpoint_architecture(checkpoint)
    target = getattr(model, "arch", "radiownet")
    if target in REFINEMENT_ARCHITECTURES:
        if source == target:
            checkpoint_input_config(checkpoint)
            return load_model_state(model, checkpoint, initialize=initialize)
        if not initialize or source != "radiownet":
            raise ValueError(f"{target} resume/evaluation requires an exact architecture match")
        first, mode, _ = checkpoint_input_config(checkpoint)
        if first != "los" or mode not in ("los", "los_distance"):
            raise ValueError(f"Initialize {target} only from the established LOS RadioWNet weights")
        if target in ("radiownet_path_profile", "radiownet_biscale") and (
                mode != "los" or checkpoint.get("args", {}).get("phase") != "firstU"):
            raise ValueError(f"{target} initialization requires the historical LOS firstU checkpoint")
        state = checkpoint["model"].copy()
        added = []
        prefix = {"radiownet_tx_direction": "Wlayer20.0.direction_",
                  "radiownet_path_profile": "Wlayer10.0.profile_",
                  "radiownet_biscale": "Wreconstruction."}[target]
        for name, value in model.state_dict().items():
            if name.startswith(prefix):
                if name in state:
                    raise ValueError(f"Legacy checkpoint unexpectedly contains {target} weights")
                state[name] = value.detach().clone()
                added.append(name)
        result = load_model_state(model, {**checkpoint, "model": state}, initialize=True)
        result.update(source_arch=source, target_arch=target, added_operator_parameters=added)
        return result
    if source != target:
        raise ValueError(f"Cannot load {source} weights into {target}; train the new architecture from scratch")
    if target == "radiownet":
        return load_model_state(model, checkpoint, initialize=initialize)
    _, _, config = checkpoint_input_config(checkpoint)
    if config != model.feature_config or checkpoint["model_config"] != model.model_config:
        raise ValueError("Checkpoint physical scales or model configuration differ")
    model.load_state_dict(checkpoint["model"], strict=True)
    return {"source_arch": source, "target_arch": target, "strict": True}
