"""Capacity-matched A/D/S/DS input ablations of the existing T backbone."""
import torch
from torch import nn

from los_features import log_distance_feature
from lunar_conv import LunarConvAligned, conv_model_config
from mechanism_features import CHANNELS, channel_mask


def mechanism_model_config(arch):
    mode = arch.removeprefix("lunar_conv_tx_mechanism_")
    channel_mask(mode)
    config = conv_model_config("lunar_conv_tx_content")
    config.update(arch=arch,version=1,
                  external_input_channels=["H","TX","F",*CHANNELS],
                  network_input_channels=["H","TX","F","rho",*CHANNELS],
                  mechanism_mode=mode)
    config["encoder"].update(implementation="lunar_mechanisms.LunarConvMechanisms",
                             stem_input_channels=8,
                             initialization="shared_T_weights_and_RNG_four_zero_input_columns")
    config["full_resolution_branch"]["channels"][0] = 8
    config["output"]["physical_input_channels"] = 8
    config["mechanism_initialization"] = {
        "sites": ["encoder.stem.0.0","full_resolution.0","output.0"],
        "weights": "append_four_zero_columns_preserve_all_old_columns_and_bias",
        "rng": "fork_cpu_rng_while_widening", "disabled_inputs": "zero"}
    return config


def _widen(original):
    if not isinstance(original,nn.Conv2d) or original.groups != 1:
        raise ValueError("Expected the original dense input convolution")
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Conv2d(original.in_channels+4,original.out_channels,
            original.kernel_size,stride=original.stride,padding=original.padding,
            dilation=original.dilation,bias=original.bias is not None,
            padding_mode=original.padding_mode)
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:,:original.in_channels].copy_(original.weight)
        if original.bias is not None:
            expanded.bias.copy_(original.bias)
    return expanded


class LunarConvMechanisms(LunarConvAligned):
    def __init__(self,feature_config,arch):
        super().__init__(feature_config,arch="lunar_conv_tx_content")
        self.arch = arch
        self.model_config = mechanism_model_config(arch)
        self.mode = self.model_config["mechanism_mode"]
        if feature_config.get("mechanism_mode") != self.mode:
            raise ValueError("Architecture and mechanism feature mode disagree")
        self.encoder.stem[0][0] = _widen(self.encoder.stem[0][0])
        self.full_resolution[0] = _widen(self.full_resolution[0])
        self.output[0] = _widen(self.output[0])
        self.register_buffer("mechanism_mask",torch.tensor(channel_mask(self.mode))[None,:,None,None],
                             persistent=False)

    def forward_features(self,input):
        if input.ndim != 4 or tuple(input.shape[1:]) != (7,256,256):
            raise ValueError("D/S input must be Bx7x256x256 [H,TX,F,D_A,S_exist,S_gain,S_extra]")
        if not input.is_floating_point():
            raise ValueError("D/S model requires normalized floating-point input")
        distance = log_distance_feature(input,self.feature_config)
        extra = input[:,3:]*self.mechanism_mask.to(dtype=input.dtype)
        physical = torch.cat((input[:,:3],distance,extra),dim=1)
        features = [self.full_resolution(physical),self.encoder.stem(physical)]
        encoded = features[-1]
        for index,block in enumerate(self.encoder.blocks):
            encoded = block(encoded)
            if index == 1 and self.tx_scene_attention is not None:
                encoded = self.tx_scene_attention(encoded,input[:,1:2],input[:,:1])
            features.append(encoded)
        decoded = []
        for decoder,skip in zip(self.decoder,reversed(features[:-1])):
            encoded = decoder(encoded,skip)
            decoded.append(encoded)
        prediction = self.output(torch.cat((encoded,physical),dim=1))
        return prediction,physical,decoded[-3:]
