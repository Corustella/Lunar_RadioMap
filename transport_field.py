"""Differentiable overlapping total-PL fields, with optional safe edge targets.

Local sides are mathematical pieces of a dB field, never LOS/non-LOS powers.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def field_config():
    return {
        "version": 1, "patch_stride_px": 4, "patch_radius_px": 4, "patch_size_px": 9,
        "patch_centres": "0,4,...,252 on both axes; fold padding4 clips exterior",
        "parameters": ["a", "minus_bx", "minus_by", "jump", "plus_bx", "plus_by", "angle", "offset", "width"],
        "planes": "Pminus=a+0.1*tanh(bminus).u; Pplus=a+jump+0.1*tanh(bplus).u; u=offset/4",
        "normal": "(cos(pi*tanh(angle)),sin(pi*tanh(angle)))",
        "offset_px": "4*tanh(offset)", "width_px": "0.15+1.85*sigmoid(width); initial1px",
        "blend": "sigmoid((normal.dot(pixel_offset_vector)-boundary_offset_px)/width_px)",
        "fold_window": "separable cos(pi*offset/10)^2; positive at all 9x9 sites; normalized exact coverage",
        "output": "ReLU(normalized overlapping total PL); FP32 rendering under AMP",
        "boundary_diagnostic": "4*w*(1-w)*sigmoid((abs(local_plane_difference)*PLscale-10dB)/2dB); contrast_gate_detached",
        "boundary_target": "valid-valid adjacent PL jump>=10dB; mark both endpoints; score only complete3x3 valid stencils",
        "boundary_loss": "mean of available positive/negative BCE means; optional training-only; defaultweight0",
    }


class LocalFieldRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        row, col = torch.meshgrid(torch.arange(-4, 5, dtype=torch.float32),
                                  torch.arange(-4, 5, dtype=torch.float32), indexing="ij")
        window = torch.cos(math.pi * row / 10).square() * torch.cos(math.pi * col / 10).square()
        self.register_buffer("x", col.flatten()[None, :, None], persistent=False)
        self.register_buffer("y", row.flatten()[None, :, None], persistent=False)
        self.register_buffer("window", window.flatten()[None, :, None], persistent=False)
        weights = window.flatten()[None, :, None].expand(1, -1, 4096)
        coverage = F.fold(weights, (256, 256), kernel_size=9, padding=4, stride=4)
        self.register_buffer("coverage", coverage, persistent=False)

    def forward(self, raw, boundary_scale=None, diagnostics=False):
        if tuple(raw.shape[1:]) != (9, 64, 64):
            raise ValueError("LocalFieldRenderer requires Bx9x64x64 parameters")
        with torch.autocast(device_type=raw.device.type, enabled=False):
            a, mx, my, jump, px, py, angle, distance, width = raw.float().flatten(2).unbind(1)
            angle = math.pi * angle.tanh()
            nx, ny = angle.cos(), angle.sin()
            distance = 4 * distance.tanh()
            width = .15 + 1.85 * width.sigmoid()
            minus = a[:, None] + .1 * (mx.tanh()[:, None] * self.x + my.tanh()[:, None] * self.y) / 4
            plus = (a + jump)[:, None] + .1 * (px.tanh()[:, None] * self.x + py.tanh()[:, None] * self.y) / 4
            weight = torch.sigmoid((nx[:, None] * self.x + ny[:, None] * self.y - distance[:, None]) / width[:, None])
            pieces = minus + weight * (plus - minus)
            output = F.fold(pieces * self.window, (256, 256), kernel_size=9, padding=4, stride=4) / self.coverage
            boundary = None
            if boundary_scale is not None:
                contrast = ((plus - minus).abs() * boundary_scale - 10).div(2).sigmoid().detach()
                strength = 4 * weight * (1 - weight) * contrast
                boundary = F.fold(strength * self.window, (256, 256), kernel_size=9, padding=4, stride=4) / self.coverage
            parameters = None
            if diagnostics:
                parameters = {"minus_at_centre": a.reshape(-1, 1, 64, 64),
                              "plus_at_centre": (a + jump).reshape(-1, 1, 64, 64),
                              "normal_x": nx.reshape(-1, 1, 64, 64), "normal_y": ny.reshape(-1, 1, 64, 64),
                              "offset_px": distance.reshape(-1, 1, 64, 64), "width_px": width.reshape(-1, 1, 64, 64)}
        return F.relu(output), boundary, parameters


def boundary_targets(target, mask, scale):
    """Targets only, never input features; invalid fill values cannot make edges."""
    with torch.no_grad(), torch.autocast(device_type=target.device.type, enabled=False):
        y, valid = target.detach().float(), mask.detach() > .5
        threshold = (10 - 1e-4) / scale
        horizontal = valid[:, :, :, 1:] & valid[:, :, :, :-1] & ((y[:, :, :, 1:] - y[:, :, :, :-1]).abs() >= threshold)
        vertical = valid[:, :, 1:, :] & valid[:, :, :-1, :] & ((y[:, :, 1:, :] - y[:, :, :-1, :]).abs() >= threshold)
        edge = (F.pad(horizontal, (1, 0)) | F.pad(horizontal, (0, 1)) |
                F.pad(vertical, (0, 0, 1, 0)) | F.pad(vertical, (0, 0, 0, 1)))
        safe = F.avg_pool2d(valid.float(), 3, stride=1, padding=1) == 1
    return edge.float(), safe


def field_boundary_loss(probability, target, mask, scale):
    edge, safe = boundary_targets(target, mask, scale)
    with torch.autocast(device_type=probability.device.type, enabled=False):
        error = F.binary_cross_entropy(probability.float().clamp(1e-6, 1 - 1e-6), edge, reduction="none")
        positive, negative = safe & (edge > .5), safe & (edge <= .5)
        np, nn = positive.sum(), negative.sum()
        lp = (error * positive).sum() / np.clamp_min(1)
        ln = (error * negative).sum() / nn.clamp_min(1)
        return (lp * (np > 0) + ln * (nn > 0)) / ((np > 0).float() + (nn > 0).float()).clamp_min(1)
