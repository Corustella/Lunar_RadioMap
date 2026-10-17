"""Standard FADE L2H upsampling with a memory-bounded PyTorch CARAFE.

Adapted from https://github.com/poppinace/fade/blob/main/
pytorch_implementation/FADE_L2H.py (Copyright (c) 2022 Hao Lu, MIT).
The license is reproduced in FADE_LICENSE. Kernel prediction and gating follow
the reference; only reassembly's tensor layout and workspace allocation differ.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


_PATCH_BYTES = 32 * 1024 * 1024


def _chunk_rows(x, kernel_size):
    # Bound one unfolded patch tensor, including training batch size. A single
    # row is the lower bound; output/kernel tensors are not part of this bound.
    b, c, h, w = x.shape
    row_bytes = b * c * kernel_size ** 2 * w * x.element_size()
    return max(1, min(h, _PATCH_BYTES // max(1, row_bytes)))


class _Reassemble(torch.autograd.Function):
    """Recompute local patches in backward instead of retaining all patches."""

    @staticmethod
    def forward(ctx, x, kernels, kernel_size):
        b, c, h, w = x.shape
        p = kernel_size // 2
        rows = _chunk_rows(x, kernel_size)
        padded = F.pad(x, (p, p, p, p))
        output = x.new_empty((b, c, 2 * h, 2 * w))
        # Explicit dtypes are also used in backward: einsum must not silently
        # change precision when called under the outer model's autocast context.
        with torch.autocast(device_type=x.device.type, enabled=False):
            for start in range(0, h, rows):
                stop = min(start + rows, h)
                patches = F.unfold(padded[:, :, start:stop + 2 * p],
                                   kernel_size=kernel_size)
                patches = patches.reshape(b, c, kernel_size ** 2,
                                          stop - start, w)
                weights = kernels[:, :, 2 * start:2 * stop].reshape(
                    b, kernel_size ** 2, stop - start, 2, w, 2)
                result = torch.einsum("bckhw,bkhrws->bchrws", patches, weights)
                output[:, :, 2 * start:2 * stop] = result.reshape(
                    b, c, 2 * (stop - start), 2 * w)
        ctx.save_for_backward(x, kernels)
        ctx.kernel_size = kernel_size
        ctx.rows = rows
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, kernels = ctx.saved_tensors
        b, c, h, w = x.shape
        k = ctx.kernel_size
        p = k // 2
        need_x, need_kernel = ctx.needs_input_grad[:2]
        # Accumulate overlapping row tiles in float32 for AMP inputs. Casting
        # back below preserves the input gradient dtype expected by autograd.
        accumulation_dtype = (torch.float32 if x.dtype in
                              (torch.float16, torch.bfloat16) else x.dtype)
        grad_padded = (torch.zeros((b, c, h + 2 * p, w + 2 * p),
                                   device=x.device, dtype=accumulation_dtype)
                       if need_x else None)
        grad_kernels = torch.empty_like(kernels) if need_kernel else None
        padded = F.pad(x, (p, p, p, p)) if need_kernel else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            for start in range(0, h, ctx.rows):
                stop = min(start + ctx.rows, h)
                length = stop - start
                grad = grad_output[:, :, 2 * start:2 * stop].reshape(
                    b, c, length, 2, w, 2)
                if need_kernel:
                    patches = F.unfold(padded[:, :, start:stop + 2 * p],
                                       kernel_size=k).reshape(b, c, k ** 2,
                                                              length, w)
                    grad_weights = torch.einsum("bckhw,bchrws->bkhrws",
                                                patches, grad)
                    grad_kernels[:, :, 2 * start:2 * stop] = (
                        grad_weights.reshape(b, k ** 2, 2 * length, 2 * w))
                    del patches, grad_weights
                if need_x:
                    weights = kernels[:, :, 2 * start:2 * stop].reshape(
                        b, k ** 2, length, 2, w, 2)
                    grad_patches = torch.einsum("bchrws,bkhrws->bckhw",
                                                grad, weights)
                    grad_tile = F.fold(grad_patches.reshape(b, c * k ** 2,
                                                           length * w),
                                       output_size=(length + 2 * p, w + 2 * p),
                                       kernel_size=k)
                    grad_padded[:, :, start:stop + 2 * p].add_(grad_tile)
        grad_x = None
        if need_x:
            grad_x = grad_padded[:, :, p:p + h, p:p + w].to(x.dtype)
        return grad_x, grad_kernels, None


class _SemiShift(nn.Module):
    def __init__(self, channels, embedding_dim, kernel_size):
        super().__init__()
        self.compressor_en = nn.Conv2d(channels, embedding_dim, 1)
        self.compressor_de = nn.Conv2d(channels, embedding_dim, 1, bias=False)
        self.content_encoder = nn.Conv2d(embedding_dim, kernel_size ** 2,
                                         kernel_size=3, padding=1)

    def forward(self, en, de):
        enc = self.compressor_en(en)
        dec = self.compressor_de(de)
        # The reference applies the shared content encoder twice, including its
        # bias both times. Moving the sum before this convolution changes that.
        return (self.content_encoder(enc) +
                F.interpolate(self.content_encoder(dec), scale_factor=2,
                              mode="nearest"))


class FADE(nn.Module):
    """Fuse equal-channel encoder/decoder features at exactly a 2x scale.

    The standard L2H predictor uses a 64-channel embedding and a shared 3x3
    content encoder. The default 5x5 kernels reassemble zero-padded decoder
    neighborhoods. Gates use the nearest-upsampled decoder's 1x1 logits.
    Reassembly supports first-order gradients, including AMP training; double
    backward is intentionally unsupported by its recomputation implementation.
    """

    def __init__(self, channels, kernel_size=5, embedding_dim=64):
        super().__init__()
        if channels < 1 or embedding_dim < 1:
            raise ValueError("FADE channels and embedding_dim must be positive")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("FADE kernel_size must be a positive odd integer")
        self.channels = channels
        self.kernel_size = kernel_size
        self.gate_generator = nn.Conv2d(channels, 1, 1)
        self.ker_generator = _SemiShift(channels, embedding_dim, kernel_size)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, encoder_high_res, decoder_low_res):
        en, de = encoder_high_res, decoder_low_res
        if (en.ndim != 4 or de.ndim != 4 or
                en.shape[:2] != de.shape[:2] or en.shape[1] != self.channels or
                en.shape[-2] != 2 * de.shape[-2] or
                en.shape[-1] != 2 * de.shape[-1]):
            raise ValueError("FADE expects equal N,C and encoder spatial size "
                             "exactly 2x decoder spatial size")
        if en.device != de.device or en.dtype != de.dtype:
            raise ValueError("FADE encoder and decoder must share device/dtype")
        gate = torch.sigmoid(F.interpolate(self.gate_generator(de),
                                           scale_factor=2, mode="nearest"))
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        # Under autocast the projections already have the same reduced dtype.
        # Outside it, kernels use the decoder dtype as in the upstream module.
        if kernels.dtype != de.dtype:
            de = de.to(kernels.dtype)
        return gate * en + (1 - gate) * _Reassemble.apply(
            de, kernels, self.kernel_size)
