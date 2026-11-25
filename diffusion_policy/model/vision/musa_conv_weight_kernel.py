"""Pack receptive fields for a batched FP32 convolution weight gradient on MUSA."""
import torch
import triton
import triton.language as tl
from triton.errors import TritonError


@triton.jit
def _pack_nhwc(src, dst, N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
               OH: tl.constexpr, OW: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
               SH: tl.constexpr, SW: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
               B: tl.constexpr, K: tl.constexpr, CHANNEL_TILE: tl.constexpr, K_TILE: tl.constexpr):
    group = tl.program_id(2)
    col = tl.program_id(1) * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)
    pos = tl.program_id(0) * K_TILE + tl.arange(0, K_TILE)
    batch = group * B + pos // (OH * OW)
    y = (pos // OW % OH * SH)[None, :] + (col // KW % KH)[:, None] - PH
    x = (pos % OW * SW)[None, :] + (col % KW)[:, None] - PW
    valid = (batch < N) & (pos < B * OH * OW)
    channel = (col // (KH * KW))[:, None]
    offset = ((batch[None, :] * C + channel) * H + y) * W + x
    value = tl.load(src + offset, valid[None, :] & (col[:, None] < C * KH * KW)
                    & (y >= 0) & (y < H) & (x >= 0) & (x < W), other=0.)
    dest = (group * K + pos[None, :]) * (C * KH * KW) + col[:, None]
    tl.store(dst + dest, value, (pos[None, :] < K) & (col[:, None] < C * KH * KW))


def grouped_weight_grad(x, dy, weight, stride, padding):
    n, ci, _, _ = x.shape
    co, _, kh, kw = weight.shape
    oh, ow = dy.shape[-2:]
    # Match the backend's split-K partials before the final FP32 reduction.
    # Calibration checks this ordering for every new convolution signature.
    batch_group = max(1, min(n, 2048 // (oh * ow)))
    groups = triton.cdiv(n, batch_group)
    k = triton.cdiv(batch_group * oh * ow, 128) * 128
    cols = ci * kh * kw
    if max(x.numel(), dy.numel(), groups * k * max(co, cols), groups * co * cols) >= 2**31:
        raise ValueError('Packed convolution exceeds the 32-bit kernel index range')

    def pack(t, kernel, step, pad):
        _, c, h, w = t.shape
        width = c * kernel[0] * kernel[1]
        result = t.new_empty((groups, k, width))
        grid = (triton.cdiv(k, 64), triton.cdiv(width, 16), groups)
        try:
            _pack_nhwc[grid](t, result, n, c, h, w, oh, ow, *kernel, *step, *pad,
                             batch_group, k, 16, 64)
        except TritonError as err:
            raise RuntimeError('MUSA convolution packing kernel unavailable') from err
        return result

    grad = pack(dy.contiguous(), (1, 1), (1, 1), (0, 0))
    inputs = pack(x, (kh, kw), stride, padding)
    partial = torch.bmm(grad.transpose(1, 2), inputs)
    return partial.sum(0).reshape_as(weight)
