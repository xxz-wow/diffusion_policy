"""Select a bitwise-checked FP32 weight-gradient implementation at first use."""
import torch
from torch import nn
from torch.nn import functional as F

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.musa_conv_backward import _time


_choices = {}


def _strict_fp32():
    matmul = getattr(getattr(torch.backends, 'musa', None), 'matmul', None)
    return (not torch.backends.mudnn.allow_tf32 and not getattr(matmul, 'allow_tf32', False)
            and torch.get_float32_matmul_precision() == 'highest')


def _select(native, trial, x, dy):
    def matches(a, g):
        return torch.equal(native(a, g).contiguous().view(torch.int32),
                           trial(a, g).contiguous().view(torch.int32))

    if not matches(x, dy):
        return False
    rng = torch.Generator(device=x.device).manual_seed(641)
    probes = [torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=rng) for t in (x, dy)]
    if not matches(*probes):
        return False
    # Include packing, batched matrix multiply, and reduction in both timing orders.
    a, b, c, d = [_time(f, (x, dy)) for f in (native, trial, trial, native)]
    return max(b, c) < 0.92 * min(a, d)


class _ConvWeightGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        ctx.save_for_backward(x, weight)
        ctx.stride, ctx.padding = stride, padding
        ctx.bias_sizes = [weight.shape[0]] if bias is not None else None
        return F.conv2d(x, weight, bias, stride, padding)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        need_dx, _, need_db = ctx.needs_input_grad[:3]
        stride, padding = ctx.stride, ctx.padding

        def native(a, g, mask):
            return torch.ops.aten.convolution_backward(
                g, a, weight, ctx.bias_sizes, stride, padding, (1, 1), False, (0, 0), 1, mask)

        if (not torch.is_grad_enabled() and x.shape[0] and dy.numel()
                and _strict_fp32() and not torch.musa.is_current_stream_capturing()):
            key = (x.device, x.shape, x.stride(), weight.shape, weight.stride(), dy.shape, dy.stride(),
                   stride, padding, ctx.bias_sizes is not None, torch.are_deterministic_algorithms_enabled(),
                   tuple(t.data_ptr() % 256 for t in (x, weight, dy)))
            if _choices.get(key) is not False:
                try:
                    from diffusion_policy.model.vision.musa_conv_weight_kernel import grouped_weight_grad

                    def trial(a, g):
                        return grouped_weight_grad(a, g, weight, stride, padding)

                    if key not in _choices:
                        if len(_choices) >= 128:
                            _choices.pop(next(iter(_choices)))
                        with torch.musa.device(x.device):
                            _choices[key] = _select(lambda a, g: native(a, g, (False, True, False))[1],
                                                    trial, x, dy)
                    if _choices[key]:
                        dw = trial(x, dy)
                        dx, db = None, None
                        if need_dx or need_db:
                            dx, _, db = native(x, dy, (need_dx, False, need_db))
                        return dx, dw, db, None, None
                except (ImportError, RuntimeError, ValueError):
                    # Optional Triton backend or workspace can be unavailable.
                    _choices[key] = False
        return (*native(x, dy, (need_dx, True, need_db)), None, None)


class _MusaWeightConv(nn.Conv2d):
    def forward(self, x):
        if (x.device.type == 'musa' and self.training and torch.is_grad_enabled() and x.ndim == 4
                and x.dtype == self.weight.dtype == torch.float32 and self.weight.requires_grad
                and x.is_contiguous() and self.weight.is_contiguous() and _strict_fp32()
                and not torch.is_autocast_enabled('musa') and not torch.musa.is_current_stream_capturing()):
            return _ConvWeightGrad.apply(x, self.weight, self.bias, self.stride, self.padding)
        return super().forward(x)


def enable_musa_conv_weight_grad(encoder):
    """Tune ordinary dense Conv2d layers. Preserve existing specialized convolution implementations."""
    def eligible(m):
        return (type(m) is nn.Conv2d and m.groups == 1 and m.dilation == (1, 1)
                and m.padding_mode == 'zeros' and isinstance(m.padding, tuple))

    def replace(m):
        conv = _MusaWeightConv(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride,
                               padding=m.padding, bias=m.bias is not None, device='meta')
        conv.weight, conv.bias = m.weight, m.bias
        return conv.train(m.training)

    return replace_submodules(encoder, eligible, replace)
