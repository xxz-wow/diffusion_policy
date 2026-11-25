import torch
from torch import nn
from torch.nn import functional as F

from diffusion_policy.common.pytorch_util import replace_submodules


# Keep only decisions, never training tensors. Device-specific timings are
# process-local and are not part of a model or its checkpoint.
_choices = {}


def _native(dy, x, weight, padding, dilation, mask):
    return torch.ops.aten.convolution_backward(
        dy, x, weight, None, (1, 1), padding, dilation, False, (0, 0), 1, mask)


def _flipped(dy, x, weight, padding, dilation):
    # For unit stride, dX is a convolution with exchanged channel axes.
    # Reversing the gradient and result preserves the spatial order of the weights.
    backward_pad = tuple(d * (k - 1) - p for d, k, p in
                         zip(dilation, weight.shape[-2:], padding))
    dx = F.conv2d(dy.flip((-2, -1)), weight.transpose(0, 1).contiguous(),
                  padding=backward_pad, dilation=dilation).flip((-2, -1))
    dw = _native(dy, x, weight, padding, dilation, (False, True, False))[1]
    return dx, dw


def _native_pair(dy, x, weight, padding, dilation):
    return _native(dy, x, weight, padding, dilation, (True, True, False))[:2]


def _matches(args):
    ref = _native_pair(*args)
    trial = _flipped(*args)
    return all(torch.equal(a.contiguous().view(torch.int32),
                           b.contiguous().view(torch.int32)) for a, b in zip(ref, trial))


def _time(fn, args):
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(4):
        fn(*args)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _select(dy, x, weight, padding, dilation):
    args = (dy, x, weight, padding, dilation)
    with torch.musa.device(x.device):
        if not _matches(args):
            return False
        # An all-zero first gradient cannot distinguish accumulation orders.
        # Probe dense data too without advancing the training RNG.
        rng = torch.Generator(device=x.device).manual_seed(641)

        def sample(t):
            return torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=rng)

        if not _matches((sample(dy), sample(x), sample(weight), padding, dilation)):
            return False
        # Both paths have already warmed up. Include weight gradients, layout
        # conversions and dispatch, and reverse the order to reduce timing bias.
        native_first = _time(_native_pair, args)
        trial_second = _time(_flipped, args)
        trial_first = _time(_flipped, args)
        native_second = _time(_native_pair, args)
        return max(trial_first, trial_second) < 0.95 * min(native_first, native_second)


class _ConvInputGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, padding, dilation):
        ctx.save_for_backward(x, weight)
        ctx.padding, ctx.dilation = padding, dilation
        return F.conv2d(x, weight, padding=padding, dilation=dilation)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        args = (dy, x, weight, ctx.padding, ctx.dilation)
        use_flip = False
        if (not torch.is_grad_enabled() and dy.is_contiguous()
                and not torch.backends.mudnn.allow_tf32
                and not torch.musa.is_current_stream_capturing()):
            key = (x.device, x.shape, x.stride(), weight.shape, weight.stride(),
                   dy.shape, dy.stride(), ctx.padding, ctx.dilation,
                   torch.are_deterministic_algorithms_enabled(),
                   tuple(t.data_ptr() % 256 for t in (x, weight, dy)))
            use_flip = _choices.get(key)
            if use_flip is None:
                try:
                    use_flip = _select(*args)
                except RuntimeError:
                    # Unsupported or insufficient workspace for an optional probe.
                    # The native call below still propagates its own failures.
                    use_flip = False
                if len(_choices) >= 128:
                    _choices.pop(next(iter(_choices)), None)
                _choices[key] = use_flip
        if use_flip:
            try:
                dx, dw = _flipped(*args)
                return dx, dw, None, None
            except torch.OutOfMemoryError:
                # A later batch can have less free workspace than calibration.
                _choices[key] = False
        dx, dw = _native_pair(*args)
        return dx, dw, None, None


class _MusaEncoderConv(nn.Conv2d):
    def forward(self, x):
        if (x.device.type == 'musa' and self.training and torch.is_grad_enabled()
                and x.ndim == 4 and x.dtype == torch.float32 and x.is_contiguous()
                and self.weight.dtype == torch.float32 and self.weight.is_contiguous()
                and x.requires_grad and self.weight.requires_grad
                and not torch.is_autocast_enabled('musa')
                and not torch.backends.mudnn.allow_tf32):
            return _ConvInputGrad.apply(x, self.weight, self.padding, self.dilation)
        return super().forward(x)


def enable_musa_conv_backward(encoder):
    """Tune eligible dense FP32 training convolutions on first use.

    Unit stride, ungrouped weights and nonnegative backward padding are required
    by the flipped-convolution implementation. No batch or channel size is assumed.
    """

    def eligible(m):
        return (type(m) is nn.Conv2d and m.stride == (1, 1) and m.groups == 1
                and isinstance(m.weight, nn.Parameter) and m.bias is None
                and m.padding_mode == 'zeros' and isinstance(m.padding, tuple)
                and all(p <= d * (k - 1) for p, d, k in zip(m.padding, m.dilation, m.kernel_size)))

    def replace(m):
        # Preserve the checkpoint keys, parameter identity and RNG sequence.
        conv = _MusaEncoderConv(m.in_channels, m.out_channels, m.kernel_size,
                                padding=m.padding, dilation=m.dilation, bias=False, device='meta')
        conv.weight = m.weight
        return conv.train(m.training)

    return replace_submodules(encoder, eligible, replace)
