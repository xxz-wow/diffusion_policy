"""End-of-backward FP32 synchronization for single-device MUSA DDP ranks."""
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn


class MusaDeferredGradientSync:
    """Average dense gradients in the original DDP buckets, once per backward.

    Wrap the DDP forward and backward together in step(). Call the optimizer
    afterwards and clear gradients with zero_grad(set_to_none=True). As with
    DDP's gradient_as_bucket_view, the synchronized gradients are bucket views.
    """

    def __init__(self, ddp):
        self.ddp = ddp
        self.group = ddp.process_group
        self.world_size = dist.get_world_size(self.group)
        self.buckets = []
        self.parameters = []
        # Dynamic unused-parameter detection keeps the original DDP bucket layout
        # fixed. Power-of-two divisors preserve the native FP32 predivision.
        if (self.world_size <= 1 or self.world_size & (self.world_size - 1)
                or not ddp.find_unused_parameters or ddp.static_graph
                or getattr(ddp, 'parameters_to_ignore', None)
                or getattr(ddp, 'mixed_precision', None) is not None
                or getattr(ddp, 'skip_all_reduce_unused_params', False)):
            return
        params = [p for p in ddp.parameters() if p.requires_grad]
        if (not params or params[0].device.type != 'musa'
                or any(p.device != params[0].device or p.dtype != torch.float32 for p in params)
                or any(isinstance(m, (nn.Embedding, nn.EmbeddingBag)) and m.sparse for m in ddp.modules())):
            return
        # Ask the reducer for its buffer geometry. Repartitioning the parameters
        # can change the summation order of the FP32 collective.
        get_buckets = getattr(ddp.reducer, '_get_zeros_like_grad_buckets', None)
        if get_buckets is None:
            return
        groups = sorted(get_buckets(), key=lambda b: b.index())
        buckets = [(b.buffer(), list(b.parameters()), list(b.gradients())) for b in groups]
        parameters = [p for _, ps, _ in buckets for p in ps]
        if (len(parameters) != len(params) or set(map(id, parameters)) != set(map(id, params))
                or any(len(ps) != len(vs) for _, ps, vs in buckets)):
            return
        self.buckets = buckets
        self.parameters = parameters

    @contextmanager
    def step(self):
        ddp = self.ddp
        if (not self.buckets or not torch.is_grad_enabled() or not ddp.require_backward_grad_sync
                or getattr(ddp, '_comm_hooks', ()) or getattr(getattr(ddp, '_join_config', None), 'enable', False)):
            yield
            return
        previous = ddp.require_forward_param_sync
        try:
            with ddp.no_sync():
                yield
            self._average()
        except BaseException:
            ddp.require_forward_param_sync = previous
            raise
        # no_sync suppresses the next forward's buffer broadcast. This backward
        # did synchronize, so restore the ordinary DDP state on success.
        ddp.require_forward_param_sync = True

    @torch.no_grad()
    def _average(self):
        used = [p.grad is not None for p in self.parameters]
        mask = torch.tensor(used, device=self.parameters[0].device, dtype=torch.int32)
        mask_request = dist.all_reduce(mask, group=self.group, async_op=True)
        buffers = []
        for buf, params, views in self.buckets:
            dest, src = [], []
            for p, view in zip(params, views):
                if p.grad is None:
                    view.zero_()
                else:
                    dest.append(view)
                    src.append(p.grad)
            if dest:
                torch._foreach_copy_(dest, src)
            buffers.append(buf)
        torch._foreach_div_(buffers, float(self.world_size))
        requests = [dist.all_reduce(buf, group=self.group, async_op=True) for buf in buffers]
        for work in requests:
            work.wait()
        mask_request.wait()
        # A rank with a locally unused parameter needs the global result if any
        # peer used it. A globally unused parameter must keep grad=None for AdamW.
        globally_used = iter(used if all(used) else mask.tolist())
        for _, params, views in self.buckets:
            for p, view in zip(params, views):
                if next(globally_used):
                    p.grad = view
