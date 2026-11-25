"""Distributed regression: torchrun --nproc_per_node=8 tests/test_musa_ddp.py."""
import copy
import os
import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from diffusion_policy.common.musa_ddp import MusaDeferredGradientSync


class Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(24, 32)
        self.extra = nn.Linear(32, 32)
        self.unused = nn.Parameter(torch.ones(8))
        self.register_buffer('offset', torch.zeros(1))

    def forward(self, x, extra):
        h = self.shared(x) + self.offset
        if extra:
            h = self.extra(h)
        return h.square().mean()


class TestMusaDeferredGradients(unittest.TestCase):
    def test_dynamic_unused_buffers_and_resume(self):
        world = int(os.environ.get('WORLD_SIZE', '1'))
        if world < 2 or world & (world - 1):
            self.skipTest('Run under torchrun on a power-of-two number of MUSA devices')
        try:
            import torch_musa
        except ImportError:
            self.skipTest('MUSA extension required')
        rank = int(os.environ['LOCAL_RANK'])
        torch.musa.set_device(rank)
        dist.init_process_group('mccl')
        self.addCleanup(dist.destroy_process_group)
        torch.manual_seed(137)
        device = torch.device('musa', rank)

        def wrap(model):
            return DistributedDataParallel(model.to(device), device_ids=[rank],
                                           bucket_cap_mb=0.002, find_unused_parameters=True)

        baseline = wrap(Branches())
        trial = wrap(copy.deepcopy(baseline.module))
        opt = torch.optim.AdamW(baseline.parameters())
        trial_opt = torch.optim.AdamW(trial.parameters())
        schedule = MusaDeferredGradientSync(trial)
        self.assertTrue(schedule.buckets)
        for i in range(5):
            # Alternate rank-dependent, globally unused, and universally used gradients.
            use_branch = (i % 3 == 0 and rank % 2 == 0) or i % 3 == 2
            x = torch.arange(24, dtype=torch.float32, device=device).reshape(1, -1) / 31 + rank / 9
            for mod in (baseline.module, trial.module):
                mod.offset.fill_(rank + i / 13)
            expected = baseline(x, use_branch)
            expected.backward()
            with schedule.step():
                actual = trial(x, use_branch)
                actual.backward()
            self.assertTrue(torch.equal(actual, expected))
            for a, b in zip(baseline.parameters(), trial.parameters()):
                self.assertEqual(a.grad is None, b.grad is None)
                if a.grad is not None:
                    self.assertTrue(torch.equal(a.grad.view(torch.int32), b.grad.view(torch.int32)))
            for optimizer in (opt, trial_opt):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for a,b in zip(baseline.parameters(),trial.parameters()):
                self.assertTrue(torch.equal(a,b))
            if i==2:
                checkpoint = copy.deepcopy((trial.module.state_dict(), trial_opt.state_dict()))
                trial = wrap(Branches())
                trial.module.load_state_dict(checkpoint[0], strict=True)
                trial_opt = torch.optim.AdamW(trial.parameters())
                trial_opt.load_state_dict(checkpoint[1])
                schedule = MusaDeferredGradientSync(trial)


if __name__ == '__main__':
    unittest.main()
