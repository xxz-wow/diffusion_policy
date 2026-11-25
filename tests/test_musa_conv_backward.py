import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from diffusion_policy.model.vision import musa_conv_backward as conv


class TestMusaConvBackward(unittest.TestCase):
    def test_replacement_preserves_geometry_parameters_and_native_fallback(self):
        block = nn.Sequential(nn.Conv2d(3, 8, (3, 5), padding=(0, 2), dilation=(2, 1), bias=False),
                              nn.Conv2d(8, 8, 3, stride=2, bias=False))
        reference = copy.deepcopy(block)
        weight, strided = block[0].weight, block[1]
        rng = torch.get_rng_state()
        conv.enable_musa_conv_backward(block)
        assert torch.equal(torch.get_rng_state(), rng)
        assert isinstance(block[0], conv._MusaEncoderConv)
        assert block[0].weight is weight and block[1] is strided
        block.load_state_dict(reference.state_dict(), strict=True)
        x = torch.randn(2, 3, 17, 23)
        assert torch.equal(block(x), reference(x))

    def test_zero_first_gradient_does_not_bypass_validation(self):
        try:
            import torch_musa
        except ImportError:
            self.skipTest('MUSA extension required')
        if not torch.musa.is_available():
            self.skipTest('MUSA device required')
        m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False)).to('musa')
        reference = copy.deepcopy(m)
        conv.enable_musa_conv_backward(m)

        # A candidate that happens to pass on all-zero gradients must be rejected.
        def bad_candidate(dy, x, weight, padding, dilation):
            dx, dw = conv._native_pair(dy, x, weight, padding, dilation)
            return torch.zeros_like(dx), dw

        with torch.backends.mudnn.flags(allow_tf32=False), patch.object(conv, '_choices', {}), \
                patch.object(conv, '_flipped', bad_candidate):
            for first in (True, False):
                x = torch.randn(2, 3, 11, 15, device='musa', requires_grad=True)
                u = x.detach().clone().requires_grad_()
                a, b = m(x), reference(u)
                dy = torch.zeros_like(a) if first else torch.randn_like(a)
                state = torch.musa.get_rng_state()
                a.backward(dy)
                b.backward(dy)
                self.assertTrue(torch.equal(state, torch.musa.get_rng_state()))
                self.assertTrue(torch.equal(x.grad, u.grad))
                self.assertTrue(torch.equal(m[0].weight.grad, reference[0].weight.grad))
                self.assertEqual(list(conv._choices.values()), [False])


if __name__ == '__main__':
    unittest.main()
