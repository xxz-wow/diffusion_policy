import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from diffusion_policy.model.vision.musa_conv_weight_grad import enable_musa_conv_weight_grad


class TestConvWeightGrad(unittest.TestCase):
    def test_preserve_parameters_and_native_paths(self):
        m = nn.Sequential(nn.Conv2d(3, 8, (3, 5), stride=(1, 2), padding=(1, 2)),
                          nn.Conv2d(8, 8, 3, groups=8, padding=1))
        ref = copy.deepcopy(m)
        weight, bias, grouped = m[0].weight, m[0].bias, m[1]
        rng = torch.get_rng_state()
        enable_musa_conv_weight_grad(m)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIs(m[0].weight, weight)
        self.assertIs(m[0].bias, bias)
        self.assertIs(m[1], grouped)
        m.load_state_dict(ref.state_dict(), strict=True)
        x = torch.randn(2, 3, 13, 17)
        m(x).sum().backward()
        ref(x).sum().backward()
        for a, b in zip(m.parameters(), ref.parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad))

    def test_packed_gradient_on_odd_spatial_and_partial_batch_tiles(self):
        try:
            import torch_musa
            from diffusion_policy.model.vision.musa_conv_weight_kernel import grouped_weight_grad
        except ImportError:
            self.skipTest('MUSA and Triton required')
        if not torch.musa.is_available():
            self.skipTest('MUSA device required')
        with torch.backends.mudnn.flags(allow_tf32=False):
            torch.set_float32_matmul_precision('highest')
            rng = torch.Generator(device='musa').manual_seed(42)
            x = torch.randn(32, 32, 17, 23, device='musa', generator=rng)
            dy = torch.randn(32, 48, 17, 23, device='musa', generator=rng)
            weight = x.new_empty(48, 32, 3, 3)
            ref = torch.ops.aten.convolution_backward(dy, x, weight, None, (1, 1), (1, 1),
                                                       (1, 1), False, (0, 0), 1, (False, True, False))[1]
            result = grouped_weight_grad(x, dy, weight, (1, 1), (1, 1))
            torch.testing.assert_close(result, ref, rtol=2e-4, atol=2e-3)
            # Different MUSA versions may accumulate in a different order.
            # The policy must still return the native gradient bit for bit.
            layer = enable_musa_conv_weight_grad(nn.Conv2d(32, 48, 3, padding=1, bias=False).to('musa'))
            layer(x).backward(dy)
            self.assertTrue(torch.equal(layer.weight.grad.view(torch.int32),
                                        ref.contiguous().view(torch.int32)))
            layer.zero_grad(set_to_none=True)
            with patch('diffusion_policy.model.vision.musa_conv_weight_grad._choices', {}), \
                    patch('diffusion_policy.model.vision.musa_conv_weight_kernel.grouped_weight_grad',
                          side_effect=RuntimeError('unavailable')):
                layer(x).backward(dy)
                self.assertTrue(torch.equal(layer.weight.grad.view(torch.int32),
                                            ref.contiguous().view(torch.int32)))


if __name__ == '__main__':
    unittest.main()
