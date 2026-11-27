import unittest

import torch
from torch import nn

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion


def make_model():
    return TransformerForDiffusion(input_dim=8, output_dim=8, horizon=5, n_obs_steps=2,
                                   cond_dim=6, n_layer=1, n_cond_layers=1, n_head=4,
                                   n_emb=32, p_drop_attn=0.3, causal_attn=True)


class TestMusaTransformerAttention(unittest.TestCase):
    def test_cpu_decoder_matches_pytorch_with_the_same_checkpoint_and_dropout(self):
        layer = make_model().decoder.layers[0]
        reference = nn.TransformerDecoderLayer(32, 4, 128, dropout=0.3, activation='gelu',
                                               batch_first=True, norm_first=True)
        reference.load_state_dict(layer.state_dict(), strict=True)
        x, memory = torch.randn(2, 5, 32), torch.randn(2, 3, 32)
        rng = torch.get_rng_state()
        expected = reference(x, memory)
        torch.set_rng_state(rng)
        self.assertTrue(torch.equal(layer(x, memory), expected))

    def test_single_visible_key_has_zero_query_and_key_gradients(self):
        try:
            import torch_musa
        except ImportError:
            self.skipTest('MUSA extension required')
        if not torch.musa.is_available():
            self.skipTest('MUSA device required')

        model = make_model().to('musa').train()
        encoder, decoder = model.encoder.layers[0], model.decoder.layers[0]
        x = torch.randn(8, 5, 32, device='musa')
        memory = torch.randn(8, 3, 32, device='musa')
        cases = [(encoder._sa_block, encoder.self_attn, (x,), 5),
                 (decoder._sa_block, decoder.self_attn, (x,), 5),
                 (decoder._mha_block, decoder.multihead_attn, (x, memory), 3)]
        for block, attn, inputs, num_keys in cases:
            with self.subTest(block=block.__qualname__):
                # With exactly one visible key, the softmax probability is always one.
                mask = x.new_full((5, num_keys), float('-inf'))
                mask[:, 0] = 0
                output = block(*inputs, mask, None)
                grad, = torch.autograd.grad(output.sum(), attn.in_proj_weight)
                dq, dk, dv = grad.chunk(3)
                self.assertEqual(dq.abs().max().item(), 0)
                self.assertEqual(dk.abs().max().item(), 0)
                self.assertGreater(dv.abs().max().item(), 0)
