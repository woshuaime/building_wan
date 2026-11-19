import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validation_cfg_pair import FirstAttentionPair


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, freqs):
        self.calls += 1
        return x.square() + freqs


class TestFirstAttentionPair(unittest.TestCase):
    @torch.inference_mode()
    def test_matches_both_branches_and_forgets_previous_step(self):
        modules = [Attention(), Attention()]
        values = torch.tensor([1., 3.])
        freq = torch.tensor([2., -1.])
        with FirstAttentionPair(*modules) as pair:
            for n in (1, 2):
                timestep = torch.tensor(n)
                latents = values + n
                for negative in (False, True):
                    with pair.branch(latents, timestep, negative):
                        for m in modules:
                            self.assertTrue(torch.equal(m(latents.clone(), freq.clone()), latents.square() + freq))
            self.assertEqual((pair.hits, pair.misses), (4, 0))
            with self.assertRaises(RuntimeError), pair.branch(latents, timestep, negative=True):
                pass
        self.assertEqual([m.calls for m in modules], [2, 2])
        for m in modules:
            self.assertNotIn("forward", vars(m))

    @torch.inference_mode()
    def test_different_inputs_recompute_and_failed_pair_clears_state(self):
        dit, vace = Attention(), Attention()
        x = torch.ones(2)
        freq = torch.tensor([.25, .75])
        timestep = torch.tensor(1)
        with FirstAttentionPair(dit, vace) as pair:
            with pair.branch(x, timestep, False):
                dit(x, freq)
                vace(x, freq)
            with pair.branch(x, timestep, True):
                self.assertTrue(torch.equal(dit(x + 1, freq), (x + 1).square() + freq))
                self.assertTrue(torch.equal(vace(x, freq + 1), x.square() + freq + 1))
            self.assertEqual((pair.hits, pair.misses), (0, 2))
            with pair.branch(x, timestep, False):
                dit(x, freq)
                vace(x, freq)
            with self.assertRaises(RuntimeError), pair.branch(x, timestep.clone(), True):
                pass
            with self.assertRaises(RuntimeError), pair.branch(x, timestep, True):
                pass
            with self.assertRaises(ValueError), pair.branch(x, timestep, False):
                dit(x, freq)
                raise ValueError("interrupted")
            with self.assertRaises(RuntimeError), pair.branch(x, timestep, True):
                pass
