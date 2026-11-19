"""CPU equivalence checks for VACE hint storage and the training diagnostics."""
import sys
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[3] / "code" / "DiffSynth-Studio"))

import integration
import training_compute_candidate as candidate
from diffsynth.models.wan_video_vace import VaceWanAttentionBlock


class HintStorage(unittest.TestCase):
    def test_fifteen_original_blocks_and_all_gradients(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261119)
            blocks = torch.nn.ModuleList([
                VaceWanAttentionBlock(False, dim=32, num_heads=2, ffn_dim=48, block_id=layer)
                for layer in range(0, 30, 2)
            ])
            def leaf(*shape):
                return torch.randn(*shape, requires_grad=True)

            c, x = leaf(1, 12, 32), leaf(1, 12, 32)
            context, t_mod = leaf(1, 5, 32), leaf(1, 6, 32)
            angles = torch.rand(12, 1, 8, dtype=torch.float64)
            freqs = torch.polar(torch.ones_like(angles), angles)
            cotangents = [torch.randn_like(c) for _ in blocks]

        inputs = [("c", c), ("x", x), ("context", context), ("t_mod", t_mod),
                  *blocks.named_parameters()]
        packed = c
        for block in blocks:
            packed = block(packed, x, context, t_mod, freqs)
        expected = packed.unbind()[:-1]
        actual = candidate.direct_hints(blocks, c, x, context, t_mod, freqs)
        self.assertEqual(len(actual), 15)
        for index, (old, new) in enumerate(zip(expected, actual)):
            with self.subTest(hint=index):
                self.assertTrue(torch.equal(old, new))

        params = [p for _, p in inputs]
        reference_grads = torch.autograd.grad(expected, params, grad_outputs=cotangents)
        candidate_grads = torch.autograd.grad(actual, params, grad_outputs=cotangents)
        for (name, _), old, new in zip(inputs, reference_grads, candidate_grads):
            with self.subTest(gradient=name):
                self.assertTrue(torch.equal(old, new))

    def test_dispatch_preserves_fallback_and_restores_pipe_on_failure(self):
        with torch.random.fork_rng(devices=[]):
            blocks = torch.nn.ModuleList([
                VaceWanAttentionBlock(False, dim=32, num_heads=2, ffn_dim=48, block_id=layer)
                for layer in range(0, 30, 2)
            ])
        for block in blocks:
            block._adapter_gt_v2_checkpoint = False
        core = SimpleNamespace(vace_blocks=blocks)
        core.forward = MethodType(integration.structured_vace_forward, core)
        pipe = SimpleNamespace(vace=core, adapter_gt_v2_checkpointing={"mode": "none"})
        batch = (torch.zeros(1),)
        old_forward = core.forward

        def fail(*args):
            self.assertIs(core.forward.__func__, candidate._direct_forward)
            raise RuntimeError("simulated model error")

        with patch.object(candidate, "_candidate_update", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "simulated model error"):
                candidate._dispatch(pipe, None, [], [], batch, 100, 101)
        self.assertIs(core.forward, old_forward)
        with patch.object(candidate, "_original_update", return_value="fallback") as original:
            for mode, size in (("full", 1), ("partial", 1), ("none", 2)):
                pipe.adapter_gt_v2_checkpointing["mode"] = mode
                with self.subTest(mode=mode, batch=size):
                    self.assertEqual(candidate._dispatch(pipe, None, [], [], (torch.zeros(size),), 100, 101), "fallback")
            pipe.adapter_gt_v2_checkpointing["mode"] = "none"
            with blocks[0].register_forward_hook(lambda _m, _i, _o: None):
                self.assertEqual(candidate._dispatch(pipe, None, [], [], batch, 100, 101), "fallback")
            self.assertEqual(original.call_count, 4)


class GradientDiagnostics(unittest.TestCase):
    @staticmethod
    def parameter(values, grad):
        p = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        if grad is not None:
            p.grad = torch.tensor(grad, dtype=torch.float32)
        return p

    @staticmethod
    def reference(adapter, lora, weights):
        def norm(parameters):
            return sum(float(p.grad.detach().float().square().sum())
                       for _, p in parameters if p.grad is not None) ** .5

        groups = {"adapter": norm(adapter), "lora": norm(lora)}
        upstream = norm([(name, p) for name, p in adapter if "structure_proj" not in name])
        return groups, upstream, bool(torch.cat(weights).sum() > 0)

    def test_stats_match_original_reductions_and_zero_weight_decision(self):
        adapter = [
            ("structure_proj.weight", self.parameter([1., -1.], [3000., -4000.])),
            ("encoder.weight", self.parameter([1., 0.], [0.0003, -0.0019])),
            ("encoder.bias", self.parameter([0.], None)),
        ]
        lora = [("lora_A", self.parameter([1., 0.], [-0.31, 0.88])),
                ("lora_B", self.parameter([0.], [0.]))]
        cases = [
            ("mixed", adapter, lora, [torch.tensor([0., .5])]),
            ("zero weight", adapter, lora, [torch.zeros(1)]),
            ("zero upstream", adapter[:1], lora, [torch.ones(1)]),
            ("missing adapter gradients", adapter[2:], lora, [torch.ones(1)]),
            ("zero lora gradient", adapter, lora[1:], [torch.ones(1)]),
        ]
        for name, a, l, weights in cases:
            with self.subTest(name=name):
                self.assertEqual(candidate.gradient_stats(a, l, weights), self.reference(a, l, weights))

    def test_parameter_finiteness_and_precision(self):
        valid = torch.ones(3, dtype=torch.float32)
        values = [torch.ones(2), torch.tensor([float("inf")]), torch.tensor([float("-inf")]),
                  torch.tensor([float("nan")]), torch.ones(2, dtype=torch.bfloat16)]
        for value in values:
            params = [valid, value]
            expected = all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in params)
            with self.subTest(value=value):
                self.assertEqual(candidate.parameters_valid(params), expected)


if __name__ == "__main__":
    unittest.main()
