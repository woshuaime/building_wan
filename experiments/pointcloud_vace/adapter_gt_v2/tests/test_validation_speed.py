"""CPU regression for qualifying candidates inside a live, sequential CFG schedule."""
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validation_speed


class ToyAttention(torch.nn.Module):
    def forward(self, x, frequencies):
        return x * frequencies


class PairProbe(TestCase):
    def sample(self, *, cost=0.5, mismatch=False, policy=None):
        dit_attn, vace_attn = ToyAttention(), ToyAttention()
        pipe = SimpleNamespace(dit=SimpleNamespace(blocks=[SimpleNamespace(self_attn=dit_attn)]),
                               device=torch.device("cpu"), training=False)
        core = SimpleNamespace(vace_blocks=[SimpleNamespace(self_attn=vace_attn)])
        selected = {} if policy is None else policy
        clock = 0.
        candidate_active = False
        calls = []
        selections = []

        @contextmanager
        def enable(_pipe):
            nonlocal candidate_active
            self.assertFalse(candidate_active)
            candidate_active = True
            try:
                yield
            finally:
                candidate_active = False

        def predict(latents, timestep, text):
            nonlocal clock
            calls.append((float(timestep), text, candidate_active))
            clock += cost if candidate_active else 1.
            x = latents + timestep
            freqs = torch.ones_like(x)
            pred = dit_attn(x, freqs) + vace_attn(x, freqs) + text
            if candidate_active and mismatch and text == -1:
                return pred + 1.
            return pred

        trajectory = []
        with (torch.inference_mode(),
              patch.object(validation_speed, "accelerated", enable),
              patch.object(validation_speed, "perf_counter", lambda: clock),
              patch.object(validation_speed.torch.cuda, "synchronize"),
              redirect_stdout(StringIO()),
              validation_speed.ValidationSpeed(pipe, core, selected, lambda p: selections.append(p.copy())) as gate):
            latents = torch.tensor([0.])
            for n in range(30):
                timestep = torch.tensor([float(n)])
                inp = dict(latents=latents, timestep=timestep)
                pos = gate.forward(predict, dict(inp, text=1))
                neg = gate.forward(predict, dict(inp, text=-1))
                # The scheduler only consumes the two returned predictions.
                latents = (neg + 5 * (pos - neg)) / 4.
                trajectory.append(latents.item())
            self.assertEqual(gate.branch_count, 60)
        self.assertFalse(candidate_active)
        self.assertNotIn("forward", vars(dit_attn))
        self.assertNotIn("forward", vars(vace_attn))
        return trajectory, calls, selections

    def test_faster_equal_candidate_uses_unchanged_thirty_step_schedule(self):
        reference, _, _ = self.sample(policy={"selected": "original"})
        trajectory, calls, saved = self.sample()
        self.assertEqual(trajectory, reference)
        self.assertEqual(len(calls), 64)  # two extra pairs, no extra scheduler steps
        self.assertEqual([p["selected"] for p in saved], ["accelerated"])
        self.assertTrue(all(active for step, _, active in calls if step >= 3))
        replay, calls, saved = self.sample(policy=saved[0])
        self.assertEqual(replay, reference)
        self.assertEqual(len(calls), 60)
        self.assertTrue(all(active for _, _, active in calls))
        self.assertEqual(saved, [])

    def test_equal_but_slower_candidate_reverts_after_warmup_and_timing(self):
        reference, _, _ = self.sample(policy={"selected": "original"})
        trajectory, calls, saved = self.sample(cost=1.1)
        self.assertEqual(trajectory, reference)
        self.assertEqual(len(calls), 64)
        self.assertEqual([p["selected"] for p in saved], ["original"])
        self.assertFalse(any(active for step, _, active in calls if step >= 3))

    def test_candidate_mismatch_never_advances_the_scheduler(self):
        reference, _, _ = self.sample(policy={"selected": "original"})
        trajectory, calls, saved = self.sample(mismatch=True)
        self.assertEqual(trajectory, reference)
        self.assertEqual(len(calls), 62)
        self.assertEqual([p["selected"] for p in saved], ["original"])
        self.assertFalse(any(active for step, _, active in calls if step >= 2))
