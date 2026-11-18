"""Verify live reporting leaves recovery writes durable and probe comparisons meaningful."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import common
import training_speed_probe as probe
import training_telemetry as telemetry
import worker


class TelemetryWrites(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run = Path(directory.name).resolve()
        # The installer binds these functions for the life of its child process.
        for module, name in ((worker, "write_json"), (common, "write_json"), (worker, "nvml_snapshot")):
            binding = patch.object(module, name, getattr(module, name))
            binding.start()
            self.addCleanup(binding.stop)
        with patch.object(telemetry, "_DeviceMemory", autospec=True):
            telemetry.install_telemetry(self.run)

    def test_live_files_atomically_replaced_without_fsync(self):
        for name in ("progress.json", "current_batch.json"):
            path = self.run / name
            path.write_text('{"update": 1}')
            with self.subTest(name=name), patch.object(os, "fsync") as sync:
                with patch.object(os, "replace", side_effect=OSError("simulated publish failure")):
                    with self.assertRaises(OSError):
                        worker.write_json(path, {"update": 2})
                self.assertEqual(json.loads(path.read_text()), {"update": 1})
                worker.write_json(path, {"update": 2})
                self.assertEqual(json.loads(path.read_text()), {"update": 2})
                sync.assert_not_called()

    def test_failures_and_checkpoints_still_fsync_before_publish(self):
        real_replace = os.replace
        with patch.object(os, "fsync", wraps=os.fsync) as sync:
            def publish(source, target):
                sync.assert_called_once()
                real_replace(source, target)

            with patch.object(os, "replace", side_effect=publish):
                failure = self.run / "failure.json"
                common.write_json(failure, {"error": "data"})
                self.assertEqual(json.loads(failure.read_text()), {"error": "data"})
                sync.reset_mock()
                checkpoint = self.run / "checkpoints" / "resume_next.pt"
                worker.atomic_torch_save(checkpoint, {"cursor": 27})
                self.assertEqual(torch.load(checkpoint, map_location="cpu", weights_only=True), {"cursor": 27})


class MatchingUpdates(unittest.TestCase):
    def test_changed_gradients_updates_or_rng_rejected(self):
        original = torch.tensor([90., 100.])
        p = torch.nn.Parameter(original + torch.tensor([0.125, -0.25]))
        p.grad = torch.tensor([-0.75, 0.5])
        state = probe.endpoint({"adapter": [("weight", p)]}, {"weight": original})
        self.assertTrue(torch.equal(state["adapter_change"], torch.tensor([0.125, -0.25])))
        self.assertTrue(torch.equal(state["adapter_gradient"], p.grad))
        sample = dict(sample_index=27, id="3", timesteps=[32.], lr=1e-4, loss=0.1,
                      grad_norm=0.4, adapter_grad=0.2, lora_grad=0.3)
        baseline = dict(checkpoint_sha256="same", rng_hashes=["cpu", "device"], trajectory=[sample])
        self.assertTrue(probe.compare(state, state, baseline, baseline)["passed"])
        for group in state:
            with self.subTest(group=group):
                different = dict(state, **{group: state[group] * 1.05})
                self.assertFalse(probe.compare(state, different, baseline, baseline)["passed"])
        different_rng = dict(baseline, rng_hashes=["cpu", "other-device-state"])
        self.assertFalse(probe.compare(state, state, baseline, different_rng)["passed"])
        different_loss = dict(baseline, trajectory=[dict(sample, loss=0.5)])
        self.assertFalse(probe.compare(state, state, baseline, different_loss)["passed"])
