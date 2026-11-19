"""Check that validation-only acceleration falls back and restores the model."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[3] / "code" / "DiffSynth-Studio"))
from diffsynth.models.wan_video_dit import RMSNorm
from validation_rmsnorm import exact_rmsnorm


class ValidationNormScope(unittest.TestCase):
    def test_unsupported_tensor_and_overridden_module_stay_on_original_paths(self):
        original = RMSNorm(16).eval()
        custom = RMSNorm(16).eval()
        custom.forward = lambda x: x + 1
        pipe = SimpleNamespace(dit=torch.nn.Sequential(original), vace=torch.nn.Sequential(custom),
                               device=torch.device("cuda:0"), training=False)
        x = torch.arange(16, dtype=torch.float32).reshape(1, 16)
        with torch.inference_mode():
            expected = original(x)
            with (patch.object(torch, "__version__", "2.11.0+cu128"),
                  patch.object(torch.cuda, "is_available", return_value=True),
                  patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
                  patch.object(torch.cuda, "get_device_name", return_value="NVIDIA RTX A6000"),
                  self.assertRaisesRegex(RuntimeError, "stop validation")):
                with exact_rmsnorm(pipe) as stats:
                    self.assertIn("forward", vars(original))
                    self.assertTrue(torch.equal(original(x), expected))
                    self.assertTrue(torch.equal(custom(x), x + 1))
                    self.assertEqual(stats["calls"], 0)
                    raise RuntimeError("stop validation")
        self.assertNotIn("forward", vars(original))
        self.assertTrue(torch.equal(custom(x), x + 1))
