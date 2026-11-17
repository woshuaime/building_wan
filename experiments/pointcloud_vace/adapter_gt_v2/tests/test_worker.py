"""CPU checks for balanced supervision, official normalization and resume RNG."""
import ast
import importlib.util
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from einops import repeat

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("adapter_gt_v2_test_worker", SCRIPTS / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_regions_normalize_per_sample_before_time_weight_and_batch_mean(self):
        values = torch.tensor([[1., 3., 5., 7.], [2., 4., 6., 8.]])
        prediction = values[:, None, None, None].repeat(1, 2, 1, 1, 1).requires_grad_()
        target = torch.zeros_like(prediction)
        occupancy = torch.tensor([[[[[1., 1., 0., 0.]]], [[[0., 0., 1., 1.]]]],
                                  [[[[0., 0., 0., 0.]]], [[[1., 1., 1., 1.]]]]])
        loss, means, areas = worker.region_loss(prediction, target, occupancy, torch.tensor([2., .25]))
        self.assertAlmostEqual(float(loss.detach()), (21 * 2 + 30 * .25) / 2)
        torch.testing.assert_close(means, torch.tensor([[5., 37.], [0., 30.]]))
        torch.testing.assert_close(areas, torch.tensor([[2., 2.], [0., 4.]]))
        loss.backward()
        expected = torch.stack([values[0] * .25, values[1] * .03125])
        torch.testing.assert_close(prediction.grad[:, 0, 0, 0], expected)
        torch.testing.assert_close(prediction.grad[:, 1, 0, 0], expected)

    def test_fractional_area_is_not_replaced_with_binary_foreground(self):
        prediction = torch.tensor([1., 3., 5.]).reshape(1, 1, 1, 1, 3)
        foreground = torch.tensor([.1, .2, .9]).reshape(1, 1, 1, 1, 3)
        occupancy = torch.cat([foreground, 1 - foreground], dim=1)
        loss, _, _ = worker.region_loss(prediction, torch.zeros_like(prediction), occupancy, torch.ones(1))
        self.assertAlmostEqual(float(loss), (24.4 / 1.2 + 10.6 / 1.8) / 2, places=5)

    def test_learning_rate_tracks_sample_progress_through_batch_changes(self):
        warmup = worker.TOTAL * .02
        self.assertAlmostEqual(worker.learning_rate(warmup), 1.e-4)
        self.assertAlmostEqual(worker.learning_rate(worker.TOTAL), 1.e-5)
        self.assertGreater(worker.learning_rate(2), worker.learning_rate(1))
        self.assertAlmostEqual(worker.learning_rate(2), 2 * worker.learning_rate(1))
        values = [worker.learning_rate(x) for x in (warmup, 1000, 10000, 20000, worker.TOTAL)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        # Both histories consume six samples; the LR cannot depend on updates.
        self.assertEqual(worker.learning_rate(sum([1] * 6)), worker.learning_rate(sum([2, 2, 2])))

    def test_full_rng_roundtrip_restores_future_samples(self):
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        fake_cuda = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
        with patch.object(torch.cuda, "get_rng_state_all", return_value=fake_cuda), \
                patch.object(torch.cuda, "set_rng_state_all") as set_cuda:
            saved = worker.rng_state()
            expected = (random.random(), np.random.random(), torch.randint(0, 1000, (4,)), torch.randn(3))
            for _ in range(5):
                random.random()
                np.random.random()
                torch.rand(8)
            worker.restore_rng(saved)
            actual = (random.random(), np.random.random(), torch.randint(0, 1000, (4,)), torch.randn(3))
            self.assertEqual(expected[:2], actual[:2])
            torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
            torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
            set_cuda.assert_called_once_with(fake_cuda)

    def test_encode_uses_installed_official_bf16_full_range_normalization(self):
        # Extract the actual installed pure preprocessing method, avoiding a
        # model load or an approximation of its dtype/operation ordering.
        official = worker.ROOT / "code/DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py"
        tree = ast.parse(official.read_text())
        node = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                    for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "preprocess_image")
        namespace = dict(torch=torch, np=np, repeat=repeat)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(official), "exec"), namespace)
        official_preprocess = namespace["preprocess_image"]
        captured = []

        class VAE:
            def encode(self, value, **kwargs):
                captured.append(value[:, :, 0].clone())
                return value[:, :1, :5, ::8, ::8].clone()

        pipe = types.SimpleNamespace(device="cpu", torch_dtype=torch.bfloat16, vae=VAE())
        line = (np.arange(832) % 256).astype(np.uint8)
        frame = np.broadcast_to(line, (480, 832)).copy()
        frames = np.broadcast_to(frame, (17, 480, 832)).copy()
        partial = np.flip(frames, axis=2).copy()
        worker.encode_raw(pipe, dict(target=frames, partial=partial,
                          occupancy=np.zeros((2, 5, 60, 104), np.float32),
                          id="example", frame_ids=list(range(17)), cache_key="test"))
        for actual, source in zip(captured, (frames, partial)):
            rgb = np.repeat(source[0, ..., None], 3, axis=2)
            expected = official_preprocess(pipe, Image.fromarray(rgb))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertEqual(float(actual.min()), -1)
            self.assertEqual(float(actual.max()), 1)


if __name__ == "__main__":
    unittest.main()
