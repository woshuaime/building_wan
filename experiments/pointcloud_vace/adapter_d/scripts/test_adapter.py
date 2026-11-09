"""CPU checks for leakage boundaries, camera layout, injection and gradient flow."""
import unittest
from types import MethodType

import numpy as np
import torch

from adapter import MultiViewStructureAdapter, projections
from geometry import grouped_occupancy, orbit, project, fps, geometry_bundle
from integration import structured_vace_forward
from train import region_loss


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(72)

    def test_parameter_count_and_zero_projection(self):
        a = MultiViewStructureAdapter()
        self.assertEqual(sum(p.numel() for p in a.parameters()), 142929)
        self.assertEqual(sum(p.numel() for p in a.structure_proj.parameters()), 99840)
        self.assertTrue(torch.equal(a.structure_proj(torch.randn(2, 7, 64)), torch.zeros(2, 7, 1536)))

    def test_temporal_area_mapping(self):
        x = np.zeros((17, 3, 480, 832), np.float32)
        x[0, 0] = 1
        x[1:5, 1, :8, :8] = 1
        x[5:9, 2, :4, :4] = 1
        y = grouped_occupancy(x, 8)
        self.assertEqual(y.shape, (3, 5, 60, 104))
        self.assertEqual(float(y[0, 0].mean()), 1)
        self.assertEqual(y[1, 1, 0, 0], 1)
        self.assertEqual(y[2, 2, 0, 0], .25)
        self.assertTrue((y.sum(0) <= 1).all())
        self.assertEqual(grouped_occupancy(x[:, :1], 16).shape, (1, 5, 30, 52))

    def test_unknown_ignored_and_area_normalized(self):
        pred = torch.tensor([[[[[2., 4., 100.]]]]], requires_grad=True)
        labels = torch.tensor([[[[[1., 0., 0.]]], [[[0., 0., 0.]]], [[[0., 1., 0.]]]]])
        loss, terms = region_loss(pred, torch.zeros_like(pred), labels)
        self.assertAlmostEqual(loss.item(), 4 + .3 * 16, places=5)
        loss.backward()
        self.assertEqual(pred.grad.flatten()[-1], 0)
        self.assertEqual(terms[1], 0)

    def test_camera_projection_and_reference_alignment(self):
        points = np.random.default_rng(2).normal(size=(40, 3))
        cameras, scale = orbit(points)
        g = geometry_bundle(points, cameras, scale, [0])
        tensors = {k: torch.from_numpy(np.asarray(v).copy())[None] for k, v in g.items()}
        uv, depth, valid = projections(tensors["anchors"], tensors["ref_C"], tensors["ref_R"], tensors["K"])
        expected, z = project(points[fps(points)], cameras[0])
        np.testing.assert_allclose(uv[0, 0].numpy(), expected, atol=1.e-4)
        np.testing.assert_allclose(depth[0, 0].numpy(), z / scale, atol=1.e-6)
        self.assertTrue(valid.all())

    def test_global_fallback_and_target_layout(self):
        points = np.random.default_rng(4).uniform(-2, 2, (32, 3))
        cameras, scale = orbit(points)
        g = geometry_bundle(points, cameras, scale, list(range(17)))
        g = {k: torch.from_numpy(np.asarray(v).copy())[None] for k, v in g.items()}
        g["reference_views"].zero_()
        module = MultiViewStructureAdapter()
        tokens = module(g)
        self.assertEqual(tokens.shape, (1, 7800, 64))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(module.source(g).shape, (1, 33, 64))

    def test_actual_vace_zero_equivalence_and_two_step_gradients(self):
        from diffsynth.models.wan_video_vace import VaceWanModel
        model = VaceWanModel(vace_layers=(0, 2), dim=32, num_heads=4, ffn_dim=64)
        model.requires_grad_(False)
        x, cond, context = torch.randn(1, 4, 32), torch.randn(1, 96, 1, 4, 4), torch.randn(1, 3, 32)
        t, freqs = torch.randn(1, 6, 32), torch.ones(4, 1, 4, dtype=torch.complex64)
        original = model(x, cond, context, t, freqs)
        model.forward = MethodType(structured_vace_forward, model)
        projection = torch.nn.Linear(64, 32)
        torch.nn.init.zeros_(projection.weight)
        torch.nn.init.zeros_(projection.bias)
        encoder = torch.nn.Linear(3, 64)
        inputs = torch.randn(1, 4, 3)
        gate = torch.tensor([[[0.], [1.], [.5], [0.]]])
        optimizer = torch.optim.AdamW(list(projection.parameters()) + list(encoder.parameters()), lr=1.e-3)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(x, cond, context, t, freqs, use_gradient_checkpointing=True,
                           structure_tokens=encoder(inputs), structure_mask=gate, structure_projection=projection)
            if step == 0:
                for a, b in zip(original, output):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            loss = sum(v.square().mean() for v in output)
            loss.backward()
            self.assertGreater(float(projection.weight.grad.abs().sum()), 0)
            if step == 0:
                self.assertEqual(float(encoder.weight.grad.abs().sum()), 0)
            else:
                self.assertGreater(float(encoder.weight.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            optimizer.step()
        for a, b in zip(original, model(x, cond, context, t, freqs)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_source_geometry_is_mesh_independent(self):
        # Interface contains P/cameras only, no mesh or weak candidate input.
        import inspect
        self.assertEqual(list(inspect.signature(geometry_bundle).parameters), ["points", "cameras", "scale", "frame_ids"])
        self.assertEqual(list(inspect.signature(orbit).parameters), ["points"])

    def test_actual_vace_lora_targets_and_optimizer_ownership_without_weights(self):
        from diffsynth.models.wan_video_vace import VaceWanModel
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        from integration import attach_trainable_adapter, parameter_groups
        pipe = torch.nn.Module()
        with torch.device("meta"):
            pipe.vace = VaceWanModel()
            pipe.dit = torch.nn.Linear(1536, 1536)
        pipe.torch_dtype = torch.bfloat16
        pipe.device = "cpu"
        pipe.scheduler = FlowMatchScheduler(template="Wan")
        pipe.freeze_except = lambda names: pipe.requires_grad_(False)
        attach_trainable_adapter(pipe, 72)
        a, l = parameter_groups(pipe)
        self.assertEqual(len(l), 15 * 2 * 4 * 2)
        self.assertEqual(sum(p.numel() for _, p in l), 2949120)
        self.assertTrue(all(not p.requires_grad for p in pipe.dit.parameters()))
        self.assertEqual(sum(p.numel() for _, p in a + l), 3092049)


if __name__ == "__main__":
    unittest.main(verbosity=2)
