"""CPU regressions for actual Adapter batch geometry and block policies."""
import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("adapter_gt_v2_test_integration", SCRIPTS / "integration.py")
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)


def geometry(seed):
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.rand((1, 8, 3), generator=generator)
    anchors[..., :2] -= .5
    anchors[..., 2] = 2 + anchors[..., 2] * .3
    views = torch.rand((1, 8, 2, 240, 416), generator=generator) * .1
    views[:, :, 0] += 2.1
    views[:, :, 1] = 1
    depths = torch.empty((1, 17, 2, 240, 416))
    depths[:, :, 0] = 2.15
    depths[:, :, 1] = 1
    return dict(
        reference_views=views, anchors=anchors,
        anchor_density=torch.rand((1, 8), generator=generator),
        ref_R=torch.eye(3).reshape(1, 1, 3, 3).expand(1, 8, 3, 3).clone(),
        ref_C=torch.randn((1, 8, 3), generator=generator) * .02,
        target_R=torch.eye(3).reshape(1, 1, 3, 3).expand(1, 17, 3, 3).clone(),
        target_C=torch.randn((1, 17, 3), generator=generator) * .02,
        target_depths=depths, depth_tolerance=torch.tensor([.4]),
        K=torch.tensor([[[514.7, 0., 415.5], [0., 514.7, 239.5], [0., 0., 1.]]]))


class AdapterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_parameter_count_and_zero_residual_initialization(self):
        model = integration._architecture()()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 142929)
        self.assertEqual(torch.count_nonzero(model.structure_proj.weight), 0)
        self.assertEqual(torch.count_nonzero(model.structure_proj.bias), 0)

    def test_independent_geometry_batch_outputs_and_all_parameter_gradients(self):
        torch.manual_seed(10)
        model = integration._architecture()()
        with torch.no_grad():
            model.structure_proj.weight.normal_(std=.01)
        one, two = geometry(101), geometry(202)
        both = {key: torch.cat([one[key], two[key]], dim=0) for key in one}
        independent_output, expected_grad = [], {}
        for sample in (one, two):
            model.zero_grad(set_to_none=True)
            tokens = model(sample)
            independent_output.append(tokens.detach())
            (model.structure_proj(tokens[:, ::257]).square().mean() / 2).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                expected_grad[name] = expected_grad.get(name, 0) + parameter.grad.clone()
        model.zero_grad(set_to_none=True)
        tokens = model(both)
        self.assertEqual(tuple(tokens.shape), (2, 7800, 64))
        torch.testing.assert_close(tokens, torch.cat(independent_output), rtol=2.e-5, atol=2.e-6)
        model.structure_proj(tokens[:, ::257]).square().mean().backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, expected_grad[name],
                                       rtol=2.e-4, atol=2.e-6, msg=name)
        self.assertGreater(float(model.cnn[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.view_score[0].weight.grad.abs().sum()), 0)

    def test_checkpoint_policy_selects_explicit_even_blocks(self):
        fake_pipeline = types.ModuleType("diffsynth.pipelines.wan_video")
        fake_diffsynth = types.ModuleType("diffsynth")
        fake_pipelines = types.ModuleType("diffsynth.pipelines")
        fake_diffsynth.pipelines = fake_pipelines
        fake_pipelines.wan_video = fake_pipeline
        modules = {"diffsynth": fake_diffsynth, "diffsynth.pipelines": fake_pipelines,
                   "diffsynth.pipelines.wan_video": fake_pipeline}
        pipe = types.SimpleNamespace(dit=types.SimpleNamespace(blocks=[torch.nn.Linear(2, 2) for _ in range(30)]),
                                     vace=types.SimpleNamespace(vace_blocks=[torch.nn.Linear(2, 2) for _ in range(15)]))
        with patch.dict(sys.modules, modules):
            selected = integration.configure_checkpointing(pipe, "partial")
            self.assertEqual(selected["blocks"]["dit"], list(range(0, 30, 2)))
            self.assertEqual(selected["blocks"]["vace"], list(range(0, 15, 2)))
            self.assertFalse(selected["use_reentrant"])
            integration.configure_checkpointing(pipe, "none")
            self.assertFalse(any(block._adapter_gt_v2_checkpoint for block in pipe.dit.blocks))
            integration.configure_checkpointing(pipe, "full")
            self.assertTrue(all(block._adapter_gt_v2_checkpoint for block in pipe.dit.blocks))

    def test_model_entry_refuses_implicit_text_or_time_broadcast(self):
        fake = types.ModuleType("diffsynth.pipelines.wan_video")
        fake.model_fn_wan_video = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {"diffsynth.pipelines.wan_video": fake}):
            with self.assertRaisesRegex(ValueError, "same batch"):
                integration.model_fn_wan_video(latents=torch.zeros(2, 16, 5, 2, 2),
                                               context=torch.zeros(1, 2, 4), timestep=torch.ones(2))
            with self.assertRaisesRegex(ValueError, "same batch"):
                integration.model_fn_wan_video(latents=torch.zeros(2, 16, 5, 2, 2),
                                               context=torch.zeros(2, 2, 4), timestep=torch.ones(1))

    def test_installed_nonreentrant_checkpoint_preserves_trainable_internal_gradients(self):
        path = SCRIPTS.parents[3] / "code/DiffSynth-Studio/diffsynth/core/gradient/gradient_checkpoint.py"
        spec = importlib.util.spec_from_file_location("adapter_gt_v2_vendor_checkpoint_test", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        original = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.SiLU(), torch.nn.Linear(5, 2))
        checkpointed = copy.deepcopy(original)
        frozen_input = torch.tensor([[.1, .3, -.5], [.7, -.2, .4]])
        expected = original(frozen_input)
        expected.square().mean().backward()
        actual = helper.gradient_checkpoint_forward(checkpointed, True, False, frozen_input)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected)
        for actual_parameter, expected_parameter in zip(checkpointed.parameters(), original.parameters()):
            self.assertIsNotNone(actual_parameter.grad)
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


if __name__ == "__main__":
    unittest.main()
