"""CPU regression tests using tiny real Wan/VACE modules, not model weights."""

import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_roof_training as training


def settings():
    return dict(
        lora_base_model="vace", lora_target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
        lora_rank=16, lora_alpha=16, lora_dropout=0.0, lora_bias="none",
        lora_use_rslora=False, lora_use_dora=False, lora_init_lora_weights=True,
        lora_parameter_dtype="float32", backbone_dtype="bfloat16",
        learning_rate=5e-5, weight_decay=0.0, adam_betas=[0.9, 0.999], adam_eps=1e-8,
        scheduler_training_steps=1000, tiled=True, tile_size=[30, 52], tile_stride=[15, 26],
        gradient_checkpointing=True, gradient_checkpointing_offload=False,
        min_timestep_boundary=0, max_timestep_boundary=1, max_grad_norm=1.0,
        max_updates=4, save_updates=[2, 4],
    )


def tiny_pipe():
    from diffsynth.models.wan_video_dit import WanModel
    from diffsynth.models.wan_video_vace import VaceWanModel
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32)
    pipe.dit = WanModel(
        dim=32, in_dim=16, ffn_dim=64, out_dim=16, text_dim=8, freq_dim=16, eps=1e-6,
        patch_size=(1, 2, 2), num_heads=4, num_layers=2, has_image_input=False,
    )
    pipe.vace = VaceWanModel(vace_layers=(0,), dim=32, num_heads=4, ffn_dim=64)
    return pipe


class TinyVAE(torch.nn.Module):
    def encode(self, pixels, **kwargs):
        reduced = torch.nn.functional.avg_pool3d(pixels, (1, 8, 8))[:, :, ::4]
        return reduced.repeat(1, 6, 1, 1, 1)[:, :16]


class TrainingInputsTests(unittest.TestCase):
    def test_real_vace_unit_reads_mask_and_keeps_gt_out_of_condition(self):
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32)
        pipe.vae = TinyVAE()
        pipe.scheduler.set_timesteps(1000, training=True)
        sample = dict(
            gt=[Image.new("RGB", (16, 16), (150, 150, 150)) for _ in range(17)],
            conditioning=[Image.new("RGB", (16, 16), (90, 90, 90)) for _ in range(17)],
            model_mask=[Image.new("RGB", (16, 16), (0, 0, 0)) for _ in range(17)],
        )
        context = torch.ones(1, 4, 8)
        black = training.encode_sample(pipe, sample, settings(), context, 1.0)
        self.assertEqual(tuple(black["input_latents"].shape), (1, 16, 5, 2, 2))
        self.assertEqual(tuple(black["vace_context"].shape), (1, 96, 5, 2, 2))
        self.assertEqual(int(torch.count_nonzero(black["vace_context"][:, -64:])), 0)
        sample["gt"] = [Image.new("RGB", (16, 16), (220, 220, 220)) for _ in range(17)]
        changed_gt = training.encode_sample(pipe, sample, settings(), context, 1.0)
        self.assertFalse(torch.equal(black["input_latents"], changed_gt["input_latents"]))
        self.assertTrue(torch.equal(black["vace_context"], changed_gt["vace_context"]))
        for image in sample["model_mask"]:
            image.paste((255, 255, 255), (0, 0, 8, 8))
        changed_mask = training.encode_sample(pipe, sample, settings(), context, 1.0)
        self.assertFalse(torch.equal(changed_gt["vace_context"], changed_mask["vace_context"]))
        self.assertTrue(torch.equal(changed_gt["input_latents"], changed_mask["input_latents"]))
        self.assertGreater(int(torch.count_nonzero(changed_mask["vace_context"][:, -64:])), 0)
        self.assertFalse(changed_mask["vace_context"].requires_grad)
        self.assertFalse(changed_mask["input_latents"].requires_grad)
        self.assertNotIn("first_frame_latents", changed_mask)
        self.assertNotIn("vace_reference_image", changed_mask)


class TrainingGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261018)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_actual_flow_loss_backpropagates_through_frozen_dit_and_saves_deltas(self):
        pipe = tiny_pipe()
        pristine_vace = copy.deepcopy(pipe.vace)
        backbone_before = {name: parameter.detach().clone() for name, parameter in pipe.dit.named_parameters()}
        training.configure_lora(pipe, settings())
        adapter_before = {name: parameter.detach().clone()
                          for name, parameter in training.trainable_lora_parameters(pipe)}
        context = torch.randn(1, 4, 8)
        cached = [(pilot, dict(
            input_latents=torch.randn(1, 16, 1, 4, 4),
            vace_context=torch.randn(1, 96, 1, 4, 4),
            context=context, vace_scale=1.0, use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=False,
            min_timestep_boundary=0, max_timestep_boundary=1,
        )) for pilot in ("P04", "P06")]
        seen = []
        with tempfile.TemporaryDirectory(prefix="vace_roof_train_test_") as directory:
            result = training.train_updates(
                pipe, cached, settings(), directory,
                on_update=lambda update, loss, pilot: seen.append((update, pilot)),
            )
            self.assertEqual(result["actual_updates"], 4)
            self.assertTrue(result["gradient_check_passed"])
            self.assertTrue(result["parameters_changed"])
            self.assertEqual(seen, [(1, "P04"), (2, "P06"), (3, "P04"), (4, "P06")])
            self.assertEqual({path.name for path in Path(directory).iterdir()},
                             {"loss.csv", "lora_update0002.safetensors", "lora_update0004.safetensors"})
            with (Path(directory) / "loss.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample"] for row in rows], ["P04", "P06", "P04", "P06"])
            self.assertTrue(all(float(row["lr"]) == 5e-5 for row in rows))
            state = load_file(str(Path(directory) / "lora_update0004.safetensors"))
            self.assertTrue(all("lora_" in name and not name.startswith("vace.") for name in state))
            self.assertTrue(all(tensor.dtype == torch.float32 for tensor in state.values()))
            from diffsynth.utils.lora import GeneralLoRALoader
            GeneralLoRALoader().fuse_lora_to_base_model(pristine_vace, state, alpha=1.0)
            probe = torch.randn(2, 32)
            actual = pipe.vace.vace_blocks[0].self_attn.q(probe)
            reloaded = pristine_vace.vace_blocks[0].self_attn.q(probe)
            self.assertTrue(torch.allclose(actual, reloaded, atol=1e-6))
        self.assertTrue(all(torch.equal(backbone_before[name], parameter)
                            and not parameter.requires_grad for name, parameter in pipe.dit.named_parameters()))
        self.assertTrue(any(not torch.equal(adapter_before[name], parameter)
                            for name, parameter in training.trainable_lora_parameters(pipe)))

    def test_optimizer_rejects_trainable_backbone_parameter(self):
        pipe = tiny_pipe()
        training.configure_lora(pipe, settings())
        next(pipe.dit.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "outside VACE LoRA"):
            training.make_optimizer(pipe, settings())

    def test_nonstandard_lora_scaling_is_rejected(self):
        modified = settings()
        modified["lora_use_rslora"] = True
        with self.assertRaisesRegex(ValueError, "standard zero-delta"):
            training.configure_lora(tiny_pipe(), modified)

    def test_nonfinite_post_update_parameter_is_not_saved_and_actual_update_is_recorded(self):
        pipe = tiny_pipe()
        training.configure_lora(pipe, settings())
        parameter = next(parameter for _, parameter in training.trainable_lora_parameters(pipe))
        optimizer = training.make_optimizer(pipe, settings())
        original_step = optimizer.step

        def corrupt_step():
            original_step()
            with torch.no_grad():
                parameter.fill_(float("nan"))

        with tempfile.TemporaryDirectory(prefix="vace_roof_bad_parameter_") as directory:
            with patch.object(optimizer, "step", side_effect=corrupt_step), \
                    patch.object(training, "make_optimizer", return_value=optimizer), \
                    patch("diffsynth.diffusion.FlowMatchSFTLoss", side_effect=lambda *args, **kwargs: parameter.sum()):
                with self.assertRaisesRegex(ValueError, "Non-finite LoRA parameter"):
                    training.train_updates(pipe, [("P04", {})], settings(), directory)
            self.assertEqual([path.name for path in Path(directory).iterdir()], ["loss.csv"])
            with (Path(directory) / "loss.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["update"] for row in rows], ["1"])

    def test_nonfinite_loss_and_disconnected_loss_stop_without_checkpoint(self):
        for loss, expected in ((torch.tensor(float("nan"), requires_grad=True), "Non-finite loss"),
                               (torch.tensor(1.0), "Gradient chain disconnected")):
            with self.subTest(expected=expected):
                pipe = tiny_pipe()
                training.configure_lora(pipe, settings())
                with tempfile.TemporaryDirectory(prefix="vace_roof_train_stop_") as directory:
                    with patch("diffsynth.diffusion.FlowMatchSFTLoss", return_value=loss):
                        with self.assertRaisesRegex(ValueError, expected):
                            training.train_updates(pipe, [("P04", {})], settings(), directory)
                    self.assertEqual([path.name for path in Path(directory).iterdir()], ["loss.csv"])
                    with (Path(directory) / "loss.csv").open() as handle:
                        self.assertEqual(list(csv.DictReader(handle)), [])

    def test_zero_weight_first_timestep_is_allowed_but_all_zero_updates_are_rejected(self):
        for eventually_nonzero in (True, False):
            with self.subTest(eventually_nonzero=eventually_nonzero):
                pipe = tiny_pipe()
                training.configure_lora(pipe, settings())
                parameter = next(parameter for name, parameter in training.trainable_lora_parameters(pipe)
                                 if ".lora_B." in name)
                calls = []

                def loss_with_zero_weight(*arguments, **kwargs):
                    calls.append(True)
                    weight = 1 if eventually_nonzero and len(calls) > 1 else 0
                    return parameter.sum() * weight

                with tempfile.TemporaryDirectory(prefix="vace_roof_zero_weight_") as directory:
                    with patch("diffsynth.diffusion.FlowMatchSFTLoss", side_effect=loss_with_zero_weight):
                        if eventually_nonzero:
                            result = training.train_updates(pipe, [("P04", {})], settings(), directory)
                            self.assertTrue(result["gradient_check_passed"])
                            self.assertTrue(result["parameters_changed"])
                        else:
                            with self.assertRaisesRegex(ValueError, "No effective LoRA"):
                                training.train_updates(pipe, [("P04", {})], settings(), directory)
                    with (Path(directory) / "loss.csv").open() as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), 4)
                    self.assertEqual(float(rows[0]["loss"]), 0)


if __name__ == "__main__":
    unittest.main()
