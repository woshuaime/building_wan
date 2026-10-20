"""CPU-only source selection, GT isolation and visible-stage scheduling tests."""

import contextlib
import copy
import io
import json
import signal
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_roof_data as data
import vace_roof_smoke as smoke


class WindowTests(unittest.TestCase):
    def test_maximum_contiguous_window_and_earliest_tie(self):
        indices, score = data.best_window([0] * 12 + [10] * 17 + [0] * 51)
        self.assertEqual(indices, list(range(12, 29)))
        self.assertEqual(score, 170)
        self.assertEqual(data.best_window([1] * 80), (list(range(17)), 17))
        self.assertEqual(data.best_window([0] * 63 + [1] * 17)[0], list(range(63, 80)))
        with self.assertRaisesRegex(ValueError, "excluding B080"):
            data.best_window([1] * 81)

    def test_frozen_experiment_budget_and_prompt(self):
        config = data.load_config()
        self.assertEqual([item["source_indices"] for item in config["samples"]], [list(range(17))] * 2)
        self.assertEqual(config["training"]["save_updates"], [100, 300])
        self.assertEqual(config["evaluation"]["maximum_new_inferences"], 6)


class SourceTests(unittest.TestCase):
    def test_original_png_loader_uses_gt_only_for_supervision(self):
        config = data.load_config()
        sample = copy.deepcopy(config["samples"][0])
        sample["window_white_pixels"] = 17 * 8 * 8
        config["samples"] = [sample]
        config["source"]["archive"] = "source.zip"
        prefix = config["source"]["package_id"] + "/"
        root = prefix + sample["relative_root"] + "/"
        metadata = dict(sample_id=sample["sample_id"], source=dict(sha256="identity"))
        cameras = dict(sample_id=sample["sample_id"], frames=81, resolutions={"512": [
            dict(frame=index, frame_id=f"{index:03}", width=512, height=512) for index in range(81)]})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with zipfile.ZipFile(directory / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(prefix + "manifest.json", json.dumps({
                    "samples": [dict(sample, source_sha256="identity")]}))
                bundle.writestr(root + "sample_meta.json", json.dumps(metadata))
                bundle.writestr(root + "cameras.json", json.dumps(cameras))
                for index in range(17):
                    for role in ("damaged", "exact", "gt"):
                        image = Image.new("RGB", (512, 512), (index, 21, 43) if role == "damaged" else (91, 121, 171))
                        if role == "exact":
                            image = Image.new("L", (512, 512), 0)
                            image.paste(255, (100, 100, 108, 108))
                        with io.BytesIO() as stream:
                            image.save(stream, format="PNG")
                            bundle.writestr(root + f"512/{role}/{index:03}.png", stream.getvalue())
            original_read = zipfile.ZipFile.read

            def no_gt_read(bundle, name, *args, **kwargs):
                self.assertNotIn("/512/gt/", name)
                return original_read(bundle, name, *args, **kwargs)

            with patch.object(data, "ROOT", directory), patch.object(zipfile.ZipFile, "read", no_gt_read):
                inference_sample = data.load_samples(config, include_gt=False)[0]
            self.assertNotIn("gt", inference_sample)
            for index in range(17):
                pixels = np.asarray(inference_sample["conditioning"][index])
                np.testing.assert_array_equal(pixels[100, 100], [127, 127, 127])
                np.testing.assert_array_equal(pixels[0, 0], [index, 21, 43])
            with patch.object(data, "ROOT", directory):
                training_sample = data.load_samples(config, include_gt=True)[0]
            np.testing.assert_array_equal(np.asarray(training_sample["gt"][0])[0, 0], [91, 121, 171])
            self.assertEqual(len(training_sample["gt"]), 17)

    def test_inference_arguments_cannot_include_gt(self):
        config = data.load_config()
        sample = dict(conditioning=[object()] * 17, model_mask=[object()] * 17)
        arguments = smoke.inference_arguments(config, sample)
        self.assertIs(arguments["vace_video"], sample["conditioning"])
        self.assertIs(arguments["vace_video_mask"], sample["model_mask"])
        self.assertNotIn("input_video", arguments)
        self.assertIsNone(arguments["vace_reference_image"])
        with self.assertRaisesRegex(ValueError, "must not read GT"):
            smoke.inference_arguments(config, dict(sample, gt=[object()] * 17))


class LauncherTests(unittest.TestCase):
    def test_stage_lora_targets_only_vace_with_multiplier_one(self):
        import torch
        from safetensors.torch import save_file

        vace = torch.nn.Module()
        vace.self_attn = torch.nn.Module()
        vace.self_attn.q = torch.nn.Linear(4, 4)
        pipe = SimpleNamespace(vace=vace, dit=torch.nn.Linear(4, 4), load_lora=Mock())
        state = {"self_attn.q.lora_A.default.weight": torch.ones(16, 4),
                 "self_attn.q.lora_B.default.weight": torch.ones(4, 16)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "delta.safetensors"
            save_file(state, str(path))
            smoke.load_stage_lora(pipe, path, data.load_config())
            self.assertIs(pipe.load_lora.call_args.args[0], vace)
            self.assertEqual(pipe.load_lora.call_args.kwargs["alpha"], 1.0)
            self.assertFalse(pipe.load_lora.call_args.kwargs["hotload"])
            save_file({"dit." + key: value for key, value in state.items()}, str(path))
            with self.assertRaisesRegex(ValueError, "exactly match the VACE"):
                smoke.load_stage_lora(pipe, path, data.load_config())
            self.assertEqual(pipe.load_lora.call_count, 1)

    def test_terminal_termination_has_explicit_exit_code(self):
        for signum, expected in ((signal.SIGINT, 130), (signal.SIGTERM, 143), (signal.SIGHUP, 129)):
            with self.assertRaises(smoke.InterruptedRun) as raised:
                smoke.stop_on_signal(signum, None)
            self.assertEqual(raised.exception.exit_code, expected)

    def test_default_and_dry_run_never_start_gpu(self):
        with patch.object(smoke.data, "load_config", return_value={}), \
                patch.object(smoke.data, "prepare") as prepare, \
                patch.object(smoke, "execute") as execute, patch.object(smoke, "worker") as worker:
            self.assertEqual(smoke.main([]), 0)
            self.assertEqual(smoke.main(["--dry-run"]), 0)
            self.assertEqual(prepare.call_count, 2)
            execute.assert_not_called()
            worker.assert_not_called()
        with self.assertRaises(SystemExit):
            smoke.parser().parse_args(["--run", "--dry-run"])

    def test_native_training_failure_stops_before_lora_inference_and_preserves_checkpoint(self):
        config = data.load_config()
        called = []

        def fake_worker(command, environment, descriptor):
            stage = command[command.index("--_worker") + 1]
            called.append(stage)
            experiment_path = Path(command[command.index("--experiment") + 1])
            if stage == "train":
                (experiment_path.parent / "loss.csv").write_text(
                    "update,sample,loss,lr\n1,P04,0.2,0.00005\n2,P06,0.1,0.00005\n", encoding="utf-8")
                (experiment_path.parent / "lora_update0100.safetensors").write_bytes(b"test-preserved-checkpoint")
                return -6, False
            experiment = smoke.read_experiment(experiment_path)
            experiment["runtime"]["stages"][stage] = {"status": "worker_finished_pending_exit"}
            smoke.write_experiment(experiment_path, experiment)
            return 0, False

        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryFile() as lock:
            with patch.object(data, "ROOT", Path(temporary)), \
                    patch.object(data, "prepare", return_value=([], {}, {})), \
                    patch.object(smoke.baseline, "gpu_lock", return_value=contextlib.nullcontext(lock)), \
                    patch.object(smoke.baseline, "check_old_training"), \
                    patch.object(smoke, "code_version", return_value={}), \
                    patch.object(smoke, "visible_worker", side_effect=fake_worker):
                self.assertEqual(smoke.execute(config), 134)
            run = next((Path(temporary) / config["evaluation"]["output_root"]).iterdir())
            experiment = smoke.read_experiment(run / "training/experiment.json")
            self.assertEqual(called, ["base", "train"])
            self.assertEqual(experiment["runtime"]["status"], "failed")
            self.assertEqual(experiment["runtime"]["actual_updates"], 2)
            self.assertEqual(experiment["runtime"]["exit_code"], -6)
            self.assertEqual((run / "training/lora_update0100.safetensors").read_bytes(),
                             b"test-preserved-checkpoint")
            self.assertFalse((run / "04").exists())
            self.assertFalse((run / "06").exists())


if __name__ == "__main__":
    unittest.main()
