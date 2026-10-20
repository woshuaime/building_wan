"""CPU synthetic orchestration tests only; never start the GPU worker."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01
from test_vace_p01 import package_fixture
from test_vace_p01_results import write_fixture_video


class FrontendTests(unittest.TestCase):
    def arguments(self, output):
        return SimpleNamespace(output_dir=output, config=vace_p01.FROZEN_CONFIG, checkpoint=None,
                               memory_mode="full", composite=False)

    def test_modified_frozen_config_is_logged_as_failure_without_starting_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/vace/rejected_config"
            config = vace_p01.load_frozen_config()
            config["inference"]["seed"] = 123
            changed = root / "changed.json"
            changed.write_text(json.dumps(config))
            with patch.object(vace_p01, "ROOT", root), patch.object(vace_p01, "run_gpu") as worker, \
                 contextlib.redirect_stdout(io.StringIO()) as terminal, contextlib.redirect_stderr(io.StringIO()):
                status = vace_p01.main(["--run", "--config", str(changed), "--output-dir", str(output)])
            self.assertEqual(status, 1)
            worker.assert_not_called()
            self.assertIn("VACE P01 17-frame inference: FAILED", terminal.getvalue())
            self.assertNotIn("inference: SUCCESS", terminal.getvalue())
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "FAILED")
            self.assertIn("Frozen experiment config changed", manifest["error"])
            self.assertEqual(manifest["last_stage"]["stage"], "environment_and_input_preflight")
            self.assertFalse((output / "raw_prediction").exists())
            for path in manifest["paths"].values():
                self.assertTrue(Path(path).exists(), path)
            self.assertIn("Frozen experiment config changed", (output / "run.log").read_text())

    def test_missing_weights_are_logged_and_gpu_never_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = package_fixture(root)
            selected = vace_p01.read_inputs(config, root)
            output = root / "missing_model_run"
            with patch.object(vace_p01, "read_inputs", return_value=selected), \
                 patch.object(vace_p01, "inspect_models", return_value=dict(ready_for_gpu=False)), \
                 patch.object(vace_p01, "run_gpu") as worker, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = vace_p01.launch_run(self.arguments(output))
            self.assertEqual(status, 1)
            worker.assert_not_called()
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "FAILED")
            self.assertEqual(manifest["last_completed_stage"], "environment_and_input_preflight")
            self.assertFalse((output / "raw_prediction").exists())

    def test_success_requires_real_result_validation_and_keeps_GT_evaluation_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = package_fixture(root)
            selected = vace_p01.read_inputs(config, root)
            output = root / "synthetic_cpu_only_run"
            models = dict(ready_for_gpu=True, files_sha256=dict(checkpoint="synthetic_CPU_test_only"))

            def synthetic_cpu_worker(run_output, actual_config, frames, model_report, args, reserved=False):
                self.assertTrue(reserved)
                vace_p01.prepare_output(run_output, actual_config, frames, model_report, reserved=True)
                vace_p01.write_json(run_output / "runtime.json", dict(models=model_report,
                                    model_loading=dict(synthetic_CPU_test_only=True)))
                raw = run_output / "raw_prediction"
                raw.mkdir()
                for index in range(17):
                    Image.new("RGB", (512, 512), (80 + index,) * 3).save(raw / f"{index:03}.png")
                vace_p01.write_json(run_output / "prediction_raw_sha256.json",
                                    {path.name: vace_p01.sha256(path) for path in raw.iterdir()})
                write_fixture_video(run_output / "raw_prediction.mp4")
                (run_output / "prediction_raw").symlink_to("raw_prediction", target_is_directory=True)
                (run_output / "prediction_raw.mp4").symlink_to("raw_prediction.mp4")
                vace_p01.write_json(run_output / "worker_complete.json", dict(status="complete", prediction_frames=17,
                                    lora_loaded=False, GT_loaded_by_worker=False, synthetic_CPU_test_only=True))
                vace_p01.write_json(run_output / "exit_status.json", dict(worker_returncode=0, interrupted=False))
                return 0

            with patch.object(vace_p01, "read_inputs", return_value=selected), \
                 patch.object(vace_p01, "inspect_models", return_value=models), \
                 patch.object(vace_p01, "run_gpu", side_effect=synthetic_cpu_worker) as worker, \
                 contextlib.redirect_stdout(io.StringIO()) as terminal, contextlib.redirect_stderr(io.StringIO()):
                status = vace_p01.launch_run(self.arguments(output))
            self.assertEqual(status, 0)
            self.assertEqual(worker.call_count, 1)
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "SUCCESS")
            self.assertFalse(manifest["quality_accepted"])
            self.assertFalse(manifest["GT_model_input"])
            self.assertEqual(manifest["results"]["video_validation"]["frames"], 17)
            self.assertIn("VACE P01 17-frame inference: SUCCESS", terminal.getvalue())
            for path in manifest["paths"].values():
                self.assertTrue(Path(path).exists(), path)
            self.assertTrue((output / "comparison_before_after_B020.png").is_file())
            self.assertTrue((output / "result_summary.md").is_file())
            self.assertEqual((output / "input_damaged/000.png").read_bytes(), selected[0]["original_bytes"]["damaged"])

    def test_postprocessing_failure_is_not_reported_as_inference_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = package_fixture(root)
            selected = vace_p01.read_inputs(config, root)
            output = root / "cpu_invalid_output"
            with patch.object(vace_p01, "read_inputs", return_value=selected), \
                 patch.object(vace_p01, "inspect_models", return_value=dict(ready_for_gpu=True)), \
                 patch.object(vace_p01, "run_gpu", return_value=0), \
                 patch("vace_p01_results.generate_results", side_effect=ValueError("MP4 cannot be decoded")), \
                 contextlib.redirect_stdout(io.StringIO()) as terminal, contextlib.redirect_stderr(io.StringIO()):
                status = vace_p01.launch_run(self.arguments(output))
            self.assertEqual(status, 1)
            self.assertNotIn("inference: SUCCESS", terminal.getvalue())
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "FAILED")
            self.assertIn("cannot be decoded", manifest["error"])
            self.assertNotIn("raw_prediction_video", manifest["paths"])

    def test_final_report_write_failure_has_explicit_failure_and_log_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = package_fixture(root)
            selected = vace_p01.read_inputs(config, root)
            output = root / "cpu_report_failure"
            with patch.object(vace_p01, "read_inputs", return_value=selected), \
                 patch.object(vace_p01, "inspect_models", return_value=dict(ready_for_gpu=False)), \
                 patch.object(vace_p01, "finish_run", side_effect=OSError("Synthetic report write failure")), \
                 contextlib.redirect_stdout(io.StringIO()) as terminal, contextlib.redirect_stderr(io.StringIO()) as errors:
                status = vace_p01.launch_run(self.arguments(output))
            self.assertEqual(status, 1)
            self.assertIn("inference: FAILED", errors.getvalue())
            self.assertIn(str(output / "run.log"), errors.getvalue())
            self.assertNotIn("inference: SUCCESS", terminal.getvalue())
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "FAILED")
            self.assertEqual(manifest["exit_code"], 1)
            self.assertIn("Synthetic report write failure", manifest["finalization_error"])
            self.assertFalse((output / "raw_prediction").exists())


if __name__ == "__main__":
    unittest.main()
