"""CPU-only checks for the frozen P01 adapter, provenance and execution gates."""

import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def package_fixture(root):
    config = copy.deepcopy(vace_p01.load_frozen_config())
    config["source"]["archive"] = "fixture.zip"
    frame_root = config["source"]["sample_relative_root"]
    frames = [dict(frame_index=index, frame_id=f"{index:03}", camera_index=index,
                   **{role: f"{frame_root}/512/{role}/{index:03}.png" for role in ("gt", "damaged", "exact")})
              for index in range(81)]
    camera = dict(resolutions={"512": [dict(frame=index, frame_id=f"{index:03}") for index in range(81)]})
    camera_payload = json.dumps(camera).encode()
    payloads = {frame_root + "/cameras.json": camera_payload}
    inventory = []
    for index in range(5, 22):
        exact = Image.new("L", (512, 512), 0)
        exact.paste(255, (100, 120, 180, 200))
        images = dict(damaged=Image.new("RGB", (512, 512), (60, 90, index)),
                      gt=Image.new("RGB", (512, 512), (245, 5, 5)), exact=exact)
        for role, image in images.items():
            name = frames[index][role]
            payloads[name] = png_bytes(image)
            inventory.append(dict(relative_path=name, sha256=hashlib.sha256(payloads[name]).hexdigest()))
    manifest = dict(source_identity=dict(source_id=465, stable_id="b465_2e6fc746"),
                    frame_count=81, frames=frames, files=inventory,
                    original_source_context=dict(cameras_sha256=hashlib.sha256(camera_payload).hexdigest()))
    prefix = config["source"]["package_id"] + "/"
    with zipfile.ZipFile(root / "fixture.zip", "w") as bundle:
        bundle.writestr(prefix + "manifest.json", json.dumps(manifest))
        for name, content in payloads.items():
            bundle.writestr(prefix + name, content)
    config["source"]["archive_sha256"] = vace_p01.sha256(root / "fixture.zip")
    return config


class FrozenConfigTests(unittest.TestCase):
    def test_real_config_keeps_frozen_parameters(self):
        config = vace_p01.load_frozen_config()
        self.assertEqual(config["inference"]["seed"], 20261018)
        self.assertEqual(config["inference"]["num_inference_steps"], 30)
        self.assertEqual(config["inference"]["num_frames"], 17)
        self.assertEqual(config["source"]["frame_start_inclusive"], 5)
        self.assertIsNone(config["model"]["lora_checkpoint"])

    def test_modified_seed_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config = vace_p01.load_frozen_config()
            config["inference"]["seed"] = 1
            path = Path(directory) / "modified.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Frozen experiment"):
                vace_p01.load_frozen_config(path)

    def test_path_traversal_is_refused(self):
        for value in ("../gt.png", "/gt.png", "inputs\\gt.png"):
            with self.assertRaises(ValueError):
                vace_p01.relative_path(value)


class AdapterTests(unittest.TestCase):
    def test_white_pixels_are_gray_and_black_pixels_keep_damaged(self):
        damaged = Image.new("RGB", (512, 512), (10, 50, 90))
        exact = Image.new("L", (512, 512), 0)
        exact.paste(255, (0, 0, 100, 200))
        mask, condition = vace_p01.adapt_images(damaged, exact)
        self.assertEqual(mask.mode, "RGB")
        self.assertEqual(mask.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(condition.getpixel((0, 0)), (127, 127, 127))
        self.assertEqual(condition.getpixel((200, 200)), damaged.getpixel((200, 200)))
        self.assertEqual(exact.mode, "L")
        self.assertEqual(damaged.getpixel((0, 0)), (10, 50, 90))

    def test_legal_all_black_mask_is_not_changed(self):
        damaged = Image.new("RGB", (512, 512), (10, 20, 30))
        mask, condition = vace_p01.adapt_images(damaged, Image.new("L", (512, 512), 0))
        self.assertFalse(np.any(np.asarray(mask)))
        self.assertEqual(condition.tobytes(), damaged.tobytes())

    def test_nonbinary_mask_or_shape_change_is_refused(self):
        for exact in (Image.new("L", (512, 512), 128), Image.new("L", (256, 256), 0)):
            with self.assertRaises(ValueError):
                vace_p01.adapt_images(Image.new("RGB", (512, 512)), exact)

    def test_model_call_does_not_include_GT_or_initial_video(self):
        config = vace_p01.load_frozen_config()
        arguments = vace_p01.inference_arguments(config, ["condition"] * 17, ["mask"] * 17)
        self.assertEqual(arguments["seed"], 20261018)
        self.assertEqual(arguments["cfg_scale"], 5)
        self.assertIsNone(arguments["vace_reference_image"])
        self.assertNotIn("input_image", arguments)
        self.assertNotIn("input_video", arguments)
        self.assertNotIn("fps", arguments)
        self.assertEqual(len(arguments["vace_video_mask"]), 17)

    def test_composite_is_distinct_and_preserves_known_pixels(self):
        prediction = Image.new("RGB", (512, 512), (210, 220, 230))
        damaged = Image.new("RGB", (512, 512), (10, 20, 30))
        mask = Image.new("RGB", (512, 512), (0, 0, 0))
        mask.paste((255, 255, 255), (0, 0, 100, 100))
        composited = vace_p01.composite_frame(prediction, damaged, mask)
        self.assertEqual(composited.getpixel((0, 0)), prediction.getpixel((0, 0)))
        self.assertEqual(composited.getpixel((200, 200)), damaged.getpixel((200, 200)))
        self.assertEqual(prediction.getpixel((200, 200)), (210, 220, 230))


class PreparationTests(unittest.TestCase):
    def test_package_to_saved_inputs_keeps_alignment_hashes_and_GT_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = package_fixture(root)
            original_hash = vace_p01.sha256(root / "fixture.zip")
            selected = vace_p01.read_inputs(config, root)
            self.assertEqual([item["source_index"] for item in selected], list(range(5, 22)))
            output = root / "prepared"
            vace_p01.prepare_output(output, config, selected, dict(ready_for_gpu=False))
            records = json.loads((output / "model_input_manifest.json").read_text())
            self.assertEqual(len(records), 17)
            for record in records:
                self.assertEqual(set(record), {"source_index", "output_index", "conditioning", "model_mask"})
                self.assertNotIn("gt", json.dumps(record))
            for relative in ("inputs/damaged", "inputs/exact", "inputs/model_mask", "inputs/conditioning", "evaluation/gt"):
                self.assertEqual(len(list((output / relative).glob("*.png"))), 17)
            self.assertEqual((output / "inputs/damaged/000.png").read_bytes(), selected[0]["original_bytes"]["damaged"])
            self.assertEqual((output / "evaluation/gt/000.png").read_bytes(), selected[0]["original_bytes"]["gt"])
            inventory = json.loads((output / "artifacts_sha256.json").read_text())
            for relative, checksum in inventory.items():
                self.assertEqual(vace_p01.sha256(output / relative), checksum)
            self.assertFalse((output / "prediction_raw").exists())
            report = json.loads((output / "prepare_report.json").read_text())
            self.assertFalse(report["GPU_inference_started"])
            self.assertFalse(report["prediction_created"])
            self.assertEqual(vace_p01.sha256(root / "fixture.zip"), original_hash)
            with self.assertRaises(FileExistsError):
                vace_p01.prepare_output(output, config, selected, dict(ready_for_gpu=False))
            for relative, checksum in inventory.items():
                self.assertEqual(vace_p01.sha256(output / relative), checksum)

    def test_wrong_archive_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = package_fixture(root)
            config["source"]["archive_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "ZIP SHA256"):
                vace_p01.read_inputs(config, root)

    def test_gpu_run_refuses_missing_models_before_lock_or_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "never-created"
            with patch.object(vace_p01, "gpu_lock") as lock, patch.object(vace_p01.subprocess, "Popen") as worker:
                with self.assertRaisesRegex(ValueError, "not verified"):
                    vace_p01.run_gpu(output, {}, [], dict(ready_for_gpu=False), SimpleNamespace())
                lock.assert_not_called()
                worker.assert_not_called()
            self.assertFalse(output.exists())

    def test_shared_gpu_lock_rejects_second_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            with vace_p01.gpu_lock(Path(directory)):
                with self.assertRaises(RuntimeError):
                    with vace_p01.gpu_lock(Path(directory)):
                        self.fail("Second launcher acquired a held lock")

    def test_worker_native_failure_or_missing_completion_is_not_success(self):
        for returncode, marker, expected, interrupted in ((-11, True, 139, False), (0, False, 1, False),
                                                         (0, True, 130, True)):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = package_fixture(root)
                selected = vace_p01.read_inputs(config, root)
                output = root / "run-fixture"
                args = SimpleNamespace(config=vace_p01.FROZEN_CONFIG, memory_mode="full", composite=False)
                original_lock = vace_p01.gpu_lock

                def create_worker_fixture(*unused_arguments, **unused_options):
                    if marker:
                        (output / "worker_complete.json").write_text('{"status":"complete"}')
                    return returncode, interrupted

                with patch.object(vace_p01, "gpu_lock", lambda: original_lock(root)), \
                     patch.object(vace_p01, "check_old_training"), \
                     patch.object(vace_p01.subprocess, "check_output", return_value="fixture-head\n"), \
                     patch("vace_p01_console.run_visible_process", side_effect=create_worker_fixture):
                    actual = vace_p01.run_gpu(output, config, selected, dict(ready_for_gpu=True), args)
                self.assertEqual(actual, expected)
                status = json.loads((output / "exit_status.json").read_text())
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["worker_returncode"], returncode)
                self.assertFalse((output / "prediction_raw").exists())

    def test_abbreviated_run_is_refused_without_importing_torch(self):
        program = "\n".join([
            "import runpy, sys",
            "sys.argv = ['vace_p01.py', '--ru']",
            "try:",
            "    runpy.run_path(" + repr(str(vace_p01.ROOT / "scripts/vace_p01.py")) + ", run_name='__main__')",
            "except SystemExit as outcome:",
            "    status = outcome.code",
            "assert 'torch' not in sys.modules",
            "sys.exit(status)",
        ])
        result = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("unrecognized arguments: --ru", result.stderr)

    def test_ordinary_T2V_checkpoint_is_refused_before_payload_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordinary_t2v.safetensors"
            path.write_bytes(b'\x02\x00\x00\x00\x00\x00\x00\x00{}')
            with self.assertRaisesRegex(ValueError, "original VACE"):
                vace_p01.inspect_models(vace_p01.load_frozen_config(), checkpoint=path)


if __name__ == "__main__":
    unittest.main()
