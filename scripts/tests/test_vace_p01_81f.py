"""CPU-only coverage for full-orbit inputs and the four-file launch lifecycle."""

import builtins
import contextlib
import copy
import hashlib
import io
import json
import signal
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01
import vace_p01_81f


def png_payload(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def orbit_fixture(root):
    config = vace_p01_81f.experiment_config()
    config["source"]["archive"] = "orbit_fixture.zip"
    sample = config["source"]["sample_relative_root"]
    frames = [dict(frame_index=index, camera_index=index, frame_id=f"{index:03}",
                   **{role: f"{sample}/512/{role}/{index:03}.png" for role in ("damaged", "exact", "gt")})
              for index in range(81)]
    cameras = dict(resolutions={"512": [dict(frame=index, frame_id=f"{index:03}") for index in range(81)]})
    camera_bytes = json.dumps(cameras).encode()
    payloads = {sample + "/cameras.json": camera_bytes}
    inventory = []
    for index in range(81):
        damaged = Image.new("RGB", (512, 512), (index, 40, 90))
        exact = Image.new("L", (512, 512), 0)
        if index not in range(46, 73):
            exact.paste(255, (10, 20, 14, 25))
        for role, image in (("damaged", damaged), ("exact", exact)):
            relative = frames[index][role]
            content = png_payload(image)
            payloads[relative] = content
            inventory.append(dict(relative_path=relative, sha256=hashlib.sha256(content).hexdigest()))
    relative = frames[20]["gt"]
    payloads[relative] = png_payload(Image.new("RGB", (512, 512), (240, 5, 15)))
    inventory.append(dict(relative_path=relative, sha256=hashlib.sha256(payloads[relative]).hexdigest()))
    manifest = dict(source_identity=dict(source_id=465, stable_id=config["source"]["stable_id"]),
                    frame_count=81, frames=frames, files=inventory,
                    original_source_context=dict(cameras_sha256=hashlib.sha256(camera_bytes).hexdigest()))
    prefix = config["source"]["package_id"] + "/"
    archive = root / config["source"]["archive"]
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(prefix + "manifest.json", json.dumps(manifest))
        for name, content in payloads.items():
            bundle.writestr(prefix + name, content)
    config["source"]["archive_sha256"] = vace_p01.sha256(archive)
    return config


class ExperimentConfigTests(unittest.TestCase):
    def test_only_authorized_orbit_and_output_fields_change(self):
        frozen = vace_p01.load_frozen_config()
        expected = copy.deepcopy(frozen)
        expected["source"].update(frame_start_inclusive=0, frame_end_inclusive=80, frame_stride=1,
                                  selection_policy="all_81_including_original_B080")
        expected["inference"]["num_frames"] = 81
        expected["output_policy"] = dict(existing_directory="refuse",
                                          final_files=list(vace_p01_81f.FINAL_FILENAMES),
                                          keep_intermediates=False, keep_logs=False)
        actual = vace_p01_81f.experiment_config()
        self.assertEqual(actual, expected)
        self.assertEqual(vace_p01.load_frozen_config(), frozen)
        self.assertEqual(vace_p01.sha256(vace_p01.FROZEN_CONFIG), vace_p01.FROZEN_SHA256)
        self.assertIsNone(actual["model"]["lora_checkpoint"])


class OrbitInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="vace81_test_inputs_")
        cls.root = Path(cls.temporary.name)
        cls.config = orbit_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_81_inputs_are_aligned_without_reading_any_GT_payload(self):
        reads = []
        original_read = zipfile.ZipFile.read

        def track_read(bundle, name, *arguments, **options):
            reads.append(name)
            if "/gt/" in name:
                raise AssertionError("GT RGB was read before inference")
            return original_read(bundle, name, *arguments, **options)

        before = vace_p01.sha256(self.root / self.config["source"]["archive"])
        with patch.object(zipfile.ZipFile, "read", track_read):
            selected = vace_p01_81f.memory_inputs(self.config, self.root)
        self.assertEqual(len(selected), 81)
        self.assertEqual([item["source_index"] for item in selected], list(range(81)))
        self.assertEqual([item["output_index"] for item in selected], list(range(81)))
        self.assertTrue(any(name.endswith("/damaged/080.png") for name in reads))
        self.assertTrue(any(name.endswith("/exact/080.png") for name in reads))
        self.assertFalse(any("/gt/" in name for name in reads))
        black_frames = []
        for index, item in enumerate(selected):
            self.assertEqual(set(item["original_bytes"]), {"damaged", "exact"})
            self.assertEqual(item["camera"]["frame"], index)
            self.assertEqual(item["frame"]["frame_id"], f"{index:03}")
            with Image.open(io.BytesIO(item["original_bytes"]["damaged"])) as image:
                damaged = np.asarray(image).copy()
            with Image.open(io.BytesIO(item["original_bytes"]["exact"])) as image:
                exact = np.asarray(image).copy()
            mask = np.asarray(item["model_mask"])
            condition = np.asarray(item["conditioning"])
            self.assertEqual(tuple(damaged[0, 0]), (index, 40, 90))
            self.assertEqual(item["model_mask"].mode, "RGB")
            for channel in range(3):
                np.testing.assert_array_equal(mask[:, :, channel], exact)
            np.testing.assert_array_equal(condition[exact == 0], damaged[exact == 0])
            if np.any(exact):
                self.assertEqual(np.count_nonzero(exact), 20)
                self.assertTrue(np.all(condition[exact == 255] == 127))
            else:
                black_frames.append(index)
                np.testing.assert_array_equal(condition, damaged)
        self.assertEqual(black_frames, list(range(46, 73)))
        self.assertNotEqual(selected[0]["conditioning"].tobytes(), selected[80]["conditioning"].tobytes())
        self.assertEqual(vace_p01.sha256(self.root / self.config["source"]["archive"]), before)

    def test_shared_reader_supports_the_original_17_frame_selection_without_GT(self):
        config = copy.deepcopy(self.config)
        config["source"].update(frame_start_inclusive=5, frame_end_inclusive=21)
        config["inference"]["num_frames"] = 17
        selected = vace_p01.read_inputs(config, self.root, include_gt=False)
        self.assertEqual([item["source_index"] for item in selected], list(range(5, 22)))
        self.assertTrue(all("gt" not in item["original_bytes"] for item in selected))

    def test_mismatched_frame_count_fails_before_reading_images(self):
        config = copy.deepcopy(self.config)
        config["inference"]["num_frames"] = 80
        with self.assertRaisesRegex(ValueError, "selection does not match"):
            vace_p01.read_inputs(config, self.root, include_gt=False)

    def test_GT_B020_is_an_explicit_separate_post_inference_read(self):
        reads = []
        original_read = zipfile.ZipFile.read

        def track_read(bundle, name, *arguments, **options):
            reads.append(name)
            return original_read(bundle, name, *arguments, **options)

        with patch.object(zipfile.ZipFile, "read", track_read):
            ground_truth = vace_p01_81f.read_gt_B020(self.config, self.root)
        self.assertEqual(ground_truth.getpixel((0, 0)), (240, 5, 15))
        self.assertEqual([name for name in reads if "/gt/" in name],
                         [self.config["source"]["package_id"] + "/samples/sample_0000/512/gt/020.png"])
        self.assertFalse(any("/damaged/" in name or "/exact/" in name for name in reads))

    def test_memory_input_gate_rejects_missing_B080_or_changed_black_mask_set(self):
        original = [dict(source_index=index,
                         model_mask=Image.new("RGB", (1, 1), 0 if index in range(46, 73) else 255))
                    for index in range(81)]
        with patch.object(vace_p01, "read_inputs", return_value=original[:-1]):
            with self.assertRaisesRegex(ValueError, "B000 through B080"):
                vace_p01_81f.memory_inputs(self.config, self.root)
        changed = copy.deepcopy(original)
        changed[46]["model_mask"] = Image.new("RGB", (1, 1), 255)
        with patch.object(vace_p01, "read_inputs", return_value=changed):
            with self.assertRaisesRegex(ValueError, "27 all-black"):
                vace_p01_81f.memory_inputs(self.config, self.root)

    def test_model_arguments_keep_all_81_predictions_independent_of_GT(self):
        config = vace_p01_81f.experiment_config()
        conditions = [object() for unused in range(81)]
        masks = [object() for unused in range(81)]
        arguments = vace_p01.inference_arguments(config, conditions, masks)
        self.assertIs(arguments["vace_video"], conditions)
        self.assertIs(arguments["vace_video_mask"], masks)
        self.assertEqual(arguments["num_frames"], 81)
        self.assertEqual(arguments["seed"], 20261018)
        self.assertEqual(arguments["num_inference_steps"], 30)
        self.assertEqual(arguments["cfg_scale"], 5)
        self.assertEqual((arguments["height"], arguments["width"]), (512, 512))
        self.assertIsNone(arguments["vace_reference_image"])
        self.assertNotIn("input_video", arguments)
        self.assertNotIn("input_image", arguments)
        self.assertFalse(any("gt" in key.lower() for key in arguments))


class PublishingTests(unittest.TestCase):
    def test_only_four_files_are_copied_and_validated(self):
        with tempfile.TemporaryDirectory(prefix="vace81_test_publish_") as directory:
            root = Path(directory)
            staged = root / "private"
            staged.mkdir()
            for index, name in enumerate(vace_p01_81f.FINAL_FILENAMES):
                (staged / name).write_bytes(f"fixture-only-{index}".encode())
            foreign = root / "someone_else.txt"
            foreign.write_text("do not touch")
            destination = root / "final"
            with patch.object(vace_p01_81f, "validate_final_results") as validator:
                actual = vace_p01_81f.publish_results(staged, destination)
            self.assertEqual(actual, destination)
            validator.assert_called_once_with(destination)
            self.assertEqual(sorted(path.name for path in destination.iterdir()), sorted(vace_p01_81f.FINAL_FILENAMES))
            for name in vace_p01_81f.FINAL_FILENAMES:
                self.assertEqual((staged / name).read_bytes(), (destination / name).read_bytes())
            self.assertEqual(foreign.read_text(), "do not touch")

    def test_existing_or_symlink_destination_is_refused_without_deletion(self):
        with tempfile.TemporaryDirectory(prefix="vace81_test_publish_") as directory:
            root = Path(directory)
            staged = root / "private"
            staged.mkdir()
            existing = root / "existing"
            existing.mkdir()
            (existing / "precious.txt").write_text("keep")
            existing_link = root / "existing_link"
            existing_link.symlink_to(existing, target_is_directory=True)
            dangling_link = root / "dangling_link"
            dangling_link.symlink_to(root / "absent", target_is_directory=True)
            for destination in (existing, existing_link, dangling_link):
                with self.subTest(destination=destination.name):
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        vace_p01_81f.publish_results(staged, destination)
            self.assertEqual((existing / "precious.txt").read_text(), "keep")
            self.assertTrue(existing_link.is_symlink())
            self.assertTrue(dangling_link.is_symlink())

    def test_extra_or_symlink_staged_file_is_rejected_before_destination_creation(self):
        with tempfile.TemporaryDirectory(prefix="vace81_test_publish_") as directory:
            root = Path(directory)
            staged = root / "private"
            staged.mkdir()
            for name in vace_p01_81f.FINAL_FILENAMES:
                (staged / name).write_bytes(b"fixture-only")
            extra = staged / "unwanted.json"
            extra.write_text("keep")
            destination = root / "final"
            with self.assertRaisesRegex(ValueError, "only the four"):
                vace_p01_81f.publish_results(staged, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(extra.read_text(), "keep")
            extra.unlink()
            foreign = root / "someone_else.mp4"
            foreign.write_bytes(b"must not delete")
            first = staged / vace_p01_81f.FINAL_FILENAMES[0]
            first.unlink()
            first.symlink_to(foreign)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                vace_p01_81f.publish_results(staged, destination)
            self.assertFalse(destination.exists())
            self.assertTrue(first.is_symlink())
            self.assertEqual(foreign.read_bytes(), b"must not delete")


class LauncherLifecycleTests(unittest.TestCase):
    def run_fixture(self, returncode=0, marker=True, interrupted=False, worker_error=False,
                    validation_error=False, memory=None, publication_signal=None):
        with tempfile.TemporaryDirectory(prefix="vace81_test_launcher_") as directory:
            root = Path(directory)
            foreign = root / "someone_elses_results"
            foreign.mkdir()
            (foreign / "keep.txt").write_text("preserve")
            private_paths = []
            original_import = builtins.__import__

            def deny_parent_torch(name, *arguments, **options):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError("Parent process must not create a CUDA context")
                return original_import(name, *arguments, **options)

            def worker(command, environment, descriptor, callback):
                self.assertEqual(command[-2], "--_worker")
                self.assertIn("-u", command)
                self.assertIn("-B", command)
                self.assertEqual(environment["DIFFSYNTH_SKIP_DOWNLOAD"], "True")
                self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
                self.assertEqual(environment["WAN_VACE81_LOCK_FD"], str(descriptor))
                private = Path(command[-1])
                private_paths.append(private)
                self.assertEqual(private.parent, Path("/tmp"))
                self.assertTrue(private.name.startswith(vace_p01_81f.TEMP_PREFIX))
                self.assertTrue(private.is_dir())
                staged = private / "deliverables"
                staged.mkdir()
                for name in vace_p01_81f.FINAL_FILENAMES:
                    (staged / name).write_bytes(b"CPU-fixture-only")
                if marker:
                    callback(dict(worker_complete=True, stage="validation", last_completed_stage="validation",
                                  **(memory or dict(peak_allocated_bytes=1024 ** 3,
                                                   peak_reserved_bytes=2 * 1024 ** 3))))
                if worker_error:
                    raise RuntimeError("fixture worker failed")
                return returncode, interrupted

            def validate(destination):
                self.assertEqual(sorted(path.name for path in destination.iterdir()),
                                 sorted(vace_p01_81f.FINAL_FILENAMES))
                if publication_signal is not None:
                    signal.getsignal(publication_signal)(publication_signal, None)
                if validation_error:
                    raise ValueError("fixture final validation failed")

            captured = io.StringIO()
            original_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGHUP)}
            with (root / "fixture.lock").open("wb") as lock_stream, \
                 patch.object(vace_p01_81f, "ROOT", root), \
                 patch.object(vace_p01_81f, "check_output_volume", return_value=3 * 1024 ** 3), \
                 patch.object(vace_p01, "gpu_lock", return_value=contextlib.nullcontext(lock_stream)), \
                 patch.object(vace_p01, "check_old_training"), \
                 patch.object(vace_p01_81f, "preflight", side_effect=AssertionError("Parent called preflight")) as preflight, \
                 patch.object(vace_p01, "inspect_models", side_effect=AssertionError("Parent checked model weights")) as inspect, \
                 patch.object(vace_p01_81f, "visible_worker", side_effect=worker) as visible, \
                 patch.object(vace_p01_81f, "validate_final_results", side_effect=validate), \
                 patch.object(builtins, "__import__", side_effect=deny_parent_torch), \
                 contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                actual = vace_p01_81f.main(["--run"])
                visible.assert_called_once()
                preflight.assert_not_called()
                inspect.assert_not_called()
            self.assertEqual(len(private_paths), 1)
            self.assertFalse(private_paths[0].exists())
            for signum, handler in original_handlers.items():
                self.assertEqual(signal.getsignal(signum), handler)
            self.assertEqual((foreign / "keep.txt").read_text(), "preserve")
            output_root = root / "outputs/vace"
            finals = list(output_root.iterdir()) if output_root.exists() else []
            if actual == 0:
                self.assertEqual(len(finals), 1)
                self.assertEqual(sorted(path.name for path in finals[0].iterdir()),
                                 sorted(vace_p01_81f.FINAL_FILENAMES))
                self.assertIn("VACE 81F: SUCCESS", captured.getvalue())
                self.assertIn("Frames: 81", captured.getvalue())
                self.assertIn("Resolution: 512x512", captured.getvalue())
                self.assertIn("Peak VRAM: 1.00 GiB allocated / 2.00 GiB reserved", captured.getvalue())
                for name in vace_p01_81f.FINAL_FILENAMES:
                    self.assertIn(name, captured.getvalue())
            else:
                self.assertNotIn("VACE 81F: SUCCESS", captured.getvalue())
                self.assertIn(f"Exit code: {actual}", captured.getvalue())
                self.assertIn("VACE 81F: FAILED", captured.getvalue())
                if not validation_error and publication_signal is None:
                    self.assertEqual(finals, [])
            return actual

    def test_success_uses_one_worker_and_cleans_owned_system_temporary_directory(self):
        self.assertEqual(self.run_fixture(), 0)

    def test_native_crash_is_failed_even_with_completion_event(self):
        self.assertEqual(self.run_fixture(returncode=-11, marker=True), 139)

    def test_zero_exit_without_completion_event_is_failed(self):
        self.assertEqual(self.run_fixture(marker=False), 1)

    def test_interrupt_is_failed_even_with_zero_exit_and_completion_event(self):
        self.assertEqual(self.run_fixture(interrupted=True), 130)

    def test_worker_exception_cleans_only_owned_temporary_directory(self):
        self.assertEqual(self.run_fixture(worker_error=True), 1)

    def test_final_validation_failure_cannot_print_success(self):
        self.assertEqual(self.run_fixture(validation_error=True), 1)

    def test_publication_SIGTERM_and_SIGHUP_clean_owned_temporary_directory(self):
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signum):
                self.assertEqual(self.run_fixture(publication_signal=signum), 128 + signum)

    def test_missing_or_invalid_actual_memory_measurement_is_failed(self):
        for memory in (dict(peak_allocated_bytes=0, peak_reserved_bytes=0),
                       dict(peak_allocated_bytes=1024, peak_reserved_bytes=1)):
            with self.subTest(memory=memory):
                self.assertEqual(self.run_fixture(memory=memory), 1)


if __name__ == "__main__":
    unittest.main()
