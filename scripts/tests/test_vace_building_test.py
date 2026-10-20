"""Temporary CPU fixtures for the seven-sample batch; never execute a model."""

import contextlib
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_building_test as batch
import vace_p01_reuse as reuse


def package_fixture():
    records = []
    for number, pilot, source, kind, tier in batch.delivery.SAMPLE_MAPPING:
        records.append(dict(output_number=number, pilot=pilot, source_id=source, kind=kind, tier=tier,
                            archive="/tmp/CPU_FIXTURE_ONLY.zip", archive_sha256="a" * 64,
                            stable_id=f"b{source}_CPU_FIXTURE", relative_root=f"samples/{pilot}"))
    return dict(archive_sha256="a" * 64, records=records)


def write_four_files(directory, pilot):
    directory.mkdir(parents=True, exist_ok=False)
    for name in batch.FINAL_FILENAMES:
        (directory / name).write_bytes(f"CPU lifecycle fixture only: {pilot}: {name}".encode())


def validate_fixture_files(directory, project_root=None):
    directory = Path(directory)
    if (not directory.is_dir() or directory.is_symlink()
            or sorted(path.name for path in directory.iterdir()) != sorted(batch.FINAL_FILENAMES)
            or not all((directory / name).is_file() and not (directory / name).is_symlink()
                       and (directory / name).stat().st_size > 0 for name in batch.FINAL_FILENAMES)):
        raise ValueError("Invalid four-file CPU fixture")
    return dict(files_sha256=reuse.file_hashes(directory))


class BatchConfigTests(unittest.TestCase):
    def test_queue_is_exactly_the_seven_remaining_samples_in_fixed_order(self):
        expected = (("02", "P02", 763), ("03", "P03", 871), ("04", "P04", 1227),
                    ("05", "P05", 1259), ("06", "P06", 1848),
                    ("07", "P07", 2297), ("08", "P08", 939))
        self.assertEqual(tuple(entry[:3] for entry in batch.QUEUE), expected)
        records = batch.queue_records(package_fixture())
        self.assertEqual(tuple((item["output_number"], item["pilot"], item["source_id"])
                               for item in records), expected)
        self.assertNotIn("P01", [item["pilot"] for item in records])

    def test_each_sample_reuses_the_exact_P01_inference_model_and_adapter(self):
        original = batch.orbit.experiment_config()
        for record in batch.queue_records(package_fixture()):
            with self.subTest(pilot=record["pilot"]):
                config = batch.sample_config(record)
                for section in ("inference", "model", "adapter"):
                    self.assertEqual(config[section], original[section])
                self.assertEqual(config["source"]["source_id"], record["source_id"])
                self.assertEqual(config["source"]["sample_relative_root"], record["relative_root"])
                self.assertEqual(config["inference"]["num_frames"], 81)
                self.assertIsNone(config["model"]["lora_checkpoint"])

    def test_missing_duplicate_or_wrong_mapping_cannot_form_a_queue(self):
        for alteration in ("missing", "duplicate", "wrong_source", "wrong_output"):
            package = package_fixture()
            if alteration == "missing":
                package["records"].pop()
            elif alteration == "duplicate":
                package["records"][-1] = copy.deepcopy(package["records"][0])
            elif alteration == "wrong_source":
                package["records"][2]["source_id"] = 465
            else:
                package["records"][2]["output_number"] = "02"
            with self.subTest(alteration=alteration), self.assertRaises((ValueError, KeyError)):
                batch.queue_records(package)


class SharedExecutionTests(unittest.TestCase):
    def execute_fixture(self, prediction_count=81, include_gt=False, pipeline_error=False):
        buffer = io.BytesIO()
        Image.new("RGB", (512, 512), (30, 40, 50)).save(buffer, format="PNG")
        selected = [dict(source_index=index, conditioning=object(), model_mask=object(),
                         original_bytes=dict(damaged=buffer.getvalue())) for index in range(81)]
        if include_gt:
            selected[20]["original_bytes"]["gt"] = b"GT must never enter this kernel"
        prediction = [object() for unused in range(prediction_count)]
        events = []
        config = batch.orbit.experiment_config()

        class Pipeline:
            def __init__(self):
                self.vae = object()
                self.calls = []

            def check_resize_height_width(self, height, width, frames, verbose=0):
                return height, width, frames

            def __call__(self, **arguments):
                self.calls.append(arguments)
                events.append("prediction")
                if pipeline_error:
                    raise RuntimeError("CPU fixture model failure")
                return prediction

        pipe = Pipeline()
        cuda = SimpleNamespace(reset_peak_memory_stats=Mock(), synchronize=Mock(),
                               max_memory_allocated=Mock(return_value=1024 ** 3),
                               max_memory_reserved=Mock(return_value=2 * 1024 ** 3))
        state = SimpleNamespace(stage=Mock(), progress=Mock(),
                                vae_stages=lambda vae: contextlib.nullcontext())

        def read_gt():
            self.assertEqual(events, ["prediction"])
            events.append("GT")
            return "CPU post-inference GT fixture"

        def render(destination, damaged, masks, actual_prediction, gt):
            self.assertEqual(events, ["prediction", "GT"])
            self.assertEqual(len(damaged), 81)
            self.assertIs(actual_prediction, prediction)
            self.assertEqual(masks, [item["model_mask"] for item in selected])
            self.assertEqual(gt, "CPU post-inference GT fixture")
            events.append("render")

        with tempfile.TemporaryDirectory(prefix="vace_shared_kernel_cpu_test_", dir="/tmp") as directory, \
                patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)), \
                patch.object(batch.baseline, "load_local_vace", return_value=pipe) as loader, \
                patch.object(batch.orbit, "render_results", side_effect=render) as renderer:
            if include_gt or prediction_count != 81 or pipeline_error:
                with self.assertRaises((ValueError, RuntimeError)):
                    batch.orbit.execute_sample(config, selected, dict(paths={"CPU": "fixture"}), state,
                                               Path(directory) / "deliverables", read_gt)
                renderer.assert_not_called()
                self.assertNotIn("GT", events)
                if include_gt:
                    loader.assert_not_called()
                    self.assertFalse(pipe.calls)
                else:
                    self.assertEqual(len(pipe.calls), 1)
            else:
                peaks = batch.orbit.execute_sample(config, selected, dict(paths={"CPU": "fixture"}), state,
                                                  Path(directory) / "deliverables", read_gt)
                self.assertEqual(peaks, dict(peak_allocated_bytes=1024 ** 3, peak_reserved_bytes=2 * 1024 ** 3))
                loader.assert_called_once_with({"CPU": "fixture"}, memory_mode="full")
                self.assertEqual(events, ["prediction", "GT", "render"])
                self.assertEqual(len(pipe.calls), 1)
                arguments = pipe.calls[0]
                expected = batch.baseline.inference_arguments(config, [item["conditioning"] for item in selected],
                                                              [item["model_mask"] for item in selected])
                expected["progress_bar_cmd"] = state.progress
                self.assertEqual(arguments, expected)
                self.assertEqual(len(arguments["vace_video"]), 81)
                self.assertIs(arguments["vace_video"][-1], selected[80]["conditioning"])
                self.assertIsNot(arguments["vace_video"][-1], selected[0]["conditioning"])
                self.assertEqual(renderer.call_count, 1)

    def test_shared_kernel_makes_one_full_81_frame_call_then_reads_GT(self):
        self.execute_fixture()

    def test_GT_payload_is_rejected_before_loading_a_model(self):
        self.execute_fixture(include_gt=True)

    def test_missing_prediction_or_pipeline_failure_never_reads_GT_or_renders(self):
        self.execute_fixture(prediction_count=80)
        self.execute_fixture(pipeline_error=True)


class BatchLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vace_remaining_cpu_test_", dir="/tmp")
        self.root = Path(self.temporary.name)
        self.output = self.root / "test"
        self.original = self.root / "P01_465_81f_fixture"
        write_four_files(self.original, "P01")
        self.source_hashes = reuse.file_hashes(self.original)
        self.package = package_fixture()
        self.attempts = []
        self.private_paths = []
        self.copied = []
        self.active = False

    def tearDown(self):
        self.temporary.cleanup()

    def run_fixture(self, fail_pilot=None, returncode=0, marker=True, interrupted=False,
                    wrong_identity=None, gate_failure=False, gate_digest=None):
        def copy_p01(source, target):
            self.copied.append((Path(source), Path(target)))
            self.assertEqual(Path(source), self.original)
            validate_fixture_files(source)
            if Path(target).exists():
                self.assertEqual(reuse.file_hashes(target), self.source_hashes)
            else:
                Path(target).mkdir(parents=True, exist_ok=False)
                for name in batch.FINAL_FILENAMES:
                    with (Path(source) / name).open("rb") as reader, (Path(target) / name).open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
            return dict(status="REUSED_EXISTING")

        def visible_worker(command, environment, lock_fd, callback):
            self.assertFalse(self.active, "Sample workers overlap")
            self.assertTrue(self.root.is_dir())
            self.active = True
            try:
                if "--_gate" in command:
                    self.attempts.append("gate")
                    if gate_failure:
                        callback(dict(worker_error="CPU fixture: missing input in P08", stage="delivery_validation"))
                        return 1, False
                    callback(dict(preflight_complete=True, p01_source=str(self.original),
                                  archive_sha256=gate_digest or self.package["archive_sha256"]))
                    return 0, False
                self.assertEqual(self.attempts[0], "gate")
                self.assertTrue(self.copied, "P01 reuse must be validated before any remaining sample starts")
                record = json.loads(environment["WAN_VACE_BATCH_RECORD"])
                pilot = command[command.index("--sample") + 1]
                self.assertEqual(pilot, record["pilot"])
                self.assertNotEqual(pilot, "P01")
                self.attempts.append(pilot)
                index = int(record["output_number"])
                for completed in range(1, index):
                    validate_fixture_files(self.output / f"{completed:02}")
                temporary = Path(command[command.index("--_worker") + 1])
                self.assertEqual(temporary.parent, Path("/tmp"))
                self.assertTrue(temporary.name.startswith(batch.TEMP_PREFIX))
                self.private_paths.append(temporary)
                if pilot != fail_pilot or returncode == 0:
                    write_four_files(temporary / "deliverables", pilot)
                event = dict(pilot=pilot, source_id=record["source_id"],
                             archive_sha256=record["archive_sha256"],
                             peak_allocated_bytes=1024 ** 3, peak_reserved_bytes=2 * 1024 ** 3)
                if pilot != fail_pilot or marker:
                    event["worker_complete"] = True
                if pilot == fail_pilot and wrong_identity:
                    event[wrong_identity] = {"pilot": "P01", "source_id": 465,
                                             "archive_sha256": "b" * 64}[wrong_identity]
                callback(event)
                return (returncode, interrupted) if pilot == fail_pilot else (0, False)
            finally:
                self.active = False

        with contextlib.ExitStack() as stack:
            lock = stack.enter_context((self.root / "lock").open("a"))
            stack.enter_context(patch.object(batch, "OUTPUT_ROOT", self.output))
            stack.enter_context(patch.object(batch.delivery, "read_delivery", return_value=self.package))
            stack.enter_context(patch.object(batch.baseline, "gpu_lock", return_value=contextlib.nullcontext(lock)))
            stack.enter_context(patch.object(batch.baseline, "check_old_training"))
            stack.enter_context(patch.object(batch.orbit, "check_output_volume", return_value=100 * 1024 ** 3))
            stack.enter_context(patch.object(batch.orbit, "cleanup_signals", return_value=contextlib.nullcontext()))
            stack.enter_context(patch.object(batch.orbit, "validate_final_results", side_effect=validate_fixture_files))
            stack.enter_context(patch.object(batch, "validate_final_results", side_effect=validate_fixture_files))
            stack.enter_context(patch.object(batch, "copy_verified_p01", side_effect=copy_p01))
            stack.enter_context(patch.object(batch, "visible_worker", side_effect=visible_worker))
            forbidden = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError("Parent initialized GPU/preflight")))
                         for owner, name in ((batch, "preflight"), (batch.baseline, "inspect_models"),
                                             (batch.baseline, "check_loader_contract"), (batch.orbit, "execute_sample"))]
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                result = batch.run_batch(batch.ConsoleState())
            self.assertTrue(all(mock.call_count == 0 for mock in forbidden))
        self.assertEqual(reuse.file_hashes(self.original), self.source_hashes)
        self.assertTrue(all(not path.exists() for path in self.private_paths))
        return result, capture.getvalue()

    def test_success_is_seven_sequential_workers_and_exactly_eight_by_four_files(self):
        result, output = self.run_fixture()
        self.assertEqual(result, 0, output)
        self.assertEqual(self.attempts, ["gate", "P02", "P03", "P04", "P05", "P06", "P07", "P08"])
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), [f"{index:02}" for index in range(1, 9)])
        self.assertEqual(len(list(self.output.rglob("*"))), 8 + 8 * 4)
        for index in range(1, 9):
            validate_fixture_files(self.output / f"{index:02}")
        self.assertIn("P01: REUSED - 81F ALREADY COMPLETED", output)
        self.assertIn("VACE BUILDING TEST: SUCCESS", output)
        self.assertEqual(len(self.private_paths), 7)

    def test_all_seven_gate_failure_prevents_P01_copy_and_every_GPU_worker(self):
        result, output = self.run_fixture(gate_failure=True)
        self.assertEqual(result, 1)
        self.assertEqual(self.attempts, ["gate"])
        self.assertFalse(self.copied)
        self.assertFalse(self.output.exists())
        self.assertIn("missing input in P08", output)
        self.assertNotIn("VACE BUILDING TEST: SUCCESS", output)

    def test_changed_package_after_gate_prevents_any_copy_or_GPU_worker(self):
        result, output = self.run_fixture(gate_digest="b" * 64)
        self.assertEqual(result, 1)
        self.assertEqual(self.attempts, ["gate"])
        self.assertFalse(self.copied)
        self.assertFalse(self.output.exists())
        self.assertIn("changed between batch preflight", output)

    def test_missing_sample_cannot_reach_the_gate_or_create_outputs(self):
        self.package["records"].pop()
        result, output = self.run_fixture()
        self.assertEqual(result, 1)
        self.assertFalse(self.attempts)
        self.assertFalse(self.output.exists())

    def test_failure_at_P04_stops_before_P05_and_keeps_completed_P02_P03(self):
        result, output = self.run_fixture(fail_pilot="P04", returncode=7, marker=False)
        self.assertEqual(result, 7)
        self.assertEqual(self.attempts, ["gate", "P02", "P03", "P04"])
        for number in ("01", "02", "03"):
            validate_fixture_files(self.output / number)
        self.assertFalse((self.output / "04").exists())
        self.assertFalse((self.output / "05").exists())
        self.assertIn("P04: FAILED", output)
        self.assertIn("P05: NOT RUN", output)

    def test_native_failure_is_139_even_if_worker_sent_a_completion_marker(self):
        result, output = self.run_fixture(fail_pilot="P02", returncode=-11, marker=True)
        self.assertEqual(result, 139)
        self.assertEqual(self.attempts, ["gate", "P02"])
        self.assertFalse((self.output / "02").exists())
        self.assertIn("worker returncode=-11", output)

    def test_missing_completion_marker_prevents_publication_and_continuation(self):
        result, output = self.run_fixture(fail_pilot="P02", marker=False)
        self.assertEqual(result, 1)
        self.assertEqual(self.attempts, ["gate", "P02"])
        self.assertFalse((self.output / "02").exists())

    def test_interrupt_after_completion_marker_is_still_a_failed_run(self):
        result, output = self.run_fixture(fail_pilot="P02", interrupted=True)
        self.assertEqual(result, 130)
        self.assertEqual(self.attempts, ["gate", "P02"])
        self.assertFalse((self.output / "02").exists())

    def test_wrong_worker_sample_source_or_archive_is_rejected_before_publication(self):
        for field in ("pilot", "source_id", "archive_sha256"):
            with self.subTest(field=field):
                self.attempts.clear()
                self.private_paths.clear()
                self.copied.clear()
                result, output = self.run_fixture(fail_pilot="P02", wrong_identity=field)
                self.assertEqual(result, 1)
                self.assertEqual(self.attempts, ["gate", "P02"])
                self.assertFalse((self.output / "02").exists())
                self.assertIn("identity mismatch", output)

    def test_existing_remaining_result_is_never_overwritten_or_executed(self):
        for number in range(2, 9):
            with self.subTest(number=number):
                target = self.output / f"{number:02}"
                target.mkdir(parents=True)
                precious = target / "keep.txt"
                precious.write_text("Existing result must be kept")
                result, output = self.run_fixture()
                self.assertEqual(result, 1)
                self.assertFalse(self.attempts)
                self.assertEqual(precious.read_text(), "Existing result must be kept")
                self.assertIn("refusing overwrite", output)
                precious.unlink()
                target.rmdir()

    def test_dangling_remaining_symlink_is_rejected_without_removing_it(self):
        self.output.mkdir()
        target = self.output / "02"
        target.symlink_to(self.root / "missing", target_is_directory=True)
        result, output = self.run_fixture()
        self.assertEqual(result, 1)
        self.assertFalse(self.attempts)
        self.assertTrue(target.is_symlink())

    def test_valid_existing_P01_is_reused_without_modification(self):
        write_four_files(self.output / "01", "P01")
        before = {name: (self.output / "01" / name).stat().st_mtime_ns for name in batch.FINAL_FILENAMES}
        result, output = self.run_fixture()
        self.assertEqual(result, 0, output)
        self.assertEqual(reuse.file_hashes(self.output / "01"), self.source_hashes)
        self.assertEqual(before, {name: (self.output / "01" / name).stat().st_mtime_ns for name in batch.FINAL_FILENAMES})

    def test_p01_copy_helper_reuses_valid_target_without_opening_it_for_write(self):
        target = self.output / "01"
        write_four_files(target, "P01")
        before = {name: (target / name).stat().st_mtime_ns for name in batch.FINAL_FILENAMES}
        with patch.object(reuse, "validate_p01_results", side_effect=validate_fixture_files):
            result = reuse.copy_verified_p01(self.original, target)
        self.assertEqual(result["status"], "REUSED_EXISTING")
        self.assertEqual(before, {name: (target / name).stat().st_mtime_ns for name in batch.FINAL_FILENAMES})

    def test_prepare_copies_P01_only_and_never_calls_run_or_GPU_worker(self):
        with patch.object(batch, "OUTPUT_ROOT", self.output), \
                patch.object(batch, "preflight", return_value=(self.package, self.original)) as preflight, \
                patch.object(batch, "copy_verified_p01", return_value=dict(status="COPIED")) as copier, \
                patch.object(batch, "run_batch", side_effect=AssertionError("GPU batch not authorized")) as runner, \
                patch.object(batch, "sample_worker", side_effect=AssertionError("GPU worker not authorized")) as worker, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(["--prepare"]), 0)
        preflight.assert_called_once()
        copier.assert_called_once_with(self.original, self.output / "01")
        runner.assert_not_called()
        worker.assert_not_called()

    def test_dry_run_never_copies_P01_or_starts_a_batch(self):
        with patch.object(batch, "preflight", return_value=(self.package, self.original)), \
                patch.object(batch, "copy_verified_p01") as copier, \
                patch.object(batch, "run_batch") as runner, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(["--dry-run"]), 0)
        copier.assert_not_called()
        runner.assert_not_called()

    def test_P01_is_not_an_allowed_internal_worker_argument(self):
        with patch.object(batch, "sample_worker") as worker, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                batch.main(["--_worker", "/tmp/CPU_FIXTURE", "--sample", "P01"])
        self.assertEqual(caught.exception.code, 2)
        worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
