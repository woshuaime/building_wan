"""CPU-only explicit-resume planning and worker lifecycle regressions."""

import builtins
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_building_test as batch
from test_vace_building_test import package_fixture, validate_fixture_files, write_four_files
from vace_p01_reuse import file_hashes


def validate_completed_fixture(record, destination):
    validate_fixture_files(destination)
    if Path(destination).name != record["output_number"]:
        raise ValueError("CPU completed fixture has the wrong output number")
    for name in batch.FINAL_FILENAMES:
        expected = f"CPU lifecycle fixture only: {record['pilot']}: {name}".encode()
        if not (Path(destination) / name).read_bytes().startswith(expected):
            raise ValueError("CPU completed fixture source identity mismatch")
    return dict(files_sha256=file_hashes(destination))


class ResumePlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vace_resume_plan_cpu_", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "test"
        self.package = package_fixture()
        patchers = [patch.object(batch, "OUTPUT_ROOT", self.output),
                    patch.object(batch.orbit, "check_output_volume"),
                    patch.object(batch, "validate_completed_sample", side_effect=validate_completed_fixture)]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_resume_accepts_only_valid_contiguous_prefix_and_returns_hashes(self):
        write_four_files(self.output / "02", "P02")
        write_four_files(self.output / "03", "P03")
        pending, completed = batch.plan_samples(self.package, resume=True)
        self.assertEqual([record["pilot"] for record in pending], ["P04", "P05", "P06", "P07", "P08"])
        self.assertEqual(completed, {pilot: file_hashes(self.output / number)
                                     for number, pilot in (("02", "P02"), ("03", "P03"))})

    def test_default_start_still_refuses_completed_results(self):
        write_four_files(self.output / "02", "P02")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            batch.check_targets()
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            batch.plan_samples(self.package)

    def test_noncontiguous_prefix_is_rejected_without_deleting_results(self):
        write_four_files(self.output / "03", "P03")
        hashes = file_hashes(self.output / "03")
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            batch.plan_samples(self.package, resume=True)
        self.assertEqual(file_hashes(self.output / "03"), hashes)

    def test_wrong_sample_in_completed_folder_is_rejected(self):
        write_four_files(self.output / "02", "P03")
        with self.assertRaisesRegex(ValueError, "source identity mismatch"):
            batch.plan_samples(self.package, resume=True)

    def test_empty_or_incomplete_result_directory_is_rejected(self):
        target = self.output / "02"
        target.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Invalid four-file"):
            batch.plan_samples(self.package, resume=True)
        (target / batch.FINAL_FILENAMES[0]).write_bytes(b"incomplete historical output; keep")
        with self.assertRaisesRegex(ValueError, "Invalid four-file"):
            batch.plan_samples(self.package, resume=True)
        self.assertEqual(len(list(target.iterdir())), 1)

    def test_resume_rejects_symlink_and_unrecognized_output_without_cleanup(self):
        self.output.mkdir()
        target = self.output / "02"
        target.symlink_to(Path(self.temporary.name) / "absent", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink path component"):
            batch.plan_samples(self.package, resume=True)
        self.assertTrue(target.is_symlink())
        target.unlink()
        precious = self.output / "history.txt"
        precious.write_text("keep")
        with self.assertRaisesRegex(ValueError, "unrecognized historical"):
            batch.plan_samples(self.package, resume=True)
        self.assertEqual(precious.read_text(), "keep")

    def test_all_completed_results_produce_no_GPU_queue(self):
        for record in batch.queue_records(self.package):
            write_four_files(self.output / record["output_number"], record["pilot"])
        pending, completed = batch.plan_samples(self.package, resume=True)
        self.assertEqual(pending, [])
        self.assertEqual(list(completed), ["P02", "P03", "P04", "P05", "P06", "P07", "P08"])


class ResumeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vace_resume_lifecycle_cpu_", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "test"
        for number, pilot in (("01", "P01"), ("02", "P02"), ("03", "P03")):
            write_four_files(self.output / number, pilot)
        self.package = package_fixture()
        self.snapshots = {number: (file_hashes(self.output / number),
                                  {name: (self.output / number / name).stat().st_mtime_ns
                                   for name in batch.FINAL_FILENAMES})
                          for number in ("01", "02", "03")}
        self.attempts = []
        self.temporary_paths = []
        self.active = False

    def run_fixture(self, gate_change=None, mutate_after_gate=False, fail_pilot=None, returncode=1):
        completed = {pilot: file_hashes(self.output / number) for number, pilot in (("02", "P02"), ("03", "P03"))}

        def copy_p01(source, target):
            self.assertEqual(Path(source), self.output / "01")
            self.assertEqual(Path(target), self.output / "01")
            validate_fixture_files(source)
            return dict(status="REUSED_EXISTING")

        def visible_worker(command, environment, lock_fd, callback):
            self.assertFalse(self.active, "Resume must never overlap model workers")
            self.active = True
            try:
                if "--_gate" in command:
                    self.attempts.append("gate")
                    self.assertIn("--resume", command)
                    event = dict(preflight_complete=True, archive_sha256=self.package["archive_sha256"],
                                 p01_source=str(self.output / "01"), completed_hashes=completed,
                                 pending_pilots=["P04", "P05", "P06", "P07", "P08"])
                    if gate_change:
                        event.update(gate_change)
                    callback(event)
                    if mutate_after_gate:
                        path = self.output / "02" / batch.FINAL_FILENAMES[0]
                        path.write_bytes(path.read_bytes() + b"unexpected concurrent change")
                    return 0, False
                pilot = command[command.index("--sample") + 1]
                self.assertIn(pilot, ("P04", "P05", "P06", "P07", "P08"))
                self.attempts.append(pilot)
                record = json.loads(environment["WAN_VACE_BATCH_RECORD"])
                self.assertEqual(record["pilot"], pilot)
                temporary = Path(command[command.index("--_worker") + 1])
                self.temporary_paths.append(temporary)
                self.assertEqual(temporary.parent, Path("/tmp"))
                if pilot == fail_pilot:
                    callback(dict(worker_error="CPU fixture: explicit model failure", stage="denoising"))
                    return returncode, False
                write_four_files(temporary / "deliverables", pilot)
                callback(dict(worker_complete=True, pilot=pilot, source_id=record["source_id"],
                              archive_sha256=record["archive_sha256"],
                              peak_allocated_bytes=1024 ** 3, peak_reserved_bytes=2 * 1024 ** 3))
                return 0, False
            finally:
                self.active = False

        original_import = builtins.__import__

        def no_parent_torch(name, *arguments, **keywords):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Resume parent imported Torch")
            return original_import(name, *arguments, **keywords)

        with contextlib.ExitStack() as stack:
            lock = stack.enter_context((self.root / "CPU_lock").open("a"))
            patches = [patch.object(batch, "OUTPUT_ROOT", self.output),
                       patch.object(batch.delivery, "read_delivery", return_value=self.package),
                       patch.object(batch, "validate_completed_sample", side_effect=validate_completed_fixture),
                       patch.object(batch.baseline, "gpu_lock", return_value=contextlib.nullcontext(lock)),
                       patch.object(batch.baseline, "check_old_training"),
                       patch.object(batch.orbit, "check_output_volume"),
                       patch.object(batch.orbit, "cleanup_signals", return_value=contextlib.nullcontext()),
                       patch.object(batch, "validate_final_results", side_effect=validate_fixture_files),
                       patch.object(batch.orbit, "validate_final_results", side_effect=validate_fixture_files),
                       patch.object(batch, "copy_verified_p01", side_effect=copy_p01),
                       patch.object(batch, "visible_worker", side_effect=visible_worker),
                       patch.object(builtins, "__import__", side_effect=no_parent_torch)]
            for patcher in patches:
                stack.enter_context(patcher)
            forbidden = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError("GPU/preflight in parent")))
                         for owner, name in ((batch, "preflight"), (batch.baseline, "inspect_models"),
                                             (batch.baseline, "check_loader_contract"), (batch.orbit, "execute_sample"))]
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                result = batch.run_batch(batch.ConsoleState(), resume=True)
            self.assertTrue(all(mock.call_count == 0 for mock in forbidden))
        self.assertTrue(all(not path.exists() for path in self.temporary_paths))
        if not mutate_after_gate:
            self.assert_preserved_prefix()
        return result, capture.getvalue()

    def assert_preserved_prefix(self):
        for number, (hashes, mtimes) in self.snapshots.items():
            self.assertEqual(file_hashes(self.output / number), hashes)
            self.assertEqual({name: (self.output / number / name).stat().st_mtime_ns
                              for name in batch.FINAL_FILENAMES}, mtimes)

    def test_resume_reuses_three_completed_results_and_runs_only_P04_to_P08(self):
        result, output = self.run_fixture()
        self.assertEqual(result, 0, output)
        self.assertEqual(self.attempts, ["gate", "P04", "P05", "P06", "P07", "P08"])
        self.assertIn("P02: REUSED", output)
        self.assertIn("P03: REUSED", output)
        self.assertIn("VACE BUILDING TEST: SUCCESS", output)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), [f"{number:02}" for number in range(1, 9)])

    def test_gate_changed_pending_queue_prevents_GPU_workers(self):
        result, output = self.run_fixture(gate_change=dict(pending_pilots=["P02", "P04", "P05", "P06", "P07", "P08"]))
        self.assertEqual(result, 1)
        self.assertEqual(self.attempts, ["gate"])
        self.assertIn("resume queue changed", output)

    def test_gate_changed_completed_hashes_prevents_GPU_workers(self):
        result, output = self.run_fixture(gate_change=dict(completed_hashes={"P02": {"bad": "changed"}}))
        self.assertEqual(result, 1)
        self.assertEqual(self.attempts, ["gate"])
        self.assertIn("resume queue changed", output)

    def test_missing_gate_resume_plan_prevents_GPU_workers(self):
        result, output = self.run_fixture(gate_change=dict(completed_hashes=None, pending_pilots=None))
        self.assertEqual(result, 1)
        self.assertEqual(self.attempts, ["gate"])

    def test_completed_output_change_after_gate_stops_before_GPU(self):
        result, output = self.run_fixture(mutate_after_gate=True)
        self.assertEqual(result, 1)
        self.assertEqual(self.attempts, ["gate"])
        self.assertIn("Completed P02 changed after preflight", output)
        self.assertFalse((self.output / "04").exists())

    def test_P04_failure_preserves_completed_prefix_and_never_starts_P05(self):
        result, output = self.run_fixture(fail_pilot="P04", returncode=7)
        self.assertEqual(result, 7)
        self.assertEqual(self.attempts, ["gate", "P04"])
        self.assertIn("P04: FAILED", output)
        self.assertIn("P05: NOT RUN", output)
        self.assertFalse((self.output / "04").exists())
        self.assertFalse((self.output / "05").exists())

    def test_native_failure_exit_code_is_preserved_without_retry(self):
        result, output = self.run_fixture(fail_pilot="P04", returncode=-11)
        self.assertEqual(result, 139)
        self.assertEqual(self.attempts, ["gate", "P04"])
        self.assertIn("worker returncode=-11", output)

    def test_resume_dry_run_never_starts_GPU_or_copies_any_output(self):
        with patch.object(batch, "preflight", return_value=(self.package, self.output / "01")) as preflight, \
                patch.object(batch, "run_batch") as runner, patch.object(batch, "sample_worker") as worker, \
                patch.object(batch, "copy_verified_p01") as copier, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(["--dry-run", "--resume"]), 0)
        self.assertTrue(preflight.call_args.kwargs["resume"])
        runner.assert_not_called()
        worker.assert_not_called()
        copier.assert_not_called()
        self.assert_preserved_prefix()


if __name__ == "__main__":
    unittest.main()
