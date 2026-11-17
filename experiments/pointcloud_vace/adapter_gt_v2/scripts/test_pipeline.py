"""CPU checks of selection, bounded recovery and intentional-stop behavior."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipeline


def result(name, batch, checkpointing, seconds, peak, mode="frozen", passed=True):
    return dict(candidate=dict(name=name, batch_size=batch, checkpointing=checkpointing,
                               cache_mode=mode), projected_total_seconds=seconds,
                stable_samples_per_second=2.0, peak_nvml_bytes=peak, passed=passed)


class SelectionTests(unittest.TestCase):
    def test_within_five_percent_chooses_lower_peak(self):
        a = result("A", 2, "full", 100, 30)
        c = result("C", 1, "full", 104, 20)
        self.assertIs(pipeline.choose_result([a, c]), c)

    def test_more_than_five_percent_chooses_fastest(self):
        a = result("A", 2, "full", 100, 30)
        c = result("C", 1, "full", 106, 20)
        self.assertIs(pipeline.choose_result([a, c]), a)

    def test_failed_candidate_not_selected(self):
        a = result("A", 2, "full", 10, 30, passed=False)
        c = result("C", 1, "full", 100, 20)
        self.assertIs(pipeline.choose_result([a, c]), c)

    def test_cache_preparation_counts_toward_selection(self):
        frozen = result("A", 2, "full", 100, 30)
        frozen.pop("projected_total_seconds")
        online = result("A_online", 2, "full", 100, 30, mode="online")
        online.pop("projected_total_seconds")
        prep = dict(projected_cache_seconds=900, projected_geometry_cache_seconds=30)
        self.assertEqual(pipeline.result_total(frozen, prep) - pipeline.result_total(online, prep), 900)

    def test_recovery_checkpointing_before_smaller_batch(self):
        d = result("D", 2, "none", 90, 39)
        dp = result("D_partial", 2, "partial", 94, 35)
        a = result("A", 2, "full", 100, 30)
        c = result("C", 1, "full", 130, 20)
        failed = result("bad", 1, "none", 90, 50, passed=False)
        online = result("C_online", 1, "full", 160, 25, mode="online")
        fallbacks = pipeline.recovery_candidates(d["candidate"], [d, dp, a, c, failed, online])
        self.assertEqual([entry["name"] for entry in fallbacks], ["D_partial", "A", "C"])


class WorkerRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = pipeline.ROOT / "cache/pointcloud_vace/adapter_gt_v2/runtime/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary)
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)

    def test_non_oom_failure_not_retried(self):
        c = result("C", 1, "full", 100, 20)
        with patch.object(pipeline, "run_worker", return_value=(1, 0.1)) as worker:
            with self.assertRaisesRegex(RuntimeError, "non-OOM"):
                pipeline.formal_training(self.run, dict(candidates=[c]), c["candidate"], False)
            self.assertEqual(worker.call_count, 1)

    def test_memory_budget_exit_uses_measured_fallback_and_checkpoint(self):
        a = result("A", 2, "full", 100, 30)
        c = result("C", 1, "full", 130, 20)
        (self.run / "checkpoints").mkdir()
        (self.run / "checkpoints/resume_latest.pt").touch()
        calls = []

        def fake_worker(run, mode, candidate, resume=False):
            calls.append((candidate, resume))
            if len(calls) == 1:
                return 43, 0.1
            for role in ("adapter", "lora"):
                (run / f"{role}_final.safetensors").write_bytes(b"saved")
            (run / "training_complete.json").write_text(json.dumps(dict(
                samples_consumed=32241, epochs=3,
                adapter_final=str(run / "adapter_final.safetensors"),
                lora_final=str(run / "lora_final.safetensors"))))
            return 0, 0.1

        with patch.object(pipeline, "run_worker", side_effect=fake_worker):
            code = pipeline.formal_training(self.run, dict(candidates=[a, c]), a["candidate"], False)
        self.assertEqual(code, 0)
        self.assertEqual([entry[0]["name"] for entry in calls], ["A", "C"])
        self.assertTrue(all(entry[1] for entry in calls))

    def test_exit_zero_without_complete_marker_is_failure(self):
        c = result("C", 1, "full", 100, 20)
        with patch.object(pipeline, "run_worker", return_value=(0, 0.1)):
            with self.assertRaisesRegex(RuntimeError, "without training_complete"):
                pipeline.formal_training(self.run, dict(candidates=[c]), c["candidate"], False)

    def test_stop_does_not_spawn_gpu_worker(self):
        (self.run / "stop.request").touch()
        with patch.object(pipeline.subprocess, "Popen") as popen:
            with self.assertRaises(InterruptedError):
                pipeline.run_worker(self.run, "train")
            popen.assert_not_called()

    def test_no_checkpoint_before_first_update_uses_clean_initialization(self):
        c = result("C", 1, "full", 100, 20)

        def fake_worker(run, mode, candidate, resume=False):
            self.assertFalse(resume)
            (run / "training_complete.json").write_text("{}")
            return 0, 0.1

        with patch.object(pipeline, "run_worker", side_effect=fake_worker):
            # The empty marker is deliberately invalid, but resume must still be
            # false for a resumed pipeline that never reached a training update.
            with self.assertRaisesRegex(RuntimeError, "without training_complete"):
                pipeline.formal_training(self.run, dict(candidates=[c]), c["candidate"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
