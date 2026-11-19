import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rgb_queue as queue
from rgb_recipe import orbit_schedule


class RGBQueueTests(unittest.TestCase):
    def test_ring_covers_seam_and_epoch_windows_are_distinct(self):
        ids = [str(i) for i in range(120)]
        schedule = orbit_schedule(ids)
        windows = {building: [] for building in ids}
        for epoch in range(3):
            block = schedule[epoch * len(ids):(epoch + 1) * len(ids)]
            self.assertEqual({building for building, _ in block}, set(ids))
            for building, start in block:
                windows[building].append({(start + j) % 80 for j in range(17)})
        self.assertTrue(any({79, 0} <= window for values in windows.values() for window in values))
        for values in windows.values():
            self.assertEqual(len(set.union(*values)), 51)

    def test_pause_after_training_resumes_validation_then_reports_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            stages = []
            acquired = []
            @contextlib.contextmanager
            def lock(_):
                acquired.append(True)
                try:
                    yield
                finally:
                    acquired.pop()
            def work(_, owner, worker, *args):
                stages.append(args[-1] if args else worker)
                if worker == "train_rgb.py":
                    self.assertTrue(acquired)
                    (run / "training_complete.json").write_text("{}")
                elif args[-1] == "render":
                    self.assertTrue(acquired)
                    if stages.count("render") == 1:
                        raise InterruptedError("pause")
                    (run / "validation_rendered.json").write_text("{}")
                else:
                    self.assertFalse(acquired)
                    (run / "validation_complete.json").write_text("{}")

            with patch.object(queue, "invoke", work), patch.object(queue, "acquire_gpu", lock), \
                 patch.object(queue, "dashboard"), patch.object(queue.signal, "signal"):
                queue.supervise(run)
                self.assertEqual(json.loads((run / "progress.json").read_text())["stage"], "paused")
                queue.supervise(run)
                queue.supervise(run)
            self.assertEqual(stages, ["train_rgb.py", "render", "render", "report"])
            self.assertEqual(json.loads((run / "progress.json").read_text())["stage"], "complete")
