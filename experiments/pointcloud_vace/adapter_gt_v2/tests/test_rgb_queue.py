import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rgb_queue as queue
import rgb_dashboard as display
from rgb_recipe import orbit_schedule


class PanelTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run = Path(folder.name)
        (self.run / "progress.json").write_text(json.dumps({"stage": "training"}))
        (self.run / "train_progress.json").write_text(json.dumps(dict(
            updates=99, cached=100, building="12386", epoch=1, aux_updates=18,
            durable_updates=1, stage="training", seconds_per_update=2.41)))
        (self.run / "validation_progress.json").write_text(json.dumps({"completed": 1}))

    @staticmethod
    def cells(line):
        return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in line)

    def test_terminal_width_including_chinese_and_early_progress(self):
        for columns in (48, 80, 138, 210):
            with self.subTest(columns=columns):
                text = queue.panel(self.run, columns=columns, color=False)
                lines = text.splitlines()
                self.assertNotIn("\x1b", text)
                self.assertLessEqual(max(map(self.cells, lines)), columns)
                if columns >= 80:
                    for count in ("100/32241", "99/32241", "1/8"):
                        row, = [line for line in lines if count in re.sub(r"[\s,]", "", line)]
                        self.assertGreaterEqual(self.cells(row.rstrip()), columns * .9)
                        percent, = re.findall(r"(\d+(?:\.\d+)?)\s*%", row)
                        numerator, denominator = map(int, count.split("/"))
                        self.assertAlmostEqual(float(percent), numerator * 100 / denominator, delta=.051)

    def test_ansi_palette_preserves_numbers_and_alignment(self):
        colorful = queue.panel(self.run, columns=138, color=True)
        plain = queue.panel(self.run, columns=138, color=False)
        styles = re.compile(r"\x1b\[[\d;]*m")
        self.assertGreaterEqual(len(set(styles.findall(colorful)) - {"\x1b[0m"}), 3)
        self.assertEqual(styles.sub("", colorful), plain)

    def test_validation_shows_frame_steps_separately_from_completed_videos(self):
        (self.run / "progress.json").write_text('{"stage":"validation"}')
        (self.run / "validation_progress.json").write_text(json.dumps(dict(
            completed=2, building="5954", stage="sampling", step=9, steps=30,
            seconds_per_step=9.31, eta_seconds=1396)))
        for columns in (48, 138):
            with self.subTest(columns=columns):
                lines = queue.panel(self.run, columns=columns, color=False).splitlines()
                self.assertTrue(all(self.cells(line) <= columns for line in lines))
                counts = re.sub(r"[\s,]", "", "\n".join(lines))
                self.assertIn("2/8", counts)
                self.assertIn("9/30", counts)

    def test_live_display_keeps_last_step_across_missing_or_incomplete_snapshot(self):
        train = self.run / "train_progress.json"
        original_read = Path.read_text
        tick = 0

        def read(path, **kwargs):
            if path == train:
                if tick == 1:
                    # A previous exists() would have returned True on this file.
                    raise FileNotFoundError(str(path))
                if tick == 2:
                    return '{"updates":'
            return original_read(path, **kwargs)

        def advance(_seconds):
            nonlocal tick
            tick += 1
            if tick == 3:
                train.write_text(json.dumps(dict(updates=100, cached=101, last_fm=.023,
                                                  last_sigma=.44, peak_vram_bytes=34 * 1024**3)))
            if tick == 4:
                (self.run / "progress.json").write_text('{"stage":"paused"}')
            self.assertLessEqual(tick, 4, "Standalone dashboard should exit on the paused snapshot")

        output = io.StringIO()
        with patch.dict(os.environ, COLUMNS="138"), patch.object(Path, "read_text", read), \
             patch.object(display.threading.Event, "wait", side_effect=advance), contextlib.redirect_stdout(output):
            display.dashboard(self.run)
        frames = output.getvalue().split("V2  建筑点云训练")[1:]
        self.assertEqual(len(frames), 4)  # The two misses share one identical frame.
        self.assertIn("等待进度刷新", frames[1])
        self.assertIn("99/32,241", frames[1])
        self.assertNotIn("等待进度刷新", frames[2])
        self.assertIn("100/32,241", frames[2])
        self.assertIn("FM 损失 0.02300", frames[2])
        self.assertIn("峰值显存 34.00 GiB", frames[2])
        self.assertIn("已暂停", frames[3])

    def test_validation_read_gap_preserves_sampling_step_until_next_snapshot(self):
        (self.run / "progress.json").write_text('{"stage":"validation"}')
        validation = self.run / "validation_progress.json"
        validation.write_text('{"completed":2, "step":12}')
        previous = display.read_progress(self.run)
        validation.unlink()
        snapshot = display.read_progress(self.run, previous)
        self.assertTrue(snapshot["waiting"])
        self.assertEqual(snapshot["validation"], {"completed": 2, "step": 12})
        validation.write_text('{"completed":2, "step":13}')
        snapshot = display.read_progress(self.run, snapshot)
        self.assertFalse(snapshot["waiting"])
        self.assertEqual(snapshot["validation"]["step"], 13)


class RGBQueueTests(unittest.TestCase):
    def test_dependency_clears_when_original_process_exits_or_pid_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch.object(queue, "process_start", side_effect=["100", "100", "200"]), \
                 patch.object(queue.time, "sleep") as sleep:
                queue.wait_for_job(run, 4321, "100")
            sleep.assert_called_once_with(2)
            status = json.loads((run / "progress.json").read_text())
            self.assertEqual(status["stage"], "waiting_for_gpu")
            self.assertIn("4321", status["reason"])

    def test_stop_during_dependency_wait_does_not_acquire_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "stop.request").touch()
            with patch.object(queue, "process_start", return_value="100"), \
                 patch.object(queue, "acquire_gpu") as lock, \
                 patch.object(queue, "dashboard"), patch.object(queue.signal, "signal"):
                queue.supervise(run, 4321, "100")
            lock.assert_not_called()
            self.assertEqual(json.loads((run / "progress.json").read_text())["stage"], "paused")

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
