"""CPU-only pixel alignment and rollback tests for offline result rearrangement."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reorganize_vace_results as layout


class PixelTests(unittest.TestCase):
    def test_player_compatible_all_81_three_panel_frames_and_overview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = {role: [] for role in ("damaged", "gt", "mask")}
            prediction = []
            for index in range(81):
                frames["damaged"].append(np.full((512, 512, 3), (index, 17, 34), dtype=np.uint8))
                frames["gt"].append(np.full((512, 512, 3), (35, (index * 41) % 256, 210), dtype=np.uint8))
                frames["mask"].append(np.full((512, 512, 3), 255 if index % 2 else 0, dtype=np.uint8))
                prediction.append(np.full((512, 512, 3), (125, 65, index), dtype=np.uint8))
            raw = root / layout.FINAL_NAMES[0]
            raw.write_bytes(b"untouched original MP4 placeholder; only hash checked in staged validation")
            raw_hash = layout.baseline.sha256(raw)
            layout.encode_comparison(root / layout.FINAL_NAMES[1], frames, prediction, notify=lambda message: None)
            self.assertTrue(layout.player_compatible(layout.comparison_profile(root / layout.FINAL_NAMES[1])))
            decoded = layout.decode_video(root / layout.FINAL_NAMES[1], layout.VIDEO_SIZE)
            np.testing.assert_array_equal(decoded[0, 31, 10], [255, 255, 255])
            layout.save_png_exclusive(root / layout.FINAL_NAMES[2], layout.three_rows(frames, prediction))
            four = layout.comparison_panel((("Damaged B020", frames["damaged"][20]),
                                            ("ModelMask B020", frames["mask"][20]),
                                            ("Prediction B020", prediction[20]), ("GT B020", frames["gt"][20])))
            layout.save_png_exclusive(root / layout.FINAL_NAMES[3], four)
            four_hash = layout.baseline.sha256(root / layout.FINAL_NAMES[3])
            layout.validate_arrangement(root, frames, prediction, raw_hash, four_hash)
            with Image.open(root / layout.FINAL_NAMES[2]) as overview:
                self.assertEqual(overview.size, (2560, 1632))
                for column, index in enumerate(layout.REPRESENTATIVE_FRAMES):
                    np.testing.assert_array_equal(layout.crop_pixels(overview, column, 0), frames["damaged"][index])
                    np.testing.assert_array_equal(layout.crop_pixels(overview, column, 1), prediction[index])
                    np.testing.assert_array_equal(layout.crop_pixels(overview, column, 2), frames["gt"][index])
            shifted = dict(frames, gt=frames["gt"][1:] + frames["gt"][:1])
            with self.assertRaisesRegex(ValueError, "frame alignment"):
                layout.validate_arrangement(root, shifted, prediction, raw_hash, four_hash)
            self.assertEqual(layout.baseline.sha256(raw), raw_hash)

    def test_rgb_444_is_refused_for_final_player_delivery(self):
        with patch.object(layout, "comparison_profile", return_value=dict(codec_name="h264", pix_fmt="gbrp")), \
                patch.object(layout, "decode_video") as decoder:
            with self.assertRaisesRegex(ValueError, "player-compatible"):
                layout.validate_comparison(Path("legacy-rgb.mp4"), {}, [])
            decoder.assert_not_called()

    def test_color_cast_and_swapped_columns_are_rejected(self):
        frames = {"damaged": [np.full((512, 512, 3), 240, dtype=np.uint8)] * 81,
                  "gt": [np.full((512, 512, 3), 60, dtype=np.uint8)] * 81}
        prediction = [np.full((512, 512, 3), 130, dtype=np.uint8)] * 81
        expected = np.asarray(layout.three_columns(frames, prediction, 0))
        cast = expected.copy()
        cast[:, :, 1] = 0
        swapped = expected.copy()
        swapped[:, :512] = expected[:, 1024:]
        swapped[:, 1024:] = expected[:, :512]
        with patch.object(layout, "comparison_profile", return_value={}), \
                patch.object(layout, "player_compatible", return_value=True):
            for incorrect in (cast, swapped):
                with self.subTest(color_or_order=incorrect[100, 100].tolist()), \
                        patch.object(layout, "decode_video", return_value=np.broadcast_to(incorrect, (81, *incorrect.shape))):
                    with self.assertRaisesRegex(ValueError, "color or frame alignment"):
                        layout.validate_comparison(Path("incorrect.mp4"), frames, prediction)

    def test_missing_prediction_stops_without_a_replacement(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(layout.subprocess, "run") as runner:
            missing = Path(temporary) / "missing.mp4"
            with self.assertRaisesRegex(ValueError, "Missing or empty"):
                layout.decode_video(missing, layout.SIZE)
            runner.assert_not_called()
            self.assertFalse(missing.exists())

    def test_import_is_cpu_only_even_when_model_modules_are_blocked(self):
        code = """
import sys
class BlockModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'diffsynth'):
            raise AssertionError('Model import forbidden: ' + fullname)
sys.meta_path.insert(0, BlockModels())
sys.path.insert(0, 'scripts')
import reorganize_vace_results
reorganize_vace_results.require_cpu_only()
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=layout.baseline.ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "01"
        self.staged = self.root / "staged"
        self.backup = self.root / "previous"
        self.destination.mkdir()
        self.staged.mkdir()
        for name in layout.OLD_NAMES:
            (self.destination / name).write_bytes(("existing: " + name).encode())
        for name in layout.FINAL_NAMES[1:]:
            (self.staged / name).write_bytes(("new: " + name).encode())
        self.before = layout.inventory(self.destination, layout.OLD_NAMES)
        self.raw_stat = (self.destination / layout.OLD_NAMES[0]).stat()

    def run_replace(self, validate=lambda: None):
        layout.replace_results(self.destination, self.staged, self.backup, self.before, validate)

    def assert_restored(self):
        self.assertEqual(layout.inventory(self.destination, layout.OLD_NAMES), self.before)
        raw = (self.destination / layout.OLD_NAMES[0]).stat()
        self.assertEqual((raw.st_ino, raw.st_mtime_ns), (self.raw_stat.st_ino, self.raw_stat.st_mtime_ns))

    def test_success_replaces_only_three_derivatives_and_keeps_raw_inode(self):
        self.run_replace()
        self.assertEqual(set(path.name for path in self.destination.iterdir()), set(layout.FINAL_NAMES))
        self.assertEqual(set(path.name for path in self.backup.iterdir()), set(layout.OLD_NAMES[1:]))
        raw = self.destination / layout.OLD_NAMES[0]
        self.assertEqual(layout.baseline.sha256(raw), self.before[layout.OLD_NAMES[0]])
        self.assertEqual(raw.stat().st_ino, self.raw_stat.st_ino)
        self.assertEqual(raw.stat().st_mtime_ns, self.raw_stat.st_mtime_ns)

    def test_final_validation_failure_restores_every_old_file(self):
        def fail():
            raise ValueError("synthetic validation failure")
        with self.assertRaisesRegex(ValueError, "synthetic validation"):
            self.run_replace(fail)
        self.assert_restored()

    def test_interrupt_during_publication_restores_every_old_file(self):
        def interrupt():
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_replace(interrupt)
        self.assert_restored()

    def test_midway_move_failure_restores_old_files(self):
        original_rename = Path.rename

        def rename(path, target):
            if path == self.staged / layout.FINAL_NAMES[2]:
                raise OSError("synthetic disk error")
            return original_rename(path, target)

        with patch.object(Path, "rename", rename), self.assertRaisesRegex(OSError, "synthetic disk"):
            self.run_replace()
        self.assert_restored()

    def test_failed_rollback_preserves_recovery_files(self):
        original_rename = Path.rename

        def rename(path, target):
            if path.parent == self.backup:
                raise OSError("synthetic rollback error")
            return original_rename(path, target)

        def fail():
            raise ValueError("synthetic validation failure")

        with patch.object(Path, "rename", rename), self.assertRaisesRegex(layout.RecoveryRequired, "preserved recovery files"):
            self.run_replace(fail)
        self.assertEqual(len(list(self.backup.iterdir())), 3)
        self.assertEqual(layout.baseline.sha256(self.destination / layout.OLD_NAMES[0]), self.before[layout.OLD_NAMES[0]])

    def test_changed_old_result_is_refused_before_any_move(self):
        (self.destination / layout.OLD_NAMES[2]).write_bytes(b"someone else changed this")
        with self.assertRaisesRegex(ValueError, "changed before replacement"):
            self.run_replace()
        self.assertFalse(self.backup.exists())
        self.assertEqual((self.destination / layout.OLD_NAMES[2]).read_bytes(), b"someone else changed this")

    def test_unexpected_file_is_preserved_and_replacement_refused(self):
        protected = self.destination / "unrelated.txt"
        protected.write_text("keep")
        with self.assertRaisesRegex(ValueError, "Unexpected or incomplete"):
            self.run_replace()
        self.assertEqual(protected.read_text(), "keep")
        self.assertFalse(self.backup.exists())


class ComparisonReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "01"
        self.directory.mkdir()
        self.staged = self.root / "compatible.mp4"
        self.staged.write_bytes(b"new compatible comparison")
        self.backup = self.root / "previous"
        for name in layout.FINAL_NAMES:
            (self.directory / name).write_bytes(("unchanged existing " + name).encode())
        self.before = layout.inventory(self.directory, layout.FINAL_NAMES)
        self.protected = {name: (self.directory / name).stat()
                          for name in (layout.FINAL_NAMES[0], *layout.FINAL_NAMES[2:])}

    def test_only_comparison_video_changes(self):
        layout.replace_comparison(self.directory, self.staged, self.backup, self.before, lambda: None)
        self.assertEqual((self.directory / layout.FINAL_NAMES[1]).read_bytes(), b"new compatible comparison")
        for name, original in self.protected.items():
            self.assertEqual(layout.baseline.sha256(self.directory / name), self.before[name])
            current = (self.directory / name).stat()
            self.assertEqual((current.st_ino, current.st_mtime_ns), (original.st_ino, original.st_mtime_ns))
        self.assertEqual({entry.name for entry in self.directory.iterdir()}, set(layout.FINAL_NAMES))

    def test_video_validation_failure_restores_original_four_results(self):
        def fail():
            raise ValueError("comparison validation failure")

        with self.assertRaisesRegex(ValueError, "comparison validation failure"):
            layout.replace_comparison(self.directory, self.staged, self.backup, self.before, fail)
        self.assertEqual(layout.inventory(self.directory, layout.FINAL_NAMES), self.before)

    def test_video_publication_failure_restores_original_four_results(self):
        original_rename = Path.rename

        def rename(path, target):
            if path == self.staged:
                raise OSError("comparison publication failure")
            return original_rename(path, target)

        with patch.object(Path, "rename", rename), self.assertRaisesRegex(OSError, "publication failure"):
            layout.replace_comparison(self.directory, self.staged, self.backup, self.before, lambda: None)
        self.assertEqual(layout.inventory(self.directory, layout.FINAL_NAMES), self.before)

    def test_video_rollback_failure_preserves_recovery_copy(self):
        original_rename = Path.rename

        def rename(path, target):
            if path.parent == self.backup:
                raise OSError("comparison rollback failure")
            return original_rename(path, target)

        def fail():
            raise ValueError("comparison validation failure")

        with patch.object(Path, "rename", rename), self.assertRaises(layout.RecoveryRequired):
            layout.replace_comparison(self.directory, self.staged, self.backup, self.before, fail)
        self.assertEqual(layout.baseline.sha256(self.backup / layout.FINAL_NAMES[1]),
                         self.before[layout.FINAL_NAMES[1]])


if __name__ == "__main__":
    unittest.main()
