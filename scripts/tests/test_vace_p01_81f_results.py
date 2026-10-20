"""Synthetic CPU fixtures in system temporary directories, never model predictions."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01_81f_results as results
from vace_p01_results import ffmpeg_binary, validate_video


def fixture_frames():
    damaged = [Image.new("RGB", results.SIZE, (index, index, index)) for index in range(81)]
    masks = [Image.new("RGB", results.SIZE, "black") for index in range(81)]
    masks[20].paste("white", (100, 100, 200, 200))
    prediction = [Image.new("RGB", results.SIZE, (index + 80, index + 80, index + 80))
                  for index in range(81)]
    return damaged, masks, prediction, Image.new("RGB", results.SIZE, "red")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_pixels(path, size):
    decoded = subprocess.run(
        [ffmpeg_binary(), "-nostdin", "-v", "error", "-threads", "1", "-i", str(path),
         "-map", "0:v:0", "-vsync", "0", "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True, timeout=120,
    )
    return np.frombuffer(decoded.stdout, dtype=np.uint8).reshape((-1, size[1], size[0], 3))


class MinimalResultContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.damaged, cls.masks, cls.prediction, cls.gt = fixture_frames()

    def test_exact_four_real_encoded_files_and_decoded_81_frames(self):
        with tempfile.TemporaryDirectory(prefix="vace81_synthetic_cpu_") as temporary:
            output = Path(temporary) / "results"
            validated = results.render_results(output, self.damaged, self.masks, self.prediction, self.gt)
            self.assertEqual(sorted(path.name for path in output.iterdir()), sorted(results.FINAL_FILENAMES))
            self.assertEqual(validated["frames"], 81)
            self.assertEqual(validated["fps"], 16)
            self.assertFalse(validated["raw_prediction_postprocessed"])
            for video in validated["videos"].values():
                self.assertEqual(video["frames"], 81)
                self.assertTrue(video["decode_complete"])
            raw = decoded_pixels(output / results.FINAL_FILENAMES[0], results.SIZE)
            self.assertLess(abs(int(raw[0, 200, 200, 0]) - 80), 4)
            self.assertLess(abs(int(raw[80, 200, 200, 0]) - 160), 4)
            self.assertFalse(np.array_equal(raw[0], raw[80]))
            compared = decoded_pixels(output / results.FINAL_FILENAMES[1], results.COMPARISON_SIZE)
            for index in (0, 20, 40, 60, 80):
                self.assertLess(abs(int(compared[index, 200, 200, 0]) - index), 4)
                self.assertLess(abs(int(compared[index, 200, 712, 0]) - (index + 80)), 4)
            with Image.open(output / results.FINAL_FILENAMES[2]) as overview:
                self.assertEqual(overview.size, results.OVERVIEW_SIZE)
                for column, index in enumerate(results.REPRESENTATIVE_FRAMES):
                    self.assertEqual(overview.getpixel((column * 512, 32)), (index,) * 3)
                    self.assertEqual(overview.getpixel((column * 512, 576)), (index + 80,) * 3)
            with Image.open(output / results.FINAL_FILENAMES[3]) as four_columns:
                self.assertEqual(four_columns.size, results.VALIDATION_SIZE)
                self.assertEqual(four_columns.getpixel((100, 132)), (20,) * 3)
                self.assertEqual(four_columns.getpixel((612, 132)), (255,) * 3)
                self.assertEqual(four_columns.getpixel((1124, 132)), (100,) * 3)
                self.assertEqual(four_columns.getpixel((1636, 132)), (255, 0, 0))
            (output / "unrelated.json").write_text("untouched", encoding="utf-8")
            with patch.object(results, "validate_video") as decoder:
                with self.assertRaisesRegex(ValueError, "exactly the four"):
                    results.validate_final_results(output)
                decoder.assert_not_called()
            self.assertEqual((output / "unrelated.json").read_text(), "untouched")

    def test_gt_changes_only_four_column_png_not_raw_comparison_or_overview(self):
        with tempfile.TemporaryDirectory(prefix="vace81_synthetic_cpu_") as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            results.render_results(first, self.damaged, self.masks, self.prediction, self.gt)
            results.render_results(second, self.damaged, self.masks, self.prediction,
                                   Image.new("RGB", results.SIZE, "blue"))
            for name, size in ((results.FINAL_FILENAMES[0], results.SIZE),
                               (results.FINAL_FILENAMES[1], results.COMPARISON_SIZE)):
                with self.subTest(video=name):
                    np.testing.assert_array_equal(decoded_pixels(first / name, size),
                                                  decoded_pixels(second / name, size))
            self.assertEqual(digest(first / results.FINAL_FILENAMES[2]),
                             digest(second / results.FINAL_FILENAMES[2]))
            self.assertNotEqual(digest(first / results.FINAL_FILENAMES[3]),
                                digest(second / results.FINAL_FILENAMES[3]))
            with Image.open(first / results.FINAL_FILENAMES[3]) as before, \
                    Image.open(second / results.FINAL_FILENAMES[3]) as after:
                np.testing.assert_array_equal(np.asarray(before)[:, :3 * results.SIZE[0]],
                                              np.asarray(after)[:, :3 * results.SIZE[0]])
                self.assertFalse(np.array_equal(np.asarray(before)[:, 3 * results.SIZE[0]:],
                                                np.asarray(after)[:, 3 * results.SIZE[0]:]))

    def test_bad_frame_counts_rejected_before_creating_any_output(self):
        for role in range(3):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                arguments = [self.damaged, self.masks, self.prediction, self.gt]
                arguments[role] = arguments[role][:-1]
                output = Path(temporary) / "not_created"
                with self.assertRaisesRegex(ValueError, "exactly 81"):
                    results.render_results(output, *arguments)
                self.assertFalse(output.exists())

    def test_bad_mode_size_and_mask_channels_are_rejected(self):
        for invalid, error in ((Image.new("L", results.SIZE), "RGB 512x512"),
                               (Image.new("RGB", (256, 512)), "RGB 512x512"),
                               (Image.new("RGB", results.SIZE, (0, 0, 255)), "binary RGB"),
                               (Image.new("RGB", results.SIZE, (127, 127, 127)), "binary RGB")):
            with self.subTest(error=error):
                masks = self.masks.copy()
                masks[20] = invalid
                with self.assertRaisesRegex(ValueError, error):
                    results.validate_frames(self.damaged, masks, self.prediction, self.gt)

    def test_unmodified_all_black_masks_remain_black_and_frames_remain_unchanged(self):
        masks_before = [mask.tobytes() for mask in self.masks]
        prediction_before = [image.tobytes() for image in self.prediction]
        results.validate_frames(self.damaged, self.masks, self.prediction, self.gt)
        results.overview_image(self.damaged, self.prediction)
        results.four_column_image(self.damaged, self.masks, self.prediction, self.gt)
        self.assertEqual(masks_before, [mask.tobytes() for mask in self.masks])
        self.assertEqual(prediction_before, [image.tobytes() for image in self.prediction])
        self.assertEqual(sum(mask.getbbox() is None for mask in self.masks), 80)

    def test_nonempty_destination_is_preserved_and_never_cleaned(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            existing = output / "other_experiment.txt"
            existing.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Refuse to overwrite"):
                results.render_results(output, self.damaged, self.masks, self.prediction, self.gt)
            self.assertEqual(existing.read_text(), "preserve")
            self.assertEqual(list(output.iterdir()), [existing])

    def test_symlink_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "owned_elsewhere").mkdir()
            (root / "alias").symlink_to(root / "owned_elsewhere", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                results.render_results(root / "alias", self.damaged, self.masks, self.prediction, self.gt)
            self.assertEqual(list((root / "owned_elsewhere").iterdir()), [])

    def test_video_writer_refuses_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "previous.mp4"
            path.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                results.write_video_exclusive(path, results.SIZE, iter(self.prediction))
            self.assertEqual(path.read_bytes(), b"preserve")

    def test_video_validator_rejects_wrong_count_and_wrong_fps(self):
        import imageio_ffmpeg

        for count, fps, error in ((80, 16, "81 frames"), (81, 15, "exactly 16")):
            with self.subTest(count=count, fps=fps), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "synthetic_cpu_only.mp4"
                writer = imageio_ffmpeg.write_frames(str(path), results.SIZE, fps=fps, codec="libx264",
                    ffmpeg_log_level="error", output_params=["-threads", "1"])
                writer.send(None)
                for index in range(count):
                    writer.send(np.asarray(self.prediction[index]).tobytes())
                writer.close()
                with self.assertRaisesRegex(ValueError, error):
                    validate_video(path, expected_frames=81)

    def test_png_corruption_is_detected_without_deleting_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in results.FINAL_FILENAMES:
                (output / name).write_bytes(b"unreadable synthetic fixture")
            with patch.object(results, "validate_video", return_value=dict(frames=81)):
                with self.assertRaises(OSError):
                    results.validate_final_results(output)
            self.assertEqual(len(list(output.iterdir())), 4)


if __name__ == "__main__":
    unittest.main()
