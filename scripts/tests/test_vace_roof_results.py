"""CPU tests use synthetic colors only, not real VACE predictions."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_roof_results as results


class RoofResultsTests(unittest.TestCase):
    def setUp(self):
        self.indices = list(range(20, 37))
        self.groups = {
            role: [np.full((512, 512, 3), color, dtype=np.uint8) for _ in range(17)]
            for role, color in (("damaged", (20, 40, 60)), ("base", (220, 70, 30)),
                                ("lora100", (150, 80, 40)), ("lora300", (90, 110, 180)),
                                ("gt", (160, 180, 190)), ("mask", (0, 0, 0)))
        }
        self.groups["mask"][6][100:200, 100:200] = 255

    def render(self, destination):
        return results.render_sample(destination, self.indices, self.groups["damaged"], self.groups["mask"],
                                     self.groups["gt"], self.groups["base"], self.groups["lora100"],
                                     self.groups["lora300"], notify=lambda message: None)

    def test_nonconsecutive_source_frames_and_closure_frame_rejected(self):
        for indices in (list(range(21, 37)) + [38], list(range(64, 81))):
            with self.assertRaisesRegex(ValueError, "consecutive"):
                results.validate_sources(indices, self.groups)

    def test_malformed_rgb_and_nonbinary_mask_rejected(self):
        self.groups["mask"][0][0, 0] = (255, 0, 255)
        with self.assertRaisesRegex(ValueError, "binary RGB"):
            results.validate_sources(self.indices, self.groups)
        self.groups["mask"][0][0, 0] = 0
        self.groups["base"][0] = np.zeros((512, 512, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "RGB uint8"):
            results.validate_sources(self.indices, self.groups)

    def test_overview_preserves_five_stages_and_local_view_alignment(self):
        roles = ("damaged", "base", "lora100", "lora300", "gt")
        for row, role in enumerate(roles):
            for index in range(17):
                self.groups[role][index][:] = (20 * row, 10 * index, 50)
        image = np.asarray(results.make_overview(self.groups, self.indices))
        self.assertEqual(image.shape, (2720, 2560, 3))
        for row, role in enumerate(roles):
            for column, index in enumerate(results.VIEWS):
                panel = image[row * 544 + 32:row * 544 + 544, column * 512:(column + 1) * 512]
                np.testing.assert_array_equal(panel, self.groups[role][index])

    def test_summary_is_pixel_reference_not_automatic_success(self):
        summary = results.pixel_summary(self.groups)
        self.assertEqual(summary["assessment"], "requires_visual_review")
        self.assertNotIn("success", summary)
        self.assertGreater(summary["masked_mae_0_255"]["base"], 0)
        self.groups["mask"][6].fill(0)
        self.assertIsNone(results.pixel_summary(self.groups)["masked_mae_0_255"]["base"])

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "04"
            destination.mkdir()
            sentinel = destination / "original.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Refuse to overwrite"):
                self.render(destination)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_encoder_failure_cleans_only_own_temporary_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            sentinel = parent / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with patch.object(results, "encode_video", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    self.render(parent / "04")
            self.assertEqual(list(parent.iterdir()), [sentinel])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Existing FFmpeg required")
    def test_synthetic_render_encodes_four_files_with_rgb_and_frame_alignment(self):
        for index in range(17):
            self.groups["lora300"][index][:] = (90 + index * 3, 110, 180 - index * 4)
        before = [pixels.copy() for pixels in self.groups["lora300"]]
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "04"
            summary = self.render(destination)
            self.assertEqual({path.name for path in destination.iterdir()}, set(results.FINAL_NAMES))
            self.assertEqual(summary["mask_frame"], 26)
            self.assertEqual(summary["frames"], 17)
            with Image.open(destination / results.FINAL_NAMES[3]) as verification:
                pixels = np.asarray(verification)
                self.assertEqual(verification.size, (2560, 544))
                for column, role in enumerate(("damaged", "mask", "base", "lora300", "gt")):
                    np.testing.assert_array_equal(pixels[32:, column * 512:(column + 1) * 512],
                                                  self.groups[role][6])
            for original, current in zip(before, self.groups["lora300"]):
                np.testing.assert_array_equal(original, current)
            self.assertEqual(list(Path(temporary).iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
