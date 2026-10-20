"""Focused CPU tests for hard-mask compositing without GT leakage."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import composite_vace_results as composite


class CompositeTests(unittest.TestCase):
    def setUp(self):
        self.damaged = np.full((512, 512, 3), (12, 34, 56), dtype=np.uint8)
        self.prediction = np.full((512, 512, 3), (210, 180, 90), dtype=np.uint8)
        self.mask = np.zeros_like(self.damaged)

    def test_exact_hard_boundary_and_rgb_channels(self):
        self.mask[120:240, 220:350] = 255
        actual = composite.hard_composite(self.prediction, self.damaged, self.mask)
        np.testing.assert_array_equal(actual[120:240, 220:350], self.prediction[120:240, 220:350])
        np.testing.assert_array_equal(actual[self.mask == 0], self.damaged[self.mask == 0])
        np.testing.assert_array_equal(self.damaged[150, 250], [12, 34, 56])
        np.testing.assert_array_equal(self.prediction[150, 250], [210, 180, 90])

    def test_black_and_white_masks_do_not_use_other_frames(self):
        np.testing.assert_array_equal(composite.hard_composite(self.prediction, self.damaged, self.mask),
                                      self.damaged)
        self.mask.fill(255)
        np.testing.assert_array_equal(composite.hard_composite(self.prediction, self.damaged, self.mask),
                                      self.prediction)

    def test_nonbinary_or_channel_mismatched_mask_is_refused(self):
        for value in (127, (255, 0, 255)):
            self.mask[30, 40] = value
            with self.assertRaisesRegex(ValueError, "binary RGB"):
                composite.hard_composite(self.prediction, self.damaged, self.mask)

    def test_gt_is_not_an_input_to_composite(self):
        with self.assertRaises(TypeError):
            composite.hard_composite(self.prediction, self.damaged, self.mask, gt=self.damaged)

    def test_view_selection_includes_visible_failure_and_black_frame(self):
        masks = [np.full_like(self.mask, 255) for _ in range(81)]
        masks[52] = self.mask.copy()
        predictions = [self.damaged.copy() for _ in range(81)]
        predictions[37] = self.prediction.copy()
        views = composite.select_views({"mask": masks, "gt": [self.damaged] * 81}, predictions)
        self.assertEqual(len(set(views)), 5)
        self.assertTrue({0, 37, 52, 80}.issubset(views))

    def test_existing_output_root_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "composite_check"
            root.mkdir()
            marker = root / "old.txt"
            marker.write_text("protected")
            with patch.object(composite, "OUTPUT_ROOT", root):
                destination = composite.new_output_root()
            self.assertNotEqual(destination, root)
            self.assertTrue(destination.is_dir())
            self.assertEqual(marker.read_text(), "protected")


if __name__ == "__main__":
    unittest.main()
