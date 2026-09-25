import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from utils.screenshot import ScreenshotManager


class ScreenshotOrientationTests(unittest.TestCase):
    @staticmethod
    def _top_red_bottom_blue():
        image = np.zeros((4, 3, 3), dtype=np.uint8)
        image[:2, :, 0] = 255
        image[2:, :, 2] = 255
        return image

    def test_normal_capture_preserves_top_to_bottom_row_order(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = ScreenshotManager(folder)
            path = manager.save_screenshot_from_array(
                self._top_red_bottom_blue(),
                track_index=1,
                photo_index=1,
            )
            saved = np.asarray(Image.open(path).convert("RGB"))

        self.assertTrue(np.all(saved[0, :, 0] == 255))
        self.assertTrue(np.all(saved[-1, :, 2] == 255))

    def test_dataset_capture_defaults_to_the_same_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = ScreenshotManager(folder)
            path = os.path.join(folder, "layer.png")
            manager.save_array_to_path(self._top_red_bottom_blue(), path)
            saved = np.asarray(Image.open(path).convert("RGB"))

        self.assertTrue(np.all(saved[0, :, 0] == 255))
        self.assertTrue(np.all(saved[-1, :, 2] == 255))


if __name__ == "__main__":
    unittest.main()
