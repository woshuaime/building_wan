"""The compute probe may replace the current worker only under the original selection rule."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_compute_speed import speed_choice


class ComputePolicyTests(unittest.TestCase):
    def test_matched_timing_and_memory(self):
        base = dict(mean_step_seconds=2.1, peak_nvml_bytes=34 << 30)
        for seconds, gib, expected in (
            (1.9, 35, True), (2.03, 33, True), (2.03, 35, False),
            (2.00, 35, False), (2.03, 34, True), (2.1, 33, False), (2.2, 33, False),
        ):
            with self.subTest(seconds=seconds, gib=gib):
                trial = dict(mean_step_seconds=seconds, peak_nvml_bytes=gib << 30)
                self.assertEqual(speed_choice(base, trial), expected)


if __name__ == "__main__":
    unittest.main()
