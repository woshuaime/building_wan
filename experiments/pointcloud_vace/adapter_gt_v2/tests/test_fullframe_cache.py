"""A tiled cache or stale window must never enter full-frame training."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fullframe_cache as cache


class FullFrameInputs(unittest.TestCase):
    def setUp(self):
        self.row = dict(sample_index=3, id="41", start=4, frame_ids=list(range(4, 21)))
        self.value = dict(id="41", frame_ids=self.row["frame_ids"], cache_key="new-key", encoding_id="new-run",
                          input_latents=torch.zeros(cache.LATENT_SHAPE, dtype=torch.bfloat16),
                          reactive=torch.zeros(cache.LATENT_SHAPE, dtype=torch.bfloat16),
                          occupancy=torch.zeros(cache.AREA_SHAPE, dtype=torch.float32))

    def test_tiled_or_wrong_building_rejected(self):
        cache.validate(self.value, self.row, "new-key", "new-run")
        for edit in (dict(encoding_id="old-run"), dict(cache_key="tiled-key"),
                     dict(id="43"), dict(frame_ids=list(range(5, 22)))):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                cache.validate(dict(self.value, **edit), self.row, "new-key", "new-run")

    def test_absent_entry_cannot_fall_back_to_v1_encoding(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(cache, "cache_dir", return_value=Path(temporary)):
            with self.assertRaises(FileNotFoundError):
                cache.read_sample(Path(temporary), None, self.row, "frozen")
            with self.assertRaises(ValueError):
                cache.read_sample(Path(temporary), None, self.row, "online")

    def test_prefetch_preserves_schedule_and_surfaces_errors(self):
        rows = [2, 5, 1, 0]
        self.assertEqual(list(cache.prefetch(rows, lambda i: i + 1)), [(2, 3), (5, 6), (1, 2), (0, 1)])
        with self.assertRaises(ZeroDivisionError):
            list(cache.prefetch(rows, lambda i: 1 / i))


if __name__ == "__main__":
    unittest.main()
