"""Execution-only VAE acceleration retains windows, tiles and blend order."""
from pathlib import Path
import sys
import types
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fast_vae import tile_tasks, tiled_encode, install


class FastVaeTests(unittest.TestCase):
    def test_same_nine_spatial_tiles(self):
        tasks = tile_tasks(480, 832, (240, 416), (120, 208))
        self.assertEqual(tasks, [(h, h + 240, w, w + 416) for h in (0, 120, 240) for w in (0, 208, 416)])

    def test_batching_keeps_every_output_and_mask_order(self):
        calls = []
        class Model:
            def encode(self, x, scale):
                calls.append(len(x))
                # Each item has a different border-dependent output, ensuring
                # wrong tile order/placement cannot hide behind a constant.
                return x[:, :1, :1, ::2, ::2] + x[:, :1, :1].mean((3, 4), keepdim=True)
        class Vae:
            upsampling_factor, z_dim, scale, model = 2, 1, None, Model()
            def build_mask(self, tile, is_bound, border_width):
                return torch.ones((1, 1, 1, *tile.shape[-2:]))
        video = torch.arange(1 * 3 * 1 * 16 * 24, dtype=torch.float32).reshape(1, 3, 1, 16, 24)
        a = tiled_encode(Vae(), video, "cpu", (8, 12), (4, 6), tile_batch=1)
        calls.clear()
        b = tiled_encode(Vae(), video, "cpu", (8, 12), (4, 6), tile_batch=3)
        self.assertEqual(calls, [3, 3, 3])
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_boundary_tiles_are_not_reshaped_into_another_sample(self):
        tasks = tile_tasks(18, 26, (8, 12), (4, 6))
        self.assertTrue(all(0 <= h < h2 <= 18 and 0 <= w < w2 <= 26 for h, h2, w, w2 in tasks))

    def test_original_encoder_can_be_restored(self):
        def original(*args):
            return "original"
        vae = types.SimpleNamespace(tiled_encode=original)
        install(vae, dict(tile_batch=2))
        self.assertIsNot(vae.tiled_encode, original)
        install(vae, dict(original=True))
        self.assertIs(vae.tiled_encode, original)


if __name__ == "__main__":
    unittest.main()
