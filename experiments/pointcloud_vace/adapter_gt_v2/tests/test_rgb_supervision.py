from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import rgb_supervision as rgb


class TestRGBSupervision(unittest.TestCase):
    def test_shared_roof_wall_edge_survives_but_triangulation_diagonal_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "house.obj"
            mesh.write_text(
                "v 0 0 1\nv 2 0 1\nv 2 2 1\nv 0 2 1\nv 0 0 0\nv 2 0 0\n"
                "f 1 2 3\nf 1 3 4\nf 5 6 2\nf 5 2 1\n")
            segments = rgb._roof_segments(str(mesh))
        np.testing.assert_array_equal(segments, [[[0, 0, 1], [2, 0, 1]]])

    def test_applied_square_affine_matches_mask_and_overlaps_are_capped(self):
        origin = np.array([652170.38, 4862170.18, 0.])
        line = np.array([[[-1., 0., 0.], [1., 0., 0.]]]) + origin
        camera = dict(world_to_local_origin=origin, camera_poses_local=[
            dict(position=[0., 0., -10.], right=[1., 0., 0.],
                 up=[0., -1., 0.], forward=[0., 0., 1.])])
        scale = 512 / 832
        transform = (scale, scale, (scale - 1) / 2, (176 + .5)*scale - .5)
        with patch.object(rgb, "_roof_segments", return_value=np.repeat(line, 2, axis=0)):
            weights = rgb.structure_weights("mock.obj", camera, [0], (512, 512), transform)
        self.assertEqual(weights.shape, (1, 512, 512))
        self.assertEqual(weights.max(), 1.)
        self.assertEqual(weights[0, 256, 256], 1.)
        self.assertTrue(0 < weights[0, 259, 256] < 1)
        self.assertEqual(weights[0, 267, 256], 0.)
        with patch.object(rgb, "_roof_segments", return_value=np.empty((0, 2, 3))):
            self.assertFalse(rgb.structure_weights("mock.obj", camera, [0], (512, 512), transform).any())

    def test_no_exact_hidden_point_matching(self):
        gt = torch.full((5, 64, 64), 240/255)
        gt[:, 22, 21] = .35
        pred = gt.roll(1, dims=-1)[:, None].repeat(1, 3, 1, 1)
        blank = torch.full_like(gt, 240/255)
        loss = rgb.rgb_losses(pred, gt, blank, torch.ones_like(gt), torch.zeros_like(gt), 1)
        self.assertLess(loss["appearance"].item(), 1.e-6)
        self.assertEqual(loss["structure"].item(), 0.)
        self.assertEqual(loss["temp"].item(), 0.)
        pred[:, 1, 22, 22] += .15
        changed = rgb.rgb_losses(pred, gt, blank, torch.ones_like(gt), torch.zeros_like(gt), 1)
        self.assertGreater(changed["appearance"].item(), 1.e-3)

    def test_motion_matches_gt_and_known_pixels_receive_no_auxiliary_derivatives(self):
        gt = torch.linspace(.6, .76, 5)[:, None, None].expand(5, 64, 64)
        partial = torch.full_like(gt, 240/255)
        partial[:, 24, 24] = .35
        edit = torch.ones_like(gt)
        edit[:, :12] = 0
        pred = (gt[:, None].repeat(1, 3, 1, 1) + .025).requires_grad_()
        static_bias = rgb.rgb_losses(pred, gt, partial, edit, torch.ones_like(gt), 1000)
        self.assertLess(static_bias["temp"].item(), 1.e-6)
        with torch.no_grad():
            pred[1, :, 32:48, 32:48] += .1
        loss = rgb.rgb_losses(pred, gt, partial, edit, torch.ones_like(gt), 1000)
        self.assertGreater(loss["temp"].item(), 1.e-3)
        torch.testing.assert_close(loss["aux"], .1*loss["appearance"] + .1*loss["structure"] + .02*loss["temp"])
        loss["aux"].backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().max().item(), 0)
        self.assertEqual(pred.grad[:, :, :12].abs().max().item(), 0)
        self.assertEqual(pred.grad[:, :, 23:26, 23:26].abs().max().item(), 0)
        self.assertEqual(rgb.ramp(1), .1)
        self.assertEqual(rgb.ramp(1000), 1.)
        self.assertEqual(rgb.ramp(32241), 1.)

    def test_only_first_two_latents_supply_real_frames_zero_to_four(self):
        class Causal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)

            def decode(self, z, scale):
                self.indices = z.shape[2]
                return torch.cat([z[:, :3, :1], z[:, :3, 1:].repeat_interleave(4, dim=2)], dim=2)

            def clear_cache(self):
                pass

        model = Causal()
        vae = type("VAE", (), dict(model=model, scale=None))()
        latent = torch.randn(1, 16, 5, 8, 8, requires_grad=True)
        prefix = rgb.decode_prefix(vae, latent)
        prefix.sum().backward()
        self.assertEqual(model.indices, 2)
        self.assertEqual(prefix.shape, (5, 3, 8, 8))
        self.assertEqual(latent.grad[:, :, 2:].abs().sum().item(), 0)
        self.assertGreater(latent.grad[:, :, :2].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
