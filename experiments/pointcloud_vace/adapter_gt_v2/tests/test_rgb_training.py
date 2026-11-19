import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import data
import rgb_window as windows
import train_rgb
import validation_delivery as delivery


class RGBTraining(unittest.TestCase):
    def test_original_bf16_timestep_and_sigma(self):
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        scheduler = FlowMatchScheduler(template='Wan')
        scheduler.set_timesteps(1000, training=True, shift=5)
        times, nearest, eligible = train_rgb.lookup(scheduler, (.1, .6))
        x = torch.ones(1, dtype=torch.float32)
        noise = torch.zeros_like(x)
        first_pool = []
        for index, time in enumerate(times):
            sigma = scheduler.sigmas[nearest[index]]
            self.assertTrue(torch.equal(scheduler.add_noise(x, noise, time), (1.-sigma)*x))
            self.assertTrue(torch.equal(scheduler.training_weight(time), scheduler.linear_timesteps_weights[nearest[index]]))
            if .1 <= sigma <= .6:
                first_pool.append(index)
        self.assertEqual(eligible.tolist(), first_pool)

    def test_square_camera_center_and_solid_gray_border(self):
        plane = np.full((1, data.HEIGHT, data.WIDTH), 240, np.uint8)
        plane[:, 237:243, 413:419] = 80
        rgb = windows.square_frames(plane)[0]
        sx, sy, tx, ty = windows.AFFINE
        np.testing.assert_allclose((415.5*sx+tx, 239.5*sy+ty), (255.5, 255.5), atol=1e-12, rtol=0.)
        self.assertEqual(rgb[256, 256], 80)
        self.assertTrue((rgb[:windows.VALID_START] == 240).all())
        self.assertTrue((rgb[windows.VALID_STOP:] == 240).all())

    def test_wrap_matches_independently_read_source_windows(self):
        path = delivery.DATASET / 'partial/1.mp4'
        if not path.exists():
            self.skipTest('V2 source video is unavailable')
        frames = [*range(74, 80), *range(11)]
        ring = windows.read_window(path, frames)
        head = data.decode_gray_window(path, range(17))
        tail = data.decode_gray_window(path, range(63, 80))
        expected = windows.square_frames(np.concatenate((tail[-6:], head[:11])))
        np.testing.assert_array_equal(ring, expected)


if __name__ == '__main__':
    unittest.main()
