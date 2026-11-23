import sys
import unittest
from pathlib import Path

import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))
sys.path.insert(0, str(EXPERIMENT.parent / "adapter_gt_v2_parallax"))
from adapter import MultiViewStructureAdapter
import parallax_geometry as geometry
import rgb_supervision as rgb
from rgb_recipe import orbit_schedule
from tracks import measure


class ParallaxTests(unittest.TestCase):
    def test_depth_at_sparse_boundary_and_weight_migration(self):
        adapter = MultiViewStructureAdapter()
        original = adapter.state_dict()
        maps = torch.tensor([[[[[30., 30.], [30., 0.]], [[1., 1.], [1., 0.]]]]])
        xy = torch.full((1, 1, 1, 2), 1.5)
        self.assertEqual(adapter.sample_depth(maps, xy)[0].item(), 22.5)
        adapter.enable_parallax()
        depth, support = adapter.sample_depth(maps, xy)
        self.assertEqual((depth.item(), support.item()), (30., .75))
        self.assertEqual(set(adapter.state_dict()) - set(original), {"parallax_proj.weight"})
        self.assertEqual(adapter.parallax_proj.weight.count_nonzero().item(), 0)
        restored = MultiViewStructureAdapter()
        restored.enable_parallax()
        restored.load_state_dict(adapter.state_dict(), strict=True)

    def test_soft_projection_and_ordered_frame_slots(self):
        uv = np.array([[4., 4.], [4., 4.]])
        z = np.array([-.6, .6])
        bins = geometry.subcells(uv, z, (16, 16)).reshape(16, 2, 2)
        np.testing.assert_allclose(np.expm1(bins[..., 0]).sum(0), [1., 1.], rtol=1.e-6)
        self.assertTrue((bins[..., 1].sum(0) * z > 0).all())
        self.assertFalse(np.array_equal(bins.ravel(), geometry.subcells(uv+.1, z, (16, 16)).ravel()))
        xyz = np.array([[0., 0., 10.], [1., 0., 11.]])
        centers = np.zeros((5, 3))
        centers[:, 0] = np.arange(5) * .5
        rot = np.tile(np.eye(3), (5, 1, 1))
        K = np.array([[100., 0, 416], [0, 100., 240], [0, 0, 1.]])
        out = geometry.hints(xyz, centers, rot, K, range(5), 1.).reshape(2, 1560, 4, 64)
        for slot in range(4):
            np.testing.assert_array_equal(out[0,:,slot], out[0,:,0])
            projected = geometry.hints(xyz, centers, rot, K, [slot+1], 1.).reshape(1560,4,64)
            np.testing.assert_array_equal(out[1,:,slot], projected[:,0])
        self.assertFalse(np.array_equal(out[1,:,0], out[1,:,3]))

    def test_point_shape_versus_contrast_and_known_edges(self):
        cfg = rgb.PARALLAX_FEATURES
        truth = torch.full((1, 1, 64, 64), 240.) / 255.
        truth[..., ::4, ::4] = 80/255.
        lighter = 240/255. - (240/255.-truth)*.6
        clump = torch.full_like(truth, 240/255.)
        for y in (0, 1):
            for x in (0, 1):
                clump[..., y::8, x::8] = 80/255.
        region = torch.ones_like(truth)
        self.assertLess(rgb.point_pairs(lighter, truth, region, cfg).item(), .002)
        clump.requires_grad_()
        loss = rgb.point_pairs(clump, truth, region, cfg)
        self.assertGreater(loss.item(), .02)
        loss.backward()
        self.assertTrue(torch.isfinite(clump.grad).all() and clump.grad.abs().sum() > 0)
        obs = torch.tensor([80., 180., 240.]).reshape(1,1,1,3)/255
        keep = lambda values: rgb.known_contribution(torch.tensor(values).reshape_as(obs)/255.,
                                                     obs, torch.ones_like(obs), cfg).item()
        self.assertEqual(keep([80., 100., 120.]), 0.)
        self.assertGreater(keep([40., 180., 240.]), 0.)
        self.assertGreater(keep([150., 200., 240.]), 0.)

    def test_shifted_prefix_and_pair_tracking(self):
        old = orbit_schedule(["1"])
        new = orbit_schedule(["1"], offset=13)
        old_views = {(s+i)%80 for _, s in old for i in range(5)}
        self.assertFalse(old_views.intersection((s+i)%80 for _, s in new for i in range(5)))
        rng = np.random.default_rng(3)
        templates = rng.integers(100, 220, (2,9,9))
        templates[:,4,4] = 30
        uv = np.array([[[40+2*t,48], [92-t,64]] for t in range(5)])
        def clip(trajectory):
            images = np.full((5,128,144),240.,np.float32)
            for t, points in enumerate(trajectory):
                for n,(x,y) in enumerate(points):
                    images[t,y-4:y+5,x-4:x+5] = templates[n]
            return images
        pairs = np.array([[0,1]])
        self.assertEqual(measure(clip(uv), uv, pairs)["passed"], 1)
        flat = uv.copy()
        flat[:,1,0] = 92+2*np.arange(5)
        self.assertEqual(measure(clip(flat), uv, pairs)["passed"], 0)
        lost = clip(uv)
        lost[3:] = 240
        m = measure(lost, uv, pairs)
        self.assertEqual((m["total"], m["passed"], len(m["far"])), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
