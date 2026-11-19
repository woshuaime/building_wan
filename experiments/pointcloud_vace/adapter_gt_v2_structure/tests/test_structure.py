import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from structure_data import (TRAIN_COUNT, VALIDATION_IDS, PreparedData, data,
                            frame_window, load_observation, load_targets, delivery)
from mesh_labels import RIDGE, structural_edges
from frame_model import BuildingAdapter, FEATURES, pack_tokens, project_xyz, splat
from objectives import (CROP, camera_tensor, choose_rgb_window, decode_pixels, framework_loss,
                        observed_tensor, point_loss, rgb_losses, target_tensor)
from quality import geometry_scores
from train import geometry_for_window, learning_rate, rng_state, restore_rng


class LabelsTest(unittest.TestCase):
    def test_real_roof_welds_faces_and_keeps_a_single_ridge(self):
        xyz = np.array([[-1., 0., 0.], [0., 0., 1.], [1., 0., 0.],
                        [-1., 2., 0.], [0., 2., 1.], [1., 2., 0.],
                        [0., 0., 1.]])
        triangles = np.array([[0, 1, 3], [6, 4, 3], [1, 2, 4], [2, 5, 4]])
        result = structural_edges(xyz, triangles)
        self.assertEqual(result["kinds"].tolist(), [RIDGE])
        self.assertEqual(result["edges"].tolist(), [[0, 1]])
        self.assertAlmostEqual(np.linalg.norm(np.diff(result["vertices"], axis=0)), 2.)
        self.assertTrue((result["supports"][0, :, 2] > 0).all())

    def test_flat_triangles_do_not_create_an_edge(self):
        xyz = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        result = structural_edges(xyz, np.array([[0, 1, 2], [0, 2, 3]]))
        self.assertEqual(result["edges"].shape, (0, 2))
        self.assertEqual(result["vertices"].shape, (0, 3))

    def test_real_gt_visualization_uses_the_original_display_subset(self):
        row = delivery.source_row("2825")
        from structure_data import CLOUDS, MESHES

        row.update(mesh=str(MESHES / "2825.obj"), gt_cloud=str(CLOUDS / "2825.ply"))
        camera, record = (data.read_json(row[key]) for key in ("camera", "record"))
        target = load_targets(row, camera, record)
        self.assertEqual(len(target["display_xyz"]), 4096)
        self.assertEqual(len(target["full_xyz"]), 16384)
        self.assertGreater(np.count_nonzero(target["kinds"] == RIDGE), 0)
        obs = load_observation(row, camera, record)
        uv, z = data.project(target["display_xyz"], obs["centers"][0], obs["rotations"][0],
                             obs["intrinsics"])
        # A center can coincide with an antialiased footprint, but projected
        # points cannot require a solid surface in this source video.
        source = data.decode_gray_window(row["GT_video"], list(range(17)))[0]
        xy = np.rint(uv).astype(int)
        inside = (z > 0) & ((xy >= 0) & (xy < [832, 480])).all(-1)
        self.assertTrue((source[xy[inside, 1], xy[inside, 0]] < 240).all())


def camera(frames=17):
    return dict(center=torch.tensor([[0., 0., -5.]]).expand(frames, -1),
                rotation=torch.eye(3).expand(frames, -1, -1),
                intrinsics=torch.tensor([[514., 0., 415.5], [0., 514., 239.5], [0., 0., 1.]]))


class ModelTest(unittest.TestCase):
    def test_same_projection_preserves_far_points_and_pixel_centers(self):
        x = torch.tensor([[0., 0., 0.], [0., 0., 1.]], requires_grad=True)
        uv, depth = project_xyz(x, camera(frames=1))
        torch.testing.assert_close(uv, torch.tensor([[[415.5, 239.5], [415.5, 239.5]]]))
        grid = splat(uv, depth, torch.tensor([[[1., 5.], [1., 6.]]]), 60, 104, stride=8)
        torch.testing.assert_close(grid.sum((1, 2)), torch.tensor([[2., 11.]]))
        grid[0, 29, 51, 0].backward()
        self.assertTrue(torch.count_nonzero(x.grad))

    def test_token_shapes_cover_both_training_and_orbit(self):
        for frames, count in ((17, 7800), (81, 32760)):
            packed = pack_tokens(torch.zeros(frames, 60, 104, FEATURES))
            self.assertEqual(packed.shape, (count, 16 * FEATURES))
        model = BuildingAdapter(width=32)
        torch.manual_seed(19)
        with torch.no_grad():
            building = model(torch.randn(128, 3) * .8)
            projected = model.project(building, camera(frames=17))
        self.assertEqual(projected.shape, (1, 7800, 1536))
        self.assertTrue(torch.isfinite(projected).all())
        self.assertEqual(int(building.point_weights(True).sum()), int(building.count.round()))

    def test_actual_branch_losses_and_projection_backpropagate(self):
        torch.manual_seed(19)
        model = BuildingAdapter(width=32)
        xyz = torch.randn(128, 3) * .7
        result = model(xyz)
        density = dict(display=xyz[:64], surface=xyz, count=128,
                       point_kind=torch.zeros(64, dtype=torch.long))
        shape = dict(vertices=xyz[:2], edges=torch.tensor([[0, 1]]), kinds=torch.tensor([RIDGE]),
                     edge_normals=torch.tensor([[[.6, 0., .8], [-.6, 0., .8]]]))
        loss = (framework_loss(result, shape) + point_loss(result, density)
                + model.project(result, camera()).square().mean())
        loss.backward()
        for parameter in (model.corner_output[-1].weight, model.edge_output[-1].weight,
                          model.surface_output[-1].weight, model.point_offsets[-1].weight,
                          model.count_output[-1].weight, model.condition[-1].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertTrue(torch.count_nonzero(parameter.grad))

    def test_ridge_report_can_reject_a_straight_displaced_edge(self):
        straight = np.array([[0., 0., 0.], [3., 0., 0.]])
        predicted = dict(vertices=straight + [0, .75, 0], edge_scores=np.array([[0., 1., 0., 0., 0.]]),
                         pairs=np.array([[0, 1]]), corner_scores=np.ones(2), points=straight)
        target = dict(vertices=straight, kinds=np.array([RIDGE]), edges=np.array([[0, 1]]),
                      display_xyz=straight, full_xyz=straight)
        report = geometry_scores(predicted, target, 1.)
        self.assertEqual(report["ridge_gt"], 1)
        self.assertEqual(report["ridge_predicted"], 1)
        self.assertEqual(report["ridge_matches"], 0)


class RGBTest(unittest.TestCase):
    def test_neutral_color_and_target_motion_have_zero_error(self):
        gt = torch.full((17, 64, 64), 240 / 255)
        gt[:, 25:36:2, 25:36:2] = 80 / 255
        tracks = torch.tensor([[[29., 29.], [31., 31.]]]).expand(17, -1, -1)
        exact = gt[None, None].expand(1, 3, -1, -1, -1)
        losses = rgb_losses(exact, gt, tracks)
        self.assertLess(float(losses["pixel"]), 1.e-6)
        self.assertLess(float(losses["chroma"]), 1.e-6)
        self.assertEqual(float(losses["track"]), 0.)
        tinted = (exact + torch.tensor([0., .03, .05]).view(1, 3, 1, 1, 1)).requires_grad_()
        losses = rgb_losses(tinted, gt, tracks)
        self.assertGreater(float(losses["chroma"].detach()), .01)
        sum(losses.values()).backward()
        self.assertTrue(torch.count_nonzero(tinted.grad))

    def test_checkpoint_preserves_causal_vae_input_gradients(self):
        from diffsynth.models.wan_video_vae import VideoVAE_
        from types import SimpleNamespace

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            core = VideoVAE_(dim=8, z_dim=16, num_res_blocks=1).eval().requires_grad_(False)
            vae = SimpleNamespace(model=core, scale=[torch.zeros(16), torch.ones(16)])
            z = torch.randn(1, 16, 2, 5, 5, requires_grad=True)
            reference = decode_pixels(vae, z, False)
            expected, = torch.autograd.grad(reference.square().mean(), (z,))
            actual = decode_pixels(vae, z, True)
            got, = torch.autograd.grad(actual.square().mean(), (z,))
            self.assertEqual(actual.shape, (1, 3, 5, 40, 40))
            torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(got, expected)
            self.assertTrue(torch.count_nonzero(got))
            self.assertTrue(all(p.grad is None for p in core.parameters()))


class RunTest(unittest.TestCase):
    def test_five_distinct_windows_and_continuous_resume_rate(self):
        for building in VALIDATION_IDS:
            windows = [tuple(frame_window(building, epoch)) for epoch in range(5)]
            self.assertEqual(len(set(windows)), 5)
            self.assertTrue(all(len(seq) == 17 and max(seq) < 80 for seq in windows))
        boundary = 3 * TRAIN_COUNT
        self.assertAlmostEqual(learning_rate(boundary, .0001), .00002)
        self.assertAlmostEqual(learning_rate(boundary + 1, .0001), .00002, places=10)
        self.assertAlmostEqual(learning_rate(5 * TRAIN_COUNT, .0001), .00001)

    def test_checkpoint_files_allow_weights_only_and_recover_rng_on_cpu(self):
        from unittest.mock import patch
        import random

        np.random.seed(72)
        random.seed(71)
        torch.manual_seed(73)
        with tempfile.TemporaryDirectory() as workspace:
            file = Path(workspace) / "rng.pt"
            with patch("torch.cuda.get_rng_state_all", return_value=[]):
                torch.save(rng_state(), file)
            result = (random.random(), np.random.random(), torch.rand(()))
            with patch("torch.cuda.set_rng_state_all") as cuda:
                restore_rng(torch.load(file, weights_only=True))
            cuda.assert_called_once_with([])
            self.assertEqual((random.random(), np.random.random(), torch.rand(())), result)

    def test_pause_resume_and_extension_commit_each_window_once(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch

        from common import atomic_torch_save
        from run import Paused, Progress
        import train as engine

        frames = list(range(17))
        train_data = SimpleNamespace(schedule=[(str(i), frames) for i in range(10)])
        steps = []

        class FakeTrainer:
            encoder = None

            def __init__(self, run, _data):
                self.run = run

            def update(self, _raw, _encoded, index):
                steps.append(index)
                if len(steps) == 1:
                    (self.run / "stop.request").touch()
                return dict(loss=.1)

            def save(self, path, count, epochs, log_bytes, elapsed, peak):
                value = dict(updates=count, goal_epochs=epochs, log_bytes=log_bytes, elapsed_seconds=elapsed,
                             peak_vram_bytes=peak, dataset="test")
                value.update({key: {"weight": torch.zeros(1)} for key in ("adapter", "lora", "structure")})
                atomic_torch_save(path, value)
                return value

            def restore(self, value):
                if value["updates"] != len(steps):
                    raise AssertionError("Checkpoint is not at the committed update")

            def epoch_review(self, epoch, _progress):
                (self.run / f"epoch_{epoch}_review.json").write_text("{}")

        class FakeWindows:
            def __init__(self, *_):
                pass

            def build(self, *_):
                return {}, {}

            def extract(self, _index, _raw, encoded):
                return encoded

            def discard_committed(self, _count):
                pass

        with tempfile.TemporaryDirectory() as local, patch.multiple(
                engine, PreparedData=lambda _path: train_data, Trainer=FakeTrainer,
                RollingWindows=FakeWindows, TRAIN_COUNT=2), patch("torch.cuda.max_memory_reserved", return_value=7):
            run = Path(local)
            device = dict(device_name="CPU test", used_bytes=0)
            with self.assertRaises(Paused):
                engine.train(run, 3, Progress(run, "testing"), device)
            (run / "stop.request").unlink()
            engine.train(run, 3, Progress(run, "testing"), device)
            engine.train(run, 5, Progress(run, "testing"), device)
            self.assertEqual(steps, list(range(10)))
            records = [json.loads(line) for line in (run / "loss.jsonl").read_text().splitlines()]
            self.assertEqual([row["update"] for row in records], list(range(1, 11)))
            self.assertTrue(all((run / f"checkpoints/epoch_{epoch}.pt").is_file() for epoch in range(1, 6)))
            self.assertTrue((run / "weights_3/structure.safetensors").is_file())
            self.assertTrue((run / "weights_5/structure.safetensors").is_file())

    @unittest.skipUnless(os.environ.get("STRUCTURE_PREPARED_RUN"), "requires the prepared experiment DB")
    def test_real_prepared_window_to_finite_geometry_and_rgb_targets(self):
        run = Path(os.environ["STRUCTURE_PREPARED_RUN"]).resolve()
        prepared = PreparedData(run)
        building, frames = prepared.schedule[0]
        raw = prepared.window(building, frames)
        xyz = observed_tensor(raw["observed"], "cpu")
        target = target_tensor(raw, "cpu")
        top, left, tracks = choose_rgb_window(raw, 0)
        self.assertEqual(tracks.shape[0], 17)
        self.assertEqual(raw["target"][:, top:top+CROP, left:left+CROP].shape, (17, CROP, CROP))
        self.assertTrue(torch.isfinite(xyz).all())
        self.assertEqual(len(prepared.schedule), TRAIN_COUNT * 5)
        self.assertEqual(len(set(i for i, _ in prepared.schedule[:TRAIN_COUNT])), TRAIN_COUNT)
        self.assertEqual(len(target["surface"]), 1024)
        geometry = geometry_for_window(raw["observed"], frames)
        self.assertEqual(geometry["target_depths"].shape, (17, 2, 240, 416))
        self.assertTrue(all(np.isfinite(value).all() for value in geometry.values()))


if __name__ == "__main__":
    unittest.main()
