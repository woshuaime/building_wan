"""CPU checks for frozen pairing, world precision, and true VAE loss layout."""
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

import numpy as np

import data


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_root = data.ROOT / "cache/pointcloud_vace/adapter_gt_v2/cpu_tests"
        cls.temporary_root.mkdir(parents=True, exist_ok=True)

    def test_schedule_every_building_once_each_epoch_and_three_distinct_windows(self):
        ids = [str(i) for i in range(10747)]
        schedule = data.schedule_for(ids)
        self.assertEqual(len(schedule), 32241)
        self.assertEqual(schedule, data.schedule_for(ids))
        windows = {i: [] for i in ids}
        for epoch in range(3):
            subset = schedule[epoch * len(ids):(epoch + 1) * len(ids)]
            self.assertEqual({r["id"] for r in subset}, set(ids))
            self.assertTrue(all(r["epoch"] == epoch for r in subset))
            for row in subset:
                self.assertEqual(row["frame_ids"], list(range(row["start"], row["start"] + 17)))
                self.assertLessEqual(row["frame_ids"][-1], 79)
                windows[row["id"]].append(row["start"])
        self.assertTrue(all(len(set(starts)) == 3 for starts in windows.values()))
        self.assertEqual([r["sample_index"] for r in schedule], list(range(32241)))

    def test_loss_first_frame_then_four_frame_groups_and_area_normalization(self):
        target = np.full((17, 480, 832), 240, dtype=np.uint8)
        target[0, 11, 11] = 80
        target[5, 11, 11] = 80
        weights = data.occupancy_for(target)
        self.assertEqual(weights.shape, (2, 5, 60, 104))
        np.testing.assert_allclose(weights.sum(axis=0), 1)
        self.assertAlmostEqual(float(weights[0, 0, 1, 1]), 49 / 64)
        self.assertEqual(float(weights[0, 1].sum()), 0)
        self.assertAlmostEqual(float(weights[0, 2, 1, 1]), 49 / (64 * 4))
        self.assertEqual(float(weights[0, 3:].sum()), 0)

    def test_world_origin_subtracted_before_float32_and_frozen_indices_preserved(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            path = Path(temporary) / "17.xyz"
            source = np.array([[474308.001, 6533276.002, 8.003],
                               [474308.104, 6533276.205, 8.306],
                               [474308.407, 6533276.508, 8.609]], np.float64)
            np.savetxt(path, source, fmt="%.10f")
            row = dict(id="17", pointcloud=str(path), source_identity=data.identity(path))
            camera = dict(world_to_local_origin=[474308., 6533276., 8.],
                          GT_PLY_local_to_world_offset=[999999., 9999999., 999.])
            record = dict(partial_original_points=3, partial_display_points=3,
                          selected_source_point_indices=dict(partial=[2, 0, 1]))
            actual = data.selected_partial(row, camera, record)
            np.testing.assert_allclose(actual, source[[2, 0, 1]] - camera["world_to_local_origin"], atol=1.e-10)
            self.assertAlmostEqual(float(actual[1, 1]), .002, places=8)
            self.assertEqual(actual.dtype, np.float64)

    def test_immutable_source_change_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            path = Path(temporary) / "2.xyz"
            path.write_text("0 0 0\n1 0 0\n0 1 0\n")
            row = dict(id="2", pointcloud=str(path), source_identity=data.identity(path))
            path.write_text("0 0 0\n1 0 0\n0 2 0\n3 3 3\n")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                data.selected_partial(row, {}, {})

    def test_projection_uses_pixel_center_and_depth_scales_consistently(self):
        pose = dict(position=[0, 0, 0], right=[1, 0, 0], up=[0, -1, 0], forward=[0, 0, 1])
        centers, rotations, K = data.camera_arrays(dict(camera_poses_local=[pose]))
        uv, z = data.project(np.array([[0., 0., 10.], [0., 0., 20.]]), centers[0], rotations[0], K)
        np.testing.assert_allclose(uv, [[415.5, 239.5], [415.5, 239.5]])
        depth = data.depth_view(np.array([[0., 0., 10.], [0., 0., 20.]]), centers[0], rotations[0], K, 5.)
        self.assertEqual(depth.shape, (2, 240, 416))
        self.assertEqual(float(depth[0, 120, 208]), 2.)
        self.assertEqual(float(depth[1].sum()), 5.)

    def test_manifest_cannot_select_preview_or_escape_data_root(self):
        root = data.ROOT / "data/pointcloud_vace/pointcloud_video_pairs_v2"
        for value in ("check_20261115/1.mp4", "../gt/1.mp4", "/tmp/1.mp4"):
            with self.assertRaises(ValueError):
                data._relative(root, value, "GT_video", "1")

    def test_frame_080_cannot_enter_training_window(self):
        with self.assertRaises(ValueError):
            data.decode_gray_window("unused", list(range(64, 81)))

    def test_uploaded_zip_equal_size_corruption_is_detected_and_archive_retained(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            base = root / "data/pointcloud_vace/pointcloud_video_pairs_v2"
            target = base / "partial/1.mp4"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"different")
            upload = root / "data/pointcloud_vace/gt_video_v2_upload"
            upload.mkdir()
            archive = upload / "GT视频V2_A6000_3epoch_完整训练上传包_20261115.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("pointcloud_video_pairs_v2/partial/1.mp4", b"unchanged")
            run = root / "run"
            run.mkdir()
            with patch.object(data, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Extracted upload CRC mismatch"):
                    data._uploaded_archives(run, base)
            self.assertTrue(archive.exists())
            self.assertEqual(target.read_bytes(), b"different")

    def test_stop_during_import_is_explicit_interruption(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            run = Path(temporary)
            (run / "stop.request").write_text("stop")
            with self.assertRaises(InterruptedError):
                data._progress(run, "checking_manifest_headers_and_sources")


if __name__ == "__main__":
    unittest.main()
