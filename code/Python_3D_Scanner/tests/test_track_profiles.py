import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from core.track import TrackSystem


class TrackProfileTests(unittest.TestCase):
    def setUp(self):
        self.system = TrackSystem()
        self.system.set_center([0.0, 0.0, 0.0])
        self.system.set_camera_distance(10.0)

    def test_cad_profile_keeps_four_complete_circles(self):
        blocks = self.system.get_all_camera_positions(
            elev_deg=0.0,
            azim_deg=0.0,
            num_positions=16,
        )

        self.assertEqual([len(block["positions"]) for block in blocks], [16] * 4)
        for block in blocks:
            points = np.asarray(block["positions"], dtype=np.float64)
            self.assertTrue(np.any(points[:, 0] > 1e-6))
            self.assertTrue(np.any(points[:, 0] < -1e-6))

    def test_building_profile_uses_ground_parallel_horizontal_and_upper_halves(self):
        blocks = self.system.get_building_camera_positions(num_positions=16)

        self.assertEqual([len(block["positions"]) for block in blocks], [16] * 4)

        horizontal = np.asarray(blocks[0]["positions"], dtype=np.float64)
        self.assertTrue(np.allclose(horizontal[:, 0], 0.0, atol=1e-9))
        self.assertTrue(np.allclose(horizontal[0], [0.0, 10.0, 0.0], atol=1e-9))

        for block in blocks[1:]:
            points = np.asarray(block["positions"], dtype=np.float64)
            self.assertTrue(np.all(points[:, 0] > 0.0))
            self.assertTrue(np.any(points[:, 1] > 1e-6) or np.any(points[:, 2] > 1e-6))
            self.assertTrue(np.any(points[:, 1] < -1e-6) or np.any(points[:, 2] < -1e-6))

            unit = points / np.linalg.norm(points, axis=1, keepdims=True)
            step_angles = np.degrees(
                np.arccos(
                    np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
                )
            )
            self.assertTrue(np.allclose(step_angles, 11.25, atol=1e-7))

    def test_building_profile_has_64_unique_positions_on_distinct_planes(self):
        blocks = self.system.get_building_camera_positions(num_positions=16)
        points_by_track = [
            np.asarray(block["positions"], dtype=np.float64)
            for block in blocks
        ]
        all_points = np.vstack(points_by_track)
        unique_points = np.unique(np.round(all_points, decimals=9), axis=0)
        self.assertEqual(len(unique_points), 64)

        vertical, diagonal_left, diagonal_right = points_by_track[1:]
        self.assertTrue(np.allclose(vertical[:, 1], 0.0, atol=1e-9))
        self.assertTrue(
            np.allclose(diagonal_left[:, 1], -diagonal_left[:, 2], atol=1e-9)
        )
        self.assertTrue(
            np.allclose(diagonal_right[:, 1], diagonal_right[:, 2], atol=1e-9)
        )

        for photo_index in range(16):
            positions = np.vstack(
                [track[photo_index] for track in points_by_track[1:]]
            )
            self.assertEqual(
                len(np.unique(np.round(positions, decimals=9), axis=0)),
                3,
            )

    def test_model_top_axis_matches_legacy_four_track_anchor(self):
        anchor = self.system.get_intersection_point()
        direction = anchor / np.linalg.norm(anchor)
        self.assertTrue(
            np.allclose(direction, [1.0, 0.0, 0.0], atol=1e-9)
        )

    def test_building_preview_track1_is_parallel_to_virtual_ground(self):
        preview_tracks = self.system.get_building_track_points_for_rendering(
            num_points=257,
        )

        self.assertEqual(len(preview_tracks), 4)
        self.assertTrue(
            np.allclose(preview_tracks[0][:, 0], self.system.center[0], atol=1e-9)
        )
        for points in preview_tracks[1:]:
            self.assertTrue(np.all(points[:, 0] >= -1e-9))

    def test_building_camera_up_keeps_horizontal_track_upright(self):
        self.assertTrue(
            np.array_equal(
                self.system.get_building_camera_view_up(0),
                np.array([1.0, 0.0, 0.0]),
            )
        )
        for track_index in (1, 2, 3):
            self.assertTrue(
                np.array_equal(
                    self.system.get_building_camera_view_up(track_index),
                    np.array([0.0, 0.0, 1.0]),
                )
            )

    def test_building_counts_remain_evenly_split(self):
        for total in (32, 64, 96):
            per_track = total // 4
            blocks = self.system.get_building_camera_positions(
                num_positions=per_track,
            )
            self.assertEqual(sum(len(block["positions"]) for block in blocks), total)

if __name__ == "__main__":
    unittest.main()
