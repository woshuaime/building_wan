import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from core.track import TrackSystem
from scripts.capture_building_video_trials import (
    MAX_ANGULAR_STEP_DEG,
    build_video_sequence,
    upper_hemisphere_transition,
)


class BuildingVideoSequenceTests(unittest.TestCase):
    def setUp(self):
        self.system = TrackSystem()
        self.system.set_center([0.0, 0.0, 0.0])
        self.system.set_camera_distance(10.0)

    def test_sequence_keeps_all_track_frames_and_adds_three_transitions(self):
        sequence, transitions = build_video_sequence(self.system)

        track_frames = [frame for frame in sequence if frame["kind"] == "track"]
        transition_frames = [
            frame for frame in sequence if frame["kind"] == "transition"
        ]
        self.assertEqual(len(track_frames), 64)
        self.assertEqual([item["frame_count"] for item in transitions], [4, 7, 7])
        self.assertEqual(len(transition_frames), 18)
        self.assertEqual(len(sequence), 82)
        self.assertEqual(
            [frame["video_frame_index"] for frame in sequence],
            list(range(1, 83)),
        )
        self.assertEqual(
            [item["boundary"] for item in transitions],
            ["track1_to_track3", "track3_to_track2", "track2_to_track4"],
        )

    def test_all_video_cameras_stay_on_upper_hemisphere_and_orbit(self):
        sequence, _ = build_video_sequence(self.system)
        positions = np.asarray(
            [frame["position"] for frame in sequence],
            dtype=np.float64,
        )

        self.assertTrue(np.all(positions[:, 0] >= -1e-9))
        radii = np.linalg.norm(positions, axis=1)
        self.assertTrue(np.allclose(radii, 10.0, atol=1e-8))

    def test_transition_step_does_not_exceed_track_step(self):
        blocks = self.system.get_building_camera_positions(16)
        for block_index in range(3):
            start = blocks[block_index]["positions"][-1]
            end = blocks[block_index + 1]["positions"][0]
            transition = upper_hemisphere_transition(
                start,
                end,
                self.system.center,
                self.system.get_building_camera_view_up(block_index),
                self.system.get_building_camera_view_up(block_index + 1),
            )
            chain = [start] + transition["positions"] + [end]
            unit = np.asarray(chain) / 10.0
            steps = np.degrees(
                np.arccos(
                    np.clip(
                        np.sum(unit[:-1] * unit[1:], axis=1),
                        -1.0,
                        1.0,
                    )
                )
            )
            self.assertTrue(np.all(steps <= MAX_ANGULAR_STEP_DEG + 1e-8))

    def test_continuous_view_up_is_valid_for_every_camera(self):
        sequence, _ = build_video_sequence(self.system)
        for frame in sequence:
            position = np.asarray(frame["position"], dtype=np.float64)
            view_up = np.asarray(frame["view_up"], dtype=np.float64)
            look = -position / np.linalg.norm(position)
            self.assertAlmostEqual(float(np.linalg.norm(view_up)), 1.0, places=8)
            self.assertAlmostEqual(float(np.dot(view_up, look)), 0.0, places=8)

    def test_video_track_plan_uses_nearest_endpoints(self):
        sequence, _ = build_video_sequence(self.system)
        track_frames = [frame for frame in sequence if frame["kind"] == "track"]
        groups = []
        for frame in track_frames:
            if not groups or groups[-1][0] != frame["track_index"]:
                groups.append((frame["track_index"], []))
            groups[-1][1].append(frame["photo_index"])

        self.assertEqual([group[0] for group in groups], [1, 3, 2, 4])
        self.assertEqual(groups[0][1], list(range(1, 17)))
        self.assertEqual(groups[1][1], list(range(1, 17)))
        self.assertEqual(groups[2][1], list(range(16, 0, -1)))
        self.assertEqual(groups[3][1], list(range(1, 17)))


if __name__ == "__main__":
    unittest.main()
