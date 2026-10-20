"""CPU synthetic fixtures only: no model predictions or project outputs are created."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01_results as results


ROOT = Path(__file__).resolve().parents[2]


def json_file(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_fixture_video(path, frames=17, size=(512, 512), fps=16):
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, quality=5, codec="libx264", ffmpeg_log_level="error",
                                        output_params=["-threads", "1"])
    writer.send(None)
    for index in range(frames):
        pixels = np.full((size[1], size[0], 3), 80 + index, dtype=np.uint8)
        writer.send(pixels.tobytes())
    writer.close()


def synthetic_run(root, gt_offset=0, with_video=False):
    output = root / "synthetic_cpu_test_run"
    output.mkdir()
    shutil.copyfile(ROOT / "configs/vace_p01_465_17f_v1.json", output / "frozen_config.json")
    inventory = {}
    raw_hashes = {}
    frame_map = []
    for directory in ("inputs/damaged", "inputs/model_mask", "inputs/conditioning", "evaluation/gt", "prediction_raw"):
        (output / directory).mkdir(parents=True)
    for output_index, source_index in enumerate(range(5, 22)):
        damaged = np.full((512, 512, 3), source_index, dtype=np.uint8)
        prediction = np.full((512, 512, 3), 80 + output_index, dtype=np.uint8)
        ground_truth = np.full((512, 512, 3), 150 + gt_offset, dtype=np.uint8)
        mask = np.zeros((512, 512, 3), dtype=np.uint8)
        mask[100:200, 120:220] = 255
        condition = damaged.copy()
        condition[100:200, 120:220] = 127
        originals = {}
        for directory, pixels in (("inputs/damaged", damaged), ("inputs/model_mask", mask),
                                  ("inputs/conditioning", condition), ("evaluation/gt", ground_truth),
                                  ("prediction_raw", prediction)):
            relative = f"{directory}/{output_index:03}.png"
            path = output / relative
            Image.fromarray(pixels).save(path)
            digest = results.sha256(path)
            if directory == "prediction_raw":
                raw_hashes[path.name] = digest
            else:
                inventory[relative] = digest
            if directory in ("inputs/damaged", "evaluation/gt"):
                originals["damaged" if directory == "inputs/damaged" else "gt"] = digest
        frame_map.append(dict(output_index=output_index, source_index=source_index,
            source_frame=dict(frame_index=source_index, camera_index=source_index, frame_id=f"{source_index:03}"),
            camera=dict(frame=source_index, frame_id=f"{source_index:03}"), original_sha256=originals))
    json_file(output / "frame_map.json", frame_map)
    json_file(output / "artifacts_sha256.json", inventory)
    json_file(output / "prediction_raw_sha256.json", raw_hashes)
    json_file(output / "worker_complete.json", dict(status="complete", prediction_frames=17,
        lora_loaded=False, GT_loaded_by_worker=False, synthetic_CPU_unit_test_fixture=True))
    if with_video:
        write_fixture_video(output / "prediction_raw.mp4")
    return output


class ResultsContractTests(unittest.TestCase):
    def test_frame_map_independently_maps_005_to_021(self):
        with tempfile.TemporaryDirectory() as directory:
            output = synthetic_run(Path(directory))
            frames, hashes = results.validate_inputs(output)
            self.assertEqual([frame["source_index"] for frame in frames], list(range(5, 22)))
            self.assertEqual(len(hashes), 85)
            self.assertEqual(frames[15]["source_index"], 20)

    def test_wrong_frame_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = synthetic_run(Path(directory))
            mapping = results.read_json(output / "frame_map.json")
            mapping[15]["source_index"] = 19
            json_file(output / "frame_map.json", mapping)
            with self.assertRaisesRegex(ValueError, "independently align"):
                results.validate_inputs(output)

    def test_output_aliases_are_supported_without_reading_external_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = synthetic_run(Path(directory))
            (output / "prediction_raw").rename(output / "raw_prediction")
            (output / "prediction_raw").symlink_to("raw_prediction", target_is_directory=True)
            frames, hashes = results.validate_inputs(output)
            self.assertEqual(len(frames), 17)
            self.assertEqual(len(hashes), 85)

    def test_raw_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = synthetic_run(Path(directory))
            Image.new("RGB", (512, 512), "red").save(output / "prediction_raw/015.png")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                results.validate_inputs(output)

    def test_missing_or_extra_raw_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = synthetic_run(Path(directory))
            Image.new("RGB", (512, 512)).save(output / "prediction_raw/017.png")
            with self.assertRaisesRegex(ValueError, "exactly 17"):
                results.validate_inputs(output)

    def test_worker_gt_or_lora_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = synthetic_run(Path(directory))
            record = results.read_json(output / "worker_complete.json")
            record["GT_loaded_by_worker"] = True
            json_file(output / "worker_complete.json", record)
            with self.assertRaisesRegex(ValueError, "exclude LoRA and GT"):
                results.validate_inputs(output)

    def test_nonidentical_mask_channels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = synthetic_run(Path(directory))
            relative = "inputs/model_mask/000.png"
            pixels = np.zeros((512, 512, 3), dtype=np.uint8)
            pixels[0, 0, 0] = 255
            Image.fromarray(pixels).save(output / relative)
            inventory = results.read_json(output / "artifacts_sha256.json")
            inventory[relative] = results.sha256(output / relative)
            json_file(output / "artifacts_sha256.json", inventory)
            with self.assertRaisesRegex(ValueError, "identical binary RGB"):
                results.validate_inputs(output)

    def test_existing_result_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "metrics.json"
            path.write_text("existing evidence", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Refuse to overwrite"):
                results.generate_results(output)
            self.assertEqual(path.read_text(), "existing evidence")

    def test_gt_changes_do_not_change_before_after_or_grid(self):
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first = synthetic_run(Path(first_directory), gt_offset=0, with_video=True)
            second = synthetic_run(Path(second_directory), gt_offset=40)
            shutil.copyfile(first / "prediction_raw.mp4", second / "prediction_raw.mp4")
            skipped = dict(status="not_created", reason="Optional encoding disabled in CPU comparison-isolation test")
            with patch.object(results, "save_comparison_video", return_value=skipped):
                first_result = results.generate_results(first)
                results.generate_results(second)
            for filename in ("comparison_before_after_B020.png", "comparison_before_after_grid.png"):
                self.assertEqual(results.sha256(first / filename), results.sha256(second / filename))
            self.assertNotEqual(results.sha256(first / "comparison_4col_B020.png"),
                                results.sha256(second / "comparison_4col_B020.png"))
            with Image.open(first_result["paths"]["before_after_image"]) as image:
                self.assertEqual(image.size, (1024, 544))
                self.assertEqual(image.getpixel((0, results.HEADER_HEIGHT)), (20, 20, 20))
                self.assertEqual(image.getpixel((512, results.HEADER_HEIGHT)), (95, 95, 95))
            with Image.open(first / "comparison_before_after_grid.png") as image:
                self.assertEqual(image.size, (2048, 1088))
                for column, source_index in enumerate((5, 10, 15, 20)):
                    self.assertEqual(image.getpixel((512 * column, 32)), (source_index,) * 3)
                    self.assertEqual(image.getpixel((512 * column, 576)), (75 + source_index,) * 3)
            metrics = results.read_json(first / "metrics.json")
            self.assertIsNone(metrics["automatic_quality_pass"])
            self.assertFalse(metrics["postprocessed_prediction_used"])
            self.assertEqual(first_result["video_validation"]["frames"], 17)


class MetricsTests(unittest.TestCase):
    def test_region_denominator_is_pixel_channel_weighted(self):
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        reference = pixels.copy()
        pixels[0, 0] = (0, 30, 60)
        region = np.zeros((2, 2), dtype=bool)
        region[0, 0] = True
        measured = results.error_summary(pixels, reference, region)
        self.assertEqual(measured["pixels"], 1)
        self.assertEqual(measured["mae_0_255"], 30)
        self.assertAlmostEqual(measured["mae_normalized"], 30 / 255)

    def test_empty_and_perfect_regions_use_json_safe_null(self):
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        empty = results.error_summary(pixels, pixels, np.zeros((2, 2), dtype=bool))
        perfect = results.error_summary(pixels, pixels, np.ones((2, 2), dtype=bool))
        self.assertIsNone(empty["mae_0_255"])
        self.assertIsNone(perfect["psnr_db"])
        self.assertTrue(perfect["perfect_match"])
        json.dumps(dict(empty=empty, perfect=perfect), allow_nan=False)

    def test_aggregate_baseline_and_raw_change_are_distinct(self):
        mask = np.zeros((2, 2, 3), dtype=np.uint8)
        mask[0, 0] = 255
        frames = [dict(output_index=index, source_index=index + 5, mask=mask,
            damaged=np.zeros((2, 2, 3), dtype=np.uint8), prediction=np.full((2, 2, 3), 20, dtype=np.uint8),
            gt=np.full((2, 2, 3), 30, dtype=np.uint8)) for index in range(17)]
        metrics = results.calculate_metrics(frames)
        inside = metrics["aggregate"]["inside_mask"]
        self.assertEqual(inside["raw_prediction"]["mae_0_255"], 10)
        self.assertEqual(inside["damaged_baseline"]["mae_0_255"], 30)
        self.assertEqual(inside["raw_vs_damaged_change"]["mae_0_255"], 20)
        self.assertEqual(inside["damaged_minus_raw_mae_0_255"], 20)
        self.assertEqual(metrics["temporal"]["raw_prediction"]["mean_absolute_consecutive_delta"], 0)
        self.assertEqual(len(metrics["temporal"]["raw_prediction"]["second_difference"]), 15)


class VideoValidationTests(unittest.TestCase):
    def test_actual_cpu_encoded_video_and_comparison_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = synthetic_run(Path(directory), with_video=True)
            raw = results.validate_video(output / "prediction_raw.mp4")
            self.assertEqual((raw["frames"], raw["width"], raw["height"], raw["fps"]), (17, 512, 512, 16))
            frames, input_hashes = results.validate_inputs(output)
            self.assertEqual(len(input_hashes), 85)
            comparison = results.save_comparison_video(output, frames)
            self.assertEqual((comparison["frames"], comparison["width"], comparison["height"]), (17, 1024, 544))

    def test_wrong_actual_frame_count_and_fps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_cpu_video.mp4"
            write_fixture_video(path, frames=16)
            with self.assertRaisesRegex(ValueError, "17 frames"):
                results.validate_video(path)
            path.unlink()
            write_fixture_video(path, fps=8)
            with self.assertRaisesRegex(ValueError, "exactly 16"):
                results.validate_video(path)

    def test_broken_video_is_rejected_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken_cpu_fixture.mp4"
            path.write_bytes(b"intentionally invalid CPU test fixture")
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "inspection failed"):
                results.validate_video(path)
            self.assertEqual(path.read_bytes(), original)

    def test_wrong_video_resolution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_cpu_wrong_size.mp4"
            write_fixture_video(path, size=(256, 256))
            with self.assertRaisesRegex(ValueError, "resolution mismatch"):
                results.validate_video(path)


if __name__ == "__main__":
    unittest.main()
