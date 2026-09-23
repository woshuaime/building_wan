import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.experiment_config import (
    ConfigError,
    MODEL_ID,
    model_origin_spec,
    read_metadata,
    validate_shape,
)
from training.prepare_splits import Sample, audit_split_leakage, build_splits
from training.prepare_temporal_clips import build_clip_windows, select_clip_windows
from training.train_split_domain_lora import expected_optimizer_steps, normalize_gpu_ids
from training.wan_train_entry import (
    _SharedTextContextCacheLoader,
    _bf16_cpu_context,
    _compact_sft_cache,
    _save_shared_text_context,
)


class TrainingToolTests(unittest.TestCase):
    def test_shared_context_cache_keeps_context_out_of_sample_files(self):
        inputs = (
            {"input_latents": torch.randn(1, 2, 3, dtype=torch.float32)},
            {"context": torch.randn(1, 4, 5, dtype=torch.float32)},
            {"context": torch.randn(1, 4, 5, dtype=torch.float32)},
        )
        context = _bf16_cpu_context(inputs)
        compact = _compact_sft_cache(inputs, include_text_context=False)

        self.assertEqual(compact[0]["input_latents"].dtype, torch.bfloat16)
        self.assertEqual(compact[1], {})
        self.assertEqual(compact[2], {})

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            context_path = root / ".shared-context.pt"
            sample_path = root / "0.pth"
            _save_shared_text_context(context_path, context)
            torch.save(compact, sample_path)

            restored = _SharedTextContextCacheLoader(context_path)(sample_path)
            self.assertTrue(torch.equal(restored[1]["context"], context))
            self.assertNotIn("context", torch.load(sample_path, weights_only=False)[1])

    def test_shared_context_cache_rejects_different_prompts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / ".shared-context.pt"
            first = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
            second = torch.ones(1, 2, 3, dtype=torch.bfloat16)
            _save_shared_text_context(path, first)
            with self.assertRaises(ValueError):
                _save_shared_text_context(path, second)

    def test_wan_video_shapes_follow_model_constraints(self):
        validate_shape(256, 256, 17)
        validate_shape(480, 480, 81)
        with self.assertRaises(ConfigError):
            validate_shape(250, 256, 17)
        with self.assertRaises(ConfigError):
            validate_shape(256, 256, 18)

    def test_model_origin_spec_contains_all_required_components(self):
        spec = model_origin_spec()
        self.assertEqual(spec.count(MODEL_ID), 3)
        self.assertIn("diffusion_pytorch_model*.safetensors", spec)
        self.assertIn("models_t5_umt5-xxl-enc-bf16.pth", spec)
        self.assertIn("Wan2.1_VAE.pth", spec)

    def test_metadata_reader_validates_files_and_required_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "sample.mp4").write_bytes(b"placeholder")
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
                writer.writeheader()
                writer.writerow({"video": "sample.mp4", "prompt": "a building"})
            rows = read_metadata(metadata, root)
            self.assertEqual(rows, [{"video": "sample.mp4", "prompt": "a building"}])

    def test_metadata_reader_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outside = root.parent / "outside.mp4"
            outside.write_bytes(b"placeholder")
            try:
                metadata = root / "metadata.csv"
                metadata.write_text(
                    "video,prompt\n../outside.mp4,a building\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ConfigError):
                    read_metadata(metadata, root)
            finally:
                outside.unlink(missing_ok=True)

    def test_grouped_split_is_deterministic_and_has_no_leakage(self):
        samples = [
            Sample(
                video=f"model{index:05d}.mp4",
                prompt="a building",
                source_obj=f"D:/models/{index // 2}.obj",
                model_folder=f"model{index:05d}",
            )
            for index in range(20)
        ]
        first = build_splits(samples, 42, 0.6, 0.2, 4)
        second = build_splits(samples, 42, 0.6, 0.2, 4)
        self.assertEqual(
            [sample.video for sample in first["train"]],
            [sample.video for sample in second["train"]],
        )
        self.assertEqual(len(first["pilot_4"]), 4)
        self.assertTrue(
            {sample.video for sample in first["pilot_4"]}
            <= {sample.video for sample in first["train"]}
        )
        self.assertTrue(all(value == 0 for value in audit_split_leakage(first).values()))

    def test_temporal_windows_cover_all_81_frames(self):
        windows = build_clip_windows(81, clip_frames=17, stride=16)
        self.assertEqual(
            [(window.start, window.end) for window in windows],
            [(0, 17), (16, 33), (32, 49), (48, 65), (64, 81)],
        )
        covered = {frame for window in windows for frame in range(window.start, window.end)}
        self.assertEqual(covered, set(range(81)))

    def test_temporal_windows_include_non_aligned_tail(self):
        windows = build_clip_windows(90, clip_frames=17, stride=16)
        self.assertEqual((windows[-1].start, windows[-1].end), (73, 90))

    def test_balanced_temporal_selection_cycles_through_orbit(self):
        windows = build_clip_windows(81, clip_frames=17, stride=16)
        selected = [
            select_clip_windows(windows, "balanced", source_index)[0].start
            for source_index in range(10)
        ]
        self.assertEqual(selected, [0, 16, 32, 48, 64, 0, 16, 32, 48, 64])

    def test_split_training_accepts_one_or_more_unique_gpu_ids(self):
        self.assertEqual(normalize_gpu_ids({"gpu": 0}), [0])
        self.assertEqual(normalize_gpu_ids({"gpus": [0, 1]}), [0, 1])
        with self.assertRaises(ConfigError):
            normalize_gpu_ids({"gpus": [0, 0]})
        with self.assertRaises(ConfigError):
            normalize_gpu_ids({"gpus": []})

    def test_expected_ddp_steps_uses_global_batch(self):
        self.assertEqual(expected_optimizer_steps(5, 2, 1, 1, 1), 3)
        self.assertEqual(expected_optimizer_steps(1000, 2, 1, 2, 1), 1000)


if __name__ == "__main__":
    unittest.main()
