"""CPU-only completed-result identity checks using private synthetic fixtures."""

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_building_reuse as reuse
import vace_p01_81f_results as results


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class CompletedSampleReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(prefix="vace_completed_source_cpu_", dir="/tmp")
        cls.fixture_path = Path(cls.fixture.name) / "02"
        cls.fixture_path.mkdir()
        damaged = [Image.new("RGB", (512, 512), (index, 60, 90)) for index in range(81)]
        prediction = [Image.new("RGB", (512, 512), (index, 140, 190)) for index in range(81)]
        masks = [Image.new("RGB", (512, 512), "black") for unused in range(81)]
        masks[20].paste("white", (100, 100, 200, 200))
        cls.gt = Image.new("RGB", (512, 512), (150, 180, 210))
        cls.selected = [dict(original_bytes=dict(damaged=png_bytes(image)), model_mask=mask)
                        for image, mask in zip(damaged, masks)]
        for name in results.FINAL_FILENAMES[:2]:
            (cls.fixture_path / name).write_bytes(b"CPU synthetic lifecycle video fixture; not a prediction")
        results.overview_image(damaged, prediction).save(cls.fixture_path / results.FINAL_FILENAMES[2])
        results.four_column_image(damaged, masks, prediction, cls.gt).save(cls.fixture_path / results.FINAL_FILENAMES[3])

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vace_completed_validate_cpu_", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = self.root / "02"
        shutil.copytree(self.fixture_path, self.target)
        self.record = dict(output_number="02", pilot="P02", source_id=763, kind="A1", tier="Medium",
                           archive_sha256="a" * 64)
        self.video = patch.object(results, "validate_video", return_value=dict(frames=81)).start()
        self.source = patch.object(reuse.delivery, "read_sample", return_value=self.selected).start()
        self.read_gt = patch.object(reuse.delivery, "read_gt_B020", return_value=self.gt).start()
        self.addCleanup(patch.stopall)

    def alter_pixel(self, filename, point):
        path = self.target / filename
        with Image.open(path) as image:
            altered = image.copy()
        altered.putpixel(point, (255, 0, 0))
        altered.save(path)

    def test_valid_completed_result_is_read_only_and_source_bound(self):
        hashes = reuse.file_hashes(self.target)
        stats = {name: (self.target / name).stat().st_mtime_ns for name in results.FINAL_FILENAMES}
        report = reuse.validate_completed_sample(self.record, self.target)
        self.assertEqual(report["pilot"], "P02")
        self.assertEqual(report["source_id"], 763)
        self.assertEqual(report["output_number"], "02")
        self.assertEqual(report["files_sha256"], hashes)
        self.assertTrue(report["source_identity_verified"])
        self.assertFalse(report["historical_parameters_independently_recoverable"])
        self.source.assert_called_once_with(self.record, include_gt=False)
        self.read_gt.assert_called_once_with(self.record)
        self.assertEqual(self.video.call_count, 2)
        for call in self.video.call_args_list:
            self.assertEqual(call.kwargs["expected_frames"], 81)
        self.assertEqual(hashes, reuse.file_hashes(self.target))
        self.assertEqual(stats, {name: (self.target / name).stat().st_mtime_ns for name in results.FINAL_FILENAMES})

    def test_every_representative_damaged_view_is_checked(self):
        for column, frame_index in enumerate(results.REPRESENTATIVE_FRAMES):
            with self.subTest(frame=frame_index):
                shutil.copy2(self.fixture_path / results.FINAL_FILENAMES[2], self.target / results.FINAL_FILENAMES[2])
                self.alter_pixel(results.FINAL_FILENAMES[2], (column * 512, 32))
                with self.assertRaisesRegex(ValueError, f"P02:.*identity mismatch: Damaged B{frame_index:03}"):
                    reuse.validate_completed_sample(self.record, self.target)
        self.read_gt.assert_not_called()

    def test_four_column_damaged_mismatch_is_rejected(self):
        self.alter_pixel(results.FINAL_FILENAMES[3], (0, 32))
        with self.assertRaisesRegex(ValueError, "P02:.*Damaged B020 four-column"):
            reuse.validate_completed_sample(self.record, self.target)
        self.read_gt.assert_not_called()

    def test_model_mask_mismatch_is_rejected_before_posthoc_GT_read(self):
        self.alter_pixel(results.FINAL_FILENAMES[3], (512, 32))
        with self.assertRaisesRegex(ValueError, "P02:.*ModelMask B020"):
            reuse.validate_completed_sample(self.record, self.target)
        self.read_gt.assert_not_called()

    def test_GT_identity_mismatch_is_rejected(self):
        self.alter_pixel(results.FINAL_FILENAMES[3], (1536, 32))
        with self.assertRaisesRegex(ValueError, "P02:.*GT B020"):
            reuse.validate_completed_sample(self.record, self.target)

    def test_wrong_fixed_source_mapping_is_rejected_before_reading_files(self):
        self.record["source_id"] = 871
        with self.assertRaisesRegex(ValueError, "fixed Pilot8 mapping"):
            reuse.validate_completed_sample(self.record, self.target)
        self.video.assert_not_called()
        self.source.assert_not_called()

    def test_wrong_output_number_directory_is_rejected(self):
        different = self.root / "03"
        self.target.rename(different)
        with self.assertRaisesRegex(ValueError, "directory number"):
            reuse.validate_completed_sample(self.record, different)
        self.source.assert_not_called()

    def test_omitted_file_is_rejected_without_cleanup(self):
        missing = self.target / results.FINAL_FILENAMES[0]
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            reuse.validate_completed_sample(self.record, self.target)
        self.assertEqual(len(list(self.target.iterdir())), 3)

    def test_extra_file_is_preserved_and_rejected(self):
        extra = self.target / "history.txt"
        extra.write_text("preserve this unrelated historical file")
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            reuse.validate_completed_sample(self.record, self.target)
        self.assertTrue(extra.is_file())

    def test_zero_byte_file_is_rejected(self):
        (self.target / results.FINAL_FILENAMES[0]).write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "nonempty regular file"):
            reuse.validate_completed_sample(self.record, self.target)

    def test_symlink_file_is_rejected_even_when_target_bytes_are_valid(self):
        path = self.target / results.FINAL_FILENAMES[0]
        path.unlink()
        path.symlink_to(self.fixture_path / results.FINAL_FILENAMES[0])
        with self.assertRaisesRegex(ValueError, "nonempty regular file"):
            reuse.validate_completed_sample(self.record, self.target)

    def test_symlink_directory_and_ancestor_are_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink path component"):
            reuse.validate_completed_sample(self.record, alias / "02")
        self.target.rename(self.root / "real")
        self.target.symlink_to(self.root / "real", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink path component"):
            reuse.validate_completed_sample(self.record, self.target)
        self.source.assert_not_called()

    def test_bad_video_decode_stops_before_source_loading(self):
        self.video.side_effect = ValueError("Incomplete 81-frame video decode")
        with self.assertRaisesRegex(ValueError, "Incomplete 81-frame"):
            reuse.validate_completed_sample(self.record, self.target)
        self.source.assert_not_called()

    def test_wrong_png_dimensions_are_rejected(self):
        Image.new("RGB", (512, 512)).save(self.target / results.FINAL_FILENAMES[2])
        with self.assertRaisesRegex(ValueError, "unexpected dimensions"):
            reuse.validate_completed_sample(self.record, self.target)

    def test_result_changed_during_validation_is_rejected(self):
        def read_gt(record):
            (self.target / results.FINAL_FILENAMES[0]).write_bytes(b"unexpected concurrent change")
            return self.gt

        self.read_gt.side_effect = read_gt
        with self.assertRaisesRegex(ValueError, "hashes changed during validation"):
            reuse.validate_completed_sample(self.record, self.target)

    def test_wrong_frame_count_in_source_is_rejected(self):
        self.source.return_value = self.selected[:-1]
        with self.assertRaisesRegex(ValueError, "must have 81 frames"):
            reuse.validate_completed_sample(self.record, self.target)
        self.read_gt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
