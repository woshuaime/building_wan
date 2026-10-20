"""CPU-only P01 identity and no-overwrite reuse tests in isolated temporary paths."""

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01_reuse as reuse
from vace_p01_81f_results import FINAL_FILENAMES, four_column_image, overview_image


def image_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def make_fixture(directory):
    directory.mkdir()
    damaged = [Image.new("RGB", (512, 512), (index, 80, 120)) for index in range(81)]
    prediction = [Image.new("RGB", (512, 512), (index, 180, 120)) for index in range(81)]
    masks = [Image.new("RGB", (512, 512), "black") for index in range(81)]
    masks[20].paste("white", (100, 100, 200, 200))
    gt = Image.new("RGB", (512, 512), (100, 180, 220))
    (directory / FINAL_FILENAMES[0]).write_bytes(b"synthetic_raw_video_fixture_not_prediction")
    (directory / FINAL_FILENAMES[1]).write_bytes(b"synthetic_comparison_fixture_not_prediction")
    overview_image(damaged, prediction).save(directory / FINAL_FILENAMES[2])
    four_column_image(damaged, masks, prediction, gt).save(directory / FINAL_FILENAMES[3])
    selected = [dict(original_bytes=dict(damaged=image_bytes(image)), model_mask=mask)
                for image, mask in zip(damaged, masks)]
    return selected, gt


class P01ReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(prefix="building_wan_p01_reuse_cpu_")
        cls.fixture_source = Path(cls.fixture.name) / "source"
        cls.selected, cls.gt = make_fixture(cls.fixture_source)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="building_wan_p01_copy_cpu_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "P01_465_81f_completed"
        shutil.copytree(self.fixture_source, self.source)
        self.patches = [
            patch.object(reuse, "validate_final_results", return_value=dict(frames=81)),
            patch.object(reuse.experiment, "experiment_config", return_value=dict(source={})),
            patch.object(reuse.experiment, "memory_inputs", return_value=self.selected),
            patch.object(reuse.experiment, "read_gt_B020", return_value=self.gt),
        ]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_source_identity_uses_five_damaged_views_mask_and_gt(self):
        validated = reuse.validate_p01_results(self.source)
        self.assertEqual(validated["source_id"], 465)
        self.assertTrue(validated["source_identity_verified"])
        self.assertFalse(validated["historical_parameters_independently_recoverable"])
        self.assertEqual(set(validated["files_sha256"]), set(FINAL_FILENAMES))

    def test_wrong_damaged_identity_is_rejected(self):
        with Image.open(self.source / FINAL_FILENAMES[2]) as image:
            altered = image.copy()
        altered.putpixel((0, 32), (255, 0, 0))
        altered.save(self.source / FINAL_FILENAMES[2])
        with self.assertRaisesRegex(ValueError, "identity mismatch.*Damaged B000"):
            reuse.validate_p01_results(self.source)

    def test_wrong_gt_identity_is_rejected(self):
        with Image.open(self.source / FINAL_FILENAMES[3]) as image:
            altered = image.copy()
        altered.putpixel((1536, 32), (255, 0, 0))
        altered.save(self.source / FINAL_FILENAMES[3])
        with self.assertRaisesRegex(ValueError, "identity mismatch.*GT B020"):
            reuse.validate_p01_results(self.source)

    def test_copy_exactly_four_files_preserves_original_and_reuses_identical_target(self):
        original_hashes = reuse.file_hashes(self.source)
        target = self.root / "test" / "01"
        result = reuse.copy_verified_p01(self.source, target)
        self.assertEqual(result["status"], "COPIED")
        self.assertEqual(reuse.file_hashes(target), original_hashes)
        before_stats = {name: (target / name).stat().st_mtime_ns for name in FINAL_FILENAMES}
        self.assertEqual(reuse.copy_verified_p01(self.source, target)["status"], "REUSED_EXISTING")
        self.assertEqual(before_stats, {name: (target / name).stat().st_mtime_ns for name in FINAL_FILENAMES})
        self.assertEqual(reuse.file_hashes(self.source), original_hashes)
        self.assertEqual(sorted(path.name for path in target.iterdir()), sorted(FINAL_FILENAMES))

    def test_different_valid_existing_target_is_not_overwritten(self):
        target = self.root / "01"
        shutil.copytree(self.source, target)
        (target / FINAL_FILENAMES[0]).write_bytes(b"different_previous_raw_result")
        before = reuse.file_hashes(target)
        with self.assertRaisesRegex(ValueError, "differs.*refuse to overwrite"):
            reuse.copy_verified_p01(self.source, target)
        self.assertEqual(reuse.file_hashes(target), before)

    def test_invalid_existing_target_is_not_cleaned(self):
        target = self.root / "01"
        target.mkdir()
        previous = target / "history.txt"
        previous.write_text("preserve", encoding="utf-8")
        with patch.object(reuse, "validate_final_results", side_effect=[dict(frames=81), ValueError("not four files")]):
            with self.assertRaisesRegex(ValueError, "not four files"):
                reuse.copy_verified_p01(self.source, target)
        self.assertEqual(previous.read_text(), "preserve")
        self.assertEqual(list(target.iterdir()), [previous])

    def test_source_validation_failure_creates_no_destination(self):
        target = self.root / "test" / "01"
        with patch.object(reuse, "validate_p01_results", side_effect=ValueError("invalid source")):
            with self.assertRaisesRegex(ValueError, "invalid source"):
                reuse.copy_verified_p01(self.source, target)
        self.assertFalse(target.parent.exists())

    def test_symlink_target_and_symlink_parent_are_rejected(self):
        owned_elsewhere = self.root / "elsewhere"
        owned_elsewhere.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(owned_elsewhere, target_is_directory=True)
        for target in (alias, alias / "01"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "symlink path component"):
                reuse.copy_verified_p01(self.source, target)
        self.assertFalse(any(owned_elsewhere.iterdir()))

    def test_source_target_ancestor_relationship_is_rejected(self):
        for target in (self.source / "child", self.source.parent):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "independent directories"):
                reuse.copy_verified_p01(self.source, target)

    def test_same_path_is_validated_and_reused_without_original_directory_dependency(self):
        target = self.root / "test" / "01"
        target.parent.mkdir()
        shutil.copytree(self.source, target)
        shutil.rmtree(self.source)
        before = {name: (target / name).stat().st_mtime_ns for name in FINAL_FILENAMES}
        hashes = reuse.file_hashes(target)
        self.assertEqual(reuse.copy_verified_p01(target, target)["status"], "REUSED_EXISTING")
        self.assertEqual(reuse.file_hashes(target), hashes)
        self.assertEqual({name: (target / name).stat().st_mtime_ns for name in FINAL_FILENAMES}, before)

    def test_finder_rejects_multiple_valid_results_and_allows_explicit_choice(self):
        second = self.root / "P01_465_81f_another"
        shutil.copytree(self.source, second)
        with self.assertRaisesRegex(ValueError, "Multiple valid.*never guess"):
            reuse.find_verified_p01(self.root)
        self.assertEqual(reuse.find_verified_p01(self.root, source=self.source), self.source)

    def test_finder_reports_missing_and_invalid_without_running_any_model(self):
        with patch.object(reuse, "validate_p01_results", side_effect=ValueError("incomplete")):
            with self.assertRaisesRegex(ValueError, "No valid.*do not rerun P01.*incomplete"):
                reuse.find_verified_p01(self.root)

    def test_finder_selects_only_matching_completed_P01_results(self):
        ignored = self.root / "P02_763_81f_other"
        ignored.mkdir()
        self.assertEqual(reuse.find_verified_p01(self.root), self.source)


if __name__ == "__main__":
    unittest.main()
