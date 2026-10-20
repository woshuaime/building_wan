"""CPU checks for GT isolation, B080, and stopping before unauthorized later batches."""

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_lora300_eval81 as evaluation


class EvaluationTests(unittest.TestCase):
    def test_pil_predictions_render_and_survive_cpu_failure(self):
        prediction = [Image.new("RGB", (512, 512), (80 + i, 95, 110)) for i in range(81)]
        damaged = [Image.new("RGB", (512, 512), (80, 90, 100)) for _ in range(81)]
        mask = Image.new("RGB", (512, 512), "black")
        mask.paste("white", (100, 100, 200, 200))
        sample = dict(number="04", pilot="P04", damaged=damaged, model_mask=[mask] * 81)
        base = np.stack([np.asarray(image) for image in damaged])
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "04"
            with patch.object(evaluation, "read_gt", side_effect=RuntimeError("CPU comparison failed")):
                with self.assertRaisesRegex(RuntimeError, "CPU comparison failed"):
                    evaluation.render_sample(destination, sample, base, prediction)
            raw = destination / evaluation.FINAL_NAMES[0]
            raw_bytes = raw.read_bytes()
            decoded = evaluation.decode_video(raw, (512, 512))
            self.assertEqual(len(decoded), 81)
            with patch.object(evaluation, "read_gt", return_value=damaged):
                evaluation.render_sample(destination, sample, base, decoded, raw_saved=True)
            self.assertEqual(raw.read_bytes(), raw_bytes)
            self.assertEqual({p.name for p in destination.iterdir()}, set(evaluation.FINAL_NAMES))
            with Image.open(destination / evaluation.FINAL_NAMES[2]) as overview:
                self.assertEqual(overview.size, (2560, 2176))

    def test_loader_preserves_all_81_frames_and_never_reads_gt(self):
        prefix = evaluation.delivery.PACKAGE_ID + "/"
        root = prefix + "samples/P04/"
        sample = dict(pilot="P04", source_id=1227, kind="A2", tier="Medium",
                      relative_root="samples/P04", stable_id="b1227_12345678",
                      source_sha256="12345678identity", sample_id="sample04")
        metadata = dict(sample_id="sample04", source=dict(sha256=sample["source_sha256"]))
        cameras = dict(sample_id="sample04", frames=81, fps_metadata=16, resolutions={"512": [
            dict(frame=i, frame_id=f"{i:03}", width=512, height=512) for i in range(81)]})
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "inputs.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for name, data in (("manifest.json", dict(samples=[sample])),
                                   ("samples/P04/sample_meta.json", metadata),
                                   ("samples/P04/cameras.json", cameras)):
                    bundle.writestr(prefix + name, json.dumps(data))
                for index in range(81):
                    for role in ("damaged", "exact"):
                        image = Image.new("RGB", (512, 512), (index, 40, 60))
                        if role == "exact":
                            image = Image.new("L", (512, 512), 0)
                            if index != 40:
                                image.paste(255, (100, 100, 110, 110))
                        with io.BytesIO() as stream:
                            image.save(stream, format="PNG")
                            bundle.writestr(root + f"512/{role}/{index:03}.png", stream.getvalue())
            # No GT files exist: any GT read would fail.
            with patch.object(evaluation.delivery, "DEFAULT_ARCHIVE", archive):
                loaded = evaluation.read_conditions("04")
            self.assertNotIn("gt", loaded)
            self.assertEqual(len(loaded["conditioning"]), 81)
            for index in range(81):
                actual = np.asarray(loaded["conditioning"][index])
                np.testing.assert_array_equal(actual[0, 0], (index, 40, 60))
                np.testing.assert_array_equal(actual[105, 105],
                                              (index, 40, 60) if index == 40 else (127, 127, 127))
            self.assertFalse(np.asarray(loaded["model_mask"][40]).any())
            self.assertEqual(np.asarray(loaded["damaged"][80])[0, 0, 0], 80)

    def test_default_does_not_start_gpu(self):
        with patch.object(evaluation, "dry_run") as check, patch.object(evaluation, "run_batch") as run:
            self.assertEqual(evaluation.main([]), 0)
            check.assert_called_once()
            run.assert_not_called()

    def test_first_batch_stops_after_pair_and_uses_new_directory(self):
        for returncodes, expected_calls, expected_code in (
            ([(0, False), (0, False)], ["04", "06"], 0),
            ([(-11, False)], ["04"], 139),
            ([(0, True)], ["04"], 130),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                output = directory / "lora300_eval81"
                output.mkdir()
                (output / "old.txt").write_text("protected")
                calls = []
                original_lock = evaluation.baseline.gpu_lock

                def fake_worker(command, environment, lock_fd):
                    calls.append(command[command.index("--worker") + 1])
                    return returncodes[len(calls) - 1]

                with patch.object(evaluation, "OUTPUT_ROOT", output), \
                        patch.object(evaluation.baseline, "gpu_lock",
                                     side_effect=lambda: original_lock(directory)), \
                        patch.object(evaluation, "visible_worker", side_effect=fake_worker):
                    self.assertEqual(evaluation.run_batch(False, None), expected_code)
                self.assertEqual(calls, expected_calls)
                self.assertEqual((output / "old.txt").read_text(), "protected")
                self.assertEqual(len(list(output.glob("run_*"))), 1)


if __name__ == "__main__":
    unittest.main()
