"""A completed delivery must remain usable after removing every training intermediate."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validation_delivery as delivery
import validation_queue


class DeliveryInputs(unittest.TestCase):
    def test_retained_weights_only_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / delivery.RUN_NAME
            weights = run / "交付_训练完成"
            weights.mkdir(parents=True)
            checksums = {}
            for role in ("adapter", "lora"):
                path = weights / f"{role}_final.safetensors"
                path.write_text(role)
                checksums[role] = delivery.sha(path)
            with patch.object(delivery, "FINAL_SHA256", checksums), patch.object(
                    delivery, "validation_rows", return_value={"12": {}}):
                result = validation_queue.check_inputs(run, ["12"])
                self.assertEqual(result["lora_sha256"], checksums["lora"])
                with self.assertRaisesRegex(ValueError, "distinct validation"):
                    validation_queue.check_inputs(run, ["12", "12"])
                self.assertEqual(list(run.iterdir()), [weights])
                (weights / "adapter_final.safetensors").write_text("replaced")
                with self.assertRaisesRegex(ValueError, "Final adapter differs"):
                    validation_queue.check_inputs(run, ["12"])

    def test_source_gate_keeps_gt_out_of_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subdir in ("partial", "gt", "cameras", "records"):
                (root / subdir).mkdir()
            paths = [root / name for name in ("cameras/12.json", "records/12.json", "12.ply", "partial/12.mp4")]
            for path in paths:
                path.write_text(path.name)
            gt = root / "gt/12.mp4"
            gt.write_text("reference")
            manifest = root / "manifest.csv"
            manifest.write_text("id,partial_video,GT_video,camera,record,split_use\n"
                                "12,partial/12.mp4,gt/12.mp4,cameras/12.json,records/12.json,validation\n"
                                "13,partial/13.mp4,gt/13.mp4,cameras/13.json,records/13.json,train\n")
            combined = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in paths)).hexdigest()
            spec = root / "hashes.csv"
            spec.write_text(f"id,conditioning_sha256,gt_video_sha256\n12,{combined},{delivery.sha(gt)}\n")
            with patch.multiple(delivery, DATASET=root, POINTS=root, SOURCES=spec,
                                MANIFEST_SHA256=delivery.sha(manifest)):
                with patch.object(delivery, "sha", wraps=delivery.sha) as read_hash:
                    result = delivery.source_row("12")
                    self.assertNotIn(gt, [Path(call.args[0]) for call in read_hash.call_args_list])
                self.assertEqual(result["gt_video_sha256"], delivery.sha(gt))
                with self.assertRaisesRegex(ValueError, "held-out"):
                    delivery.source_row("13")
                paths[0].write_text("changed")
                with self.assertRaisesRegex(ValueError, "conditioning inputs changed"):
                    delivery.source_row("12")
