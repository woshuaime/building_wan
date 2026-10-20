"""CPU-only fixtures for the in-memory Pilot8 receiver and model inputs."""

import copy
import csv
import hashlib
import io
import json
import stat
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import buildingfiller_pilot8 as pilot8


def encoded_json(value):
    return json.dumps(value, ensure_ascii=False).encode()


def fixture_payloads():
    payloads = {"evidence/frozen/quality_config.json": b"fixture-quality", "atlas.pdf": b"fixture-reviewed-atlas"}
    digest = pilot8.payload_hash
    png = {}
    for resolution in pilot8.RESOLUTIONS:
        for mode in ("RGB", "L"):
            stream = io.BytesIO()
            Image.new(mode, (resolution, resolution), 0).save(stream, format="PNG")
            png[resolution, mode] = stream.getvalue()
    samples, source_map, rows = [], [], []
    columns = ("pilot", "source_id", "sample_id", "resolution", "frame_id", "camera_index", "azimuth_degrees",
               "elevation_degrees", "time_seconds", "gt", "gt_sha256", "damaged", "damaged_sha256", "exact", "exact_sha256")
    for output_number, pilot, source_id, kind, tier in pilot8.SAMPLE_MAPPING:
        root = "samples/" + pilot
        sample_id = "sample_" + f"{int(output_number) - 1:04}"
        models = {}
        for role in ("gt", "damaged", "removed"):
            content = f"fixture-{source_id}-{role}".encode()
            payloads[f"{root}/models/{role}.obj"] = content
            models[role] = dict(relative_path=f"models/{role}.obj", sha256=digest(content))
        source_hash = models["gt"]["sha256"]
        stable_id = f"b{source_id}_{source_hash[:8]}"
        family_id = source_id if pilot not in ("P05", "P07") else {"P05": 7, "P07": 357}[pilot]
        meta = dict(sample_id=sample_id, source=dict(sha256=source_hash), kind=kind, size=tier,
                    removed_original_surface_ratio=0.05, models=models, human_QA="pending", render_status="not_run",
                    training_ready=False)
        if pilot != "P01":
            meta["pilot_source_identity"] = dict(source_id=source_id, stable_id=stable_id, family_id=family_id)
        payloads[root + "/sample_meta.json"] = encoded_json(meta)
        cameras = dict(sample_id=sample_id, frames=81, fps_metadata=16,
                       view_ids=[f"{index:03}" for index in range(81)],
                       trajectory=dict(closure="repeat", orbit_degrees=360, fps_metadata=16), resolutions={},
                       source_geometry_sha256=models, input_metadata_sha256=digest(payloads[root + "/sample_meta.json"]),
                       output_files=[])
        for resolution in pilot8.RESOLUTIONS:
            label = str(resolution)
            poses = []
            for index in range(81):
                pose = dict(frame=index, frame_id=f"{index:03}", azimuth_degrees=4.5 * index,
                            elevation_degrees=20, time_seconds=index / 16, width=resolution, height=resolution)
                pose.update({key: [[1, 0], [0, 1]] for key in pilot8.POSE_KEYS})
                poses.append(pose)
                row = dict(pilot=pilot, source_id=source_id, sample_id=sample_id, resolution=resolution,
                           frame_id=f"{index:03}", camera_index=index, azimuth_degrees=4.5 * index,
                           elevation_degrees=20, time_seconds=index / 16)
                for role in pilot8.ROLES:
                    relative = f"{label}/{role}/{index:03}.png"
                    content = png[resolution, "L" if role == "exact" else "RGB"]
                    payloads[root + "/" + relative] = content
                    row[role], row[role + "_sha256"] = root + "/" + relative, digest(content)
                    cameras["output_files"].append(dict(relative=relative, sha256=digest(content)))
                rows.append(row)
            cameras["resolutions"][label] = poses
        payloads[root + "/cameras.json"] = encoded_json(cameras)
        sample = dict(source_id=source_id, stable_id=stable_id, family_id=family_id, sample_id=sample_id, kind=kind,
                      tier=tier, removed_ratio=0.05, source_sha256=source_hash, pilot=pilot, relative_root=root,
                      sample_meta_sha256=digest(payloads[root + "/sample_meta.json"]),
                      cameras_sha256=digest(payloads[root + "/cameras.json"]),
                      acceptance_record=f"evidence/acceptance/{pilot}_accepted.json", human_QA="accepted",
                      training_ready=False, generation_verified=False, generation_evidence_verified=True,
                      pixel_check_record=f"evidence/pixel_checks/{pilot}.json")
        event = {key: sample[key] for key in ("source_id", "stable_id", "family_id", "sample_id", "kind", "tier", "source_sha256",
                                            "training_ready", "generation_verified", "generation_evidence_verified")}
        event.update(decision="accepted", review_number=pilot, bindings=[], reviewed_atlas=dict(
            path="original-atlas", sha256=digest(payloads["atlas.pdf"]), size=len(payloads["atlas.pdf"])))
        for relative in (root + "/sample_meta.json", root + "/cameras.json", *(f"{root}/models/{role}.obj" for role in models)):
            original = "original/" + relative
            binding = dict(path=original, sha256=digest(payloads[relative]), size=len(payloads[relative]))
            event["bindings"].append(binding)
            source_map.append(dict(original_path=original, package_relative_path=relative, sha256=binding["sha256"], size=binding["size"]))
        payloads[sample["acceptance_record"]] = encoded_json(event)
        sample["acceptance_record_sha256"] = digest(payloads[sample["acceptance_record"]])
        payloads[sample["pixel_check_record"]] = encoded_json(dict(status="passed", pilot=pilot, PNG_fully_decoded=729,
            Exact_binary_checks=243, empty_exact_frame_indices={str(resolution): list(range(81)) for resolution in pilot8.RESOLUTIONS},
            Exact_white_pixel_counts={str(resolution): [0] * 81 for resolution in pilot8.RESOLUTIONS}))
        samples.append(sample)
    source_map.append(dict(original_path="original-atlas", package_relative_path="atlas.pdf",
                           sha256=digest(payloads["atlas.pdf"]), size=len(payloads["atlas.pdf"])))
    payloads["provenance/path_map.json"] = encoded_json(dict(files=source_map))
    payloads["acceptance_index.json"] = encoded_json(dict(accepted=8, pending=0, samples=samples))
    csv_stream = io.StringIO()
    writer = csv.DictWriter(csv_stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payloads["frame_order.csv"] = csv_stream.getvalue().encode()
    manifest = dict(schema="BuildingFiller.portable_Pilot8_AllRes.v1", package_id=pilot8.PACKAGE_ID,
                    sample_count=8, frames_per_sample=81, resolutions=list(pilot8.RESOLUTIONS), dataset_PNG=5832,
                    ModelMask_created=False, QF_1_3_sha256=digest(payloads["evidence/frozen/quality_config.json"]), samples=samples)
    return payloads, manifest


class Pilot8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads, cls.manifest = fixture_payloads()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pilot8-cpu-test-")
        self.addCleanup(self.temporary.cleanup)
        self.archive = Path(self.temporary.name) / "delivery.zip"
        self.payloads = dict(type(self).payloads)
        self.manifest = copy.deepcopy(type(self).manifest)
        self.write_archive()

    def write_archive(self, extra=None, update_checksums=True):
        manifest = copy.deepcopy(self.manifest)
        manifest["files"] = [dict(relative_path=relative, size=len(content), sha256=pilot8.payload_hash(content))
                             for relative, content in sorted(self.payloads.items())]
        payloads = dict(self.payloads, **{"manifest.json": encoded_json(manifest)})
        payloads["SHA256SUMS"] = "".join(f"{pilot8.payload_hash(content)}  {relative}\n" for relative, content in sorted(payloads.items())).encode()
        if not update_checksums:
            payloads["atlas.pdf"] = b"corrupted-after-hashing"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for relative, content in payloads.items():
                    bundle.writestr(pilot8.PACKAGE_ID + "/" + relative, content)
                if extra:
                    bundle.writestr(*extra)
        self.digest = pilot8.baseline.sha256(self.archive)
        Path(str(self.archive) + ".sha256").write_text(self.digest + "  " + self.archive.name + "\n")

    def read(self, verify_images=False):
        with mock.patch.object(pilot8, "ARCHIVE_SHA256", self.digest), mock.patch.object(pilot8, "QUALITY_SHA256", self.manifest["QF_1_3_sha256"]):
            return pilot8.read_delivery(self.archive, verify_images=verify_images)

    def test_mapping_status_and_compact_serializable_records(self):
        delivery = self.read()
        self.assertEqual([record["source_id"] for record in delivery["records"]], [465, 763, 871, 1227, 1259, 1848, 2297, 939])
        self.assertEqual(delivery["records"][4]["family_id"], 7)
        self.assertFalse(delivery["original_data_modified"])
        for record in delivery["records"]:
            self.assertLess(len(json.dumps(record).encode()), 120000)
            self.assertEqual(record["metadata"]["original_human_QA"], "pending")
            self.assertEqual(record["metadata"]["effective_human_QA"], "accepted")

    def test_full_png_decode_checks_all_resolutions(self):
        self.assertTrue(self.read(verify_images=True)["images_fully_decoded"])

    def test_model_input_never_reads_gt_by_default(self):
        record = self.read()["records"][1]
        read_member = zipfile.ZipFile.read
        seen = []

        def observed(bundle, name, *arguments, **keywords):
            seen.append(name)
            return read_member(bundle, name, *arguments, **keywords)

        with mock.patch.object(pilot8, "ARCHIVE_SHA256", self.digest), mock.patch.object(zipfile.ZipFile, "read", observed):
            selected = pilot8.read_sample(record)
        self.assertEqual(len(selected), 81)
        self.assertEqual(selected[-1]["source_index"], 80)
        self.assertFalse(any("/gt/" in name for name in seen))
        self.assertTrue(all(set(item["original_bytes"]) == {"damaged", "exact"} for item in selected))
        self.assertEqual(selected[0]["model_mask"].getextrema(), ((0, 0), (0, 0), (0, 0)))
        with mock.patch.object(pilot8, "ARCHIVE_SHA256", self.digest):
            self.assertEqual(pilot8.read_gt_B020(record).size, (512, 512))

    def test_worker_rejects_altered_record_and_archive(self):
        record = self.read()["records"][1]
        record["source_id"] = 465
        with self.assertRaisesRegex(ValueError, "identity mapping"):
            pilot8.read_sample(record)
        record["source_id"] = 763
        with self.assertRaisesRegex(ValueError, "Delivery changed"):
            pilot8.read_sample(record)

    def test_wrong_archive_and_sidecar_rejected(self):
        with self.assertRaisesRegex(ValueError, "ZIP SHA256 mismatch"):
            pilot8.read_delivery(self.archive)
        Path(str(self.archive) + ".sha256").write_text("0" * 64 + "  delivery.zip\n")
        with self.assertRaisesRegex(ValueError, "sidecar mismatch"):
            self.read()

    def test_payload_corruption_rejected(self):
        self.write_archive(update_checksums=False)
        with self.assertRaisesRegex(ValueError, "Payload SHA256 mismatch"):
            self.read()

    def test_duplicate_zip_member_rejected(self):
        self.write_archive(extra=(pilot8.PACKAGE_ID + "/atlas.pdf", b"duplicate"))
        with self.assertRaisesRegex(ValueError, "Duplicate ZIP"):
            self.read()

    def test_symlink_zip_member_rejected(self):
        entry = zipfile.ZipInfo(pilot8.PACKAGE_ID + "/link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.write_archive(extra=(entry, b"/outside"))
        with self.assertRaisesRegex(ValueError, "Non-regular ZIP"):
            self.read()

    def test_traversal_zip_member_rejected(self):
        self.write_archive(extra=(pilot8.PACKAGE_ID + "/../escape", b"unsafe"))
        with self.assertRaisesRegex(ValueError, "Unsafe member"):
            self.read()

    def test_wrong_sample_identity_rejected(self):
        self.manifest["samples"][1]["source_id"] = 465
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "mapping mismatch"):
            self.read()

    def test_wrong_effective_qa_rejected(self):
        self.manifest["samples"][1]["human_QA"] = "pending"
        acceptance = json.loads(self.payloads["acceptance_index.json"])
        acceptance["samples"][1]["human_QA"] = "pending"
        self.payloads["acceptance_index.json"] = encoded_json(acceptance)
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "QA/status mismatch"):
            self.read()

    def test_missing_frame_rejected(self):
        del self.payloads["samples/P02/512/exact/080.png"]
        self.write_archive()
        with self.assertRaises((ValueError, KeyError)):
            self.read()

    def test_non_binary_and_bad_modes_rejected(self):
        for mode, color, role, resolution in (("L", 127, "exact", 512), ("RGB", 0, "exact", 512), ("RGB", 0, "damaged", 768)):
            stream = io.BytesIO()
            Image.new(mode, (512, 512), color).save(stream, format="PNG")
            with self.subTest(mode=mode, color=color, resolution=resolution), self.assertRaises(ValueError):
                pilot8.png_pixels(stream.getvalue(), resolution, role, True)

    def test_mask_semantics_reuses_successful_adapter(self):
        damaged = Image.new("RGB", (512, 512), (20, 30, 40))
        mask = Image.new("L", (512, 512), 0)
        mask.putpixel((20, 20), 255)
        model_mask, condition = pilot8.baseline.adapt_images(damaged, mask)
        self.assertEqual(model_mask.getpixel((20, 20)), (255, 255, 255))
        self.assertEqual(condition.getpixel((20, 20)), (127, 127, 127))
        self.assertEqual(condition.getpixel((21, 21)), damaged.getpixel((21, 21)))
        self.assertTrue(np.array_equal(np.asarray(mask), np.asarray(model_mask)[:, :, 0]))


if __name__ == "__main__":
    unittest.main()
