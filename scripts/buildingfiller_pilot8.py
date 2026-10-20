"""Read-only, in-memory validation and loading of the hash-bound Pilot8 delivery."""

import csv
import hashlib
import io
import json
import re
import stat
import struct
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image

import vace_p01 as baseline


DEFAULT_ARCHIVE = baseline.ROOT / "data/buildingfiller_import/P02_P08_20261020/BuildingFiller_Pilot8_AllRes_v1.zip"
ARCHIVE_SHA256 = "7f8eff9ddb3c05e43fe5a8623258f9dceb6fa61d547b244c4f2d441099b4fcf4"
PACKAGE_ID = "BuildingFiller_Pilot8_AllRes_v1"
QUALITY_SHA256 = "af8f5e170d25ead847ca4c5dd7d40facca8f4c3ab79625f0bc06495399700594"
SAMPLE_MAPPING = (("01", "P01", 465, "A1", "Medium"),
                  ("02", "P02", 763, "A1", "Medium"),
                  ("03", "P03", 871, "A1", "Medium"),
                  ("04", "P04", 1227, "A2", "Medium"),
                  ("05", "P05", 1259, "A2", "Large"),
                  ("06", "P06", 1848, "A3", "Medium"),
                  ("07", "P07", 2297, "A3", "Large"),
                  ("08", "P08", 939, "A4", "Medium"))
RESOLUTIONS = (512, 768, 1024)
ROLES = ("gt", "damaged", "exact")
POSE_KEYS = ("position", "right", "up", "forward", "K", "world_to_camera_4x4", "camera_to_world_4x4")


def safe_relative(value):
    baseline.require(isinstance(value, str) and value and "\\" not in value, f"Unsafe member: {value}")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    baseline.require(not posix.is_absolute() and not windows.drive and ".." not in posix.parts
                     and str(posix) == value and "\x00" not in value, f"Unsafe member: {value}")
    return value


def checked_members(bundle, prefix):
    members = {}
    for entry in bundle.infolist():
        name = safe_relative(entry.filename)
        baseline.require(name.startswith(prefix) and not entry.is_dir(), f"Unexpected ZIP member: {name}")
        baseline.require(name not in members, f"Duplicate ZIP member: {name}")
        mode = entry.external_attr >> 16
        baseline.require(not stat.S_ISLNK(mode) and (stat.S_IFMT(mode) in (0, stat.S_IFREG)),
                         f"Non-regular ZIP member: {name}")
        members[name] = entry
    return members


def payload_hash(content):
    return hashlib.sha256(content).hexdigest()


def json_member(bundle, prefix, relative):
    return json.loads(bundle.read(prefix + safe_relative(relative)))


def png_pixels(content, resolution, role, decode):
    baseline.require(len(content) >= 33 and content[:8] == b"\x89PNG\r\n\x1a\n"
                     and content[12:16] == b"IHDR", "Invalid PNG header")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", content[16:29])
    baseline.require((width, height, depth, color) == (resolution, resolution, 8, 0 if role == "exact" else 2),
                     "PNG resolution/mode mismatch")
    if not decode:
        return None
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        baseline.require(image.format == "PNG" and image.size == (resolution, resolution)
                         and image.mode == ("L" if role == "exact" else "RGB"), "Decoded PNG mismatch")
        if role == "exact":
            counts = image.getcolors(257)
            baseline.require(counts is not None and {value for count, value in counts} <= {0, 255},
                             "Exact is not binary 0/255")
            return sum(count for count, value in counts if value == 255)
    return None


def check_status(value, label):
    baseline.require(value.get("human_QA") == "accepted" and value.get("training_ready") is False
                     and value.get("generation_verified") is False
                     and value.get("generation_evidence_verified") is True, f"{label}: effective QA/status mismatch")


def read_delivery(archive=DEFAULT_ARCHIVE, verify_images=True):
    archive = Path(archive)
    baseline.require(archive.is_file() and not archive.is_symlink(), "Missing or symlinked delivery archive")
    archive = archive.absolute()
    digest = baseline.sha256(archive)
    baseline.require(digest == ARCHIVE_SHA256, "Delivery ZIP SHA256 mismatch")
    sidecar = Path(str(archive) + ".sha256")
    baseline.require(sidecar.is_file() and not sidecar.is_symlink(), "Missing or symlinked ZIP SHA256 sidecar")
    checksum_parts = sidecar.read_text(encoding="utf-8-sig").strip().split()
    baseline.require(len(checksum_parts) == 2 and checksum_parts[0] == digest
                     and checksum_parts[1].lstrip("*") == archive.name, "ZIP SHA256 sidecar mismatch")
    prefix = PACKAGE_ID + "/"
    with zipfile.ZipFile(archive) as bundle:
        members = checked_members(bundle, prefix)
        baseline.require(bundle.testzip() is None, "ZIP CRC failure")
        checksums = {}
        for line in bundle.read(prefix + "SHA256SUMS").decode("utf-8").splitlines():
            checksum, relative = line.split("  ", 1)
            safe_relative(relative)
            baseline.require(re.fullmatch("[0-9a-f]{64}", checksum) and relative not in checksums,
                             "Invalid or duplicate payload checksum")
            baseline.require(payload_hash(bundle.read(prefix + relative)) == checksum,
                             f"Payload SHA256 mismatch: {relative}")
            checksums[relative] = checksum
        baseline.require(set(members) == {prefix + relative for relative in checksums} | {prefix + "SHA256SUMS"},
                         "ZIP payload inventory mismatch")
        manifest = json_member(bundle, prefix, "manifest.json")
        baseline.require(manifest["schema"] == "BuildingFiller.portable_Pilot8_AllRes.v1"
                         and manifest["package_id"] == PACKAGE_ID and manifest["sample_count"] == 8
                         and manifest["frames_per_sample"] == 81 and manifest["resolutions"] == list(RESOLUTIONS)
                         and manifest["dataset_PNG"] == 5832 and manifest["ModelMask_created"] is False,
                         "Unexpected Pilot8 manifest schema/counts")
        inventory = {entry["relative_path"]: entry for entry in manifest["files"]}
        baseline.require(len(inventory) == len(manifest["files"])
                         and set(checksums) == set(inventory) | {"manifest.json"}, "Manifest inventory mismatch")
        for relative, entry in inventory.items():
            baseline.require(entry["sha256"] == checksums[relative]
                             and entry["size"] == members[prefix + relative].file_size,
                             f"Manifest size/hash mismatch: {relative}")
        baseline.require(manifest["QF_1_3_sha256"] == checksums["evidence/frozen/quality_config.json"] == QUALITY_SHA256,
                         "Frozen QF-1.3 hash mismatch")
        acceptance = json_member(bundle, prefix, "acceptance_index.json")
        baseline.require(acceptance["accepted"] == 8 and acceptance["pending"] == 0
                         and len(acceptance["samples"]) == 8, "Effective acceptance counts mismatch")
        accepted = {entry["pilot"]: entry for entry in acceptance["samples"]}
        baseline.require(len(accepted) == 8 and len(manifest["samples"]) == 8, "Duplicate or missing Pilot sample")
        source_entries = json_member(bundle, prefix, "provenance/path_map.json")["files"]
        source_map = {entry["original_path"]: entry for entry in source_entries}
        baseline.require(len(source_map) == len(source_entries), "Duplicate original source mapping")
        rows = list(csv.DictReader(io.StringIO(bundle.read(prefix + "frame_order.csv").decode("utf-8"))))
        indexed = {(row["pilot"], row["resolution"], row["frame_id"]): row for row in rows}
        baseline.require(len(rows) == len(indexed) == 8 * 3 * 81, "CSV frame inventory mismatch")
        records = []
        for expected, sample in zip(SAMPLE_MAPPING, manifest["samples"]):
            output_number, pilot, source_id, kind, tier = expected
            baseline.require((sample["pilot"], sample["source_id"], sample["kind"], sample["tier"])
                             == (pilot, source_id, kind, tier), "Fixed Pilot/source/kind/tier mapping mismatch")
            baseline.require(sample["relative_root"] == "samples/" + pilot, "Unexpected sample root")
            baseline.require(sample["stable_id"] == f"b{source_id}_" + sample["source_sha256"][:8],
                             f"{pilot}: stable source identity mismatch")
            baseline.require(accepted[pilot] == sample, f"{pilot}: effective acceptance identity mismatch")
            check_status(sample, pilot)
            root = sample["relative_root"]
            meta = json_member(bundle, prefix, root + "/sample_meta.json")
            cameras = json_member(bundle, prefix, root + "/cameras.json")
            baseline.require(checksums[root + "/sample_meta.json"] == sample["sample_meta_sha256"]
                             == cameras["input_metadata_sha256"]
                             and checksums[root + "/cameras.json"] == sample["cameras_sha256"],
                             f"{pilot}: metadata/camera hash binding mismatch")
            baseline.require(meta["sample_id"] == cameras["sample_id"] == sample["sample_id"]
                             and meta["source"]["sha256"] == sample["source_sha256"]
                             and meta["kind"] == kind and meta["size"] == tier
                             and meta["removed_original_surface_ratio"] == sample["removed_ratio"],
                             f"{pilot}: original metadata identity mismatch")
            identity = meta.get("pilot_source_identity")
            baseline.require((pilot == "P01" and identity is None) or identity == {
                "source_id": source_id, "stable_id": sample["stable_id"], "family_id": sample["family_id"]},
                f"{pilot}: original identity binding mismatch")
            for role in ("gt", "damaged", "removed"):
                entry = meta["models"][role]
                baseline.require(entry["relative_path"] == f"models/{role}.obj"
                                 and checksums[root + "/" + entry["relative_path"]] == entry["sha256"]
                                 and cameras["source_geometry_sha256"][role] == entry,
                                 f"{pilot}: geometry/model camera binding mismatch")
            baseline.require(meta["models"]["gt"]["sha256"] == sample["source_sha256"], f"{pilot}: GT source mismatch")
            event = json_member(bundle, prefix, sample["acceptance_record"])
            baseline.require(checksums[sample["acceptance_record"]] == sample["acceptance_record_sha256"]
                             and event["decision"] == "accepted" and event["review_number"] == pilot,
                             f"{pilot}: hash-bound human acceptance mismatch")
            for key in ("source_id", "stable_id", "family_id", "sample_id", "kind", "tier", "source_sha256"):
                baseline.require(event[key] == sample[key], f"{pilot}: acceptance {key} mismatch")
            check_status(dict(event, human_QA=event["decision"]), pilot + " acceptance")
            for binding in [*event["bindings"], event["reviewed_atlas"]]:
                original = source_map[binding["path"]]
                relative = safe_relative(original["package_relative_path"])
                baseline.require(checksums[relative] == original["sha256"] == binding["sha256"]
                                 and members[prefix + relative].file_size == original["size"] == binding["size"],
                                 f"{pilot}: accepted source binding mismatch")
            baseline.require(cameras["frames"] == 81 and cameras["fps_metadata"] == 16
                             and cameras["view_ids"] == [f"{index:03}" for index in range(81)]
                             and cameras["trajectory"]["closure"] == "repeat"
                             and cameras["trajectory"]["orbit_degrees"] == 360
                             and cameras["trajectory"]["fps_metadata"] == 16, f"{pilot}: camera trajectory mismatch")
            originals = {entry["relative"]: entry["sha256"] for entry in cameras["output_files"]}
            baseline.require(len(originals) == len(cameras["output_files"]) == 729, f"{pilot}: original image inventory mismatch")
            pixel = json_member(bundle, prefix, sample["pixel_check_record"])
            baseline.require(pixel["status"] == "passed" and pixel["pilot"] == pilot
                             and pixel["PNG_fully_decoded"] == 729 and pixel["Exact_binary_checks"] == 243,
                             f"{pilot}: original pixel evidence mismatch")
            empty = {}
            for resolution in RESOLUTIONS:
                label = str(resolution)
                poses = cameras["resolutions"][label]
                baseline.require(len(poses) == 81 and all(poses[0][key] == poses[-1][key] for key in POSE_KEYS),
                                 f"{pilot}: camera count/closure mismatch")
                empty[label] = []
                white_counts = []
                for index, pose in enumerate(poses):
                    row = indexed[(pilot, label, f"{index:03}")]
                    baseline.require(pose["frame"] == int(row["camera_index"]) == index
                                     and pose["frame_id"] == row["frame_id"] == f"{index:03}"
                                     and row["sample_id"] == sample["sample_id"] and int(row["source_id"]) == source_id
                                     and pose["azimuth_degrees"] == float(row["azimuth_degrees"]) == 4.5 * index
                                     and pose["elevation_degrees"] == float(row["elevation_degrees"]) == 20
                                     and pose["time_seconds"] == float(row["time_seconds"]) == index / 16
                                     and pose["width"] == pose["height"] == resolution, f"{pilot}: frame/pose alignment mismatch")
                    for role in ROLES:
                        relative = f"{label}/{role}/{index:03}.png"
                        full = root + "/" + relative
                        baseline.require(row[role] == full and checksums[full] == originals[relative]
                                         == row[role + "_sha256"], f"{pilot}: frame/role hash alignment mismatch")
                        count = png_pixels(bundle.read(prefix + full), resolution, role, verify_images)
                        if role == "exact" and verify_images:
                            white_counts.append(count)
                            if count == 0:
                                empty[label].append(index)
                for role in ROLES:
                    baseline.require(checksums[f"{root}/{label}/{role}/000.png"] == checksums[f"{root}/{label}/{role}/080.png"],
                                     f"{pilot}: original input closure mismatch")
                if verify_images:
                    baseline.require(empty[label] == pixel["empty_exact_frame_indices"][label]
                                     and white_counts == pixel["Exact_white_pixel_counts"][label],
                                     f"{pilot}: Exact pixel evidence mismatch")
                else:
                    empty[label] = pixel["empty_exact_frame_indices"][label]
            record = dict(sample, output_number=output_number, review_number=pilot, archive=str(archive),
                          archive_sha256=digest, member_prefix=prefix,
                          empty_exact_frame_indices=empty["512"],
                          sha256={relative: checksum for relative, checksum in checksums.items()
                                  if relative.startswith(root + "/512/") or relative in (root + "/cameras.json", root + "/sample_meta.json")},
                          metadata=dict(original_human_QA=meta["human_QA"], original_render_status=meta["render_status"],
                                        original_training_ready=meta["training_ready"], effective_human_QA=sample["human_QA"],
                                        generation_evidence_verified=sample["generation_evidence_verified"]))
            records.append(record)
    return dict(archive=str(archive), archive_sha256=digest, member_prefix=prefix, records=records,
                files_sha256_checked=len(checksums), zip_crc="passed", dataset_png=5832,
                images_fully_decoded=bool(verify_images), original_data_modified=False)


def open_record(record):
    baseline.require(any((record["output_number"], record["pilot"], record["source_id"], record["kind"], record["tier"]) == entry
                         for entry in SAMPLE_MAPPING), "Worker record identity mapping mismatch")
    archive = Path(record["archive"])
    baseline.require(archive.is_file() and not archive.is_symlink()
                     and baseline.sha256(archive) == record["archive_sha256"] == ARCHIVE_SHA256, "Delivery changed after preflight")
    baseline.require(record["member_prefix"] == PACKAGE_ID + "/"
                     and record["relative_root"] == "samples/" + record["pilot"], "Worker record path mismatch")
    return zipfile.ZipFile(archive)


def read_sample(record, include_gt=False):
    selected = []
    prefix, root = record["member_prefix"], record["relative_root"]
    with open_record(record) as bundle:
        camera_bytes = bundle.read(prefix + root + "/cameras.json")
        baseline.require(payload_hash(camera_bytes) == record["cameras_sha256"], "Worker camera hash mismatch")
        cameras = json.loads(camera_bytes)
        baseline.require(cameras["sample_id"] == record["sample_id"] and cameras["frames"] == 81,
                         "Worker sample/camera identity mismatch")
        black_frames = []
        for index, pose in enumerate(cameras["resolutions"]["512"]):
            baseline.require(pose["frame"] == index and pose["frame_id"] == f"{index:03}", "Worker frame alignment mismatch")
            images, payload, frame = {}, {}, dict(frame_index=index, camera_index=index, frame_id=f"{index:03}")
            for role in (("damaged", "exact", "gt") if include_gt else ("damaged", "exact")):
                relative = f"{root}/512/{role}/{index:03}.png"
                content = bundle.read(prefix + safe_relative(relative))
                baseline.require(payload_hash(content) == record["sha256"][relative], f"Worker PNG hash mismatch: {relative}")
                png_pixels(content, 512, role, True)
                payload[role], frame[role] = content, relative
                if role != "gt":
                    with Image.open(io.BytesIO(content)) as image:
                        images[role] = image.copy()
            model_mask, conditioning = baseline.adapt_images(images["damaged"], images["exact"])
            if images["exact"].getextrema() == (0, 0):
                black_frames.append(index)
            selected.append(dict(source_index=index, output_index=index, frame=frame, camera=pose,
                                 original_bytes=payload, model_mask=model_mask, conditioning=conditioning))
        baseline.require(len(selected) == 81 and black_frames == record["empty_exact_frame_indices"],
                         "Worker original mask distribution/frame count mismatch")
    return selected


def read_gt_B020(record):
    relative = record["relative_root"] + "/512/gt/020.png"
    with open_record(record) as bundle:
        content = bundle.read(record["member_prefix"] + safe_relative(relative))
    baseline.require(payload_hash(content) == record["sha256"][relative], "GT B020 SHA256 mismatch")
    png_pixels(content, 512, "gt", True)
    with Image.open(io.BytesIO(content)) as image:
        return image.copy()
