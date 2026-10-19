"""Inspect a portable BuildingFiller ZIP without loading models or changing inputs."""

import argparse
import csv
import hashlib
import io
import json
import re
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    require(name and "\\" not in name and not path.is_absolute()
            and ".." not in path.parts, f"Unsafe ZIP path: {name}")
    return path


def audit(archive, expected_sha256=None):
    sidecar = Path(str(archive) + ".sha256")
    checksum_line = sidecar.read_text(encoding="utf-8-sig").strip()
    match = re.fullmatch(r"([a-fA-F0-9]{64})\s+\*?(.+)", checksum_line)
    require(match is not None and match[2] == archive.name, "Invalid ZIP SHA256 sidecar")
    checksum = digest(archive)
    require(checksum == match[1].lower(), "ZIP SHA256 mismatch")
    if expected_sha256 is not None:
        require(checksum == expected_sha256.lower(), "ZIP differs from expected source SHA256")

    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [item.filename for item in entries]
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        require(sum(item.file_size for item in entries) <= 1024 ** 3,
                "Package exceeds the 1 GiB inspection limit")
        for item in entries:
            safe_relative(item.filename)
            require(not stat.S_ISLNK(item.external_attr >> 16), "ZIP contains a symlink")
        prefix = archive.stem + "/"
        require(all(name.startswith(prefix) for name in names), "Unexpected ZIP root")
        require(bundle.testzip() is None, "ZIP CRC failure")

        def read(relative):
            safe_relative(relative)
            return bundle.read(prefix + relative)

        def load(relative):
            return json.loads(read(relative))

        manifest = load("manifest.json")
        require(manifest["schema"] == "BuildingFiller.portable_P01_512.v1",
                "Unsupported portable package schema")
        checksums = {}
        for line in read("SHA256SUMS").decode("utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            require(re.fullmatch(r"[a-f0-9]{64}", expected), "Invalid payload checksum")
            require(relative not in checksums, "Duplicate payload checksum")
            require(hashlib.sha256(read(relative)).hexdigest() == expected,
                    f"Payload SHA256 mismatch: {relative}")
            checksums[relative] = expected
        actual = {item.filename[len(prefix):] for item in entries if not item.is_dir()}
        require(actual == set(checksums) | {"SHA256SUMS"}, "ZIP file inventory mismatch")
        inventory = {item["relative_path"]: item for item in manifest["files"]}
        require(len(inventory) == len(manifest["files"]), "Duplicate manifest path")
        require(set(inventory) | {"manifest.json"} == set(checksums),
                "Manifest/checksum inventory mismatch")
        for relative, item in inventory.items():
            require(checksums[relative] == item["sha256"]
                    and len(read(relative)) == item["size"], f"Inventory mismatch: {relative}")

        sample_root = manifest["sample_relative_root"]
        camera = load(sample_root + "/cameras.json")
        metadata = load(sample_root + "/sample_meta.json")
        require(camera["input_metadata_sha256"] == checksums[sample_root + "/sample_meta.json"],
                "Camera/metadata hash binding mismatch")
        for role, item in metadata["models"].items():
            require(checksums[sample_root + "/" + item["relative_path"]] == item["sha256"],
                    f"Geometry hash mismatch: {role}")
        frames = manifest["frames"]
        poses = camera["resolutions"]["512"]
        require(len(frames) == len(poses) == manifest["frame_count"] == 81,
                "Expected 81 aligned frames")
        csv_frames = list(csv.DictReader(io.StringIO(read("frame_order.csv").decode("utf-8"))))
        require(len(csv_frames) == 81, "CSV frame count mismatch")
        camera_hashes = {item["relative"]: item["sha256"] for item in camera["output_files"]}
        counts = dict(gt=0, damaged=0, exact=0)
        coverage = []
        for index, (frame, pose, csv_frame) in enumerate(zip(frames, poses, csv_frames)):
            require(frame["frame_index"] == frame["camera_index"] == pose["frame"] == index,
                    "Frame/camera index mismatch")
            require(frame["frame_id"] == pose["frame_id"] == csv_frame["frame_id"] == f"{index:03}",
                    "Frame ID mismatch")
            require(frame["azimuth_degrees"] == pose["azimuth_degrees"] == index * 4.5
                    and frame["time_seconds"] == pose["time_seconds"] == index / 16,
                    "Frame angle/time mismatch")
            for role in counts:
                relative = f"{sample_root}/512/{role}/{index:03}.png"
                require(frame[role] == csv_frame[role] == relative, "Role/frame path mismatch")
                require(checksums[relative] == camera_hashes[f"512/{role}/{index:03}.png"],
                        "Image/camera hash binding mismatch")
                with Image.open(io.BytesIO(read(relative))) as image:
                    image.load()
                    require(image.format == "PNG" and image.size == (512, 512),
                            "Expected a 512x512 PNG")
                    require(image.mode == ("L" if role == "exact" else "RGB"),
                            "Unexpected image channel mode")
                    if role == "exact":
                        values = image.getcolors(maxcolors=257)
                        require(values is not None and all(value in (0, 255) for _, value in values),
                                "Nonbinary Exact image")
                        white_pixels = sum(count for count, value in values if value == 255)
                        coverage.append(dict(frame_index=index, frame_id=f"{index:03}",
                                             exact_white_pixels=white_pixels))
                counts[role] += 1
        for role in counts:
            require(checksums[f"{sample_root}/512/{role}/000.png"]
                    == checksums[f"{sample_root}/512/{role}/080.png"], "Closure image mismatch")
        pose_keys = ("position", "right", "up", "forward", "K",
                     "world_to_camera_4x4", "camera_to_world_4x4")
        require(all(poses[0][key] == poses[-1][key] for key in pose_keys), "Closure pose mismatch")
        window_start = max(range(64), key=lambda start: sum(
            row["exact_white_pixels"] for row in coverage[start:start + 17]))
        selection = []
        for output_index, frame in enumerate(frames[window_start:window_start + 17]):
            selected = dict(frame, output_index=output_index,
                            output_frame_id=f"{output_index:03}",
                            source_sha256={role: checksums[frame[role]] for role in counts})
            selection.append(selected)
        report = dict(
            status="passed", checked_at_utc=datetime.now(timezone.utc).isoformat(),
            archive=str(archive.resolve()), archive_bytes=archive.stat().st_size,
            archive_sha256=checksum, ZIP_CRC="passed", payload_SHA256_checked=len(checksums),
            zip_file_count=len(actual), image_count=sum(counts.values()), role_counts=counts,
            frame_count=81, distinct_orbit_positions=80, closure_image_pairs=3,
            source_identity=manifest["source_identity"], human_QA=manifest["human_QA"],
            training_ready=manifest["training_ready"], camera_metadata_binding="passed",
            all_PNG_fully_decoded=True, Exact_values=[0, 255],
            all_black_exact_frame_ids=[row["frame_id"] for row in coverage
                                       if row["exact_white_pixels"] == 0],
            exact_coverage=coverage, selection_policy="max_sum_Exact_17_contiguous_exclude_closure",
            selection_uses_GT_oracle_mask=True, selected_frames=selection,
            ModelMask_created=False, model_loaded=False, GPU_inference_started=False,
        )
    return report


def save_report(archive, report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = output_dir / "package" / str(safe_relative(item.filename))
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(bundle.read(item.filename))
        prefix = archive.stem + "/"
        frames = json.loads(bundle.read(prefix + "manifest.json"))["frames"]
        selected_ids = [0, 20, 40, 60, 46, 5, 13, 21]
        sheet = Image.new("RGB", (3 * 192, len(selected_ids) * 214), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, frame_index in enumerate(selected_ids):
            for column, role in enumerate(("gt", "damaged", "exact")):
                with Image.open(io.BytesIO(bundle.read(prefix + frames[frame_index][role]))) as image:
                    preview = image.convert("RGB").resize((192, 192), Image.Resampling.NEAREST)
                sheet.paste(preview, (column * 192, row_index * 214 + 22))
                draw.text((column * 192 + 4, row_index * 214 + 4),
                          f"{frame_index:03} {role}", fill="black")
        sheet.save(output_dir / "contact_sheet.png")
    (output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    selection = dict(policy=report["selection_policy"], source_archive_sha256=report["archive_sha256"],
                     frames=report["selected_frames"], GT_usage="evaluation_only")
    (output_dir / "frame_selection_17.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output-dir", type=Path,
                        help="Create a NEW audit directory with an extracted copy and reports")
    args = parser.parse_args()
    try:
        report = audit(args.archive, args.expected_sha256)
        if args.output_dir is not None:
            save_report(args.archive, report, args.output_dir)
        summary = {key: value for key, value in report.items()
                   if key not in ("exact_coverage", "selected_frames")}
        summary["selected_source_frames"] = [frame["frame_id"] for frame in report["selected_frames"]]
        if args.output_dir is not None:
            summary["report_directory"] = str(args.output_dir.resolve())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(json.dumps(dict(status="failed", error=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
