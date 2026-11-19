"""Immutable V2 video pairs, uploaded cameras, and observed-only geometry.

This module deliberately has no mesh/GT-PLY reader and never imports CUDA.  The
Y plane of the full-range source stream is copied directly: a video-library
RGB/gray conversion must not silently reinterpret these files as limited range.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
WIDTH, HEIGHT, FRAMES, FPS = 832, 480, 81, 16
TRAIN_COUNT, TOTAL_SAMPLES, SEED = 10747, 32241, 20261115
SPLITS = dict(train=10747, validation=905, test=661, diagnostic_only=147,
              exclude_boundary=3421, unpaired_GT_only=9)
LOSS_VERSION = "target-lt239-square-radius3-area8-time-first1-then4-v1"
GEOMETRY_VERSION = "selected-partial-f64-frozen-camera-pixel-centers-d-depth-v2"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino, device=stat.st_dev)


def settings(run):
    run = Path(run)
    return read_json(run / "config.json") if (run / "config.json").exists() else {}


def directories(run):
    config = settings(run)
    paths = config.get("paths", {})
    data = Path(paths.get("data", config.get("data_directory",
                ROOT / "data/pointcloud_vace/pointcloud_video_pairs_v2"))).resolve()
    cache = Path(paths.get("cache", config.get("cache_directory",
                 ROOT / "cache/pointcloud_vace/adapter_gt_v2"))).resolve()
    return data, cache


def _progress(run, phase, **values):
    if (Path(run) / "stop.request").exists():
        raise InterruptedError(f"Stop requested during import: {phase}")
    record = dict(stage="preparing", phase=phase, **values)
    write_json(Path(run) / "import_progress.json", record)
    from common import event
    event(run, "preparing", detail=phase, **values)


def _relative(base, value, role, building):
    """Manifest paths alone select training input; no preview discovery."""
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe {role} path for {building}: {value}")
    expected = {"partial_video": "partial", "GT_video": "gt",
                "camera": "cameras", "record": "records"}[role]
    if relative.parts[0] != expected:
        raise ValueError(f"Unexpected {role} directory for {building}: {value}")
    path = (base / str(relative)).resolve()
    if base not in path.parents or path.stem != building:
        raise ValueError(f"Mismatched {role} identity for {building}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role} for building {building}: {path}")
    return path


def video_metadata(path):
    """Inspect container/codec headers, without a second full-video review."""
    import av
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        codec = stream.codec_context
        result = dict(width=stream.width, height=stream.height, frames=stream.frames,
                      fps=str(stream.average_rate), codec=codec.name,
                      pixel_format=codec.format.name, color_range=int(codec.color_range),
                      profile=str(codec.profile))
    if (result["width"], result["height"], result["frames"], result["fps"]) != (832, 480, 81, "16"):
        raise ValueError(f"Unexpected video dimensions/timing {path}: {result}")
    # Monochrome x264 is reported by FFmpeg as yuvj420p on this uploaded corpus.
    # Raw full-range Y is unchanged and no chroma or range conversion is used.
    if result["codec"] != "h264" or result["color_range"] != 2:
        raise ValueError(f"Expected full-range lossless H264 source {path}: {result}")
    if result["pixel_format"] not in ("gray", "gray8", "yuvj420p", "yuvj444p", "yuv420p", "yuv444p"):
        raise ValueError(f"Unsupported raw-luma format {path}: {result}")
    if result["profile"] != "High 4:4:4 Predictive":
        raise ValueError(f"Unexpected x264 lossless profile {path}: {result}")
    return result


def validate_camera(camera, building, split):
    if str(camera.get("id")) != building or camera.get("split_use") != split:
        raise ValueError(f"Camera ID/split mismatch for {building}")
    expected = dict(width=832, height=480, fps=16, frames=81, version=2,
                    point_RGB=[80, 80, 80], background_RGB=[240, 240, 240],
                    pixel_format="gray", color_range="full", crf=0,
                    point_radius_pixels=.75, antialias_samples_per_pixel=16)
    for key, value in expected.items():
        if camera.get(key) != value:
            raise ValueError(f"Camera/render field mismatch {building}/{key}: {camera.get(key)}")
    for key in ("vertical_fov_degrees", "vertical_fov", "fov_y_degrees"):
        if key in camera and float(camera[key]) != 50.:
            raise ValueError(f"Camera vertical FOV must remain 50 degrees: {building}/{key}")
    origin = np.asarray(camera["world_to_local_origin"], dtype=np.float64)
    poses = camera["camera_poses_local"]
    if origin.shape != (3,) or not np.isfinite(origin).all() or len(poses) != 81:
        raise ValueError(f"Invalid camera origin/pose count for {building}")
    if [int(p["frame"]) for p in poses] != list(range(81)):
        raise ValueError(f"Camera frame order mismatch for {building}")
    rotations, positions = [], []
    for pose in poses:
        if pose.get("projection") != "perspective" or pose.get("pixel_center") != [.5, .5]:
            raise ValueError(f"Unexpected camera projection/center for {building}")
        rotations.append([pose["right"], -np.asarray(pose["up"]), pose["forward"]])
        positions.append(pose["position"])
    rotations, positions = np.asarray(rotations, np.float64), np.asarray(positions, np.float64)
    if rotations.shape != (81, 3, 3) or positions.shape != (81, 3):
        raise ValueError(f"Invalid camera matrices for {building}")
    if not np.isfinite(rotations).all() or not np.isfinite(positions).all():
        raise ValueError(f"Non-finite camera matrices for {building}")
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1.e-9):
        raise ValueError(f"Non-orthonormal frozen camera for {building}")
    if not np.allclose(rotations[0], rotations[80], atol=1.e-9) or not np.allclose(positions[0], positions[80], atol=1.e-9):
        raise ValueError(f"Closing camera differs from first for {building}")


def schedule_for(ids, seed=SEED):
    """Exactly one window/building/epoch, stable across Python processes."""
    ids = list(map(str, ids))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate training building IDs")
    rows = []
    for epoch in range(3):
        order = ids.copy()
        random.Random(seed + epoch).shuffle(order)
        for building in order:
            u = int(hashlib.sha256(f"{seed}:{building}".encode()).hexdigest(), 16) % 64
            start = (u + 21 * epoch) % 64
            rows.append(dict(id=building, epoch=epoch, start=start,
                             frame_ids=list(range(start, start + 17)), sample_index=len(rows)))
    return rows


def _uploaded_archives(run, data):
    """Extract missing members into final locations, validate CRC, then remove ZIP.

    Existing directories are reused; files with correct sizes are never replaced.
    The allowed roots narrowly identify this upload rather than historical ZIPs.
    """
    upload = ROOT / "data/pointcloud_vace/gt_video_v2_upload"
    archives = sorted(upload.glob("GT视频V2_A6000_3epoch_完整训练上传包_*.zip"))
    checked = []
    destination = data.parent
    allowed = {"pointcloud_video_pairs_v2", "微调任务_GT视频V2_3epoch"}
    for archive in archives:
        _progress(run, "checking_uploaded_archive", archive=str(archive))
        archive_identity = identity(archive)
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            if not members or {PurePosixPath(m.filename).parts[0] for m in members} - allowed:
                raise ValueError(f"Unexpected uploaded ZIP members: {archive}")
            byte_count = 0
            for i, member in enumerate(members):
                if (Path(run) / "stop.request").exists():
                    raise InterruptedError(f"Stop requested during uploaded archive check: {member.filename}")
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe ZIP member: {member.filename}")
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if target.exists() and target.stat().st_size != member.file_size:
                    raise ValueError(f"Existing imported file has wrong size: {target}")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".import.writing")
                    with handle.open(member) as source, temporary.open("wb") as output:
                        while chunk := source.read(8 * 1024 * 1024):
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, target)
                # Reading members verifies stored CRCs; it does not decode video.
                with handle.open(member) as source:
                    while chunk := source.read(8 * 1024 * 1024):
                        byte_count += len(chunk)
                # Compare extracted bytes too: equal sizes alone cannot detect
                # transfer damage or extraction corruption in an existing file.
                import zlib
                crc = 0
                with target.open("rb") as extracted:
                    while chunk := extracted.read(8 * 1024 * 1024):
                        crc = zlib.crc32(chunk, crc)
                if (crc & 0xffffffff) != member.CRC:
                    raise ValueError(f"Extracted upload CRC mismatch: {target}")
                if i % 1000 == 0:
                    _progress(run, "checking_uploaded_archive", members_checked=i + 1,
                              total_members=len(members), uncompressed_bytes_checked=byte_count)
        checked.append(dict(identity=archive_identity, members=len(members),
                            uncompressed_bytes=byte_count, archive_crc_verified=True,
                            extracted_member_sizes_verified=True, extracted_member_crcs_verified=True))
    return checked


def prepare(run):
    """Full import preflight, immutable manifests, and all 32,241 windows."""
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    data, cache = directories(run)
    began = time.monotonic()
    checked_archives = _uploaded_archives(run, data)
    render_path = data / "render_config.json"
    render = read_json(render_path)
    render_signature = hashlib.sha256(json.dumps(render, sort_keys=True).encode("utf-8")).hexdigest()
    if any(render.get(k) != v for k, v in dict(width=832, height=480, fps=16, frames=81,
            pixel_format="gray", color_range="full", crf=0,
            point_RGB=[80, 80, 80], background_RGB=[240, 240, 240]).items()):
        raise ValueError("Uploaded render_config is not the frozen grayscale V2 dataset")
    with (data / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "partial_video", "GT_video", "camera", "record", "split_use"]:
            raise ValueError(f"Unexpected manifest columns: {reader.fieldnames}")
        original = list(reader)
    counts = Counter(row["split_use"] for row in original)
    if dict(counts) != SPLITS or len({r["id"] for r in original}) != 15890:
        raise ValueError(f"Frozen split counts/unique IDs differ: {dict(counts)}")
    point_root = ROOT / "data/pointcloud_vace/sample-data-organized/pointcloud"
    source_index = {}
    for source in point_root.rglob("*"):
        if source.is_file() and source.suffix.lower() in (".ply", ".xyz"):
            source_index.setdefault(source.stem, []).append(source.resolve())
    mapping, train_rows, all_videos, codec_counts = {}, [], set(), Counter()
    imports = []
    for index, row in enumerate(original):
        if (run / "stop.request").exists():
            raise InterruptedError(f"Stop requested during manifest import at building {row['id']}")
        building, split = str(row["id"]), row["split_use"]
        if not building.isdigit():
            raise ValueError(f"Unexpected building ID: {building}")
        paths = {key: _relative(data, row[key], key, building)
                 for key in ("GT_video", "camera", "record")}
        partial = split != "unpaired_GT_only"
        if partial:
            paths["partial_video"] = _relative(data, row["partial_video"], "partial_video", building)
        elif row["partial_video"].strip():
            raise ValueError(f"GT-only building has a partial-video entry: {building}")
        camera_bytes, record_bytes = paths["camera"].read_bytes(), paths["record"].read_bytes()
        camera, record = json.loads(camera_bytes), json.loads(record_bytes)
        validate_camera(camera, building, split)
        if str(record["id"]) != building or record["split_use"] != split or bool(record["paired"]) != partial:
            raise ValueError(f"Record ID/split/paired mismatch: {building}")
        if record["frames_written"] != 81 or record["fps"] != 16 or record["config_version"] != 2:
            raise ValueError(f"Record timing/version mismatch: {building}")
        if record.get("config_signature") != render_signature:
            raise ValueError(f"Record/render configuration signature mismatch: {building}")
        if record["first_last_frame_rule"] != "last_frame_is_exact_copy_of_first":
            raise ValueError(f"Unexpected closing-frame rule: {building}")
        for key in ("partial_source", "gt_source"):
            if record.get(key) != camera.get(key):
                raise ValueError(f"Record/camera source identity mismatch: {building}/{key}")
        if PurePosixPath(record["camera_json"].replace("\\", "/")).stem != building:
            raise ValueError(f"Record camera identity mismatch: {building}")
        for role, field in (("gt", "GT_video"), ("partial", "partial_video")):
            if role == "partial" and not partial:
                continue
            video = paths[field]
            if video.stat().st_size != int(record["video_bytes"][role]):
                raise ValueError(f"Record video_bytes mismatch: {building}/{role}")
            meta = video_metadata(video)
            codec_counts[(meta["codec"], meta["profile"], meta["pixel_format"], meta["color_range"])] += 1
            all_videos.add(video)
        item = dict(id=building, split_use=split, **{k: str(v) for k, v in paths.items()},
                    camera_sha256=digest(camera_bytes), record_sha256=digest(record_bytes),
                    camera_identity=identity(paths["camera"]), record_identity=identity(paths["record"]),
                    render_sha256=digest(render_path.read_bytes()))
        mapping[record["camera_json"]] = str(paths["camera"])
        if partial:
            candidates = source_index.get(building, [])
            if not candidates:
                raise FileNotFoundError(f"Missing observed pointcloud for building {building} under {point_root}")
            basename = PurePosixPath(record["partial_source"].replace("\\", "/")).name
            if Path(basename).stem != building:
                raise ValueError(f"Observed record source ID differs: {building}/{basename}")
            exact = [p for p in candidates if p.name == basename]
            candidates = exact or candidates
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous observed source for building {building}: {candidates}")
            source = candidates[0]
            if source.suffix.lower() == ".ply":
                with source.open("rb") as handle:
                    header_count = None
                    for _ in range(200):
                        words = handle.readline().split()
                        if words[:2] == [b"element", b"vertex"]:
                            header_count = int(words[2])
                        if words == [b"end_header"]:
                            break
                    if header_count != int(record["partial_original_points"]):
                        raise ValueError(f"Observed PLY header count differs: {building}/{source}")
            selected = np.asarray(record["selected_source_point_indices"]["partial"], dtype=np.int64)
            count = int(record["partial_original_points"])
            if (selected.ndim != 1 or len(selected) != int(record["partial_display_points"])
                    or not len(selected) or len(np.unique(selected)) != len(selected)
                    or selected.min() < 0 or selected.max() >= count
                    or count != int(camera["partial_point_count"])):
                raise ValueError(f"Invalid observed display indices/counts: {building}")
            mapping[record["partial_source"]] = str(source)
            item.update(pointcloud=str(source), source_identity=identity(source),
                        partial_original_points=count, partial_display_points=len(selected))
        imports.append(item)
        if split == "train":
            train_rows.append(item)
        if index % 100 == 0 or index + 1 == len(original):
            _progress(run, "checking_manifest_headers_and_sources", buildings_checked=index + 1,
                      total_buildings=len(original), videos_checked=len(all_videos), train_count=len(train_rows))
    actual_videos = {p.resolve() for role in ("partial", "gt") for p in (data / role).rglob("*")
                     if p.is_file() and p.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov")}
    if actual_videos != all_videos or len(all_videos) != 31771:
        raise ValueError(f"Formal video inventory mismatch: actual={len(actual_videos)}, manifest={len(all_videos)}, "
                         f"extra={list(map(str, actual_videos - all_videos))[:10]}, missing={list(map(str, all_videos - actual_videos))[:10]}")
    if len(list((data / "cameras").glob("*.json"))) != 15890 or len(list((data / "records").glob("*.json"))) != 15890:
        raise ValueError("Camera/record inventory must contain exactly 15,890 JSON files each")
    schedule = schedule_for([r["id"] for r in train_rows])
    if len(train_rows) != TRAIN_COUNT or len(schedule) != TOTAL_SAMPLES:
        raise ValueError("Formal training sample budget mismatch")
    # A resumed run must not replace its frozen identities with a new source
    # snapshot. Its optimizer/RNG checkpoint belongs to the original schedule.
    for name, value in (("train_manifest.json", train_rows), ("schedule.json", schedule)):
        old = run / name
        if old.exists() and read_json(old) != value:
            raise ValueError(f"Frozen run identity changed; refusing to replace {name}")
    write_json(run / "import_map.json", mapping)
    write_json(run / "import_inventory.json", imports)
    write_json(run / "train_manifest.json", train_rows)
    write_json(run / "schedule.json", schedule)
    for archive in checked_archives:
        archive_path = Path(archive["identity"]["path"])
        if identity(archive_path) != archive["identity"]:
            raise ValueError(f"Uploaded archive changed during validation: {archive_path}")
        archive_path.unlink()
        archive["uploaded_zip_deleted"] = True
    report = dict(status="prepared", data_directory=str(data), cache_directory=str(cache),
                  split_counts=dict(counts), formal_videos=len(all_videos),
                  formal_video_bytes=sum(p.stat().st_size for p in all_videos),
                  metadata_profiles=[dict(codec=k[0], profile=k[1], pixel_format=k[2], color_range=k[3], count=v)
                                     for k, v in codec_counts.items()],
                  train_buildings=len(train_rows), scheduled_windows=len(schedule),
                  schedule_seed=SEED, schedule_sha256=digest((run / "schedule.json").read_bytes()),
                  window_derivation="u=int(SHA256(UTF8(str(seed)+':'+id)),16)%64; start=(u+21*epoch)%64",
                  epoch_shuffle="independent random.Random(seed+epoch).shuffle(manifest_train_order)",
                  manifest_sha256=digest((run / "train_manifest.json").read_bytes()),
                  source_manifest_sha256=digest((data / "manifest.csv").read_bytes()),
                  loss_version=LOSS_VERSION, geometry_version=GEOMETRY_VERSION,
                  archive_checks=checked_archives, elapsed_seconds=time.monotonic() - began,
                  no_mesh_or_gt_ply_loaded=True, full_visual_review=False)
    write_json(run / "data_prepared.json", report)
    _manifest.cache_clear()
    _progress(run, "prepared", **{k: report[k] for k in ("formal_videos", "train_buildings", "scheduled_windows", "elapsed_seconds")})
    return report


@lru_cache(maxsize=4)
def _manifest(run_string):
    return {str(row["id"]): row for row in read_json(Path(run_string) / "train_manifest.json")}


def _row(run, row):
    if "pointcloud" in row:
        return row
    try:
        return _manifest(str(Path(run).resolve()))[str(row["id"])]
    except KeyError as exc:
        raise ValueError(f"Unknown training building: {row.get('id')}") from exc


def decode_gray_window(path, frame_ids):
    import av
    ids = list(map(int, frame_ids))
    if len(ids) != 17 or ids != list(range(ids[0], ids[0] + 17)) or not 0 <= ids[0] <= 63:
        raise ValueError(f"Training requires an aligned 17-frame window in 000–079: {ids}")
    values = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        for number, frame in enumerate(container.decode(stream)):
            if number < ids[0]:
                continue
            if (frame.width, frame.height) != (832, 480) or int(frame.color_range) != 2:
                raise ValueError(f"Decoded range/shape changed: {path}, frame={number}")
            if frame.format.name not in ("gray", "gray8", "yuvj420p", "yuvj444p", "yuv420p", "yuv444p"):
                raise ValueError(f"Decoded raw-luma format changed: {path}, frame={number}, {frame.format.name}")
            plane = frame.planes[0]
            luma = np.frombuffer(plane, np.uint8).reshape(frame.height, plane.line_size)[:, :frame.width]
            values.append(luma.copy())
            if number == ids[-1]:
                break
    if len(values) != 17:
        raise ValueError(f"Truncated/corrupt video {path}: wanted {ids}, decoded {len(values)}")
    return np.stack(values)


def read_xyz(path):
    """Read original point order as float64, before subtracting world origin."""
    path = Path(path)
    if path.suffix.lower() != ".ply":
        with path.open(encoding="utf-8-sig") as stream:
            values = np.loadtxt(stream, usecols=(0, 1, 2), dtype=np.float64, ndmin=2)
    else:
        scalar_types = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
                        "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
                        "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
                        "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}
        with path.open("rb") as stream:
            if stream.readline().strip() != b"ply":
                raise ValueError(f"Invalid PLY header: {path}")
            properties, count, format_name, in_vertex = [], None, None, False
            while True:
                line = stream.readline()
                if not line:
                    raise ValueError(f"Truncated PLY header: {path}")
                words = line.decode("ascii").split()
                if not words:
                    continue
                if words[0] == "format":
                    format_name = words[1]
                elif words[0] == "element":
                    in_vertex = words[1] == "vertex"
                    if in_vertex:
                        count = int(words[2])
                elif words[0] == "property" and in_vertex:
                    if words[1] == "list":
                        raise ValueError(f"Unsupported list vertex property: {path}")
                    properties.append((words[-1], words[1]))
                elif words[0] == "end_header":
                    break
            names = [p[0] for p in properties]
            if count is None or any(key not in names for key in ("x", "y", "z")):
                raise ValueError(f"PLY lacks XYZ vertices: {path}")
            if format_name == "ascii":
                values = np.loadtxt(stream, usecols=[names.index(k) for k in ("x", "y", "z")],
                                    max_rows=count, dtype=np.float64, ndmin=2)
            elif format_name in ("binary_little_endian", "binary_big_endian"):
                order = "<" if format_name == "binary_little_endian" else ">"
                typed = np.dtype([(name, order + scalar_types[kind]) for name, kind in properties])
                vertices = np.fromfile(stream, dtype=typed, count=count)
                values = np.column_stack([vertices[k].astype(np.float64) for k in ("x", "y", "z")])
            else:
                raise ValueError(f"Unsupported PLY encoding: {path}/{format_name}")
            if len(values) != count:
                raise ValueError(f"Truncated pointcloud {path}: expected {count}, got {len(values)}")
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid observed XYZ: {path}")
    return values


def selected_partial(row, camera, record):
    source = Path(row["pointcloud"])
    if row.get("source_identity") and identity(source) != row["source_identity"]:
        raise ValueError(f"Observed source identity changed: {row['id']}/{source}")
    xyz = read_xyz(source)
    indices = np.asarray(record["selected_source_point_indices"]["partial"], dtype=np.int64)
    if len(xyz) != int(record["partial_original_points"]) or len(indices) != int(record["partial_display_points"]):
        raise ValueError(f"Observed source/selected count mismatch: {row['id']}")
    if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= len(xyz) or len(np.unique(indices)) != len(indices):
        raise ValueError(f"Observed source selection invalid: {row['id']}")
    # Essential: do not cast absolute world coordinates to float32 first.
    return xyz[indices] - np.asarray(camera["world_to_local_origin"], dtype=np.float64)


def camera_arrays(camera):
    poses = camera["camera_poses_local"]
    centers = np.asarray([p["position"] for p in poses], np.float64)
    rotations = np.asarray([[p["right"], -np.asarray(p["up"]), p["forward"]] for p in poses], np.float64)
    focal = HEIGHT / (2 * np.tan(np.deg2rad(25.)))
    K = np.array([[focal, 0, WIDTH / 2 - .5], [0, focal, HEIGHT / 2 - .5], [0, 0, 1]], np.float64)
    return centers, rotations, K


def project(points, center, rotation, K):
    q = (points - center) @ rotation.T
    return q[:, :2] / np.maximum(q[:, 2:], 1.e-12) * np.diag(K)[:2] + K[:2, 2], q[:, 2]


def depth_view(points, center, rotation, K, scale, image_size=(HEIGHT, WIDTH)):
    """D-style radius-one half-size depth support, exact uploaded projection."""
    height, width = image_size
    uv, z = project(points, center, rotation, K)
    # Preserve center convention when both resolution and intrinsics halve.
    pixels = np.rint((uv + .5) / 2 - .5).astype(np.int64)
    valid = (z > 0) & (pixels[:, 0] >= -1) & (pixels[:, 0] <= width // 2)
    valid &= (pixels[:, 1] >= -1) & (pixels[:, 1] <= height // 2)
    pixels, z = pixels[valid], z[valid]
    depth = np.full((height // 2, width // 2), np.inf, dtype=np.float64)
    # PIL's radius-one ellipse used by D corresponds to this five-pixel cross.
    for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        x, y = pixels[:, 0] + dx, pixels[:, 1] + dy
        inside = (x >= 0) & (x < width // 2) & (y >= 0) & (y < height // 2)
        np.minimum.at(depth, (y[inside], x[inside]), z[inside])
    valid = np.isfinite(depth)
    return np.stack([np.where(valid, depth / scale, 0), valid]).astype(np.float32)


def fps(points, maximum=128):
    indices, distances = [], np.full(len(points), np.inf)
    index = int(np.argmax(np.linalg.norm(points - points.mean(0), axis=1)))
    for _ in range(min(maximum, len(points))):
        indices.append(index)
        distances = np.minimum(distances, np.sum((points - points[index]) ** 2, axis=1))
        distances[indices] = -1
        index = int(np.argmax(distances))
    return np.asarray(indices, dtype=np.int64)


def cache_key(run, row, manifest=None):
    manifest = manifest or _row(run, row)
    config = settings(run)
    for field, path_field in (("source_identity", "pointcloud"),
                              ("camera_identity", "camera"), ("record_identity", "record")):
        if field in manifest and identity(manifest[path_field]) != manifest[field]:
            raise ValueError(f"Frozen {path_field} identity changed: {row['id']}")
    prompt = (Path(run) / "prompt.txt").read_text() if (Path(run) / "prompt.txt").exists() else config.get("prompt", "")
    return digest(dict(building=str(row["id"]), frame_ids=row["frame_ids"],
        source=manifest.get("source_identity", identity(manifest["pointcloud"])),
        partial_video=identity(manifest["partial_video"]), target_video=identity(manifest["GT_video"]),
        camera_sha256=manifest["camera_sha256"], record_sha256=manifest["record_sha256"],
        render_sha256=manifest["render_sha256"], prompt=prompt,
        vae_identity=config.get("models_verified", config.get("models", {})),
        encoding=dict(width=832, height=480, frames=17, raw_full_range_Y=True,
                      tile_size=[30, 52], tile_stride=[15, 26]),
        condition="partial-reactive-all-one-mask-v1", loss=LOSS_VERSION,
        geometry=GEOMETRY_VERSION))


def geometry_for(run, row):
    run = Path(run)
    manifest = _row(run, row)
    frame_ids = list(map(int, row["frame_ids"]))
    if (len(frame_ids) != 17 or frame_ids != list(range(frame_ids[0], frame_ids[0] + 17))
            or not 0 <= frame_ids[0] <= 63):
        raise ValueError(f"Invalid structural training window: {row['id']}/{frame_ids}")
    if manifest.get("source_identity") and identity(manifest["pointcloud"]) != manifest["source_identity"]:
        raise ValueError(f"Observed source changed: {row['id']}")
    # Persist one building's eight fixed views; target depths are cheap to
    # rasterize and must not occupy hundreds of GiB over 32,241 windows.
    key = cache_key(run, dict(row, frame_ids=[]), manifest)
    _, cache = directories(run)
    path = cache / "geometry" / key[:2] / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["cache_key"].item()) != key:
                raise ValueError(f"Geometry cache identity mismatch: {path}")
            base = {name: stored[name].copy() for name in stored.files if name != "cache_key"}
        return _window_geometry(base, frame_ids)
    camera_bytes, record_bytes = Path(manifest["camera"]).read_bytes(), Path(manifest["record"]).read_bytes()
    if digest(camera_bytes) != manifest["camera_sha256"] or digest(record_bytes) != manifest["record_sha256"]:
        raise ValueError(f"Uploaded camera/selection identity changed: {row['id']}")
    camera, record = json.loads(camera_bytes), json.loads(record_bytes)
    points = selected_partial(manifest, camera, record)
    centers, rotations, K = camera_arrays(camera)
    # Scale uses partial points only.  Camera positions themselves stay frozen.
    scale = max(float(np.linalg.norm(points - (points.min(0) + points.max(0)) / 2, axis=1).max()), 1.)
    anchors = points[fps(points)]
    # A fixed 128-anchor tensor permits true heterogeneous B=2, even for the
    # occasional 64-point display. Repeat valid anchors, never invent geometry.
    if len(anchors) < 128:
        anchors = anchors[np.arange(128) % len(anchors)]
    density = cKDTree(points).query_ball_point(anchors, .5, return_length=True)
    refs = list(range(0, 80, 10))
    base = dict(reference_views=np.stack([depth_view(points, centers[i], rotations[i], K, scale) for i in refs]),
                  anchors=(anchors / scale).astype(np.float32),
                  anchor_density=np.log1p(density).astype(np.float32),
                  ref_C=(centers[refs] / scale).astype(np.float32),
                  ref_R=rotations[refs].astype(np.float32),
                  depth_tolerance=np.asarray(.45 / scale, dtype=np.float32), K=K.astype(np.float32),
                  _points_f64=points, _centers_f64=centers, _rotations_f64=rotations,
                  _K_f64=K, _scale=np.asarray(scale))
    if not all(np.isfinite(value).all() for value in base.values()):
        raise ValueError(f"Non-finite structural input: {row['id']}/{frame_ids}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.writing")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, cache_key=np.asarray(key), **base)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _window_geometry(base, frame_ids)


def _window_geometry(base, frame_ids):
    points, centers, rotations = base["_points_f64"], base["_centers_f64"], base["_rotations_f64"]
    scale, K = float(base["_scale"]), base["_K_f64"]
    result = {name: value for name, value in base.items() if not name.startswith("_")}
    result.update(target_C=(centers[frame_ids] / scale).astype(np.float32),
                  target_R=rotations[frame_ids].astype(np.float32),
                  target_depths=np.stack([depth_view(points, centers[i], rotations[i], K, scale)
                                          for i in frame_ids]))
    return result


def occupancy_for(target, image_size=(HEIGHT, WIDTH)):
    """Two fractional areas on VAE [5,60,104], never token [5,30,52]."""
    height, width = image_size
    if target.shape != (17, height, width) or target.dtype != np.uint8:
        raise ValueError(f"Unexpected target video shape/dtype: {target.shape}/{target.dtype}")
    foreground = np.stack([binary_dilation(frame < 239, structure=np.ones((7, 7), dtype=bool))
                           for frame in target])
    spatial = foreground.reshape(17, height // 8, 8, width // 8, 8).mean(axis=(2, 4), dtype=np.float32)
    temporal = np.stack([spatial[0]] + [spatial[i:i + 4].mean(axis=0)
                                     for i in range(1, 17, 4)])
    return np.stack([temporal, 1 - temporal]).astype(np.float32)


def load_raw(run, row):
    manifest = _row(run, row)
    frames = row["frame_ids"]
    partial = decode_gray_window(manifest["partial_video"], frames)
    target = decode_gray_window(manifest["GT_video"], frames)
    return dict(partial=partial, target=target, geometry=geometry_for(run, row),
                occupancy=occupancy_for(target), id=str(row["id"]), frame_ids=frames,
                cache_key=cache_key(run, row, manifest))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    options = parser.parse_args()
    if options.prepare:
        prepare(options.run)
    else:
        summary = read_json(options.run / "data_prepared.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
