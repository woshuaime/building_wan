"""Frozen D-3000 inference identities and shared CPU/video utilities."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
TRAIN_SOURCE = ROOT / "experiments/pointcloud_vace/adapter_d/scripts"
TRAIN_RUN = ROOT / "outputs/pointcloud_vace/adapter_d/run_20261110_050724"
CASES = ROOT / "data/pointcloud_vace/微调任务/cases"
PYTHON = "/home/shi/miniconda3/envs/scj/bin/python"
sys.path.insert(0, str(TRAIN_SOURCE))
sys.path.insert(0, str(ROOT / "scripts"))
EXPECTED_WEIGHTS = {
    "adapter": "8720f3a3d2883aaa9b985e77f74961c7669d84c71ddd652e8129e565490aaaf4",
    "lora": "bcc21c2ac547781b0da6954fe3ab09998f7b57b99b6aa2741a6c3400ad181544",
}


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def event(run, stage, **extra):
    value = dict(time=now(), stage=stage, pid=os.getpid(), **extra)
    write(Path(run) / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def decode(path):
    import numpy as np
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
        "-map", "0:v:0", "-vsync", "0", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(81, 480, 832, 3)


def validate_inputs(video, mask):
    import numpy as np
    if video.shape != mask.shape or video.shape != (81, 480, 832, 3):
        raise ValueError("Expected aligned 81-frame 832x480 RGB video/mask")
    if not np.isin(mask, [0, 255]).all() or not np.all(mask == mask[..., :1]):
        raise ValueError("Mask is not binary grayscale")
    white = mask[..., 0] == 255
    if not white.any() or not (~white).any() or not (video[white] == 127).all():
        raise ValueError("White region must mean generate, with input RGB127")
    if not np.array_equal(video[0], video[-1]) or not np.array_equal(mask[0], mask[-1]):
        raise ValueError("Provided input no longer has the expected matching end view")
    return white


def observed_points(case):
    """Load P then discard withheld original IDs before any geometry is formed."""
    import numpy as np
    from geometry import read_cloud
    case = Path(case)
    record = json.loads((case / "case.json").read_text(encoding="utf-8"))
    paths = list((ROOT / "data/pointcloud_vace/sample-data-organized/pointcloud").glob(record["id"] + ".*"))
    if len(paths) != 1 or sha(paths[0]) != record["point_cloud_sha256"]:
        raise ValueError("Original point cloud identity changed")
    world, colors = read_cloud(paths[0])
    hidden = np.asarray(record["hidden_original_point_ids"], dtype=int)
    if len(world) != record["cloud_size"] or len(hidden) != record["count"]:
        raise ValueError("Source/withheld point count mismatch")
    if len(np.unique(hidden)) != len(hidden) or (len(hidden) and (hidden.min() < 0 or hidden.max() >= len(world))):
        raise ValueError("Invalid withheld original point IDs")
    keep = np.ones(len(world), dtype=bool)
    keep[hidden] = False
    points = world[keep] - np.asarray(record["world_to_local_origin"])
    cameras = json.loads((case / "reference/camera_poses.json").read_text())
    if len(cameras) != 81:
        raise ValueError("Expected 81 original camera poses")
    center = (points.min(0) + points.max(0)) / 2
    scale = max(float(np.linalg.norm(points - center, axis=1).max()), 1.)
    actual_cameras = []
    for i, camera in enumerate(cameras):
        if camera["frame"] != i or camera["projection"] != "perspective":
            raise ValueError("Camera frame ordering/projection mismatch")
        rotation = np.array([camera["right"], -np.array(camera["up"]), camera["forward"]])
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.e-6):
            raise ValueError("Invalid camera basis")
        actual_cameras.append(dict(position=np.asarray(camera["position"]) - center, R=rotation, frame=i))
    return points - center, colors[keep], actual_cameras, scale, dict(
        building_id=record["id"], original_points=len(world), withheld_points=len(hidden),
        geometry_points=int(keep.sum()), original_cloud=str(paths[0]), point_sha256=record["point_cloud_sha256"],
        normalization_center_in_case_coordinates=center.tolist(), normalization_scale=scale,
        original_point_ids_used=np.flatnonzero(keep).tolist(), withheld_points_used=False,
        mesh_used=False, target_video_decoded=False)


def verify_frozen(run):
    c = json.loads((Path(run) / "config.json").read_text())
    expected = hashlib.sha256(json.dumps({k: v for k, v in c.items() if k != "config_id"}, sort_keys=True).encode()).hexdigest()
    if expected != c["config_id"]:
        raise ValueError("Frozen inference configuration changed")
    for path, digest in c["frozen_files"].items():
        if sha(path) != digest:
            raise ValueError("Frozen source/input/weight changed: " + path)
    return c
