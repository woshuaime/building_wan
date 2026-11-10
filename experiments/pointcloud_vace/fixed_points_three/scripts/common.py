"""Shared, CPU-only identities for the three fixed-3D-point experiments."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/pointcloud_vace"
CASES = DATA / "微调任务/cases"
D_RUN = ROOT / "outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749"
PYTHON = "/home/shi/miniconda3/envs/scj/bin/python"
SAMPLES = [("01", "eave_3230"), ("04", "strip_5016"), ("07", "wall_10416")]
sys.path.insert(0, str(ROOT / "experiments/pointcloud_vace/adapter_d/scripts"))
sys.path.insert(0, str(ROOT / "scripts"))


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    p = Path(path)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(t, p)


def prepare_observed(case, out):
    """Return only remaining P and the supplied coarse box, never hidden coordinates."""
    import numpy as np
    from geometry import read_cloud
    record = json.loads((case / "case.json").read_text())
    paths = list((DATA / "sample-data-organized/pointcloud").glob(record["id"] + ".*"))
    if len(paths) != 1 or sha(paths[0]) != record["point_cloud_sha256"]:
        raise ValueError("Input point-cloud identity mismatch")
    world, rgb = read_cloud(paths[0])
    keep = np.ones(len(world), bool)
    removed = np.asarray(record["hidden_original_point_ids"], dtype=int)
    keep[removed] = False
    origin = np.array(record["world_to_local_origin"])
    observed = dict(points=world[keep] - origin, colors=rgb[keep], original_ids=np.flatnonzero(keep),
                    world_origin=origin, box_center=np.array(record["center"]) - origin,
                    box_axes=np.array(record["axes"]), box_half=np.array(record["half_sizes_m"]))
    np.savez_compressed(out / "observed_input.npz", **observed)
    write(out / "input_identity.json", dict(time=now(), case=case.name, source=str(paths[0]), source_sha256=sha(paths[0]),
        original_points=len(world), observed_points=int(keep.sum()), removed_count=len(removed),
        retained_original_ids=np.flatnonzero(keep).tolist(), removed_original_ids=removed.tolist(),
        hidden_coordinates_passed_to_generator=False, mesh_used=False,
        region_semantics="supplied coarse box only; box axes are not adopted as a fitted surface normal"))
    return observed
