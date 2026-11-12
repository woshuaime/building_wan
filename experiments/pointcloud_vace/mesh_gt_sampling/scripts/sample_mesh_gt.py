"""All-OBJ geometry GT: double surface candidates -> CUDA FPS indices -> binary PLY."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from zoneinfo import ZoneInfo

from build_fps import PROJECT, DATA, RUNTIME, DEPS, configure
configure()
tempfile.tempdir = str(RUNTIME / "tmp")

import numpy as np

CANDIDATES, POINTS = 50000, 16384
PLY_DTYPE = np.dtype([(key, "<f8") for key in ("x", "y", "z")] +
                     [(key, "<f4") for key in ("nx", "ny", "nz")])
PLY_HEADER = ("ply\nformat binary_little_endian 1.0\n"
              "comment mesh_gt: 50000 surface candidates -> GPU FPS 16384\n"
              "element vertex 16384\nproperty double x\nproperty double y\nproperty double z\n"
              "property float nx\nproperty float ny\nproperty float nz\nend_header\n").encode("ascii")
FPS_NAME = "pytorch3d.ops.sample_farthest_points (official CUDA FPS, FPS-only build)"


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seed_for_id(building_id):
    if building_id.isdecimal():
        return int(building_id) % (2 ** 64)
    return int.from_bytes(hashlib.blake2b(building_id.encode(), digest_size=8).digest(), "little")


def read_triangles(path):
    import trimesh
    mesh = trimesh.load(str(path), force="mesh", process=False, skip_materials=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError("OBJ has no finite triangle mesh")
    center = vertices.min(0) + (vertices.max(0) - vertices.min(0)) * .5
    # Exact geometric duplicates, including faces with different vertex IDs or winding.
    _, vertex_ids = np.unique(vertices, axis=0, return_inverse=True)
    canonical = np.sort(vertex_ids[faces], axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    unique_faces = faces[np.sort(first)]
    triangles = (vertices - center)[unique_faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(twice_area) & (twice_area > 0)
    triangles, cross, twice_area = triangles[valid], cross[valid], twice_area[valid]
    if not len(triangles):
        raise ValueError("All triangles have zero area")
    return triangles, cross / twice_area[:, None], twice_area * .5, center, {
        "input_triangles": len(faces), "duplicate_triangles": len(faces) - len(unique_faces),
        "zero_area_triangles": int((~valid).sum()), "surface_triangles": len(triangles)}


def surface_candidates(path, seed):
    triangles, normals, area, center, counts = read_triangles(path)
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(area, dtype=np.float64)
    face_ids = np.searchsorted(cdf, rng.random(CANDIDATES) * cdf[-1], side="right")
    selected = triangles[face_ids]
    uv = rng.random((CANDIDATES, 2), dtype=np.float64)
    u = np.sqrt(uv[:, 0])
    # Area-weighted face selection and uniform barycentric sampling, as in the author's surface sampler.
    local = ((1 - u)[:, None] * selected[:, 0] +
             (u * (1 - uv[:, 1]))[:, None] * selected[:, 1] +
             (u * uv[:, 1])[:, None] * selected[:, 2])
    return local, center, normals[face_ids], counts


def fps_indices(local):
    import torch
    from pytorch3d.ops import sample_farthest_points
    coordinates = torch.from_numpy(np.ascontiguousarray(local, dtype=np.float32))[None].to("cuda:0")
    with torch.no_grad():
        _, indices = sample_farthest_points(coordinates, K=POINTS, random_start_point=False)
    return indices[0].cpu().numpy()


def valid_ply(path):
    if not path.is_file() or path.stat().st_size != len(PLY_HEADER) + POINTS * PLY_DTYPE.itemsize:
        return False
    with path.open("rb") as stream:
        return stream.read(len(PLY_HEADER)) == PLY_HEADER


def write_ply(path, xyz, normals):
    if xyz.shape != (POINTS, 3) or normals.shape != xyz.shape:
        raise ValueError("Exactly 16384 XYZ/normal pairs are required")
    records = np.empty(POINTS, dtype=PLY_DTYPE)
    for column, key in enumerate(("x", "y", "z")):
        records[key] = xyz[:, column]
    for column, key in enumerate(("nx", "ny", "nz")):
        records[key] = normals[:, column]
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(PLY_HEADER)
            records.tofile(stream)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def convert(path, output):
    seed = seed_for_id(path.stem)
    local, center, normals, counts = surface_candidates(path, seed)
    world_candidates = local + center
    indices = fps_indices(local)
    if indices.shape != (POINTS,) or (indices < 0).any() or (indices >= CANDIDATES).any():
        raise ValueError("CUDA FPS returned invalid indices")
    # Reconstruct from the original double candidate coordinates, never the GPU float32 results.
    world = world_candidates[indices]
    selected_normals = normals[indices]
    write_ply(output, world, selected_normals)
    return dict(seed=seed, **counts), world, selected_normals


def self_test():
    import torch
    import trimesh
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="mesh_gt_debug_", dir=RUNTIME / "tmp") as name:
        directory = Path(name)
        mesh = trimesh.creation.box(extents=(10., 8., 6.))
        mesh.vertices += np.array([1000000.125, 6000000.25, 20.5], dtype=np.float64)
        source = directory / "synthetic_cube.obj"
        with source.open("w") as stream:
            for vertex in mesh.vertices:
                stream.write("v " + " ".join(format(float(x), ".17g") for x in vertex) + "\n")
            for face in mesh.faces:
                stream.write("f " + " ".join(str(int(x) + 1) for x in face) + "\n")
            stream.write("f " + " ".join(str(int(x) + 1) for x in mesh.faces[0][::-1]) + "\n")
            stream.write("f 1 1 2\n")
        target = directory / "synthetic_cube.ply"
        started = time.monotonic()
        counts, xyz, normals = convert(source, target)
        assert valid_ply(target)
        decoded = np.fromfile(target, dtype=PLY_DTYPE, offset=len(PLY_HEADER))
        np.testing.assert_array_equal(np.column_stack([decoded[k] for k in ("x", "y", "z")]), xyz)
        np.testing.assert_array_equal(np.column_stack([decoded[k] for k in ("nx", "ny", "nz")]), normals.astype(np.float32))
        assert counts["surface_triangles"] == 12 and counts["duplicate_triangles"] == 1 and counts["zero_area_triangles"] == 1
        record = dict(time=now(), test="synthetic_cube_only", success=True, backend=FPS_NAME,
                      gpu=torch.cuda.get_device_name(), candidates=CANDIDATES, output_points=POINTS,
                      xyz_dtype="float64", normals_dtype="float32", binary_ply=True,
                      elapsed_seconds=time.monotonic() - started, **counts)
        write_json(RUNTIME / "logs/self_test.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)


@contextmanager
def gpu_lock():
    directory = PROJECT / "logs"
    directory.mkdir(exist_ok=True)
    with (directory / ".wan_a6000_gpu.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def generate(log_prefix):
    import torch
    from pytorch3d.ops import sample_farthest_points
    torch.set_num_threads(2)
    source = DATA / "sample-data-organized/mesh"
    output = DATA / "mesh_gt_pointcloud"
    summary_path = Path(str(log_prefix) + ".summary.json")
    started = time.monotonic()
    state = dict(started_at=now(), status="running", pid=os.getpid(), backend=FPS_NAME,
                 input=str(source), output=str(output), candidates=CANDIDATES, points_per_building=POINTS,
                 xyz_dtype="double", normals_dtype="float", total=0, success=0, skipped=0, failed=0)
    with gpu_lock():
        meshes = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".obj")
        if not meshes:
            raise RuntimeError("No OBJ files found")
        state["total"] = len(meshes)
        output.mkdir(exist_ok=True)
        torch.cuda.init()
        state["gpu"] = torch.cuda.get_device_name()
        write_json(summary_path, state)
        print(json.dumps(dict(event="BATCH_STARTED", **state), ensure_ascii=False), flush=True)
        for index, path in enumerate(meshes, 1):
            target = output / (path.stem + ".ply")
            item_started = time.monotonic()
            try:
                if valid_ply(target):
                    state["skipped"] += 1
                    result = "SKIP"
                else:
                    counts, _, _ = convert(path, target)
                    state["success"] += 1
                    result = "SUCCESS"
                print(f"{now()} {result} {index}/{len(meshes)} id={path.stem} seconds={time.monotonic()-item_started:.3f}", flush=True)
            except Exception as error:
                state["failed"] += 1
                print(f"{now()} FAILED {index}/{len(meshes)} id={path.stem} error={error!r}", flush=True)
                traceback.print_exc()
            state.update(processed=index, last_building=path.stem, updated_at=now(), elapsed_seconds=time.monotonic()-started)
            write_json(summary_path, state)
        state.update(status="complete" if state["failed"] == 0 else "complete_with_failures", ended_at=now())
        write_json(summary_path, state)
        print(json.dumps(dict(event="BATCH_FINISHED", **state), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("self-test", "generate"))
    parser.add_argument("--log-prefix", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test":
        with gpu_lock():
            self_test()
    else:
        if args.log_prefix is None:
            parser.error("--log-prefix is required for generation")
        generate(args.log_prefix)
