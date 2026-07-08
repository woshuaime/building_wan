"""Read-only import-pose regression checks.

Examples:
    python scripts/validate_import_pose.py --file D:\path\to\26.obj
    python scripts/validate_import_pose.py --batch D:\path\to\建筑物模型
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.obj_loader import OBJLoader  # noqa: E402
from core.track import TrackSystem  # noqa: E402
from ui.tk_widget import (  # noqa: E402
    MAX_TRIANGLES,
    PyVistaView,
    _obj_to_pv_coords,
    build_pyvista_mesh,
)


def _load_pose(path: Path, verbose: bool = False):
    loader = OBJLoader()
    stream = None if verbose else io.StringIO()
    with contextlib.redirect_stdout(stream):
        verts, _, faces = loader.load(str(path))

    mesh = build_pyvista_mesh(_obj_to_pv_coords(verts), faces, MAX_TRIANGLES)
    if mesh is None:
        raise RuntimeError(f"failed to build mesh: {path}")

    mesh.points = np.asarray(mesh.points, dtype=np.float64) - np.asarray(mesh.center, dtype=np.float64)
    track = TrackSystem()
    track.set_center([0.0, 0.0, 0.0])

    fake = types.SimpleNamespace(
        _mesh=mesh,
        _faces=faces,
        _track_sys=track,
        _loader_normalization_report=loader.get_normalization_report(),
    )
    fake.get_model_center = lambda m=mesh: np.asarray(m.center, dtype=np.float64)
    fake._get_track_center = lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    fake._validate_boundary_faces_parallel = types.MethodType(
        PyVistaView._validate_boundary_faces_parallel, fake
    )

    with contextlib.redirect_stdout(stream):
        ok = PyVistaView._validate_import_pose(fake)

    return ok, fake._last_import_pose_report, loader.get_normalization_report(), mesh, faces


def _bottom_angle_deg(mesh, faces) -> float | None:
    pts = np.asarray(mesh.points, dtype=np.float64)
    if pts.size == 0 or not faces:
        return None

    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    tol = max((x_max - x_min) * 0.03, 1e-5)
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    normal_sum = np.zeros(3, dtype=np.float64)

    for face in faces:
        if len(face) < 3:
            continue
        for i in range(1, len(face) - 1):
            tri = [face[0], face[i], face[i + 1]]
            p0, p1, p2 = pts[tri]
            normal = np.cross(p1 - p0, p2 - p0)
            area2 = float(np.linalg.norm(normal))
            if area2 <= 1e-12:
                continue
            normal /= area2
            centroid_x = float((p0[0] + p1[0] + p2[0]) / 3.0)
            alignment = abs(float(np.dot(normal, axis)))
            if centroid_x <= x_min + tol and alignment >= 0.80:
                if float(np.dot(normal, axis)) < 0.0:
                    normal = -normal
                normal_sum += normal * area2

    norm = float(np.linalg.norm(normal_sum))
    if norm <= 1e-12:
        return None
    normal_sum /= norm
    alignment = max(-1.0, min(1.0, abs(float(np.dot(normal_sum, axis)))))
    return math.degrees(math.acos(alignment))


def _validate_file(path: Path, max_bottom_angle: float, verbose: bool = False) -> tuple[bool, str]:
    ok, report, loader_report, mesh, faces = _load_pose(path, verbose=verbose)
    angle = _bottom_angle_deg(mesh, faces)
    angle_ok = angle is None or angle <= max_bottom_angle
    source = loader_report.get("normalizer_source", "n/a")
    axis = loader_report.get("ground_axis_alignment_before", "n/a")
    message = (
        f"{path.name}: ok={ok} source={source} ground_axis={axis} "
        f"bottom_angle={angle} | {report}"
    )
    return bool(ok and angle_ok), message


def _find_default_26() -> Path | None:
    root = Path("D:/scj/down_load")
    if not root.exists():
        return None
    matches = list(root.rglob("26.obj"))
    preferred = [p for p in matches if p.parent.name == "建筑物模型"]
    if preferred:
        return preferred[0]
    return matches[-1] if matches else None


def _run_synthetic_open_mesh() -> tuple[bool, str]:
    import pyvista as pv

    points = np.array(
        [
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    mesh = pv.PolyData(points)
    faces = [[0, 1, 2, 3]]
    track = TrackSystem()
    track.set_center([0.0, 0.0, 0.0])
    fake = types.SimpleNamespace(
        _mesh=mesh,
        _faces=faces,
        _track_sys=track,
        _loader_normalization_report={
            "status": "ground_normalized",
            "fallback_used": False,
            "normalizer_source": "mesh_bottom_boundary",
            "ground_inliers": 4,
            "ground_axis_alignment_before": 1.0,
        },
    )
    fake.get_model_center = lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    fake._get_track_center = lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    fake._validate_boundary_faces_parallel = types.MethodType(
        PyVistaView._validate_boundary_faces_parallel, fake
    )
    with contextlib.redirect_stdout(io.StringIO()):
        ok = PyVistaView._validate_import_pose(fake)
    return bool(ok), f"synthetic_open_mesh: ok={ok} | {fake._last_import_pose_report}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, action="append", default=[], help="OBJ file to validate")
    parser.add_argument("--batch", type=Path, help="Directory of OBJ files to validate")
    parser.add_argument("--max-bottom-angle", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    files = list(args.file)
    if not files:
        default_26 = _find_default_26()
        if default_26 is not None:
            files.append(default_26)
        cube = REPO_ROOT / "test_cube.obj"
        if cube.exists():
            files.append(cube)

    failures = []
    for path in files:
        passed, message = _validate_file(path, args.max_bottom_angle, verbose=args.verbose)
        print(message)
        if not passed:
            failures.append(message)

    passed, message = _run_synthetic_open_mesh()
    print(message)
    if not passed:
        failures.append(message)

    if args.batch:
        batch_files = sorted(args.batch.glob("*.obj"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        source_counts = {}
        state_counts = {}
        for path in batch_files:
            passed, message = _validate_file(path, args.max_bottom_angle, verbose=False)
            if not passed:
                failures.append(message)
            parts = message.split("|", 1)
            report = parts[1] if len(parts) > 1 else ""
            source = message.split("source=", 1)[1].split(" ", 1)[0] if "source=" in message else "n/a"
            state = report.split("boundary_status=", 1)[1].rstrip(")") if "boundary_status=" in report else "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1
            state_counts[state] = state_counts.get(state, 0) + 1
        print(f"batch_count={len(batch_files)} fail_count={len(failures)}")
        print(f"sources={source_counts}")
        print(f"states={state_counts}")

    if failures:
        print("FAILURES:")
        for failure in failures[:20]:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
