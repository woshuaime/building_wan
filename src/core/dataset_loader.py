from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.obj_loader import parse_obj_file


PREALIGNED_TRIPLET = "prealigned_triplet"
SHARED_TRIPLET = "shared_triplet"
RAW_TRIPLET = "raw_triplet"

BOTTOM_TRIANGLE_MIN_ALIGNMENT = 0.80
BOTTOM_NORMAL_MIN_ALIGNMENT = 0.95
BOTTOM_MIN_AREA_RATIO = 0.03


@dataclass
class MeshData:
    role: str
    path: str
    vertices: np.ndarray
    faces: list


@dataclass
class DatasetTriplet:
    mode: str
    root_dir: str
    complete: MeshData
    incomplete: MeshData
    removed: MeshData
    validation_warnings: list
    alignment_report: dict = field(default_factory=dict)


def load_dataset_triplet(folder, mode=PREALIGNED_TRIPLET):
    """Load a dataset triplet.

    prealigned_triplet:
        Normal mode prepared and froze aligned_complete.obj first. Blender then
        exported complete/incomplete/removed from that frozen model while keeping
        one world coordinate system. This function only parses them. It must not
        normalize, recenter, rescale, reorient, or compute any mesh-dependent
        transform.

    shared_triplet:
        Non-default experimental mode.
        The three meshes share one source coordinate system, but the complete
        model may not yet sit at the capture center. Compute one rigid alignment
        from complete only, then apply the same rotation+translation to all
        three meshes. Scale is always 1.0.

    raw_triplet:
        Reserved for later. It will compute one normalization matrix from raw
        complete and apply the same matrix to all three meshes. It must never
        compute transforms from incomplete or removed.
    """
    if mode == PREALIGNED_TRIPLET:
        return _load_prealigned_triplet(folder)
    if mode == SHARED_TRIPLET:
        return _load_shared_triplet(folder)
    if mode == RAW_TRIPLET:
        raise NotImplementedError(
            "raw_triplet is reserved: compute one transform from complete, "
            "then apply that same transform to complete/incomplete/removed."
        )
    raise ValueError(f"Unsupported dataset mode: {mode}")


def _load_prealigned_triplet(folder):
    root, meshes = _read_triplet_meshes(folder)
    warnings = _validate_triplet_bounds(meshes, mode=PREALIGNED_TRIPLET)

    return DatasetTriplet(
        mode=PREALIGNED_TRIPLET,
        root_dir=str(root),
        complete=meshes["complete"],
        incomplete=meshes["incomplete"],
        removed=meshes["removed"],
        validation_warnings=warnings,
        alignment_report={
            "mode": PREALIGNED_TRIPLET,
            "scale": 1.0,
            "transform": "identity",
            "normalize_applied": False,
            "recenter_applied": False,
            "rescale_applied": False,
            "reorient_applied": False,
            "geometry_based_alignment_applied": False,
        },
    )


def _load_shared_triplet(folder):
    root, meshes = _read_triplet_meshes(folder)
    transform, report = _compute_shared_alignment(
        meshes["complete"].vertices,
        meshes["complete"].faces,
    )

    aligned = {}
    for role, mesh in meshes.items():
        aligned[role] = MeshData(
            role=role,
            path=mesh.path,
            vertices=transform(mesh.vertices),
            faces=mesh.faces,
        )

    warnings = _validate_triplet_bounds(aligned, mode=SHARED_TRIPLET)
    return DatasetTriplet(
        mode=SHARED_TRIPLET,
        root_dir=str(root),
        complete=aligned["complete"],
        incomplete=aligned["incomplete"],
        removed=aligned["removed"],
        validation_warnings=warnings,
        alignment_report=report,
    )


def _read_triplet_meshes(folder):
    root = Path(folder)
    files = {
        "complete": root / "complete.obj",
        "incomplete": root / "incomplete.obj",
        "removed": root / "removed.obj",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset triplet files: " + ", ".join(missing))

    meshes = {}
    for role, path in files.items():
        vertices, _, faces = parse_obj_file(str(path))
        if not vertices:
            raise ValueError(f"{path} has no vertices")
        meshes[role] = MeshData(
            role=role,
            path=str(path),
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=faces,
        )

    return root, meshes


def _mesh_bounds(vertices):
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    size = hi - lo
    return lo, hi, size


def _compute_shared_alignment(complete_vertices, complete_faces):
    verts = np.asarray(complete_vertices, dtype=np.float64)
    if len(verts) < 3:
        raise ValueError("complete.obj has too few vertices for shared alignment")

    lo, hi, size = _mesh_bounds(verts)
    input_bbox_center = (lo + hi) * 0.5
    centered = verts - input_bbox_center

    R, orient_report = _estimate_complete_orientation(centered, complete_faces)
    rotated = centered @ R.T
    r_lo, r_hi, _ = _mesh_bounds(rotated)
    post_rotation_bbox_center = (r_lo + r_hi) * 0.5

    def apply_transform(vertices):
        v = np.asarray(vertices, dtype=np.float64)
        return ((v - input_bbox_center) @ R.T) - post_rotation_bbox_center

    final_complete = apply_transform(verts)
    final_validation = _validate_shared_alignment_result(
        final_complete,
        complete_faces,
        expected_source_bottom_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    report = {
        "mode": SHARED_TRIPLET,
        "rule": (
            "Rigid alignment is computed from complete.obj only, then applied "
            "unchanged to complete/incomplete/removed. Scale is not changed."
        ),
        "scale": 1.0,
        "input_bbox_center": [float(v) for v in input_bbox_center],
        "post_rotation_bbox_center": [float(v) for v in post_rotation_bbox_center],
        "final_bbox_center": final_validation["final_bbox_center"],
        "final_validation": final_validation,
        "rotation_matrix": [[float(x) for x in row] for row in R],
    }
    report.update(orient_report)
    return apply_transform, report


def _estimate_complete_orientation(centered_vertices, faces):
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ground_normal, ground_report = _estimate_bottom_normal_from_faces(centered_vertices, faces)
    R_ground = _rotation_between_vectors(ground_normal, z_axis)
    ground_pts = centered_vertices @ R_ground.T

    pts_xy = ground_pts[:, :2]
    lo = pts_xy.min(axis=0)
    hi = pts_xy.max(axis=0)
    dx = float(hi[0] - lo[0])
    dy = float(hi[1] - lo[1])
    if abs(dx - dy) > 1e-6:
        main_dir = np.array([1.0, 0.0], dtype=np.float64) if dx >= dy else np.array([0.0, 1.0], dtype=np.float64)
    else:
        cov = np.cov((pts_xy - pts_xy.mean(axis=0)).T)
        _, eigvecs = np.linalg.eigh(cov)
        main_dir = np.asarray(eigvecs[:, -1], dtype=np.float64)
        main_dir /= max(float(np.linalg.norm(main_dir)), 1e-12)

    target_xy = np.array([1.0, 0.0], dtype=np.float64)
    cross_val = float(target_xy[0] * main_dir[1] - target_xy[1] * main_dir[0])
    dot_val = float(target_xy[0] * main_dir[0] + target_xy[1] * main_dir[1])
    yaw_angle = float(np.arctan2(cross_val, dot_val))
    c, s = np.cos(yaw_angle), np.sin(yaw_angle)
    R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R = R_yaw @ R_ground

    top_axis_after = R @ ground_normal
    report = {
        "orientation_source": ground_report["source"],
        "ground_normal_before": [float(v) for v in ground_normal],
        "ground_axis_alignment_before": float(ground_report["alignment"]),
        "ground_area_ratio": float(ground_report["area_ratio"]),
        "yaw_angle_deg": float(np.degrees(yaw_angle)),
        "main_axis_xy_before_yaw": [float(v) for v in main_dir],
        "top_axis_after_alignment": [float(v) for v in top_axis_after],
    }
    return R, report


def _validate_shared_alignment_result(vertices, faces, expected_source_bottom_normal):
    lo, hi, size = _mesh_bounds(vertices)
    center = (lo + hi) * 0.5
    max_extent = max(float(np.max(size)), 1e-9)
    center_err = float(np.linalg.norm(center))
    center_tol = max(max_extent * 1e-6, 1e-8)
    if center_err > center_tol:
        raise RuntimeError(
            "shared_triplet final validation failed: complete bbox center is not at track center "
            f"(err={center_err:.6g}, tol={center_tol:.6g})"
        )

    bottom_normal, bottom_report = _estimate_bottom_normal_from_faces(vertices, faces)
    bottom_parallel = float(abs(np.dot(bottom_normal, expected_source_bottom_normal)))
    if bottom_parallel < BOTTOM_NORMAL_MIN_ALIGNMENT:
        raise RuntimeError(
            "shared_triplet final validation failed: complete bottom is not parallel to the expected "
            f"source capture plane (score={bottom_parallel:.6f})"
        )

    app_top_dir = np.array([
        expected_source_bottom_normal[2],
        expected_source_bottom_normal[0],
        expected_source_bottom_normal[1],
    ], dtype=np.float64)
    app_top_dir /= max(float(np.linalg.norm(app_top_dir)), 1e-12)
    app_top_to_camera = float(np.dot(app_top_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64)))
    if app_top_to_camera < 0.999:
        raise RuntimeError(
            "shared_triplet final validation failed: mapped top direction does not face the +X camera start "
            f"(score={app_top_to_camera:.6f})"
        )

    return {
        "final_bbox_center": [float(v) for v in center],
        "center_error": center_err,
        "center_tolerance": center_tol,
        "source_bottom_parallel_score": bottom_parallel,
        "app_top_to_camera_score": app_top_to_camera,
        "bottom_source": bottom_report["source"],
        "bottom_area_ratio": float(bottom_report["area_ratio"]),
    }


def _estimate_bottom_normal_from_faces(vertices, faces):
    pts = np.asarray(vertices, dtype=np.float64)
    if not faces:
        raise ValueError("complete.obj has no faces; cannot detect bottom plane")

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_span = float(np.ptp(pts[:, 0])) if len(pts) else 0.0
    y_span = float(np.ptp(pts[:, 1])) if len(pts) else 0.0
    z_span = float(np.ptp(pts[:, 2])) if len(pts) else 0.0
    footprint_area2 = max(2.0 * x_span * y_span, 1e-9)
    bottom_z = float(np.min(pts[:, 2]))
    bottom_tol = max(z_span * 0.03, 1e-8)
    normal_sum = np.zeros(3, dtype=np.float64)
    area_sum = 0.0

    for face in faces:
        face_indices = [int(idx) for idx in face if 0 <= int(idx) < len(pts)]
        if len(face_indices) < 3:
            continue
        for i in range(1, len(face_indices) - 1):
            p0 = pts[face_indices[0]]
            p1 = pts[face_indices[i]]
            p2 = pts[face_indices[i + 1]]
            n = np.cross(p1 - p0, p2 - p0)
            area2 = float(np.linalg.norm(n))
            if area2 < 1e-12:
                continue
            n /= area2
            if np.dot(n, z_axis) < 0.0:
                n = -n
            alignment = float(abs(np.dot(n, z_axis)))
            centroid = (p0 + p1 + p2) / 3.0
            if (
                centroid[2] <= bottom_z + bottom_tol
                and alignment >= BOTTOM_TRIANGLE_MIN_ALIGNMENT
            ):
                normal_sum += n * area2
                area_sum += area2

    area_ratio = float(area_sum / footprint_area2)
    if area_sum <= 1e-12 or area_ratio < BOTTOM_MIN_AREA_RATIO:
        raise ValueError(
            "complete.obj bottom plane could not be detected; "
            "shared_triplet requires a recognizable bottom face because it uses complete.obj "
            "to compute one shared rotation for the whole triplet. Please check that the model "
            "has a visible bottom/ground face and has not been exported with broken faces."
        )

    normal = normal_sum / max(float(np.linalg.norm(normal_sum)), 1e-12)
    alignment = float(abs(np.dot(normal, z_axis)))
    if alignment < BOTTOM_NORMAL_MIN_ALIGNMENT:
        raise ValueError(
            "complete.obj bottom plane is not close enough to the expected source horizontal plane "
            f"(alignment={alignment:.6f}). shared_triplet does not guess a new scale or independent "
            "alignment for incomplete/removed; please inspect this triplet before dataset capture."
        )

    return normal, {
        "source": "complete_bottom_faces",
        "alignment": alignment,
        "area_ratio": area_ratio,
    }


def _rotation_between_vectors(v_from, v_to):
    v_from = np.asarray(v_from, dtype=np.float64)
    v_to = np.asarray(v_to, dtype=np.float64)
    v_from /= max(float(np.linalg.norm(v_from)), 1e-12)
    v_to /= max(float(np.linalg.norm(v_to)), 1e-12)

    cross = np.cross(v_from, v_to)
    dot = float(np.dot(v_from, v_to))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-9:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, v_from))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, v_from) * v_from
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return _rotation_axis_angle(axis, np.pi)

    s = float(np.linalg.norm(cross))
    k = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + k + (k @ k) * ((1.0 - dot) / (s * s))


def _rotation_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ], dtype=np.float64)


def _validate_triplet_bounds(meshes, mode=PREALIGNED_TRIPLET):
    """Sanity-check shared coordinates without modifying any mesh."""
    warnings = []
    complete = meshes["complete"].vertices
    c_lo, c_hi, c_size = _mesh_bounds(complete)
    c_extent = float(np.max(c_size))
    if not np.isfinite(complete).all() or c_extent <= 1e-9:
        raise ValueError("complete.obj has invalid or degenerate coordinates")

    center = (c_lo + c_hi) * 0.5
    centroid = complete.mean(axis=0)
    center_tol = max(c_extent * 0.08, 1e-6)
    if float(np.linalg.norm(center)) > center_tol:
        warnings.append(
            "complete.obj bounding-box center is not close to origin; "
            f"{mode} expects the shared track center at (0, 0, 0)"
        )
    if float(np.linalg.norm(centroid)) > center_tol * 2.0:
        warnings.append(
            "complete.obj centroid is far from origin; verify that the triplet "
            "was exported from the normalized complete model"
        )

    pad = max(c_extent * 0.08, 1e-6)
    for role in ("incomplete", "removed"):
        verts = meshes[role].vertices
        if not np.isfinite(verts).all():
            raise ValueError(f"{role}.obj has invalid coordinates")
        lo, hi, size = _mesh_bounds(verts)
        extent = float(np.max(size))
        if extent <= 1e-9:
            warnings.append(f"{role}.obj is nearly degenerate")
        outside_low = lo < (c_lo - pad)
        outside_high = hi > (c_hi + pad)
        if bool(np.any(outside_low) or np.any(outside_high)):
            raise ValueError(
                f"{role}.obj bounds are outside complete.obj bounds; "
                "the triplet may not share one coordinate system"
            )
        if role == "incomplete" and extent > c_extent * 1.20:
            warnings.append("incomplete.obj is larger than complete.obj by more than 20%")
        if role == "removed" and extent > c_extent * 1.20:
            warnings.append("removed.obj is larger than complete.obj by more than 20%")
    return warnings
