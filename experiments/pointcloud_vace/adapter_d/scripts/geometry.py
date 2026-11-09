"""Observed-only geometry/conditions; paired mesh is confined to weak supervision."""
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import binary_dilation

W, H = 832, 480
BG = np.array([237, 240, 244], dtype=np.uint8)
FOCAL = H / (2 * np.tan(np.deg2rad(25)))
FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
         (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def read_cloud(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix.lower() == ".ply":
            properties, vertex, count, ascii_format = [], False, None, False
            for line in stream:
                words = line.split()
                if words[:1] == ["format"]:
                    ascii_format = words[1] == "ascii"
                elif words[:1] == ["element"]:
                    vertex = words[1] == "vertex"
                    if vertex:
                        count = int(words[2])
                elif words[:1] == ["property"] and vertex:
                    properties.append(words[-1])
                elif words == ["end_header"]:
                    break
            if not ascii_format or count is None:
                raise ValueError(f"Unsupported source PLY: {path}")
            columns = [properties.index(k) for k in ("x", "y", "z", "red", "green", "blue")]
            values = np.loadtxt(stream, usecols=columns, max_rows=count, ndmin=2)
        else:
            values = np.loadtxt(stream, usecols=range(6), ndmin=2)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid source cloud: {path}")
    if not ((values[:, 3:] >= 0) & (values[:, 3:] <= 255)).all():
        raise ValueError(f"Invalid source colors: {path}")
    return values[:, :3], values[:, 3:].astype(np.uint8)


def read_mesh(path):
    vertices, triangles = [], []
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            if words[0] == "v":
                vertices.append([float(v) for v in words[1:4]])
            elif words[0] == "f":
                indices = [int(v.split("/")[0]) for v in words[1:]]
                indices = [v - 1 if v > 0 else len(vertices) + v for v in indices]
                triangles.extend([[indices[0], indices[k], indices[k + 1]] for k in range(1, len(indices) - 1)])
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=int)
    if not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid mesh: {path}")
    return vertices[triangles]


def project(points, camera, factor=1):
    q = (points - camera["position"]) @ camera["R"].T
    uv = q[:, :2] / np.maximum(q[:, 2:], 1.e-8) * (FOCAL / factor)
    return uv + np.array([W / (2 * factor), H / (2 * factor)]), q[:, 2]


def orbit(points):
    """Same 20-degree, 80-view perspective orbit and 50-degree vertical FOV."""
    center = (points.min(0) + points.max(0)) / 2
    radius = np.linalg.norm(points - center, axis=1).max()
    distance = max(radius, 1.) / np.sin(np.deg2rad(25)) * 1.12
    cameras = []
    for i in range(81):
        az = 2 * np.pi * (i % 80) / 80
        outward = np.array([np.cos(az) * np.cos(np.deg2rad(20)),
                            np.sin(az) * np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))])
        right = np.array([-np.sin(az), np.cos(az), 0.])
        forward = -outward
        up = np.cross(right, forward)
        cameras.append(dict(position=center + distance * outward,
                            R=np.stack([right, -up, forward]), frame=i))
    return cameras, max(float(radius), 1.)


def render_ids(points, camera, factor=1):
    uv, depth = project(points, camera, factor)
    image = Image.new("I", (W // factor, H // factor), -1)
    draw = ImageDraw.Draw(image)
    positions = np.rint(uv).astype(int)
    visible = (depth > 0) & (positions[:, 0] >= -2) & (positions[:, 0] < W // factor + 2)
    visible &= (positions[:, 1] >= -2) & (positions[:, 1] < H // factor + 2)
    radius = 2 if factor == 1 else 1
    for i in np.argsort(-depth, kind="stable"):
        if visible[i]:
            x, y = positions[i]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=int(i))
    return np.array(image), depth


def depth_view(points, camera, scale, factor=2):
    ids, depths = render_ids(points, camera, factor)
    valid = ids >= 0
    depth = np.where(valid, depths[np.maximum(ids, 0)] / scale, 0).astype(np.float32)
    return np.stack([depth, valid.astype(np.float32)])


def fps(points, maximum=128):
    indices, distances = [], np.full(len(points), np.inf)
    index = int(np.argmax(np.linalg.norm(points - points.mean(0), axis=1)))
    for _ in range(min(maximum, len(points))):
        indices.append(index)
        distances = np.minimum(distances, np.sum((points - points[index]) ** 2, axis=1))
        distances[indices] = -1
        index = int(np.argmax(distances))
    return np.array(indices)


def inside_polygon(uv, hull):
    return (uv @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1.e-7).all(axis=1)


def observed_gap(points, seed):
    """Choose a low-density, bounded surface region using P alone, never mesh."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(points)
    sampled = rng.choice(len(points), min(384, len(points)), replace=False)
    neighbors = tree.query(points[sampled], k=min(16, len(points)))[1]
    neighborhoods = points[neighbors]
    centered = neighborhoods - neighborhoods.mean(1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered))
    normals = eigenvectors[:, :, 0]
    flatness = eigenvalues[:, 0] / np.maximum(eigenvalues.sum(1), 1.e-8)
    candidate_indices = np.flatnonzero((flatness < .025) & ((abs(normals[:, 2]) < .2) | (abs(normals[:, 2]) > .55)))
    planes, choices = [], []
    for k in candidate_indices[np.argsort(flatness[candidate_indices])]:
        normal, origin = normals[k], points[sampled[k]]
        if any(abs(normal @ n) > .97 and abs((origin - o) @ n) < .4 for n, o in planes):
            continue
        support = np.abs((points - origin) @ normal) < .23
        if support.sum() < 24:
            continue
        planes.append((normal, origin))
        if abs(normal[2]) < .2:
            u = np.cross([0., 0., 1.], normal)
        else:
            u = np.array([1., 0., 0.]) - normal * normal[0]
        u /= np.linalg.norm(u)
        axes = np.stack([u, np.cross(normal, u), normal])
        surface = (points[support] - origin) @ axes.T
        try:
            hull = ConvexHull(surface[:, :2])
        except Exception:
            continue
        lo, hi = surface[:, :2].min(0), surface[:, :2].max(0)
        half = np.minimum([1.5, 1.0], (hi - lo) / 5)
        if half.min() < .5:
            continue
        grid = np.stack(np.meshgrid(np.arange(lo[0], hi[0], .5), np.arange(lo[1], hi[1], .5)), -1).reshape(-1, 2)
        if len(grid) > 20000:
            grid = grid[rng.choice(len(grid), 20000, replace=False)]
        corners = np.array(list(product((-1, 1), repeat=2))) * half
        valid = np.logical_and.reduce([inside_polygon(grid + corner, hull) for corner in corners])
        grid = grid[valid]
        if not len(grid):
            continue
        stree = cKDTree(surface[:, :2] / half)
        counts = stree.query_ball_point(grid / half, 1., p=np.inf, return_length=True)
        dense_i = int(np.argmax(counts))
        if counts[dense_i] < 12:
            continue
        sparse = np.flatnonzero(counts <= max(1, counts[dense_i] * .15))
        if not len(sparse):
            continue
        nearest = cKDTree(surface[:, :2]).query(grid[sparse])[0]
        gap_i = sparse[int(np.argmax(nearest))]
        if nearest.max() < .45:
            continue
        center = origin + np.r_[grid[gap_i], 0.] @ axes
        donor = origin + np.r_[grid[dense_i], 0.] @ axes
        choices.append(dict(center=center, axes=axes, half=np.r_[half, .35], donor=donor,
                            observed_in_gap=int(counts[gap_i]), donor_count=int(counts[dense_i]),
                            kind="wall" if abs(normal[2]) < .2 else "roof",
                            score=float(counts[dense_i] / (1 + counts[gap_i]))))
        if len(planes) >= 8:
            break
    return max(choices, key=lambda x: x["score"]) if choices else None


def clip_polygon(poly, axis, bound, sign):
    result = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = sign * a[axis] - bound, sign * b[axis] - bound
        if da <= 1.e-8:
            result.append(a)
        if (da < 0) != (db < 0):
            result.append(a + (b - a) * da / (da - db))
    return np.asarray(result)


def weak_target(points, colors, gap, mesh_path, origin):
    """Transfer measured same-plane texture; mesh covers/supports target only."""
    if gap is None:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8), [], {"weak_reliable": False, "reason": "no_observed_bounded_gap"}
    try:
        triangles = read_mesh(mesh_path) - origin
    except (ValueError, OSError) as exc:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8), [], {"weak_reliable": False, "reason": str(exc)}
    local = (triangles - gap["center"]) @ gap["axes"].T
    vec = np.cross(local[:, 1] - local[:, 0], local[:, 2] - local[:, 0])
    norm = np.linalg.norm(vec, axis=1)
    aligned = (abs(vec[:, 2]) / np.maximum(norm, 1.e-8) > .97) & (abs(local[:, :, 2]).max(1) <= .3)
    polygons = []
    for triangle in local[aligned]:
        poly = triangle
        for axis in (0, 1):
            for sign in (-1, 1):
                poly = clip_polygon(poly, axis, gap["half"][axis], sign) if len(poly) else poly
        if len(poly) >= 3:
            polygons.append(poly)
    # Require coverage of the full candidate surface, not only a few isolated vertices.
    coverage_grid = np.stack(np.meshgrid(np.linspace(-gap["half"][0], gap["half"][0], 25),
                                        np.linspace(-gap["half"][1], gap["half"][1], 25)), -1).reshape(-1, 2)
    coverage = np.zeros(len(coverage_grid), bool)
    for poly in polygons:
        try:
            coverage |= inside_polygon(coverage_grid, ConvexHull(poly[:, :2]))
        except Exception:
            pass
    if coverage.mean() < .9:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8), [], {"weak_reliable": False, "reason": "mesh_coverage_below_90_percent", "coverage": float(coverage.mean())}
    donor_local = (points - gap["donor"]) @ gap["axes"].T
    donor_ids = np.flatnonzero((abs(donor_local) <= gap["half"]).all(1))
    proposed, copied_colors = [], []
    for i in donor_ids:
        xy = donor_local[i, :2]
        for poly in polygons:
            try:
                if not inside_polygon(xy[None], ConvexHull(poly[:, :2]))[0]:
                    continue
            except Exception:
                continue
            normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
            if abs(normal[2]) < 1.e-8:
                continue
            z = poly[0, 2] - (normal[:2] @ (xy - poly[0, :2])) / normal[2]
            proposed.append(gap["center"] + np.r_[xy, z] @ gap["axes"])
            copied_colors.append(colors[i])
            break
    proposed = np.asarray(proposed).reshape(-1, 3)
    copied_colors = np.asarray(copied_colors, dtype=np.uint8).reshape(-1, 3)
    if len(proposed):
        donor_spacing = np.median(cKDTree(points[donor_ids]).query(points[donor_ids], k=2)[0][:, 1])
        keep = cKDTree(points).query(proposed)[0] > max(.08, .5 * donor_spacing)
        proposed, copied_colors = proposed[keep], copied_colors[keep]
    reliable = len(proposed) >= 8
    audit = dict(weak_reliable=reliable, candidate_points=len(proposed), mesh_surface_coverage=float(coverage.mean()),
                 donor_points=len(donor_ids), donor_source="same_building_same_observed_plane_translation",
                 target_semantics="weak_estimate_not_real_complementary_scan")
    return (proposed if reliable else proposed[:0], copied_colors if reliable else copied_colors[:0],
            [p @ gap["axes"] + gap["center"] for p in polygons] if reliable else [], audit)


def projected_region(gap, camera, observed_ids):
    if gap is None:
        return np.zeros((H, W), bool)
    corners = np.array(list(product((-1, 1), repeat=3))) * gap["half"]
    corners = corners @ gap["axes"] + gap["center"]
    uv, depth = project(corners, camera)
    image = Image.new("L", (W, H))
    if (depth > 0).all():
        draw = ImageDraw.Draw(image)
        for face in FACES:
            draw.polygon([tuple(uv[i]) for i in face], fill=255)
    mask = np.array(image.filter(ImageFilter.MaxFilter(7))) > 0
    # Every observed sprite is retained; no measured point is artificially removed.
    return mask & (observed_ids < 0)


def geometry_bundle(points, cameras, scale, frame_ids):
    anchors = points[fps(points)]
    counts = cKDTree(points).query_ball_point(anchors, .5, return_length=True)
    refs = [cameras[i] for i in range(0, 80, 10)]
    targets = [cameras[i] for i in frame_ids]
    views = np.stack([depth_view(points, c, scale) for c in refs])
    target_depths = np.stack([depth_view(points, c, scale) for c in targets])
    return {"reference_views": views, "anchors": (anchors / scale).astype(np.float32),
            "anchor_density": np.log1p(counts).astype(np.float32),
            "ref_R": np.stack([c["R"] for c in refs]).astype(np.float32),
            "ref_C": np.stack([c["position"] / scale for c in refs]).astype(np.float32),
            "target_R": np.stack([c["R"] for c in targets]).astype(np.float32),
            "target_C": np.stack([c["position"] / scale for c in targets]).astype(np.float32),
            "target_depths": target_depths, "depth_tolerance": np.array(.45 / scale, np.float32),
            "K": np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1]], np.float32)}


def make_sample(row, frame_ids, gap_seed):
    path = Path(row["pointcloud"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["point_sha256"]:
        raise ValueError(f"Point identity mismatch: {row['id']}")
    world, rgb = read_cloud(path)
    origin = (world.min(0) + world.max(0)) / 2
    points = world - origin
    gap = observed_gap(points, gap_seed)
    cameras, scale = orbit(points)
    added, added_rgb, polygons, weak_audit = weak_target(points, rgb, gap, row["mesh"], origin)
    all_points, all_rgb = np.vstack([points, added]), np.vstack([rgb, added_rgb, BG])
    conditions, masks, targets, labels = [], [], [], []
    for frame in frame_ids:
        cam = cameras[frame]
        observed_ids, _ = render_ids(points, cam)
        observed = np.vstack([rgb, BG])[observed_ids]
        editable = projected_region(gap, cam, observed_ids)
        full_ids, _ = render_ids(all_points, cam)
        target = all_rgb[full_ids]
        support = Image.new("L", (W, H))
        draw = ImageDraw.Draw(support)
        for poly in polygons:
            uv, depth = project(poly, cam)
            if (depth > 0).all():
                draw.polygon([tuple(v) for v in uv], fill=255)
        weak = editable & (np.array(support) > 0)
        # Known supervision is confined to observed points and their immediate background.
        # Empty natural gaps elsewhere never become a full-image 'stay empty' target.
        known = binary_dilation(observed_ids >= 0, iterations=3) & ~editable
        target[~weak] = observed[~weak]
        conditioning = observed.copy()
        conditioning[editable] = 127
        conditions.append(Image.fromarray(conditioning))
        masks.append(Image.fromarray(np.repeat((editable * 255).astype(np.uint8)[..., None], 3, -1)))
        targets.append(Image.fromarray(target))
        labels.append(np.stack([known, np.zeros_like(known), weak]))
    labels = np.stack(labels)
    audit = {**weak_audit, "building_id": row["id"], "source": row["source"],
             "frame_ids": list(frame_ids), "input_points": len(points), "gap_seed": gap_seed,
             "origin": origin.tolist(), "normalization_scale": scale,
             "point_sha256": digest, "mesh_sha256": hashlib.sha256(Path(row["mesh"]).read_bytes()).hexdigest(),
             "label_pixel_counts": labels.sum(axis=(0, 2, 3)).tolist(),
             "mask_pixels": sum(int((np.array(m)[..., 0] > 0).sum()) for m in masks),
             "real_complementary_observations": False,
             "gap": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in gap.items()} if gap else None}
    return dict(gt=targets, conditioning=conditions, model_mask=masks), labels, geometry_bundle(points, cameras, scale, frame_ids), audit


def grouped_occupancy(values, block):
    """[T,C,H,W] -> [C,1+(T-1)/4,H/block,W/block], fractional areas."""
    t, c, h, w = values.shape
    if (t - 1) % 4 or h % block or w % block:
        raise ValueError("Invalid 4n+1 or spatial block layout")
    spatial = values.reshape(t, c, h // block, block, w // block, block).mean(axis=(3, 5))
    temporal = [spatial[0]] + [spatial[i:i + 4].mean(0) for i in range(1, t, 4)]
    return np.stack(temporal, axis=1).astype(np.float32)
