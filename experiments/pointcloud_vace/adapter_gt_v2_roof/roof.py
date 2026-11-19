"""Observed roof tracks for VACE and held-out roof metrics."""
from pathlib import Path

import av
import numba
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
import torch
from torch import nn

import data
from common import ROOT

HEIGHT, WIDTH = data.HEIGHT, data.WIDTH
MESHES = ROOT / "data/pointcloud_vace/sample-data-organized/mesh"
GT_POINTS = ROOT / "data/pointcloud_vace/mesh_gt_pointcloud_stable"


def observed_roof(points):
    """Select stable candidate IDs once in the observed building coordinates."""
    if len(points) < 12:
        return points[:0]
    nearest = cKDTree(points).query(points, k=12)[1]
    neighbors = points[nearest]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    values, normals = np.linalg.eigh(centered.transpose(0, 2, 1) @ centered)
    floor, top = np.quantile(points[:, 2], [.02, .98])
    keep = (np.abs(normals[:, 2, 0]) >= .5) & (values[:, 1] > 0)
    keep &= values[:, 0] < .4 * values[:, 1]
    keep &= points[:, 2] >= floor + .2 * (top - floor)
    return points[keep]


def pack_frames(frames):
    """Match the Wan latent temporal layout, retaining all four frame slots."""
    tiles = np.concatenate((np.repeat(frames[:1], 4, axis=0), frames[1:]))
    count, height, width, channels = tiles.shape
    return (tiles.reshape(count // 4, 4, height, width, channels)
            .transpose(0, 2, 3, 1, 4).reshape(-1, 4 * channels).copy())


def tracks(points, camera, frames):
    """Each 16-pixel token carries 4×4 subcell counts and local XYZ averages."""
    roof = observed_roof(points)
    scale = max(float(np.linalg.norm(points - (points.min(0) + points.max(0)) / 2, axis=1).max()), 1.)
    centers, rotations, intrinsics = data.camera_arrays(camera)
    cells = (HEIGHT // 4) * (WIDTH // 4)
    views = []
    for frame in frames:
        uv, depth = data.project(roof, centers[frame], rotations[frame], intrinsics)
        ij = np.floor((uv + .5) / 4).astype(np.int64)
        keep = ((depth > 0) & (ij[:, 0] >= 0) & (ij[:, 0] < WIDTH // 4)
                & (ij[:, 1] >= 0) & (ij[:, 1] < HEIGHT // 4))
        index = ij[keep, 1] * (WIDTH // 4) + ij[keep, 0]
        count = np.bincount(index, minlength=cells)
        coords = roof[keep] / scale
        summed = [np.bincount(index, weights=coords[:, axis], minlength=cells) for axis in range(3)]
        image = np.column_stack([np.log1p(count), *[v / np.maximum(count, 1) for v in summed]])
        views.append(image.reshape(HEIGHT // 16, 4, WIDTH // 16, 4, 4)
                     .transpose(0, 2, 1, 3, 4).reshape(HEIGHT // 16, WIDTH // 16, 64))
    return pack_frames(np.asarray(views, dtype=np.float32))


class RoofAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.density = nn.Linear(128, 8)
        self.project = nn.Linear(136, 1536, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, hints):
        occupied = hints[..., ::4].reshape(*hints.shape[:-1], 4, 16).amax(-1) > 0
        features = self.encode(hints)
        fractions = self.density(features).float().sigmoid()
        residual = self.project(torch.cat((features, fractions.to(features.dtype)), dim=-1))
        return residual * occupied.any(-1, keepdim=True), fractions, occupied


def mesh_triangles(building, camera):
    """OBJ geometry is used for supervision and scoring only."""
    vertices, normals, faces, labels = [], [], [], []
    path = MESHES / f"{building}.obj"
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "vn":
            normals.append([float(v) for v in fields[1:4]])
        elif fields[0] == "f":
            corners = [value.split("/") for value in fields[1:]]
            indices = [int(value[0]) - 1 if int(value[0]) > 0 else len(vertices) + int(value[0])
                       for value in corners]
            for j in range(1, len(indices) - 1):
                tri = [indices[0], indices[j], indices[j + 1]]
                faces.append(tri)
                refs = [corners[k] for k in (0, j, j + 1)]
                if all(len(v) > 2 and v[2] for v in refs):
                    norm = [int(v[2]) - 1 if int(v[2]) > 0 else len(normals) + int(v[2]) for v in refs]
                    n = np.mean([normals[k] for k in norm], axis=0)
                else:
                    a, b, c = np.asarray([vertices[k] for k in tri])
                    n = np.cross(b - a, c - a)
                nz = n[2] / max(float(np.linalg.norm(n)), 1.e-10)
                labels.append(1 if nz > .5 else 2 if abs(nz) < .2 else 0)
    if not faces:
        raise ValueError(f"No mesh faces for {building}")
    return (np.asarray(vertices, dtype=np.float64) - camera["world_to_local_origin"],
            np.asarray(faces, dtype=np.int32), np.asarray(labels, dtype=np.uint8))


@numba.njit(nogil=True, cache=True)
def raster(uv, depth, triangles, triangle_labels):
    """Nearest surface by perspective-correct depth at source pixel centers."""
    z = np.full((HEIGHT, WIDTH), np.inf, np.float32)
    labels = np.zeros((HEIGHT, WIDTH), np.uint8)
    for i, (a, b, c) in enumerate(triangles):
        if min(depth[a], depth[b], depth[c]) <= 0:
            continue
        xa, ya = uv[a]
        xb, yb = uv[b]
        xc, yc = uv[c]
        denom = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        if abs(denom) < 1.e-12:
            continue
        x0 = max(0, int(np.ceil(min(xa, xb, xc))))
        x1 = min(WIDTH - 1, int(np.floor(max(xa, xb, xc))))
        y0 = max(0, int(np.ceil(min(ya, yb, yc))))
        y1 = min(HEIGHT - 1, int(np.floor(max(ya, yb, yc))))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                u = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / denom
                v = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / denom
                w = 1 - u - v
                if min(u, v, w) >= -1.e-9:
                    distance = 1 / (u / depth[a] + v / depth[b] + w / depth[c])
                    if distance < z[y, x]:
                        z[y, x] = distance
                        labels[y, x] = triangle_labels[i]
    return labels, z


def surface_views(building, camera, frame_ids):
    vertices, triangles, triangle_labels = mesh_triangles(building, camera)
    centers, rotations, intrinsics = data.camera_arrays(camera)
    for i in frame_ids:
        uv, depth = data.project(vertices, centers[i], rotations[i], intrinsics)
        yield raster(uv, depth, triangles, triangle_labels)


def tile_sum(mask, edge):
    return mask.reshape(HEIGHT // edge, edge, WIDTH // edge, edge).sum((1, 3))


def density_targets(building, camera, frame_ids, images):
    fractions, weights = [], []
    for (surface, _), image in zip(surface_views(building, camera, frame_ids), images):
        mask = surface == 1
        coverage = tile_sum(mask, 16)
        fractions.append(np.stack([tile_sum(mask & (image < level), 16) / np.maximum(coverage, 1)
                                   for level in (220, 150)], axis=-1))
        weights.append(coverage[:, :, None] / 256)
    return (pack_frames(np.asarray(fractions, dtype=np.float32)),
            pack_frames(np.asarray(weights, dtype=np.float32)))


def video_luma(path, *, comparison=False):
    result = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        for i, frame in enumerate(container.decode(stream)):
            if frame.is_corrupt or frame.pts is None or frame.pts * frame.time_base * 16 != i:
                raise ValueError(f"Invalid video frame {i}: {path}")
            plane = frame.planes[0]
            pixels = np.frombuffer(plane, np.uint8).reshape(frame.height, plane.line_size)
            area = pixels[32:32 + HEIGHT, WIDTH:2 * WIDTH] if comparison else pixels[:, :WIDTH]
            if area.shape != (HEIGHT, WIDTH):
                raise ValueError(f"Wrong video dimensions: {path}")
            image = area.astype(np.float32)
            if int(frame.color_range) != 2:
                image = np.clip((image - 16) * (255 / 219), 0, 255)
            result.append(image)
    if len(result) != 81:
        raise ValueError(f"Incomplete 81-frame video: {path}")
    return result


def rgb_luma(frames):
    return [np.asarray(frame.convert("L"), dtype=np.float32) for frame in frames]


def scores(building, camera, selection, gt, clips):
    """Score all clips on the same nearest roof/wall surfaces and visible point tracks."""
    centers, rotations, intrinsics = data.camera_arrays(camera)
    points = data.read_xyz(GT_POINTS / f"{building}.ply")[selection["selected_source_point_indices"]["gt"]]
    points += np.asarray(camera["GT_PLY_local_to_world_offset"]) - camera["world_to_local_origin"]
    all_clips = {"gt": gt, **clips}
    values = {name: dict(cv=[], motion=0., pairs=0, intersect=0, union=0) for name in all_clips}
    previous = {}
    for i, (surface, z) in enumerate(surface_views(building, camera, range(81))):
        roof = surface == 1
        wall = surface == 2
        gt_wall = tile_sum(wall & (gt[i] < 220), 8) > 0
        coverage = tile_sum(roof, 16) >= 224
        uv, depth = data.project(points, centers[i], rotations[i], intrinsics)
        pixel = np.rint(uv).astype(np.int64)
        px = pixel[:, 0].clip(0, WIDTH - 1)
        py = pixel[:, 1].clip(0, HEIGHT - 1)
        visible = ((pixel[:, 0] == px) & (pixel[:, 1] == py)
                   & (surface[py, px] == 1) & (np.abs(z[py, px] - depth) < .06))
        for name, clip in all_clips.items():
            image, score = clip[i], values[name]
            if i == 0:
                first_mask = roof
            if i < 80:
                local = tile_sum(roof & (image < 220), 16)[coverage] / 256
                if len(local) > 1:
                    # A blank roof will be rejected by the dark coverage error below.
                    score["cv"].append(float(local.std() / max(local.mean(), 1.e-6)))
                wall_tiles = tile_sum(wall & (image < 220), 8) > 0
                score["intersect"] += int((wall_tiles & gt_wall).sum())
                score["union"] += int((wall_tiles | gt_wall).sum())
                score.setdefault("ink", []).append(float((image[roof] < 220).mean()) if roof.any() else 0.)
                score.setdefault("deep_ink", []).append(float((image[roof] < 150).mean()) if roof.any() else 0.)
            samples = map_coordinates(gaussian_filter(image, 1.5), [uv[:, 1], uv[:, 0]], order=1)
            if i:
                joint = previous_visible & visible
                score["motion"] += float(np.abs(samples[joint] - previous[name][joint]).sum())
                score["pairs"] += int(joint.sum())
            previous[name] = samples
            if i == 80:
                score["loop"] = float(np.abs(image[first_mask] - clip[0][first_mask]).mean())
        previous_visible = visible
    results = {}
    for name, score in values.items():
        if not score["cv"] or not score["pairs"] or not score["union"]:
            raise ValueError(f"Insufficient visible roof/wall support in {building}")
        results[name] = dict(cv=float(np.mean(score["cv"])), motion=score["motion"] / score["pairs"],
                             wall_iou=score["intersect"] / score["union"], loop=score["loop"],
                             ink=float(np.mean(score["ink"])), deep_ink=float(np.mean(score["deep_ink"])))
    return results
