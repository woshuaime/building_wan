"""Structural edges of the original mesh, without its triangulation diagonals."""

from collections import defaultdict

import numpy as np

RIDGE, VALLEY, EAVE, WALL_CORNER = 1, 2, 3, 4
PLANAR_ANGLE = np.cos(np.deg2rad(3.0))
CREASE_ANGLE = np.cos(np.deg2rad(8.0))
STRAIGHT_ANGLE = np.cos(np.deg2rad(2.0))
PLANE_TOLERANCE = .02


def read_obj(path):
    points, normals, faces, face_normals = [], [], [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            kind, *values = fields
            if kind == "v":
                points.append(tuple(map(float, values[:3])))
            if kind == "vn":
                normals.append(tuple(map(float, values[:3])))
            if kind == "f":
                corners = [value.split("/") for value in values]
                indices = []
                for corner in corners:
                    index = int(corner[0])
                    if index == 0:
                        raise ValueError(f"Zero OBJ index in {path}")
                    indices.append(index - 1 if index > 0 else len(points) + index)
                for j in range(1, len(indices) - 1):
                    selector = (0, j, j + 1)
                    faces.append(tuple(indices[i] for i in selector))
                    face_normals.append(tuple(int(corners[i][2]) if len(corners[i]) > 2
                                              and corners[i][2] else 0 for i in selector))
    vertices = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or not len(triangles) or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError(f"Invalid OBJ geometry in {path}")
    reference = np.zeros((len(faces), 3), dtype=np.float64)
    if normals:
        normals = np.asarray(normals, dtype=np.float64)
        indices = np.asarray(face_normals)
        valid = ((indices != 0) & (indices <= len(normals)) & (indices >= -len(normals))).all(1)
        chosen = indices[valid]
        reference[valid] = normals[np.where(chosen > 0, chosen - 1, chosen)].mean(1)
    return vertices, triangles, reference


def _planes(vertices, triangles, normals, adjacency, offsets):
    """Cluster connected, nearly coplanar faces against one plane per patch."""
    neighbors = [[] for _ in triangles]
    for faces in adjacency.values():
        if len(faces) == 2:
            a, b = faces
            neighbors[a].append(b)
            neighbors[b].append(a)
    plane = np.full(len(triangles), -1)
    for seed in range(len(triangles)):
        if plane[seed] >= 0:
            continue
        plane[seed] = seed
        pending = [seed]
        while pending:
            for candidate in neighbors[pending.pop()]:
                if plane[candidate] >= 0 or normals[candidate] @ normals[seed] < PLANAR_ANGLE:
                    continue
                distance = vertices[triangles[candidate]] @ normals[seed] - offsets[seed]
                if np.abs(distance).max() <= PLANE_TOLERANCE:
                    plane[candidate] = seed
                    pending.append(candidate)
    return plane


def _merge_straight(vertices, segments):
    """Suppress only collinear degree-two nodes with the same incident patches."""
    incident = defaultdict(set)
    for a, b in segments:
        incident[a].add(b)
        incident[b].add(a)
    # Sorted processing makes connected labels independent of hash iteration order.
    for middle in sorted(incident):
        if len(incident[middle]) != 2:
            continue
        a, b = sorted(incident[middle])
        ka, kb, joined = (min(a, middle), max(a, middle)), (min(b, middle), max(b, middle)), (a, b)
        if joined in segments or segments[ka] != segments[kb]:
            continue
        u, v = vertices[a] - vertices[middle], vertices[middle] - vertices[b]
        if u @ v < STRAIGHT_ANGLE * np.linalg.norm(u) * np.linalg.norm(v):
            continue
        if np.linalg.norm(np.cross(u, v)) > PLANE_TOLERANCE * np.linalg.norm(u + v):
            continue
        segments[joined] = segments.pop(ka)
        del segments[kb]
        for end, other in ((a, b), (b, a)):
            incident[end].remove(middle)
            incident[end].add(other)
        incident[middle].clear()


def structural_edges(vertices, triangles, reference_normals=None):
    """Return shared endpoints, edge types and the normals of their two incident faces."""
    # Weld OBJ vertices at their original precision before finding shared sides.
    _, first, lookup = np.unique(np.rint(vertices * 1.e5).astype(np.int64), axis=0,
                                 return_index=True, return_inverse=True)
    vertices = vertices[first]
    triangles = lookup[triangles]
    surface = vertices[triangles]
    cross = np.cross(surface[:, 1] - surface[:, 0], surface[:, 2] - surface[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    _, unique = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    keep = np.sort(unique[lengths[unique] > 1.e-9])
    triangles, normals = triangles[keep], cross[keep] / lengths[keep, None]
    if reference_normals is not None:
        normals *= np.where((normals * reference_normals[keep]).sum(1) < 0, -1, 1)[:, None]
    adjacency = defaultdict(list)
    for i, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (a, c), (b, c)):
            adjacency[tuple(sorted((u, v)))].append(i)
    offsets = (normals * vertices[triangles[:, 0]]).sum(1)
    planes = _planes(vertices, triangles, normals, adjacency, offsets)
    segments = {}
    for (a, b), faces in adjacency.items():
        # Boundary holes and non-manifold sides have no known second surface.
        if len(faces) != 2:
            continue
        left, right = faces
        nl, nr = normals[left], normals[right]
        if planes[left] == planes[right] or nl @ nr > CREASE_ANGLE:
            continue
        roof_l, roof_r = nl[2] > .25, nr[2] > .25
        wall_l, wall_r = abs(nl[2]) < .25, abs(nr[2]) < .25
        if roof_l and roof_r:
            opposite = next(v for v in triangles[right] if v not in (a, b))
            kind = VALLEY if (vertices[opposite] - vertices[a]) @ nl > 0 else RIDGE
        elif (roof_l and wall_r) or (roof_r and wall_l):
            kind = EAVE
        elif wall_l and wall_r:
            kind = WALL_CORNER
        else:
            continue
        segments[a, b] = (kind, *sorted((int(planes[left]), int(planes[right]))))
    _merge_straight(vertices, segments)
    edges = sorted(segments)
    endpoints = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    used = np.unique(endpoints)
    endpoints = np.searchsorted(used, endpoints)
    kinds = np.array([segments[edge][0] for edge in edges], dtype=np.int64)
    supports = np.array([normals[list(segments[edge][1:])] for edge in edges]).reshape(-1, 2, 3)
    return dict(vertices=vertices[used], edges=endpoints, kinds=kinds, supports=supports,
                triangles=vertices[triangles], normals=normals)
