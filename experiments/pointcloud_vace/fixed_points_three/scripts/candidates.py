"""Fit observed surfaces and extend measured scan lines into a supplied coarse box.

No mesh, reference image, withheld point, case identity or D prediction is an
argument to this module. All positions/colors are fixed once in 3D.
"""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import gaussian_filter1d

SETTINGS = dict(seed=20261110, plane_distance_m=.075, plane_min_support=12,
                plane_local_margin_m=4., plane_maximum=12, ransac_hypotheses=3200,
                minimum_plane_span_m=.55, scan_angle_bins=180, scan_line_min_points=3,
                normal_neighbors=12, normal_planarity_ratio=.5,
                wall_max_exterior_fraction=.15, wall_boundary_tolerance_m=.5,
                boundary_line_hypotheses=4000, boundary_line_distance_m=.12,
                boundary_line_min_support=8, boundary_line_min_span_m=1.5,
                boundary_line_max_vertical_slope=.12,
                facade_envelope_support_m=.18, facade_envelope_min_edge_m=.8,
                scan_line_angle_tolerance_deg=12., scan_line_residual_m=.055,
                maximum_line_extension_m=1.5, minimum_spacing_m=.08,
                maximum_spacing_m=1.0, duplicate_spacing_fraction=.45)


def in_box(points, center, axes, half, margin=0.):
    return (np.abs((points - center) @ axes.T) <= half + margin + 1.e-8).all(-1)


def plane_basis(normal):
    if abs(normal[2]) < .25:
        u = np.cross([0., 0., 1.], normal)
    else:
        u = np.array([1., 0., 0.]) - normal * normal[0]
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(normal, u)])


def box_polygon(origin, basis, normal, center, axes, half):
    signs = np.array(list(product((-1, 1), repeat=3)))
    corners = center + (signs * half) @ axes
    distance = (corners - origin) @ normal
    intersections = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(signs[i] != signs[j]) != 1:
                continue
            if abs(distance[i]) < 1.e-9:
                intersections.append(corners[i])
            if distance[i] * distance[j] < 0:
                intersections.append(corners[i] + (corners[j] - corners[i]) * distance[i] / (distance[i] - distance[j]))
    if len(intersections) < 3:
        return None
    uv = (np.unique(np.round(intersections, 9), axis=0) - origin) @ basis.T
    try:
        hull = ConvexHull(uv)
    except Exception:
        return None
    return uv[hull.vertices]


def fit_surfaces(points, center, axes, half, settings=SETTINGS):
    rng = np.random.default_rng(settings["seed"])
    nearby = np.flatnonzero(in_box(points, center, axes, half, settings["plane_local_margin_m"]))
    if len(nearby) < settings["plane_min_support"]:
        return []
    planes = []
    # Normals from real neighboring P distinguish a roof scan line from a wall
    # line. Otherwise a vertical plane through two unrelated roof strips can
    # obtain more votes than a genuinely sparse facade.
    neighbors = cKDTree(points).query(points, k=min(settings["normal_neighbors"], len(points)))[1]
    patch = points[neighbors]
    patch = patch - patch.mean(1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(np.einsum("nki,nkj->nij", patch, patch))
    local_normals = eigenvectors[:, :, 0]
    planar = eigenvalues[:, 0] / np.maximum(eigenvalues[:, 1], 1.e-9) < settings["normal_planarity_ratio"]
    roof_evidence = planar & (abs(local_normals[:, 2]) > .5)
    for kind, eligible in (("roof", roof_evidence),):
        pool = nearby[eligible[nearby]]
        if len(pool) < settings["plane_min_support"]:
            continue
        p = points[pool]
        normals, origins = [], []
        for _ in range(settings["ransac_hypotheses"]):
            a, b, c = p[rng.choice(len(p), 3, replace=False)]
            n = np.cross(b - a, [0., 0., 1.]) if kind == "wall" else np.cross(b - a, c - a)
            length = np.linalg.norm(n)
            if length < .02:
                continue
            n /= length
            if kind == "roof" and abs(n[2]) < .5:
                continue
            normals.append(n)
            origins.append(a)
        if not normals:
            continue
        normals, origins = np.asarray(normals), np.asarray(origins)
        distances = np.abs(p @ normals.T - np.einsum("ij,ij->i", origins, normals))
        support = distances < settings["plane_distance_m"]
        if kind == "roof":
            support &= np.abs(local_normals[pool] @ normals.T) > .94
        else:
            sides = points @ normals.T - np.einsum("ij,ij->i", origins, normals)
            exterior = np.minimum(np.mean(sides > settings["wall_boundary_tolerance_m"], axis=0),
                                  np.mean(sides < -settings["wall_boundary_tolerance_m"], axis=0))
            support[:, exterior > settings["wall_max_exterior_fraction"]] = False
        weights = np.ones(len(p)) if kind == "roof" else np.where(roof_evidence[pool], .25, 3.)
        available = np.ones(len(p), bool)
        for _ in range(settings["plane_maximum"] * 3):
            counts = support[available].sum(0)
            scores = (support[available] * weights[available, None]).sum(0)
            minimum = settings["plane_min_support"] if kind == "roof" else 8
            scores[counts < minimum] = 0
            k = int(np.argmax(scores))
            if scores[k] <= 0:
                break
            normal = normals[k]
            mask = support[:, k] & available
            selected = p[mask]
            for _ in range(3):
                origin = selected.mean(0)
                if kind == "wall":
                    _, _, v = np.linalg.svd(selected[:, :2] - origin[:2], full_matrices=False)
                    refined = np.r_[v[-1], 0.]
                else:
                    _, _, v = np.linalg.svd(selected - origin, full_matrices=False)
                    refined = v[-1]
                normal = refined if refined @ normal >= 0 else -refined
                mask = (abs((p - origin) @ normal) < settings["plane_distance_m"]) & available
                if kind == "roof":
                    mask &= abs(local_normals[pool] @ normal) > .94
                selected = p[mask]
            available[mask] = False
            if len(selected) < minimum:
                continue
            basis = plane_basis(normal)
            uv = (selected - origin) @ basis.T
            spans = np.ptp(uv, axis=0)
            covariance = np.linalg.eigvalsh(np.cov(uv.T))
            line_only = covariance.min() < .025
            if kind == "roof" and (line_only or spans.min() < settings["minimum_plane_span_m"]):
                continue
            if kind == "roof" and np.median(selected[:, 2]) < np.quantile(points[:, 2], .4) - .3:
                continue  # Do not turn a low horizontal base/ground strip into a roof.
            if kind == "wall":
                side = (points - origin) @ normal
                exterior = min(np.mean(side > settings["wall_boundary_tolerance_m"]),
                               np.mean(side < -settings["wall_boundary_tolerance_m"]))
                if exterior > settings["wall_max_exterior_fraction"] or spans[0] < .8:
                    continue
            polygon = box_polygon(origin, basis, normal, center, axes, half)
            if polygon is None or (kind == "wall" and line_only):
                continue
            local_support = in_box(selected, center, axes, half, 1.5)
            if local_support.sum() < 6:
                continue
            if any(abs(normal @ v["normal"]) > .985 and abs((origin - v["origin"]) @ v["normal"]) < .15 for v in planes):
                continue
            planes.append(dict(origin=origin, normal=normal, basis=basis, point_ids=pool[mask],
                polygon=polygon, kind=kind, support=len(selected), local_support=int(local_support.sum()),
                spans=spans, residual_median=float(np.median(abs((selected - origin) @ normal)))))
    walls = envelope_surfaces(points, center, axes, half, settings)
    if not walls:
        walls = boundary_scan_surfaces(points, center, axes, half, settings)
    return planes + walls


def envelope_surfaces(points, center, axes, half, settings=SETTINGS):
    """Use surviving exterior wall/eave lines, rather than a missing-face box normal.

    The observed XY envelope can bridge a missing stretch when both exterior
    corners survive. Every edge endpoint is a real, non-withheld observation.
    No candidate is accepted outside the supplied coarse volume.
    """
    try:
        boundary = ConvexHull(points[:, :2]).vertices
    except Exception:
        return []
    surfaces = []
    for i, j in zip(boundary, np.roll(boundary, -1)):
        delta = points[j] - points[i]
        length = np.linalg.norm(delta[:2])
        if length < settings["facade_envelope_min_edge_m"]:
            continue
        direction = np.r_[delta[:2] / length, 0.]
        normal = np.cross(direction, [0., 0., 1.])
        origin = (points[i] + points[j]) / 2
        basis = plane_basis(normal)
        polygon = box_polygon(origin, basis, normal, center, axes, half)
        if polygon is None:
            continue
        ids = np.flatnonzero((abs((points-origin) @ normal) < settings["facade_envelope_support_m"]) &
                            in_box(points, center, axes, half, settings["plane_local_margin_m"]))
        selected = points[ids]
        if len(selected) < 8:
            continue
        local_support = in_box(selected, center, axes, half, 1.5)
        if local_support.sum() < 3:
            continue
        if any(abs(normal @ s["normal"]) > .985 and abs((origin-s["origin"]) @ s["normal"]) < .15 for s in surfaces):
            continue
        surfaces.append(dict(origin=origin, normal=normal, basis=basis, point_ids=ids,
            polygon=polygon, kind="wall", support=len(ids), local_support=int(local_support.sum()),
            spans=np.ptp((selected-origin) @ basis.T, axis=0),
            residual_median=float(np.median(abs((selected-origin) @ normal))),
            scan_direction=direction @ basis.T, boundary_line_point_ids=np.array([i,j]),
            evidence="surviving measured exterior XY boundary endpoints plus gravity",
            exterior_fraction=0.))
    return surfaces


def boundary_scan_surfaces(points, center, axes, half, settings=SETTINGS):
    """Recover sparse facade planes from actual long 3D boundary scan strips.

    A broad plane vote is unsafe for a facade with only a few horizontal rows:
    unrelated roof strips can accidentally vote for it. A tight 3D line fit
    fixes its position first; gravity and the measured line set the wall normal.
    """
    ids = np.flatnonzero(in_box(points, center, axes, half, settings["plane_local_margin_m"]))
    p = points[ids]
    if len(p) < settings["boundary_line_min_support"]:
        return []
    rng = np.random.default_rng(settings["seed"] + 1)
    seeds = np.flatnonzero(in_box(p, center, axes, half, 1.5))
    if len(seeds) < 2:
        return []
    hypotheses = []
    for _ in range(settings["boundary_line_hypotheses"]):
        a, b = p[rng.choice(seeds, 2, replace=False)]
        direction = b - a
        horizontal = np.linalg.norm(direction[:2])
        if horizontal < 1. or abs(direction[2]) / horizontal > settings["boundary_line_max_vertical_slope"]:
            continue
        direction /= np.linalg.norm(direction)
        delta = p - a
        residual = np.linalg.norm(np.cross(delta, direction), axis=1)
        mask = residual < settings["boundary_line_distance_m"]
        if mask.sum() < settings["boundary_line_min_support"]:
            continue
        hypotheses.append((int(mask.sum()), a, direction, mask))
    surfaces = []
    for _, a, direction, mask in sorted(hypotheses, key=lambda h: -h[0]):
        selected = p[mask]
        origin = selected.mean(0)
        _, _, vectors = np.linalg.svd(selected - origin, full_matrices=False)
        direction = vectors[0]
        if abs(direction[2]) / np.linalg.norm(direction[:2]) > settings["boundary_line_max_vertical_slope"]:
            continue
        mask = np.linalg.norm(np.cross(p - origin, direction), axis=1) < settings["boundary_line_distance_m"]
        selected = p[mask]
        if len(selected) < settings["boundary_line_min_support"] or np.ptp((selected-origin) @ direction) < settings["boundary_line_min_span_m"]:
            continue
        # A strip several metres below the opening cannot determine the phase
        # of a row through it. Require the measured 3D row itself to reach the
        # local opening neighborhood before accepting its vertical plane.
        q, dq = (origin-center) @ axes.T, direction @ axes.T
        lower, upper = -np.inf, np.inf
        for position, velocity, bound in zip(q, dq, half + .2):
            if abs(velocity) < 1.e-9:
                if abs(position) > bound:
                    lower, upper = 1., 0.
                    break
            else:
                t = sorted(((-bound-position)/velocity, (bound-position)/velocity))
                lower, upper = max(lower,t[0]), min(upper,t[1])
        if lower >= upper:
            continue
        normal = np.cross(direction, [0., 0., 1.])
        normal /= np.linalg.norm(normal)
        side = (points - origin) @ normal
        exterior = min(np.mean(side > settings["wall_boundary_tolerance_m"]),
                       np.mean(side < -settings["wall_boundary_tolerance_m"]))
        if exterior > settings["wall_max_exterior_fraction"]:
            continue
        if any(abs(normal @ s["normal"]) > .985 and abs((origin-s["origin"]) @ s["normal"]) < .15 for s in surfaces):
            continue
        basis = plane_basis(normal)
        polygon = box_polygon(origin, basis, normal, center, axes, half)
        if polygon is None:
            continue
        surface_mask = abs((p-origin) @ normal) < settings["plane_distance_m"]
        surface_points = p[surface_mask]
        local_support = in_box(surface_points, center, axes, half, 1.5)
        if local_support.sum() < 3:
            continue
        surfaces.append(dict(origin=origin, normal=normal, basis=basis, point_ids=ids[surface_mask],
            polygon=polygon, kind="wall", support=int(surface_mask.sum()), local_support=int(local_support.sum()),
            spans=np.ptp((surface_points-origin) @ basis.T, axis=0),
            residual_median=float(np.median(abs((surface_points-origin) @ normal))),
            scan_direction=direction @ basis.T, boundary_line_point_ids=ids[mask],
            evidence="measured horizontal 3D boundary scan line plus gravity",
            exterior_fraction=float(exterior)))
        if len(surfaces) >= 6:
            break
    return surfaces


def dominant_direction(uv, settings=SETTINGS):
    distances, ids = cKDTree(uv).query(uv, k=min(7, len(uv)))
    vectors = uv[ids[:, 1:]] - uv[:, None]
    length = np.linalg.norm(vectors, axis=-1)
    nearest = distances[:, 1]
    maximum = max(.3, float(np.quantile(nearest[nearest > .02], .80)) * 3) if (nearest > .02).any() else .5
    valid = (length > settings["minimum_spacing_m"] * .5) & (length < maximum)
    angles = np.mod(np.arctan2(vectors[..., 1], vectors[..., 0]), np.pi)
    bins = settings["scan_angle_bins"]
    histogram, _ = np.histogram(angles[valid], bins=bins, range=(0, np.pi), weights=1 / np.maximum(length[valid], .05))
    smooth = gaussian_filter1d(histogram, 1.5, mode="wrap")
    peak = int(np.argmax(smooth))
    angle = (peak + .5) * np.pi / bins
    return np.array([np.cos(angle), np.sin(angle)]), dict(angle_degrees=float(np.rad2deg(angle)),
        peak_mass_fraction=float(smooth[peak] / max(smooth.sum(), 1.e-9)), nearest_spacing=float(np.median(nearest)))


def scan_lines(uv, direction, settings=SETTINGS):
    perpendicular = np.array([-direction[1], direction[0]])
    coordinate = uv @ perpendicular
    tolerance = settings["scan_line_residual_m"]
    remaining = np.ones(len(uv), bool)
    lines = []
    for _ in range(100):
        current = np.flatnonzero(remaining)
        if len(current) < settings["scan_line_min_points"]:
            break
        b = coordinate[current]
        counts = (np.abs(b[:, None] - b[None]) < tolerance).sum(1)
        seed = current[int(np.argmax(counts))]
        ids = current[np.abs(b - coordinate[seed]) < tolerance]
        if len(ids) < settings["scan_line_min_points"]:
            break
        center = uv[ids].mean(0)
        _, _, vectors = np.linalg.svd(uv[ids] - center, full_matrices=False)
        d = vectors[0]
        if d @ direction < 0:
            d = -d
        if abs(d @ direction) < np.cos(np.deg2rad(settings["scan_line_angle_tolerance_deg"])):
            remaining[ids] = False
            continue
        n = np.array([-d[1], d[0]])
        ids = current[abs((uv[current] - center) @ n) < tolerance]
        s = (uv[ids] - center) @ d
        if len(ids) >= settings["scan_line_min_points"] and np.ptp(s) >= .45:
            lines.append(dict(origin=center, direction=d, ids=ids, parameters=s, normal=n))
        remaining[ids] = False
    return lines


def line_interval(origin, direction, polygon):
    hull = ConvexHull(polygon)
    a = hull.equations[:, :2] @ direction
    b = hull.equations[:, :2] @ origin + hull.equations[:, 2]
    lower, upper = -np.inf, np.inf
    for ai, bi in zip(a, b):
        if abs(ai) < 1.e-10:
            if bi > 1.e-7:
                return None
        elif ai > 0:
            upper = min(upper, -bi / ai)
        else:
            lower = max(lower, -bi / ai)
    return (lower, upper) if lower < upper else None


def generate(points, colors, center, axes, half, settings=SETTINGS):
    planes = fit_surfaces(points, center, axes, half, settings)
    proposals, proposal_colors, metadata, surface_audit = [], [], [], []
    tree3d = cKDTree(points)
    for plane_id, plane in enumerate(planes):
        ids = plane["point_ids"]
        uv = (points[ids] - plane["origin"]) @ plane["basis"].T
        if "scan_direction" in plane:
            direction = plane["scan_direction"] / np.linalg.norm(plane["scan_direction"])
            direction_audit = dict(source=plane["evidence"],
                angle_degrees=float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
                measured_line_point_ids=plane["boundary_line_point_ids"].tolist(),
                exterior_fraction=plane["exterior_fraction"])
        else:
            direction, direction_audit = dominant_direction(uv, settings)
        lines = scan_lines(uv, direction, settings)
        # Local measured line spacings define density and phase, not the hidden count.
        intervals = []
        for line in lines:
            delta = np.diff(np.sort(line["parameters"]))
            intervals.extend(delta[(delta > settings["minimum_spacing_m"]) & (delta < settings["maximum_spacing_m"])].tolist())
        if not intervals:
            surface_audit.append(dict(plane_id=plane_id, kind=plane["kind"], support=plane["support"],
                reason="no_reliable_measured_scan_spacing", proposed=0))
            continue
        base_spacing = float(np.median(intervals))
        try:
            hull = ConvexHull(uv)
        except Exception:
            hull = None  # A measured boundary row may itself be nearly collinear.
        plane_count = 0
        for line_id, line in enumerate(lines):
            interval = line_interval(line["origin"], line["direction"], plane["polygon"])
            if interval is None:
                continue
            lo, hi = interval
            ordered = np.sort(line["parameters"])
            delta = np.diff(ordered)
            measured = delta[(delta > settings["minimum_spacing_m"]) & (delta < min(settings["maximum_spacing_m"], base_spacing * 2.2))]
            spacing = float(np.median(measured)) if len(measured) >= 2 else base_spacing
            spacing = max(settings["minimum_spacing_m"], min(settings["maximum_spacing_m"], spacing))
            before, after = ordered[ordered < lo], ordered[ordered > hi]
            bilateral = len(before) > 0 and len(after) > 0
            # Interpolate phase across a measured scan-line gap whenever both
            # boundaries survive. Otherwise allow only a short, observed-hull extension.
            if bilateral:
                a, b = before[-1], after[0]
                divisions = max(1, int(round((b - a) / spacing)))
                candidates = np.linspace(a, b, divisions + 1)[1:-1]
                candidates = candidates[(candidates >= lo) & (candidates <= hi)]
                confidence = .95
            else:
                phase = ordered[np.argmin(abs(ordered - (lo + hi) / 2))]
                candidates = phase + np.arange(np.floor((lo - phase) / spacing), np.ceil((hi - phase) / spacing) + 1) * spacing
                candidates = candidates[(candidates >= lo) & (candidates <= hi)]
                confidence = .60
            for s in candidates:
                xy = line["origin"] + s * line["direction"]
                outside_hull = hull is not None and bool(np.any(hull.equations[:, :2] @ xy + hull.equations[:, 2] > .05))
                extension = max(float(ordered[0] - s), float(s - ordered[-1]), 0.)
                xyz = plane["origin"] + xy @ plane["basis"]
                if not in_box(xyz[None], center, axes, half)[0]:
                    continue
                # Color from the same fitted surface; no generic black fill.
                neighbor = cKDTree(uv).query(xy, k=min(4, len(uv)))[1]
                color = np.rint(np.median(colors[ids[np.atleast_1d(neighbor)]], axis=0)).astype(np.uint8)
                proposals.append(xyz)
                proposal_colors.append(color)
                metadata.append(dict(plane_id=plane_id, line_id=line_id, spacing=spacing,
                                     confidence=confidence, bilateral=bilateral, kind=plane["kind"],
                                     outside_observed_hull=outside_hull, extension_m=extension))
                plane_count += 1
        surface_audit.append(dict(plane_id=plane_id, kind=plane["kind"], origin=plane["origin"].tolist(),
            normal=plane["normal"].tolist(), basis=plane["basis"].tolist(), support=plane["support"],
            local_support=plane["local_support"], spans=plane["spans"].tolist(), residual_median=plane["residual_median"],
            direction=direction_audit, fitted_lines=len(lines), base_spacing=base_spacing, proposed=plane_count,
            polygon_uv=plane["polygon"].tolist(), lines=[dict(origin=l["origin"].tolist(), direction=l["direction"].tolist(),
                point_ids=ids[l["ids"]].tolist(), measured_parameters=l["parameters"].tolist()) for l in lines]))
    xyz = np.asarray(proposals, dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(proposal_colors, dtype=np.uint8).reshape(-1, 3)
    # Distance-based deduplication preserves all actual P points, resolves surface
    # overlaps at eaves, and avoids adding a denser lattice than the measured strips.
    kept, rejected = [], []
    for index in sorted(range(len(xyz)), key=lambda i: -metadata[i]["confidence"]):
        threshold = settings["duplicate_spacing_fraction"] * metadata[index]["spacing"]
        reason = None
        if metadata[index]["outside_observed_hull"]:
            reason = "outside_observed_surface_hull"
        elif metadata[index]["extension_m"] > settings["maximum_line_extension_m"]:
            reason = "unilateral_extension_too_far"
        elif tree3d.query(xyz[index])[0] < threshold:
            reason = "near_existing_observation"
        elif kept and np.linalg.norm(xyz[kept] - xyz[index], axis=1).min() < threshold:
            reason = "candidate_duplicate_or_surface_overlap"
        if reason:
            rejected.append(dict(index=index, reason=reason))
        else:
            kept.append(index)
    return xyz, rgb, np.array(kept, dtype=int), dict(settings=settings, surfaces=surface_audit,
        proposal_metadata=metadata, generated_candidates=len(xyz), retained_candidates=len(kept),
        rejected=rejected, algorithm_inputs="observed P/RGB and coarse box only; no case identity, mesh or target",
        abstained=not len(kept))
