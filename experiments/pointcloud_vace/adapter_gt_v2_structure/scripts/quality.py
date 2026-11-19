"""Read-only paired video and geometric measurements for the eight held-out buildings."""

import av
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import data
from mesh_labels import RIDGE

LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)
VIEW_IDS = (0, 16, 32, 48, 64, 80)


def decode_prediction(path):
    result = []
    with av.open(str(path)) as container:
        codec = container.streams.video[0].codec_context
        codec.thread_count = 2
        for i, frame in enumerate(container.decode(video=0)):
            if (frame.is_corrupt or frame.pts is None or frame.pts * frame.time_base * 16 != i
                    or (frame.width, frame.height) != (data.WIDTH, data.HEIGHT)):
                raise ValueError(f"Invalid prediction frame {i} in {path}")
            result.append(frame.to_ndarray(format="rgb24"))
    if len(result) != 81:
        raise ValueError(f"Expected 81 frames in {path}")
    return np.stack(result)


def luminance(rgb):
    return rgb @ LUMA if rgb.ndim == 4 else rgb.astype(np.float32)


def appearance(rgb, gt_y):
    y = luminance(rgb)
    fg = gt_y < 220
    bg = ~ndimage.maximum_filter(fg, size=(1, 5, 5))
    chroma = rgb.max(-1).astype(np.float32) - rgb.min(-1) if rgb.ndim == 4 else np.zeros_like(y)
    dark = y < 150
    components, spacing, large = [], [], []
    for frame in VIEW_IDS:
        labels, count = ndimage.label(dark[frame])
        areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        components.append(count)
        if count:
            large.append(float(areas[areas >= 12].sum() / areas.sum()))
        if count > 1:
            xy = np.array(ndimage.center_of_mass(dark[frame], labels, range(1, count + 1)))
            spacing.extend(cKDTree(xy).query(xy, k=2)[0][:, 1])
    local = dark.reshape(81, 30, 16, 52, 16).mean((2, 4))
    support = fg.reshape(81, 30, 16, 52, 16).any((2, 4))
    active = local[support]
    return dict(point_core_fraction=float(dark.mean()), local_cv=float(active.std() / max(active.mean(), 1.e-8)),
                color_error=float(chroma[fg].mean()), background_fill=float((y[bg] < 220).mean()),
                foreground_mae=float(np.abs(y[fg] - gt_y[fg]).mean()),
                point_components=float(np.mean(components)),
                component_spacing_px=float(np.median(spacing)) if spacing else None,
                large_dark_fraction=float(np.mean(large)) if large else None)


def geometry_scores(prediction, targets, scale):
    accepted = (prediction["edge_scores"][:, 1:].sum(-1) >= .5)
    pairs = prediction["pairs"][accepted]
    known = (prediction["corner_scores"][pairs] >= .5).all(1)
    pairs = pairs[known]
    kinds = prediction["edge_scores"][accepted][known, 1:].argmax(-1) + 1
    predicted = prediction["vertices"][pairs]
    exact = targets["vertices"][targets["edges"]]
    tolerance = max(.20, .02 * scale)

    def match(guess, truth):
        guess, truth = guess[:, None], truth[None]
        same = np.linalg.norm(guess - truth, axis=-1).max(axis=-1)
        flipped = np.linalg.norm(guess - truth[:, :, ::-1], axis=-1).max(axis=-1)
        costs = np.minimum(same, flipped)
        left, right = linear_sum_assignment(np.where(costs <= tolerance, costs, tolerance * (1 + min(costs.shape))))
        good = costs[left, right] <= tolerance
        errors = costs[left[good], right[good]]
        return len(errors), float(errors.mean()) if len(errors) else None

    matches, error = match(predicted, exact)
    r_predicted, r_truth = predicted[kinds == RIDGE], exact[targets["kinds"] == RIDGE]
    ridge_matches, ridge_error = match(r_predicted, r_truth)
    distances = cKDTree(targets["display_xyz"]).query(prediction["points"])[0]
    surface_coverage = cKDTree(prediction["points"]).query(targets["full_xyz"])[0]
    return dict(edge_predicted=len(predicted), edge_gt=len(exact), edge_matches=matches,
                edge_endpoint_error_world=error, ridge_predicted=len(r_predicted), ridge_gt=len(r_truth),
                ridge_matches=ridge_matches, ridge_endpoint_error_world=ridge_error,
                tolerance_world=tolerance, point_count=len(prediction["points"]),
                gt_point_count=len(targets["display_xyz"]), point_precision_distance_world=float(distances.mean()),
                surface_coverage_distance_world=float(surface_coverage.mean()))


def edge_image_support(clips, targets, observed, frames=VIEW_IDS):
    """Measure detectable projected line support. RGB does not provide true XYZ."""
    ridges = targets["vertices"][targets["edges"][targets["kinds"] == RIDGE]]
    output = {role: dict(eligible=0, detected=0, views=[], offsets=[], curvature=[],
                         visible_px=0., matched_px=0., interior_gaps_px=0., transitions=0,
                         changes=0, previous={})
              for role in clips}
    for frame in frames:
        responses = {role: ndimage.gaussian_filter((240. - y[frame]).clip(0) / 160, 2.)
                     for role, y in clips.items()}
        for ridge_id, endpoints in enumerate(ridges):
            uv, depth = data.project(endpoints, observed["centers"][frame], observed["rotations"][frame],
                                     observed["intrinsics"])
            axis = uv[1] - uv[0]
            span = np.linalg.norm(axis)
            if min(depth) <= 0 or span < 20:
                continue
            direction = axis / span
            normal = np.array([-direction[1], direction[0]])
            along = np.linspace(.14, .86, 12)
            samples = uv[0] + along[:, None] * axis
            offsets = np.arange(-10, 11)
            grid = samples[:, None] + offsets[None, :, None] * normal
            if ((grid < 1) | (grid >= [data.WIDTH - 1, data.HEIGHT - 1])).any():
                continue
            coords = grid[..., ::-1].transpose(2, 0, 1)
            profiles = {role: ndimage.map_coordinates(response, coords, order=1)
                        for role, response in responses.items()}
            # Require a signed density gradient at the GT segment itself.
            # Texture-free, occluded and foreshortened views then contribute
            # coverage counts rather than fabricated straightness scores.
            truth_grad = np.gradient(profiles["gt"], axis=1)
            sign = np.sign(truth_grad[:, 9:12].mean())
            if not sign:
                continue
            truth_grad *= sign
            truth_peak = np.argmax(truth_grad, axis=1)
            visible = (np.abs(truth_peak - 10) <= 3) & (truth_grad.max(1) >= .008)
            if int(visible.sum()) < 5:
                continue
            for role, profile in profiles.items():
                score = output[role]
                score["eligible"] += 1
                gradient = sign * np.gradient(profile, axis=1)
                peak = gradient.argmax(1)
                usable = visible & (gradient.max(1) >= .008) & (np.abs(peak - 10) <= 8)
                # Integrate the linearly sampled, GT-readable central span. Interior
                # gaps sit between two supported samples and exclude both endpoints.
                unit_length = span * (along[-1] - along[0]) / len(along)
                score["visible_px"] += unit_length * visible.sum()
                score["matched_px"] += unit_length * usable.sum()
                gaps = visible & ~usable & np.maximum.accumulate(usable) & np.maximum.accumulate(usable[::-1])[::-1]
                score["interior_gaps_px"] += unit_length * gaps.sum()
                detected = usable.sum() >= 5
                last = score["previous"].get(ridge_id)
                if last is not None and last[0] == frame - 1:
                    score["transitions"] += 1
                    score["changes"] += int(last[1] != detected)
                score["previous"][ridge_id] = (frame, detected)
                if int(usable.sum()) < 5:
                    continue
                score["detected"] += 1
                score["views"].append(frame)
                displacement = (peak - truth_peak)[usable]
                score["offsets"].append(float(np.median(np.abs(displacement))))
                # Perpendicular deviations from this prediction's own best
                # straight line. Compare displacement against GT separately.
                solution = np.polyfit(along[usable] * span, offsets[peak[usable]], 1)
                residual = offsets[peak[usable]] - np.polyval(solution, along[usable] * span)
                score["curvature"].append(float(np.quantile(np.abs(residual), .95)))
    return {role: dict(eligible=score["eligible"], detected=score["detected"], views=sorted(set(score["views"])),
                       readable_length_px_ratio=score["matched_px"] / score["visible_px"] if score["visible_px"] else None,
                       interior_gap_ratio=score["interior_gaps_px"] / score["visible_px"] if score["visible_px"] else None,
                       visibility_flips=score["changes"], adjacent_eligible=score["transitions"],
                       position_px=float(np.mean(score["offsets"])) if score["offsets"] else None,
                       straightness_p95_px=float(np.mean(score["curvature"])) if score["curvature"] else None)
            for role, score in output.items()}
