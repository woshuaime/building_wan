"""Image-only associations on five-frame clips, evaluated against fixed camera projections."""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import correlate
from scipy.spatial import cKDTree

STARTS = (14, 21, 38, 60)
PATCH = 4


def fixed_pairs(uv, depth, middle_depth):
    """Select pairs using observed XYZ/cameras alone; no prediction-dependent denominator."""
    valid = ((depth > 0) & (uv > [28, 28]).all(-1) & (uv < [804, 452]).all(-1)).all(0)
    for frame in uv:
        valid &= cKDTree(frame).query(frame, k=2)[0][:, 1] > 3
    near = np.flatnonzero(valid & (depth < middle_depth[:, None]).all(0))
    far = np.flatnonzero(valid & (depth >= middle_depth[:, None]).all(0))
    if not len(near) or not len(far):
        return np.empty((0, 2), dtype=int)
    motion = uv[-1] - uv[0]
    candidates = []
    for n, neighbors in zip(near, cKDTree(uv[0, far]).query_ball_point(uv[0, near], 64)):
        for f in far[neighbors]:
            distance = np.linalg.norm(uv[0, n] - uv[0, f])
            if distance >= 5 and np.linalg.norm(motion[n] - motion[f]) >= 4:
                candidates.append((distance, n, f))
    used, pairs = set(), []
    for _, n, f in sorted(candidates):
        if n in used or f in used:
            continue
        pairs.append((n, f))
        used.update((n, f))
        if len(pairs) == 24:
            break
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def patch(image, xy, radius):
    x, y = np.rint(xy).astype(int)
    if min(x, y) < radius or x + radius >= image.shape[1] or y + radius >= image.shape[0]:
        return None
    return image[y-radius:y+radius+1, x-radius:x+radius+1].astype(np.float32, copy=False)


def candidates(image, template, center, radius, limit):
    """Distinct local matches. A single sparse dot can be ambiguous on one frame."""
    window = patch(image, center, radius + PATCH)
    if window is None:
        return []
    ref = template - template.mean()
    energy = np.sum(ref * ref)
    if energy < template.size * 4:
        return []
    size = template.shape[0]
    middle = np.s_[PATCH:-PATCH, PATCH:-PATCH]
    mean = uniform_filter(window, size)[middle]
    variance = np.maximum(uniform_filter(window * window, size)[middle] - mean * mean, 0)
    cov = correlate(window, ref, 'valid', method='fft')
    scores = cov / np.sqrt(np.maximum(variance, 1.) * template.size * energy)
    scores[(variance < 4) | (mean - window[middle] < 4)] = -2.
    result = []
    for _ in range(limit):
        y, x = np.unravel_index(scores.argmax(), scores.shape)
        value = float(scores[y, x])
        if value < .40:
            break
        result.append((np.rint(center).astype(int) + [x-radius, y-radius], value))
        scores[max(0,y-2):y+3, max(0,x-2):x+3] = -2.
    return result


def _unique(positions):
    active = [i for i, xy in enumerate(positions) if xy is not None]
    if len(active) > 1:
        coords = np.asarray([positions[i] for i in active])
        for a, b in cKDTree(coords).query_pairs(2.):
            positions[active[a]] = positions[active[b]] = None


def associate(clip, origins):
    positions = []
    for point in origins:
        small = patch(clip[0], point, 2)
        if small is None or small.max() - small.min() < 6:
            positions.append(None)
            continue
        y, x = np.unravel_index(small.argmin(), small.shape)
        xy = np.rint(point).astype(int) + [x-2, y-2]
        positions.append(xy if np.linalg.norm(xy - point) <= 2.5 else None)
    _unique(positions)
    first = positions.copy()
    # Resolve ambiguous first steps over the five-frame image sequence. A true
    # projected point under this orbit has slowly changing velocity. No later
    # GT positions enter the search or the choice between candidate trajectories.
    tracks = []
    for xy in first:
        paths = [] if xy is None else [(0., [xy])]
        for t in range(1, len(clip)):
            proposals = []
            for score, path in paths:
                last = path[-1]
                expected = last if t == 1 else 2*last - path[-2]
                descriptor = patch(clip[t-1], last, PATCH)
                for pos, corr in candidates(clip[t], descriptor, expected,
                                             18 if t == 1 else 3, 12 if t == 1 else 2):
                    bend = 0 if t == 1 else np.linalg.norm(pos-expected)
                    if bend <= 3:
                        proposals.append((score + corr - .10*bend, [*path, pos]))
            paths = sorted(proposals, key=lambda item: -item[0])[:12]
        unique = paths and (len(paths) == 1 or paths[0][0] - paths[1][0] >= .20)
        tracks.append(paths[0][1] if unique else None)
    rejected = set()
    for t in range(len(clip)):
        valid = [i for i, path in enumerate(tracks) if path is not None]
        if len(valid) > 1:
            for a, b in cKDTree([tracks[i][t] for i in valid]).query_pairs(2.):
                rejected.update((valid[a], valid[b]))
    movement = np.full((len(first), 2), np.nan)
    for i, path in enumerate(tracks):
        if path is not None and i not in rejected:
            movement[i] = path[-1] - path[0]
    return np.array([p is not None for p in first], dtype=bool), movement


def measure(clip, uv, pairs):
    ids = pairs.ravel()
    seen, measured = associate(clip, uv[0, ids])
    truth = (uv[-1, ids] - uv[0, ids]).reshape(-1, 2, 2)
    error = measured.reshape(-1, 2, 2) - truth
    single = np.linalg.norm(error, axis=-1)
    relative = np.linalg.norm(error[:, 0] - error[:, 1], axis=-1)
    return dict(total=len(pairs), seen=seen.reshape(-1, 2).sum(0).tolist(),
                near=single[:, 0][np.isfinite(single[:, 0])].tolist(),
                far=single[:, 1][np.isfinite(single[:, 1])].tolist(),
                diff=relative[np.isfinite(relative)].tolist(),
                passed=int(((relative <= 2) & (single <= 3).all(1)).sum()))


def summary(windows):
    count = sum(w['total'] for w in windows)
    result = dict(total=count, passed=sum(w['passed'] for w in windows),
                  seen=np.sum([w['seen'] for w in windows], axis=0).tolist())
    for key in ('near', 'far', 'diff'):
        errors = [e for w in windows for e in w[key]]
        result[key] = dict(tracked=len(errors), lost=count-len(errors),
                           median_px=float(np.median(errors)) if errors else None)
    return result
