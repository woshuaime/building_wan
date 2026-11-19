"""Supervise both the predicted geometry and the actual decoded VACE RGB."""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

import data
from mesh_labels import RIDGE, VALLEY
from structure_data import VERTEX_QUERIES

CROP = 224
RGB_EVERY = 8
LUMA = (.2126, .7152, .0722)


def _balanced(values, mask):
    mask = mask.to(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.)


def _tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(array, dtype=dtype, device=device)


def camera_tensor(camera, device):
    return dict(center=_tensor((camera["centers"] - camera["midpoint"]) / camera["scale"], device),
                rotation=_tensor(camera["rotations"], device),
                intrinsics=_tensor(camera["intrinsics"], device))


def observed_tensor(observed, device):
    return _tensor((observed["xyz"] - observed["midpoint"]) / observed["scale"], device)


def target_tensor(raw, device):
    targets = raw["targets"]
    camera = raw["camera"]
    vertices, edges, kinds, normals, full, points, point_normals = (
        targets[name] for name in ("vertices", "edges", "kinds", "supports", "full_xyz",
                                   "display_xyz", "display_normals"))
    if len(vertices) > VERTEX_QUERIES:
        span = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1)
        # Retain long structural features first. Full point supervision and
        # evaluation still use the entire building, including short edges.
        order = np.argsort(-span * np.where(kinds == RIDGE, 1.5, 1.), kind="stable")
        seen = {}
        for vertex in edges[order].ravel():
            if vertex not in seen and len(seen) < VERTEX_QUERIES:
                seen[vertex] = len(seen)
        keep = np.array(list(seen))
        lookup = np.full(len(vertices), -1)
        lookup[keep] = np.arange(len(keep))
        edges = lookup[edges]
        valid = (edges >= 0).all(1)
        vertices, edges, kinds, normals = vertices[keep], edges[valid], kinds[valid], normals[valid]
    normalized = lambda xyz: _tensor((xyz.astype(np.float64) - camera["midpoint"]) / camera["scale"], device)
    selected = np.linspace(0, len(points) - 1, min(1024, len(points)), dtype=int)
    full_offset = raw.get("sample_index", 0) % 16
    nz = point_normals[selected, 2]
    point_kind = np.where(nz > .25, 0, np.where(np.abs(nz) < .25, 1, 2))
    return dict(vertices=normalized(vertices), edges=_tensor(edges, device, torch.long),
                kinds=_tensor(kinds, device, torch.long), edge_normals=_tensor(normals, device),
                surface=normalized(full[full_offset::16]), display=normalized(points[selected]),
                point_kind=_tensor(point_kind, device, torch.long), count=len(points))


def framework_loss(building, target):
    device = building.points.device
    vertices = target["vertices"]
    presence = torch.zeros_like(building.vertex_logits)
    corner_cost = torch.cdist(building.vertices.detach(), vertices)
    rows, columns = linear_sum_assignment(corner_cost.cpu().numpy())
    detected, expected = (_tensor(value, device, torch.long) for value in (rows, columns))
    presence[detected] = 1
    confidence = F.binary_cross_entropy_with_logits(building.vertex_logits, presence, reduction="none")
    corner_loss = _balanced(confidence, presence) + _balanced(confidence, 1 - presence)
    labels = torch.zeros(len(building.edge_logits), dtype=torch.long, device=device)
    has_edges = len(target["edges"]) > 0
    if has_edges:
        corner_loss = corner_loss + 4 * F.smooth_l1_loss(
            building.vertices[detected], vertices[expected], beta=.04)
        assignment = torch.empty(len(vertices), dtype=torch.long, device=device)
        assignment[expected] = detected
        left, right = assignment[target["edges"]].sort(dim=-1).values.unbind(-1)
        edge_index = left * (2 * len(presence) - left - 1) // 2 + right - left - 1
        labels[edge_index] = target["kinds"]
        targets = target["edge_normals"]
        normals = building.edge_normals[edge_index]
        aligned = (normals * targets).sum(-1).mean(-1)
        reversed_ = (normals * targets.flip(1)).sum(-1).mean(-1)
        # Correct the segment length and angle through its common endpoints.
        vector = building.vertices[right] - building.vertices[left]
        truth = vertices[target["edges"][:, 1]] - vertices[target["edges"][:, 0]]
        direction = F.cosine_similarity(vector, truth, dim=-1).abs()
        length = F.smooth_l1_loss(vector.norm(dim=-1), truth.norm(dim=-1), beta=.04)
        planes = (1 - torch.maximum(aligned, reversed_)).mean()
        edge_loss = length + .2 * (1 - direction).mean() + .2 * planes

        # Both planes incident to a positive edge need a nearby predicted patch.
        midpoint = (building.vertices[left] + building.vertices[right]) / 2
        patches = building.faces.mean(1)
        relative = patches[None] - midpoint[:, None]
        sides = (relative[:, :, None] * targets[:, None]).sum(-1).abs()
        patch_normal = F.normalize(torch.cross(building.faces[:, 1] - building.faces[:, 0],
                                              building.faces[:, 2] - building.faces[:, 0], dim=-1),
                                   dim=-1, eps=1.e-6)
        alignment = 1 - torch.einsum("esi,pi->eps", targets, patch_normal).abs()
        edge_loss = edge_loss + .25 * (sides + .15 * alignment + .1 * relative.norm(dim=-1)[..., None]).amin(1).mean()
    else:
        edge_loss = building.vertices.sum() * 0
    classification = F.cross_entropy(building.edge_logits, labels, reduction="none")
    interior = (labels == RIDGE) | (labels == VALLEY)
    return (corner_loss + _balanced(classification, (labels > 0).float() + interior.float())
            + _balanced(classification, labels == 0) + edge_loss)


def _nearest(source, target, weights=None):
    with torch.no_grad():
        distance = torch.cdist(source.detach(), target.detach())
        if weights is not None:
            distance = distance.masked_fill(weights[None] == 0, float("inf"))
        return distance.argmin(dim=-1)


def point_loss(building, target):
    points = building.points
    weight = building.point_weights()
    display = target["display"]
    surface = target["surface"]
    predicted_to_display = _nearest(points, display)
    display_to_predicted = _nearest(display, points, weight)
    surface_to_predicted = _nearest(surface, points, weight)
    predicted_to_surface = _nearest(points, surface)
    fit = F.smooth_l1_loss(points, display[predicted_to_display], beta=.025, reduction="none").mean(-1)
    mesh_fit = F.smooth_l1_loss(points, surface[predicted_to_surface], beta=.025, reduction="none").mean(-1)
    coverage = F.smooth_l1_loss(points[display_to_predicted], display, beta=.025)
    full_coverage = F.smooth_l1_loss(points[surface_to_predicted], surface, beta=.025)
    surface_class = F.cross_entropy(building.point_scores, target["point_kind"][predicted_to_display],
                                    reduction="none")
    budget = F.smooth_l1_loss(building.count.log(), points.new_tensor(target["count"]).log(), beta=.08)

    # Compare nearest-neighbor spacing after equal-count thinning, so the
    # reference's subsampling does not reward a blur or dense cluster.
    pred_subset = torch.linspace(0, len(points) - 1, len(display), device=points.device)
    pred_subset = (pred_subset * ((building.count.detach() - 1) / (len(points) - 1))).round().long()
    pred = points[pred_subset]
    with torch.no_grad():
        pred_pair = torch.cdist(pred.detach(), pred.detach()).fill_diagonal_(float("inf")).argmin(-1)
        real_spacing = torch.cdist(display, display).fill_diagonal_(float("inf")).amin(-1)
    spacing = (pred - pred[pred_pair]).norm(dim=-1)
    percentile = torch.linspace(.1, .9, 9, device=points.device)
    distribution = F.l1_loss(torch.quantile((spacing + 1.e-3).log(), percentile),
                             torch.quantile((real_spacing + 1.e-3).log(), percentile))
    return (8 * (_balanced(fit + .5 * mesh_fit, weight) + coverage + .5 * full_coverage)
            + .1 * _balanced(surface_class, weight) + .3 * budget + .05 * distribution)


def choose_rgb_window(raw, sample):
    """Select one static crop for the entire causal decoding window."""
    frames, targets, camera = raw["target"], raw["targets"], raw["camera"]
    middle = len(frames) // 2
    selected = np.linspace(0, len(targets["display_xyz"]) - 1, min(512, len(targets["display_xyz"])), dtype=int)
    display = targets["display_xyz"][selected]
    vertices = targets["vertices"]
    ridges = vertices[targets["edges"][targets["kinds"] == RIDGE]]
    tracks = display
    if len(ridges):
        delta = ridges[:, 1] - ridges[:, 0]
        samples = ridges[:, :1] + np.linspace(.15, .85, 5)[None, :, None] * delta[:, None]
        tracks = np.concatenate((tracks, samples.reshape(-1, 3)))
    projected = np.stack([data.project(tracks, center, rotation, camera["intrinsics"])[0]
                          for center, rotation in zip(camera["centers"], camera["rotations"])])
    # Use both roof detail and wall/background crops on a fixed rotation.
    focus = display if sample % 4 != 0 or not len(ridges) else ridges.mean(1)
    uv, _ = data.project(focus, camera["centers"][middle], camera["rotations"][middle],
                         camera["intrinsics"])
    valid = ((uv >= 0) & (uv < [data.WIDTH, data.HEIGHT])).all(-1)
    choices = uv[valid]
    if not len(choices):
        point = np.array([data.WIDTH / 2, data.HEIGHT / 2])
    elif sample % 4 in (1, 3):
        point = choices[(sample * 131) % len(choices)]
    else:
        point = np.median(choices, axis=0)
    left, top = (np.clip(np.floor((point - CROP / 2) / 8), 0,
                         [(data.WIDTH - CROP) // 8, (data.HEIGHT - CROP) // 8]).astype(int) * 8)
    return int(top), int(left), projected - np.array([left, top])


def luma(rgb):
    return (rgb * rgb.new_tensor(LUMA).view(1, 3, 1, 1)).sum(1, keepdim=True)


def rgb_losses(rgb, gt, tracks):
    """Video and track losses on unclipped decoded RGB, scaled to [0,1]."""
    rgb = rgb.float().permute(0, 2, 1, 3, 4)[0]
    gt = gt[:, None].float()
    y = luma(rgb)
    chroma = (rgb - y).abs().mean(dim=1, keepdim=True)
    support = F.max_pool2d((gt < 239 / 255).float(), 5, stride=1, padding=2)
    center = torch.ones_like(gt)
    # Crops are causal and spatially bounded. Leave the artificial crop edges
    # out of the supervision, including the pooled density statistics.
    center[..., :12, :] = center[..., -12:, :] = 0
    center[..., :, :12] = center[..., :, -12:] = 0
    error = (y - gt).abs()
    pixel = _balanced(error, support * center) + .5 * _balanced(error, (1 - support) * center)
    color = (_balanced(chroma, support * center) + .5 * _balanced(chroma, (1 - support) * center))
    density = y.new_zeros(())
    for threshold in (150 / 255, 220 / 255):
        probability = torch.sigmoid(32 * (threshold - y))
        exact = torch.sigmoid(32 * (threshold - gt))
        for size in (8, 16, 32):
            pool = lambda x: F.avg_pool2d(x, size, stride=size)
            discrepancy = (pool(probability * center) - pool(exact * center)).abs()
            weight = 1 + 9 * pool(support * center)
            density = density + (discrepancy * weight).mean() / 6

    resolution = tracks.new_tensor((gt.shape[-1], gt.shape[-2]))
    grid = ((tracks + .5) * 2 / resolution - 1)[..., None, :]
    actual = F.avg_pool2d(gt, 3, stride=1, padding=1)
    predicted = F.avg_pool2d(y, 3, stride=1, padding=1)
    truth = F.grid_sample(actual, grid, align_corners=False)[:, 0, :, 0]
    estimate = F.grid_sample(predicted, grid, align_corners=False)[:, 0, :, 0]
    valid = ((tracks >= 12) & (tracks < resolution - 12)).all(-1) & (truth < .934)
    pairs = ((0, 1), (8, 9), (15, 16), (0, 8), (8, 16), (0, 16))
    motion = []
    for first, second in pairs:
        expected_delta = truth[second] - truth[first]
        predicted_delta = estimate[second] - estimate[first]
        motion.append(_balanced((expected_delta - predicted_delta).abs(), valid[first] & valid[second]))
    return dict(pixel=pixel, chroma=color, density=density, track=torch.stack(motion).mean())


def decode_pixels(vae, clean, use_checkpoint):
    """Use the official differentiable decoder with fresh causal feature state."""
    from torch.utils.checkpoint import checkpoint

    def forward(value):
        try:
            return (vae.model.decode(value, vae.scale).float() + 1.) / 2.
        finally:
            vae.model.clear_cache()

    if use_checkpoint:
        return checkpoint(forward, clean, use_reentrant=False, preserve_rng_state=False)
    return forward(clean)
