"""Predict one shared corner graph, surface patches and point set per building."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from structure_data import MAX_POINTS, SURFACE_QUERIES, VERTEX_QUERIES

POINTS_PER_SURFACE = MAX_POINTS // SURFACE_QUERIES
PROJECTED_EDGES = 128
FEATURES = 12


@dataclass
class Building:
    vertices: torch.Tensor
    vertex_logits: torch.Tensor
    pairs: torch.Tensor
    edge_logits: torch.Tensor
    edge_normals: torch.Tensor
    faces: torch.Tensor
    face_scores: torch.Tensor
    points: torch.Tensor
    point_scores: torch.Tensor
    count: torch.Tensor

    def point_weights(self, hard=False):
        ranks = torch.arange(1, len(self.points) + 1, device=self.points.device)
        if hard:
            return (ranks <= self.count.round()).float()
        # The narrow transition lets RGB/coverage losses update the count head.
        weights = ((self.count - ranks + 32) / 64).clamp(0, 1)
        return torch.where(ranks <= 64, 1., weights)


def _mlp(width, channels):
    return nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, channels))


class BuildingAdapter(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.observation = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.observation_norm = nn.LayerNorm(width)
        self.corners = nn.Parameter(torch.randn(VERTEX_QUERIES, width) / width**.5)
        self.corner_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.corner_output = _mlp(width, 4)
        self.edge_output = _mlp(width * 2, 11)
        self.corner_context = nn.Linear(5, width)
        self.surfaces = nn.Parameter(torch.randn(SURFACE_QUERIES, width) / width**.5)
        self.surface_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.surface_output = _mlp(width, 12)
        self.point_offsets = _mlp(width, POINTS_PER_SURFACE * 4)
        self.count_output = _mlp(width, 1)
        self.condition = nn.Sequential(nn.Linear(16 * FEATURES, width), nn.SiLU(),
                                       nn.Linear(width, 1536, bias=False))
        nn.init.normal_(self.condition[-1].weight, std=.001)
        self.register_buffer("pairs", torch.triu_indices(VERTEX_QUERIES, VERTEX_QUERIES, offset=1),
                             persistent=False)
        # A fixed low-discrepancy sequence starts points across a surface, not at its corners.
        ranks = torch.arange(POINTS_PER_SURFACE, dtype=torch.float32) + .5
        root = (ranks / POINTS_PER_SURFACE).sqrt()
        fraction = torch.frac(ranks * .61803398875)
        barycentric = torch.stack((1. - root, root * fraction, root * (1. - fraction)), dim=-1)
        self.register_buffer("barycentric", barycentric, persistent=False)
        with torch.no_grad():
            self.count_output[-1].bias.fill_(3.)
            self.edge_output[-1].bias[1:5].fill_(-2.)

    def forward(self, xyz):
        encoded = self.observation_norm(self.observation(xyz))[None]
        global_features = encoded.amax(dim=1)
        queries = self.corners[None] + global_features[:, None]
        corner_features = (queries + self.corner_attention(queries, encoded, encoded, need_weights=False)[0])[0]
        corner_output = self.corner_output(corner_features).float()
        vertices = corner_output[:, :3]
        vertex_logits = corner_output[:, 3]

        left, right = self.pairs
        a, b = corner_features[left], corner_features[right]
        edge_parameters = self.edge_output(torch.cat((a + b, (a - b).abs()), dim=-1)).float()
        edge_logits = edge_parameters[:, :5]
        edge_normals = F.normalize(edge_parameters[:, 5:].reshape(-1, 2, 3), dim=-1, eps=1.e-6)
        related = (1. - edge_logits.softmax(-1)[:, 0])
        degree = torch.zeros_like(vertex_logits)
        degree = degree.index_add(0, left, related).index_add(0, right, related)
        context = torch.cat((vertices, vertex_logits.sigmoid()[:, None],
                             (degree / (VERTEX_QUERIES - 1))[:, None]), dim=-1)
        frame = (corner_features + self.corner_context(context))[None]

        queries = self.surfaces[None] + global_features[:, None]
        patch = (queries + self.surface_attention(queries, frame, frame, need_weights=False)[0])[0]
        params = self.surface_output(patch).float()
        center = params[:, :3]
        # Two independently oriented axes define each local triangle.
        u, v = (.8 * params[:, 3:9].tanh()).chunk(2, dim=-1)
        faces = torch.stack((center - u - v, center + u, center + v), dim=1)
        deviations = self.point_offsets(patch).reshape(SURFACE_QUERIES, POINTS_PER_SURFACE, 4).float()
        barycentric = (self.barycentric.log() + .5 * deviations[..., :3].tanh()).softmax(-1)
        normal = F.normalize(torch.cross(u, v, dim=-1), dim=-1, eps=1.e-6)
        points = barycentric @ faces + .04 * deviations[..., 3:].tanh() * normal[:, None]
        face_scores = params[:, 9:12]
        point_scores = face_scores[None].expand(POINTS_PER_SURFACE, -1, -1).reshape(MAX_POINTS, 3)
        count = 64 + (MAX_POINTS - 64) * self.count_output(global_features).float().sigmoid().squeeze()
        return Building(vertices, vertex_logits, self.pairs, edge_logits, edge_normals, faces, face_scores,
                        points.transpose(0, 1).reshape(MAX_POINTS, 3), point_scores, count)

    def project(self, building, camera, hard=False):
        points = building.points
        edge_scores = building.edge_logits.softmax(-1)[:, 1:]
        _, chosen = edge_scores.sum(-1).topk(PROJECTED_EDGES, sorted=False)
        ends = building.vertices[building.pairs[:, chosen].T]
        t = torch.linspace(0., 1., 5, device=points.device).view(1, 5, 1)
        edge_points = torch.lerp(ends[:, :1], ends[:, 1:], t).reshape(-1, 3)
        weight = building.point_weights(hard)
        # Do not remove far points. The dataset's point sprites do not behave as
        # opaque roof triangles. Separate surface channels and depth moments
        # let the conditioned model distinguish overlapping projected surfaces.
        xy, depth = project_xyz(torch.cat((points, edge_points), dim=0), camera)
        point_depth, edge_depth = depth.split((len(points), len(edge_points)), dim=1)
        kind = building.point_scores.softmax(-1)
        point_channels = torch.cat((
            weight[None, :, None].expand(len(xy), -1, -1),
            point_depth[..., None] * weight[None, :, None],
            point_depth.square()[..., None] * weight[None, :, None],
            (kind * weight[:, None])[None].expand(len(xy), -1, -1),
            (point_depth[..., None] * (kind * weight[:, None])[None]),
            points.new_zeros(len(xy), len(points), 3),
        ), dim=-1)
        selected = edge_scores[chosen]
        # Wall corners share the eave channel. Ridge and valley remain separate.
        graph = torch.cat((selected[:, :2], selected[:, 2:].sum(-1, keepdim=True)), dim=-1)
        endpoint_weight = building.vertex_logits.sigmoid()[building.pairs[:, chosen]].prod(0)
        graph = (graph * endpoint_weight[:, None]).repeat_interleave(5, 0)
        edge_channels = torch.cat((points.new_zeros(len(xy), len(edge_points), 9),
                                   graph[None].expand(len(xy), -1, -1)), dim=-1)
        grid = splat(xy, depth, torch.cat((point_channels, edge_channels), dim=1),
                     60, 104, stride=8)
        mass = grid[..., :1]
        # Normalize depth moments by point mass to avoid making ranges depend
        # on apparent point count. The point-mass feature keeps density explicit.
        grid = torch.cat((torch.log1p(mass), grid[..., 1:3] / mass.clamp_min(1.),
                          torch.log1p(grid[..., 3:6]), grid[..., 6:9] / mass.clamp_min(1.),
                          torch.log1p(grid[..., 9:])), dim=-1)
        return self.condition(pack_tokens(grid))[None]


def project_xyz(xyz, camera):
    with torch.autocast(xyz.device.type, enabled=False):
        relative = xyz.float()[None] - camera["center"][:, None]
        view = torch.bmm(relative, camera["rotation"].transpose(1, 2))
        depth = view[..., 2]
        intrinsics = camera["intrinsics"]
        uv = view[..., :2] / depth[..., None].clamp_min(1.e-4)
        return uv * intrinsics.diagonal()[:2] + intrinsics[:2, 2], depth


def splat(xy, depth, features, height, width, stride):
    """Bilinear scattering with the same image-index pixel-center convention."""
    cell = (xy + .5) / stride - .5
    base = cell.floor().long()
    frac = cell - base
    corners = base.new_tensor(((0, 0), (1, 0), (0, 1), (1, 1)))
    indices = base[..., None, :] + corners
    x, y = indices.unbind(-1)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (depth[..., None] > 0)
    weights = torch.where(corners.bool(), frac[..., None, :], 1. - frac[..., None, :]).prod(-1)
    rows = torch.arange(len(xy), device=xy.device)[:, None, None] * (height * width)
    indices = rows + y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
    channels = features.shape[-1]
    values = features[..., None, :] * (weights * valid)[..., None]
    grid = features.new_zeros(len(xy) * height * width, channels)
    return grid.index_add(0, indices.reshape(-1), values.reshape(-1, channels)).reshape(
        len(xy), height, width, channels)


def pack_tokens(grid):
    """Use first-frame replication and the original 4n+1 temporal layout."""
    x = torch.cat((grid[:1].expand(4, -1, -1, -1), grid[1:]))
    frames, h, w, c = x.shape
    return x.reshape(frames // 4, 4, h // 2, 2, w // 2, 2, c).permute(
        0, 2, 4, 1, 3, 5, 6).reshape(-1, 16 * c)
