"""Eight-view, camera-aligned structure adapter from the uploaded specification."""
import torch
from torch import nn
from torch.nn import functional as F


def sample_map(maps, pixels):
    """Sample [B,V,C,H,W] at [B,V,A,2] image-index coordinates."""
    b, v, c, h, w = maps.shape
    grid = (pixels + .5) / pixels.new_tensor([w, h]) * 2 - 1
    sampled = F.grid_sample(maps.flatten(0, 1).float(), grid.flatten(0, 1).unsqueeze(2).float(),
                            mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(-1).transpose(1, 2).reshape(b, v, -1, c)


def projections(anchors, centers, rotations, K):
    relative = anchors[:, None] - centers[:, :, None]
    q = torch.einsum("bvij,bvaj->bvai", rotations.float(), relative.float())
    uv = q[..., :2] / q[..., 2:].clamp_min(1.e-8)
    uv = uv * torch.stack([K[:, 0, 0], K[:, 1, 1]], -1)[:, None, None]
    uv = uv + K[:, :2, 2][:, None, None]
    valid = (q[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < 832)
    valid &= (uv[..., 1] >= 0) & (uv[..., 1] < 480)
    return uv, q[..., 2], valid


class MultiViewStructureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(2, 16, 3, 2, 1), nn.SiLU(),
                                 nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU())
        self.view_score = nn.Sequential(nn.Linear(68, 32), nn.SiLU(), nn.Linear(32, 1))
        self.point_mlp = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64))
        self.query_mlp = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 64))
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.structure_proj = nn.Linear(64, 1536)
        nn.init.zeros_(self.structure_proj.weight)
        nn.init.zeros_(self.structure_proj.bias)

    def source(self, g):
        views = g["reference_views"]
        b, v = views.shape[:2]
        features = self.cnn(views.flatten(0, 1)).reshape(b, v, 64, 30, 52)
        uv, depth, visible = projections(g["anchors"], g["ref_C"], g["ref_R"], g["K"])
        observed = sample_map(views, uv / 2)
        tolerance = g["depth_tolerance"].reshape(b, 1, 1)
        visible &= (observed[..., 1] > .5) & ((observed[..., 0] - depth).abs() <= tolerance)
        sampled = sample_map(features, uv / 16)
        direction = F.normalize(g["ref_C"][:, :, None] - g["anchors"][:, None], dim=-1)
        score = self.view_score(torch.cat([sampled, direction, depth.unsqueeze(-1)], -1)).squeeze(-1)
        weights = score.float().masked_fill(~visible, -1.e4).softmax(dim=1) * visible
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1.e-8)
        fused = (sampled * weights.unsqueeze(-1)).sum(1)
        point_feature = self.point_mlp(torch.cat([g["anchors"], g["anchor_density"].unsqueeze(-1)], -1))
        local = fused + point_feature
        return torch.cat([local, local.mean(1, keepdim=True)], 1)

    def forward(self, g):
        source = self.source(g)
        keys, values = self.key(source), self.value(source)
        b, t = g["target_C"].shape[:2]
        if (t - 1) % 4:
            raise ValueError("Target frames must have 4n+1 layout")
        uv, depth, visible = projections(g["anchors"], g["target_C"], g["target_R"], g["K"])
        observed = sample_map(g["target_depths"], uv / 2)
        occluded = (observed[..., 1] > .5) & (depth > observed[..., 0] + g["depth_tolerance"].reshape(b, 1, 1))
        visible &= ~occluded
        yy, xx = torch.meshgrid(torch.arange(30, device=source.device),
                                torch.arange(52, device=source.device), indexing="ij")
        blocks = torch.stack([xx, yy], -1).reshape(-1, 2).float()
        pixels = (blocks + .5) * 16 - .5
        K = g["K"]
        camera_ray = torch.cat([(pixels[None] - K[:, None, :2, 2]) /
                                torch.stack([K[:, 0, 0], K[:, 1, 1]], -1)[:, None],
                                torch.ones(b, len(pixels), 1, device=source.device)], -1)
        frames = []
        for i in range(t):
            rays = F.normalize(torch.einsum("bji,bqj->bqi", g["target_R"][:, i].float(), camera_ray), dim=-1)
            center = g["target_C"][:, i, None].expand(-1, len(pixels), -1)
            query = self.query_mlp(torch.cat([center, rays], -1))
            scores = torch.matmul(query.float(), keys.float().transpose(1, 2)) / 8
            anchor_blocks = (uv[:, i] + .5) / 16 - .5
            bias = -(blocks[None, :, None] - anchor_blocks[:, None]).square().sum(-1) / (2 * 4 ** 2)
            local_scores = (scores[..., :-1] + bias).masked_fill(~visible[:, i, None], -1.e4)
            scores = torch.cat([local_scores, scores[..., -1:]], -1)
            frames.append(torch.matmul(scores.softmax(-1).to(values.dtype), values))
        aligned = [frames[0]] + [torch.stack(frames[i:i + 4]).mean(0) for i in range(1, t, 4)]
        return torch.stack(aligned, 1).flatten(1, 2)
