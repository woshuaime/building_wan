"""Continuous, observed-only point projections on the native VACE token grid."""
import numpy as np

import data


def subcells(uv, relative_depth, size):
    height, width = size
    ny, nx = height // 4, width // 4
    inside = ((uv[:, 0] >= -.5) & (uv[:, 0] < width - .5) &
              (uv[:, 1] >= -.5) & (uv[:, 1] < height - .5))
    uv, z = uv[inside], relative_depth[inside]
    grid = (uv + .5) / 4 - .5
    base = np.floor(grid).astype(np.int64)
    fx, fy = (grid - base).T
    # The transition spans +/- 0.1 observed-building radii around the centre plane.
    rear = np.clip(.5 + 5 * z, 0, 1)
    bins = np.zeros((ny * nx, 2, 2), dtype=np.float32)
    for dy, wy in enumerate((1 - fy, fy)):
        for dx, wx in enumerate((1 - fx, fx)):
            x = np.clip(base[:, 0] + dx, 0, nx - 1)
            y = np.clip(base[:, 1] + dy, 0, ny - 1)
            for layer, membership in enumerate((1 - rear, rear)):
                mass = wx * wy * membership
                idx = y * nx + x
                bins[:, layer, 0] += np.bincount(idx, mass, minlength=ny * nx)
                bins[:, layer, 1] += np.bincount(idx, mass * z, minlength=ny * nx)
    count = bins[..., 0]
    # A vanishing contribution should vanish continuously in both channels.
    # log(1+mass) gates its mean depth; occupied cells still encode the mean.
    mean = np.divide(bins[..., 1], count, out=np.zeros_like(count), where=count > 0)
    bins[..., 0] = np.log1p(count)
    bins[..., 1] = mean * bins[..., 0]
    return (bins.reshape(ny // 4, 4, nx // 4, 4, 4).transpose(0, 2, 1, 3, 4)
            .reshape(ny // 4 * (nx // 4), 64))


def hints(points, centers, rotations, K, frames, radius):
    """The first latent sees frame zero; each subsequent latent sees four ordered frames."""
    middle = (points.min(0) + points.max(0)) / 2
    views = []
    for frame in frames:
        uv, depth = data.project(points, centers[frame], rotations[frame], K)
        center_depth = (middle - centers[frame]) @ rotations[frame, 2]
        valid = depth > 0
        views.append(subcells(uv[valid], (depth[valid] - center_depth) / radius,
                              (data.HEIGHT, data.WIDTH)))
    packed = np.stack([views[0]] * 3 + views).reshape(-1, 4, *views[0].shape)
    return packed.transpose(0, 2, 1, 3).reshape(-1, 256).copy()
