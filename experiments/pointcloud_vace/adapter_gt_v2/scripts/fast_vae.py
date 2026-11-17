"""Batch the SAME frozen VAE spatial tiles; retain their order and blend weights.

This changes execution only: no tile extent, frame, model weight or condition is
changed. The mutable causal cache is reset by the original model.encode for each
independent batch. The inference batch is unrelated to the training batch.
"""
from types import MethodType
import torch


def tile_tasks(height, width, tile_size, tile_stride):
    size_h, size_w = tile_size
    stride_h, stride_w = tile_stride
    tasks = []
    for h in range(0, height, stride_h):
        if h - stride_h >= 0 and h - stride_h + size_h >= height:
            continue
        for w in range(0, width, stride_w):
            if w - stride_w >= 0 and w - stride_w + size_w >= width:
                continue
            tasks.append((h, min(h + size_h, height), w, min(w + size_w, width)))
    return tasks


def tiled_encode(vae, video, device, tile_size, tile_stride, tile_batch=1, gpu_blend=False):
    if video.shape[0] != 1 or tile_batch < 1:
        raise ValueError("The spatial tile executor accepts one complete source window")
    _, _, frames, height, width = video.shape
    factor = vae.upsampling_factor
    out_t = (frames + 3) // 4
    blend_device = device if gpu_blend else "cpu"
    values = torch.zeros((1, vae.z_dim, out_t, height // factor, width // factor),
                         dtype=video.dtype, device=blend_device)
    weights = torch.zeros((1, 1, out_t, height // factor, width // factor),
                          dtype=video.dtype, device=blend_device)
    # Keep the window on the GPU instead of copying each overlapping tile.
    video = video.to(device)
    tasks = tile_tasks(height, width, tile_size, tile_stride)
    cursor = 0
    while cursor < len(tasks):
        group = [tasks[cursor]]
        shape = (group[0][1] - group[0][0], group[0][3] - group[0][2])
        while cursor + len(group) < len(tasks) and len(group) < tile_batch:
            following = tasks[cursor + len(group)]
            if (following[1] - following[0], following[3] - following[2]) != shape:
                break
            group.append(following)
        patches = torch.cat([video[:, :, :, h:h2, w:w2] for h, h2, w, w2 in group], dim=0)
        encoded = vae.model.encode(patches, vae.scale).to(blend_device)
        for index, (h, h2, w, w2) in enumerate(group):
            tile = encoded[index:index + 1]
            mask = vae.build_mask(tile,
                  is_bound=(h == 0, h2 >= height, w == 0, w2 >= width),
                  border_width=((tile_size[0] - tile_stride[0]) // factor,
                                (tile_size[1] - tile_stride[1]) // factor)).to(device=blend_device, dtype=video.dtype)
            y, x = h // factor, w // factor
            values[:, :, :, y:y + tile.shape[3], x:x + tile.shape[4]] += tile * mask
            weights[:, :, :, y:y + tile.shape[3], x:x + tile.shape[4]] += mask
        cursor += len(group)
    if not bool((weights > 0).all()):
        raise ValueError("VAE spatial tiles did not cover the complete latent grid")
    return (values / weights).cpu()


def install(vae, candidate):
    if not hasattr(vae, "_gt_v2_original_tiled_encode"):
        vae._gt_v2_original_tiled_encode = vae.tiled_encode
    if hasattr(vae, "encode") and not hasattr(vae, "_gt_v2_original_encode"):
        vae._gt_v2_original_encode = vae.encode
    if candidate.get("original", False):
        vae.tiled_encode = vae._gt_v2_original_tiled_encode
        if hasattr(vae, "_gt_v2_original_encode"):
            vae.encode = vae._gt_v2_original_encode
        return
    def execute(self, video, device, tile_size, tile_stride):
        return tiled_encode(self, video, device, tile_size, tile_stride,
                            tile_batch=candidate["tile_batch"], gpu_blend=candidate.get("gpu_blend", False))
    vae.tiled_encode = MethodType(execute, vae)
    if candidate.get("gpu_input", True) and hasattr(vae, "_gt_v2_original_encode"):
        def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
            # The original wrapper first copies a GPU-normalized window to CPU.
            # For one complete window we can keep that exact tensor on GPU.
            if tiled and isinstance(videos, torch.Tensor) and videos.ndim == 5 and videos.shape[0] == 1:
                factor = self.upsampling_factor
                return self.tiled_encode(videos, device,
                        tuple(v * factor for v in tile_size), tuple(v * factor for v in tile_stride))
            return self._gt_v2_original_encode(videos, device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        vae.encode = MethodType(encode, vae)
