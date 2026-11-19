"""Image-space auxiliary supervision for the original V2 trainable increments."""

from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import data


# Load only the existing pure OBJ helpers. The separate framework experiment
# and its model, data preparation and precomputed assets are never instantiated.
_mesh_module = Path(__file__).resolve().parents[2] / "adapter_gt_v2_structure/scripts/mesh_labels.py"
_spec = importlib.util.spec_from_file_location("v2_gt_mesh_edges", _mesh_module)
_mesh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mesh)

RGB_CONFIG = {
    "appearance": 0.10, "structure": 0.10, "temp": 0.02,
    "ramp_from": 0.1, "ramp_to": 1., "ramp_updates": 1000,
    "sigma_min": 0.10, "sigma_max": 0.60,
    "causal_prefix_latents": 2, "decoded_frames": 5,
    "geometry_cache_buildings": 256, "structure_band_sigma_px": 3.,
    "structure_band_cutoff_px": 9., "structure_weight_cap": 1.,
    "coverage_levels_255": [150., 220.], "coverage_transition_halfwidth_255": 12.,
    "texture_tile_pixels": [16, 32, 64], "lowpass_pixels": 5,
    "known_partial_threshold_255": 239., "known_partial_halo_pixels": 1,
}


@lru_cache(maxsize=RGB_CONFIG["geometry_cache_buildings"])
def _roof_segments(mesh_path):
    vertices, triangles, normals = _mesh.read_obj(Path(mesh_path))
    # Work in double precision near the origin. Keep only compact 3D line
    # segments in this cache, with no projected image or complete mesh retained.
    origin = vertices[0]
    labels = _mesh.structural_edges(vertices - origin, triangles, normals)
    kinds = labels["kinds"]
    selected = np.isin(kinds, (_mesh.RIDGE, _mesh.VALLEY, _mesh.EAVE))
    return labels["vertices"][labels["edges"][selected]] + origin


def structure_weights(mesh_path, camera, frame_ids, size=(512, 512), image_affine=None):
    """Project mesh creases using the camera and exact image pixel-center transform.

    size is (height, width). image_affine=(sx, sy, tx, ty) maps source image
    indices to training image indices. Each segment defines a neighborhood in
    the existing transparent point rendering, without changing any target RGB.
    """
    height, width = size
    if image_affine is None:
        sx, sy = width / data.WIDTH, height / data.HEIGHT
        image_affine = (sx, sy, (sx - 1) / 2, (sy - 1) / 2)
    sx, sy, tx, ty = image_affine
    frames = list(frame_ids)
    output = np.zeros((len(frames), height, width), dtype=np.float32)
    edges = _roof_segments(str(mesh_path)) - np.asarray(camera["world_to_local_origin"], np.float64)
    if not len(edges):
        return output
    centers, rotations, intrinsics = data.camera_arrays(camera)
    for i, frame in enumerate(frames):
        uv, depth = data.project(edges.reshape(-1, 3), centers[frame], rotations[frame], intrinsics)
        lines = (uv * (sx, sy) + (tx, ty)).reshape(-1, 2, 2)
        valid = (depth.reshape(-1, 2) > 0).all(axis=1)
        lines = lines[valid]
        if not len(lines):
            continue
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for line in np.rint(lines).astype(np.int64).reshape(-1, 4):
            draw.line(tuple(map(int, line)), fill=1, width=1)
        bbox = image.getbbox()
        if bbox is None:
            continue
        radius = int(RGB_CONFIG["structure_band_cutoff_px"])
        x0, y0, x1, y1 = bbox
        x0, y0, x1, y1 = max(0, x0-radius), max(0, y0-radius), min(width, x1+radius), min(height, y1+radius)
        distances = distance_transform_edt(np.asarray(image)[y0:y1, x0:x1] == 0)
        band = np.exp(-.5 * (distances / RGB_CONFIG["structure_band_sigma_px"])**2)
        band[distances > radius] = 0
        # Distance to the union gives max(edge weights), including at crossings.
        output[i, y0:y1, x0:x1] = band
    return output


def decode_prefix(vae, clean):
    """Decode the first two causal latent indices into actual video frames 0..4."""
    if clean.ndim != 5 or clean.shape[0] != 1 or clean.shape[2] < 2:
        raise ValueError(f"Expected a single Wan latent video, got {tuple(clean.shape)}")
    prefix = clean[:, :, :2].to(dtype=next(vae.model.parameters()).dtype)

    def decode(z):
        try:
            return vae.model.decode(z, vae.scale)
        finally:
            vae.model.clear_cache()

    decoded = checkpoint(decode, prefix, use_reentrant=False, preserve_rng_state=False)
    if decoded.shape[1:3] != (3, 5):
        raise RuntimeError(f"Unexpected native VAE prefix layout {tuple(decoded.shape)}")
    return (decoded[0].transpose(0, 1).float() + 1.) * .5


def ramp(step):
    """Step is the one-based optimizer update. Endpoints are 1 and 1000."""
    progress = min(1., max(0., (step - 1) / (RGB_CONFIG["ramp_updates"] - 1)))
    return RGB_CONFIG["ramp_from"] + (RGB_CONFIG["ramp_to"] - RGB_CONFIG["ramp_from"]) * progress


def _image_tensor(value, device):
    value = torch.as_tensor(value, device=device, dtype=torch.float32)
    return value[:, None] if value.ndim == 3 else value


def _luma(rgb):
    if rgb.shape[1] == 1:
        return rgb
    return .2126 * rgb[:, :1] + .7152 * rgb[:, 1:2] + .0722 * rgb[:, 2:3]


def _coverage(y, level):
    half_width = RGB_CONFIG["coverage_transition_halfwidth_255"]
    t = ((level + half_width - 255 * y) / (2 * half_width)).clamp(0., 1.)
    return t.square() * (3. - 2. * t)


def _region_mean(values, region):
    return (values.mean(1, keepdim=True) * region).sum() / region.sum().clamp_min(1.)


def _moments(rgb, y, region):
    r, g, b = rgb.expand(-1, 3, -1, -1).unbind(1)
    rg, bg = (r-g)[:, None], (b-g)[:, None]
    # Per-frame brightness and signed/absolute color statistics prevent
    # opposing hue errors from canceling. Membership belongs to each image.
    features = torch.cat((y, y.square(), rg, bg, rg.abs(), bg.abs()), dim=1)
    return (features * region).sum((-2, -1)) / region.sum((-2, -1)).clamp_min(1.)


def _appearance(pred, py, target, gy, missing):
    pred_cover = torch.cat([_coverage(py, level) for level in RGB_CONFIG["coverage_levels_255"]], dim=1)
    true_cover = torch.cat([_coverage(gy, level) for level in RGB_CONFIG["coverage_levels_255"]], dim=1)
    estimate, truth = pred_cover[:, 1:2], true_cover[:, 1:2]
    moments = []
    for predicted_region, true_region in ((estimate, truth), (1.-estimate, 1.-truth)):
        p = _moments(pred, py, predicted_region * missing)
        g = _moments(target, gy, true_region * missing)
        # A truly absent target class contributes no (undefined) color statistic.
        valid = ((true_region * missing).sum((-2, -1)) >= 1.).float()
        moments.append(((p-g).abs().mean(1, keepdim=True) * valid).sum() / valid.sum().clamp_min(1.))

    fractions = []
    for tile in RGB_CONFIG["texture_tile_pixels"]:
        pool = lambda x: F.avg_pool2d(x, tile, stride=tile, ceil_mode=True, count_include_pad=False)
        valid_fraction = pool(missing)
        count = valid_fraction.clamp_min(1.e-6)
        p = pool(pred_cover * missing) / count
        g = pool(true_cover * missing) / count
        # Square root makes small differences in sparse coverage visible. Tiles
        # containing point texture and empty tiles are independently normalized.
        distance = ((p + 1.e-4).sqrt() - (g + 1.e-4).sqrt()).abs()
        foreground = (g[:, 1:2] > 0).float()
        fractions.append(.5 * (_region_mean(distance, foreground * valid_fraction)
                               + _region_mean(distance, (1.-foreground) * valid_fraction)))
    return .5 * (torch.stack(moments).mean() + torch.stack(fractions).mean())


def rgb_losses(pred, gt, partial, edit_mask, structure, step):
    """Inputs are aligned T[×C]×H×W float images scaled to [0,1]."""
    pred = pred.float()
    gt, partial, edit_mask, structure = (
        _image_tensor(x, pred.device) for x in (gt, partial, edit_mask, structure))
    py, gy = _luma(pred), _luma(gt)
    known_points = (_luma(partial) < RGB_CONFIG["known_partial_threshold_255"] / 255.).float()
    known = F.max_pool2d(known_points, 3, stride=1, padding=1)
    missing = edit_mask * (1.-known)
    appearance = _appearance(pred, py, gt, gy, missing)

    # A linear filter of the masked luminance difference is exactly D(pred) -
    # D(GT) when both images use the same observation outside editable pixels.
    # This prevents any RGB derivative on known points or the padded border.
    lowpass = F.avg_pool2d((py-gy) * missing, RGB_CONFIG["lowpass_pixels"],
                           stride=1, padding=2, count_include_pad=False)
    kernel = lowpass.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.
    kernels = torch.stack((kernel, kernel.T))[:, None]
    with torch.autocast(device_type=pred.device.type, enabled=False):
        sobel = F.conv2d(F.pad(lowpass, (1, 1, 1, 1), mode="replicate"), kernels)
    band = structure.clamp(0., RGB_CONFIG["structure_weight_cap"]) * missing
    structure_loss = _region_mean(lowpass.abs(), band) + _region_mean(sobel.abs(), band)
    pair_region = torch.minimum(missing[1:], missing[:-1])
    temporal = _region_mean((lowpass[1:] - lowpass[:-1]).abs(), pair_region)
    aux = ramp(step) * (RGB_CONFIG["appearance"] * appearance + RGB_CONFIG["structure"] * structure_loss
                        + RGB_CONFIG["temp"] * temporal)
    return dict(appearance=appearance, structure=structure_loss, temp=temporal, aux=aux)
