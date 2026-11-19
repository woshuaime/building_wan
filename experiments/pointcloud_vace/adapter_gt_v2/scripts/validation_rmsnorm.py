"""Wan's BF16 RMSNorm with the PyTorch 2.11 CUDA sum order for one validation grid."""
from contextlib import ExitStack, contextmanager
from functools import partial

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _wan_rmsnorm(X, W, Y, ROWS: tl.constexpr, VEC: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, 32 * VEC)
    base = row[:, None] * 1536
    valid = row[:, None] < 32760

    # ATen Reduce.cuh and ReduceMomentKernel.cu (v2.11.0) assign one
    # warp per row here. Each lane accumulates four independent columns.
    partials = tl.full((ROWS, 32 * VEC), 0, tl.float32)
    for group in tl.static_range(1536 // (32 * VEC)):
        x = tl.load(X + base + group * (32 * VEC) + columns[None, :], mask=valid, other=0).to(tl.float32)
        partials += x * x
    lanes = tl.arange(0, 32)[None, :] + tl.full((ROWS, 32), 0, tl.int32)
    sums = tl.gather(partials, lanes * VEC, axis=1)
    for component in tl.static_range(1, VEC):
        sums = sums + tl.gather(partials, lanes * VEC + component, axis=1)
    for depth in tl.static_range(5):
        offset = 16 >> depth
        sums = sums + tl.gather(sums, (lanes + offset) % 32, axis=1)
    total = tl.gather(sums, tl.full((ROWS, 1), 0, tl.int32), axis=1)
    inv_rms = libdevice.rsqrt(total * (1.0 / 1536.0) + EPS)

    for group in tl.static_range(1536 // (32 * VEC)):
        col = group * 32 * VEC + columns
        x = tl.load(X + base + col[None, :], mask=valid, other=0).to(tl.float32)
        weight = tl.load(W + col).to(tl.float32)
        # Wan casts back to the activation dtype before multiplying by weight.
        normalized = (x * inv_rms).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + base + col[None, :], normalized * weight[None, :], mask=valid)


def _forward(module, original, device, stats, x):
    weight = module.weight
    if (not torch.is_inference_mode_enabled() or module.training or module.use_torch_norm
            or module.eps != 1e-6 or tuple(module.normalized_shape) != (1536,)
            or x.device != device or weight.device != device
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or x.shape != (1, 32760, 1536) or weight.shape != (1536,)
            or not x.is_contiguous() or not weight.is_contiguous() or weight.requires_grad):
        return original(x)
    y = torch.empty_like(x)
    _wan_rmsnorm[(triton.cdiv(32760, 8),)](
        x, weight, y, 8, 4, 1e-6, enable_fp_fusion=False, num_warps=4)
    stats["calls"] += 1
    return y


@contextmanager
def exact_rmsnorm(pipe):
    """Use the measured kernel only on this pipeline's original RMSNorm instances."""
    stats = {"calls": 0}
    device = torch.device(pipe.device)
    if (pipe.training or device.type != "cuda" or torch.__version__ != "2.11.0+cu128"
            or not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (8, 6)
            or torch.cuda.get_device_name(device) != "NVIDIA RTX A6000"):
        yield stats
        return
    from diffsynth.models.wan_video_dit import RMSNorm

    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with ExitStack() as stack:
        for component in (pipe.dit, pipe.vace):
            for module in component.modules():
                if (type(module) is not RMSNorm or "forward" in vars(module)
                        or getattr(module.forward, "__func__", None) is not RMSNorm.forward
                        or getattr(module.norm, "__func__", None) is not RMSNorm.norm):
                    continue
                module.forward = partial(_forward, module, module.forward, device, stats)
                stack.callback(delattr, module, "forward")
        yield stats
