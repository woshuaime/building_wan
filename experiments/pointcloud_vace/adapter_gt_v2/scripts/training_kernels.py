"""Process-local CUDA rotary candidate for the fixed Wan training shape."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(X, Frequencies, Y, size, sequence: tl.constexpr,
                   heads: tl.constexpr, width: tl.constexpr, adjoint: tl.constexpr,
                   block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < size
    values = tl.load(X + offsets, mask=valid, other=0).to(tl.float64)
    position = offsets // (heads * width) % sequence
    component = offsets % width
    phase = tl.load(Frequencies + position * width + component,
                    mask=valid, other=0).to(tl.float64)
    real, imaginary = tl.split(tl.reshape(values, (block // 2, 2)))
    cosine, sine = tl.split(tl.reshape(phase, (block // 2, 2)))
    if adjoint:
        sine = -sine
    yr = real * cosine - imaginary * sine
    yi = real * sine + imaginary * cosine
    # c10::BFloat16 takes a float. Preserve the reference double -> float
    # -> BF16 rounding on Ampere for both the forward and its transpose.
    result = tl.reshape(tl.join(yr, yi), (block,)).to(tl.float32).to(tl.bfloat16)
    tl.store(Y + offsets, result, mask=valid)


class _Rotate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs, num_heads, adjoint):
        ctx.save_for_backward(freqs)
        ctx.num_heads = num_heads
        ctx.adjoint = adjoint
        x = x.contiguous()
        output = torch.empty_like(x)
        size = x.numel()
        _rotary_kernel[(triton.cdiv(size, 512),)](
            x, torch.view_as_real(freqs), output, size, x.shape[1], num_heads,
            x.shape[2] // num_heads, adjoint, 512)
        return output

    @staticmethod
    def backward(ctx, grad):
        (freqs,) = ctx.saved_tensors
        return _Rotate.apply(grad, freqs, ctx.num_heads, not ctx.adjoint), None, None, None


def _supported(x, freqs, heads):
    return (heads == 12 and x.ndim == 3 and x.shape[-1] == 1536 and x.is_cuda
            and x.dtype == torch.bfloat16 and freqs.device == x.device
            and freqs.dtype in (torch.complex64, torch.complex128)
            and freqs.shape == (x.shape[1], 1, 64) and freqs.is_contiguous()
            and not freqs.requires_grad and not freqs.is_conj())


def install_rotary():
    """Select the local implementation only for the supported Wan layout."""
    from diffsynth.models import wan_video_dit as wan

    if getattr(wan.rope_apply, "_gt_v2_fused", False):
        return
    reference = wan.rope_apply

    def apply(x, freqs, num_heads):
        if _supported(x, freqs, num_heads):
            return _Rotate.apply(x, freqs, num_heads, False)
        return reference(x, freqs, num_heads)

    apply._gt_v2_fused = True
    apply._gt_v2_reference = reference
    wan.rope_apply = apply


def verify_rotary(device="cuda"):
    """Require elementwise identical BF16 forward and input gradient on the actual grid."""
    from diffsynth.models import wan_video_dit as wan

    device = torch.device(device)
    f, h, w = 5, 30, 52
    phase_t, phase_h, phase_w = wan.precompute_freqs_cis_3d(128, end=max(f, h, w))
    freqs = torch.cat([phase_t[:f, None, None].expand(f, h, w, -1),
                       phase_h[None, :h, None].expand(f, h, w, -1),
                       phase_w[None, None, :w].expand(f, h, w, -1)], dim=-1)
    freqs = freqs.reshape(f * h * w, 1, 64).to(device)
    generator = torch.Generator(device=device).manual_seed(20261119)
    shape = (1, f * h * w, 1536)
    x = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    grad = torch.randn(shape, device=device, dtype=x.dtype, generator=generator)
    if not _supported(x, freqs, 12):
        raise RuntimeError("Rotary equivalence test requires the supported A6000 CUDA layout")
    reference = getattr(wan.rope_apply, "_gt_v2_reference", wan.rope_apply)
    with torch.enable_grad():
        x.requires_grad_()
        expected = reference(x, freqs, 12)
        actual = _Rotate.apply(x, freqs, 12, False)
        expected_grad, = torch.autograd.grad(expected, x, grad)
        actual_grad, = torch.autograd.grad(actual, x, grad)
    forward_errors = torch.count_nonzero(expected != actual).item()
    backward_errors = torch.count_nonzero(expected_grad != actual_grad).item()
    if forward_errors or backward_errors:
        raise ValueError(f"Rotary BF16 mismatch: forward={forward_errors} backward={backward_errors}")
    return dict(shape=list(shape), frequency_dtype=str(freqs.dtype), elements=x.numel(),
                forward_mismatches=forward_errors, backward_mismatches=backward_errors, passed=True)
