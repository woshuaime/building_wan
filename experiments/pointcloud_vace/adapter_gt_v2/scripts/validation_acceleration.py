"""Scoped inference optimizations for the frozen whole-frame GT V2 model."""
from contextlib import contextmanager

import torch

from integration import _core
from training_compute_candidate import _supported


@contextmanager
def accelerated(pipe, *, rotary=True):
    """Keep one immutable clip condition while recomputing all text-dependent blocks.

    Enter once per clip. The caller may enter and leave for paired comparisons.
    The pipeline's VACE input, structure tokens and mask are read-only throughout
    a trajectory; holding the tensors themselves prevents storage identity reuse.
    """
    from diffsynth.models import wan_video_dit as wan

    if pipe.training or not _supported(pipe, (None,)):
        raise ValueError("Expected the frozen GT V2 validation pipeline and its 15 official VACE blocks")
    core = _core(pipe)
    reference = core.forward
    original_rotary = wan.rope_apply
    sources = None
    before = None

    def forward(x, vace_context, context, t_mod, freqs, use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False, structure_tokens=None,
                structure_mask=None, structure_projection=None):
        nonlocal sources, before
        if (torch.is_grad_enabled() or use_gradient_checkpointing or use_gradient_checkpointing_offload
                or x.shape[0] != 1 or vace_context.shape[0] != 1 or context.shape[0] != 1
                or structure_tokens is None or structure_tokens.shape[:2] != x.shape[:2]
                or structure_mask is None or structure_mask.shape != (*x.shape[:2], 1)
                or structure_projection is None):
            return reference(x, vace_context, context, t_mod, freqs, use_gradient_checkpointing,
                             use_gradient_checkpointing_offload, structure_tokens, structure_mask,
                             structure_projection)

        current = (vace_context, structure_tokens, structure_mask, structure_projection)
        if sources is None or any(old is not new for old, new in zip(sources, current)):
            # Match the original B=1 VACE condition, including the contiguous
            # copy made by cat after the spatial-to-sequence transpose.
            c = torch.cat([core.vace_patch_embedding(vace_context[0].unsqueeze(0))
                           .flatten(2).transpose(1, 2)], dim=0)
            delta = structure_projection(structure_tokens) * structure_mask
            before = core.vace_blocks[0].before_proj(c + delta.to(c.dtype))
            sources = current
        core.structure_last_call = dict(injected_before_first_vace_block=True,
                                        tokens=list(structure_tokens.shape), mask=list(structure_mask.shape),
                                        gradient_enabled=False, batch_size=1)
        c = before + x
        hints = []
        for block in core.vace_blocks:
            c = wan.DiTBlock.forward(block, c, context, t_mod, freqs)
            hints.append(block.after_proj(c))
        return tuple(hints)

    try:
        if rotary:
            from training_kernels import install_rotary
            install_rotary()
        core.forward = forward
        yield
    finally:
        core.forward = reference
        wan.rope_apply = original_rotary
