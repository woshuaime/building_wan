"""Opt-in execution candidate for the pinned B/whole-frame training process."""
from types import MethodType

import torch

import integration
import worker


_counts = dict(vace_forwards=0, diagnostic_updates=0, fallback_updates=0)
_original_update = worker.update


def activity():
    return _counts.copy()


def direct_hints(blocks, c, x, context, t_mod, freqs):
    """Each block consumes the previous hidden state; only its skip goes to the DiT."""
    from diffsynth.models.wan_video_dit import DiTBlock
    hints = []
    for block in blocks:
        if block.block_id == 0:
            c = block.before_proj(c) + x
        c = DiTBlock.forward(block, c, context, t_mod, freqs)
        hints.append(block.after_proj(c))
    return tuple(hints)


def _direct_forward(self, x, vace_context, context, t_mod, freqs,
                            use_gradient_checkpointing=False,
                            use_gradient_checkpointing_offload=False,
                            structure_tokens=None, structure_mask=None,
                            structure_projection=None):
    if len(vace_context) != x.shape[0] or context.shape[0] != x.shape[0]:
        raise ValueError("Each VACE batch item needs its own condition and text context")
    c = torch.cat([self.vace_patch_embedding(u.unsqueeze(0)).flatten(2).transpose(1, 2)
                   for u in vace_context], dim=0)
    if structure_tokens is not None:
        if structure_tokens.shape[:2] != c.shape[:2]:
            raise ValueError(f"Structure/VACE token mismatch: {structure_tokens.shape}, {c.shape}")
        if structure_mask is None or structure_mask.shape != (*c.shape[:2], 1):
            raise ValueError("Structure mask must contain one scalar per sample and VACE token")
        if structure_projection is None:
            raise ValueError("Structure projection is required")
        delta = structure_projection(structure_tokens) * structure_mask
        c = c + delta.to(c.dtype)
    if x.shape[1] < c.shape[1]:
        raise ValueError("VACE condition is longer than the DiT token sequence")
    if x.shape[1] > c.shape[1]:
        c = torch.nn.functional.pad(c, (0, 0, 0, x.shape[1] - c.shape[1]))
    if c.shape != x.shape:
        raise ValueError("VACE and DiT token layouts differ")
    self.structure_last_call = dict(
        tokens=None if structure_tokens is None else list(structure_tokens.shape),
        mask=None if structure_mask is None else list(structure_mask.shape),
        injected_before_first_vace_block=structure_tokens is not None,
        gradient_enabled=torch.is_grad_enabled(), batch_size=x.shape[0])
    _counts["vace_forwards"] += 1
    return direct_hints(self.vace_blocks, c, x, context, t_mod, freqs)


def gradient_stats(adapter, lora, weights):
    """Transfer scalar reductions once. Retain parameter-wise FP32 sums and Python summation order."""
    named_groups = (("adapter", adapter), ("lora", lora))
    sums_of_squares = []
    positions = {"adapter": [], "lora": [], "upstream": []}
    for group, parameters in named_groups:
        for name, parameter in parameters:
            if parameter.grad is None:
                continue
            position = len(sums_of_squares)
            positions[group].append(position)
            if group == "adapter" and "structure_proj" not in name:
                positions["upstream"].append(position)
            sums_of_squares.append(parameter.grad.detach().float().square().sum())
    has_weight = (torch.cat(weights).sum() > 0).float()
    values = torch.stack([*sums_of_squares, has_weight]).tolist()
    norms = {key: sum(values[i] for i in indices) ** .5 for key, indices in positions.items()}
    return {"adapter": norms["adapter"], "lora": norms["lora"]}, norms["upstream"], bool(values[-1])


def parameters_valid(parameters):
    if any(p.dtype != torch.float32 for p in parameters):
        return False
    if not parameters:
        return True
    flat = torch.cat([p.detach().reshape(-1) for p in parameters])
    return bool(torch.isfinite(flat).all())


def _supported(pipe, clean):
    from diffsynth.models.wan_video_vace import VaceWanAttentionBlock
    if len(clean) != 1 or getattr(pipe, "adapter_gt_v2_checkpointing", {}).get("mode") != "none":
        return False
    core = integration._core(pipe)
    if getattr(core.forward, "__func__", None) is not integration.structured_vace_forward:
        return False
    if len(core.vace_blocks) != 15:
        return False
    # The official block wrapper is replaced by its operations on the hidden
    # state. Custom forward/backward hooks or an overridden forward need the wrapper.
    return all(type(b) is VaceWanAttentionBlock and b.block_id == 2 * i
               and getattr(b.forward, "__func__", None) is VaceWanAttentionBlock.forward
               and getattr(b, "_adapter_gt_v2_checkpoint", True) is False
               and not (b._forward_hooks or b._forward_pre_hooks or b._backward_hooks or b._backward_pre_hooks)
               for i, b in enumerate(core.vace_blocks))


def _candidate_update(pipe, optimizer, adapter, lora, batch, consumed, update_number):
    clean, vace, context, occupancy, geometry = batch
    optimizer.zero_grad(set_to_none=True)
    noisy, targets, times, weights = [], [], [], []
    # Scalar scheduler API is invoked separately; RNG calls retain sample order even when batch changes.
    for i in range(len(clean)):
        index = torch.randint(0, 1000, (1,), device="cpu")
        timestep = pipe.scheduler.timesteps[index].to(device=pipe.device, dtype=torch.bfloat16)
        noise = torch.randn(clean[i:i + 1].shape, device=pipe.device, dtype=torch.bfloat16)
        noisy.append(pipe.scheduler.add_noise(clean[i:i + 1], noise, timestep))
        targets.append(pipe.scheduler.training_target(clean[i:i + 1], noise, timestep))
        times.append(timestep)
        weights.append(pipe.scheduler.training_weight(timestep).to(pipe.device).reshape(1))
    timestep = torch.cat(times)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = pipe.structure_adapter(geometry)
        prediction = pipe.model_fn(**models, latents=torch.cat(noisy), timestep=timestep, context=context,
                                   vace_context=vace, vace_scale=1., use_gradient_checkpointing=True,
                                   use_gradient_checkpointing_offload=False,
                                   structure_tokens=tokens,
                                   structure_mask=tokens.new_ones((*tokens.shape[:2], 1)),
                                   structure_projection=pipe.structure_adapter.structure_proj)
        loss, terms, areas = worker.region_loss(prediction, torch.cat(targets), occupancy, torch.cat(weights))
    if not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise RuntimeError("Nonfinite/disconnected loss")
    loss.backward()
    parameters = [p for _, p in adapter + lora]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    group_norms, upstream, nonzero_time_weight = gradient_stats(adapter, lora, weights)
    if nonzero_time_weight and (group_norms["adapter"] <= 0 or group_norms["lora"] <= 0):
        raise RuntimeError(f"Trainable increment has no gradient: {group_norms}")
    if nonzero_time_weight and update_number >= 5 and upstream <= 0:
        raise RuntimeError("Adapter upstream gradient is absent after nonzero projection updates")
    watched = {}
    if update_number <= 5 and nonzero_time_weight:
        for name, group in (("adapter", adapter), ("lora", lora)):
            parameter = next(p for _, p in group if p.grad is not None and bool(p.grad.abs().max() > 0))
            watched[name] = (parameter, parameter.detach().clone())
    lr = worker.learning_rate(consumed + len(clean))
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    changed = {name: not torch.equal(parameter.detach(), before) for name, (parameter, before) in watched.items()}
    if changed and not all(changed.values()):
        raise RuntimeError(f"Optimizer did not change a trainable increment: {changed}")
    if not parameters_valid(parameters):
        raise RuntimeError("Invalid increment precision or values")
    if any(p.grad is not None for p in pipe.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen model acquired gradients")
    _counts["diagnostic_updates"] += 1
    return dict(loss=float(loss.detach()), region_losses=terms.cpu().tolist(), region_areas=areas.cpu().tolist(),
                grad_norm=float(norm), gradient_norms=group_norms, adapter_upstream_grad_norm=upstream,
                timesteps=timestep.float().cpu().tolist(), scheduler_weights=torch.cat(weights).float().cpu().tolist(),
                scheduler_zero_weight_batch=not nonzero_time_weight, parameters_changed=changed, lr=lr)


def _dispatch(pipe, optimizer, adapter, lora, batch, consumed, update_number):
    if not _supported(pipe, batch[0]):
        _counts["fallback_updates"] += 1
        return _original_update(pipe, optimizer, adapter, lora, batch, consumed, update_number)
    core = integration._core(pipe)
    normal = core.forward
    core.forward = MethodType(_direct_forward, core)
    try:
        return _candidate_update(pipe, optimizer, adapter, lora, batch, consumed, update_number)
    finally:
        core.forward = normal


def install_compute():
    """Install inside one trial or formal worker, independently of when it loads the pipeline."""
    if worker.update is not _original_update and worker.update is not _dispatch:
        raise RuntimeError("Another training update override is already installed")
    worker.update = _dispatch
