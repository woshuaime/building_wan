"""Project-local optional interfaces attached to the actual DiffSynth VACE model.

All original layers and the official model_fn are reused. Vendor files remain
unchanged; only this experiment's loaded VaceWanModel instance gains the optional
forward interface. Turning off both deltas restores the original computation.
"""
from types import MethodType
import torch


def structured_vace_forward(self, x, vace_context, context, t_mod, freqs,
                            use_gradient_checkpointing=False,
                            use_gradient_checkpointing_offload=False,
                            structure_tokens=None, structure_mask=None, structure_projection=None):
    from diffsynth.models.wan_video_vace import VaceWanModel
    from diffsynth.core.gradient import gradient_checkpoint_forward
    if structure_tokens is None:
        return VaceWanModel.forward(self, x, vace_context, context, t_mod, freqs,
                                   use_gradient_checkpointing, use_gradient_checkpointing_offload)
    c = [self.vace_patch_embedding(u.unsqueeze(0)).flatten(2).transpose(1, 2) for u in vace_context]
    c = torch.cat(c)
    if structure_tokens.shape[:2] != c.shape[:2] or structure_mask.shape != (*c.shape[:2], 1):
        raise ValueError(f"Structure/VACE token layout mismatch: {structure_tokens.shape}, {structure_mask.shape}, {c.shape}")
    delta = structure_projection(structure_tokens) * structure_mask
    c = c + delta.to(c.dtype)
    if x.shape[1] > c.shape[1]:
        c = torch.nn.functional.pad(c, (0, 0, 0, x.shape[1] - c.shape[1]))
    if x.shape != c.shape:
        raise ValueError("VACE/DiT token layout mismatch")
    self.structure_last_call = dict(tokens=list(structure_tokens.shape), mask=list(structure_mask.shape),
                                   injected_before_first_vace_block=True, gradient_enabled=torch.is_grad_enabled())
    for block in self.vace_blocks:
        c = gradient_checkpoint_forward(block, use_gradient_checkpointing,
                                        use_gradient_checkpointing_offload, c, x, context, t_mod, freqs)
    return torch.unbind(c)[:-1]


def install_structure_interface(pipe):
    # This DiffSynth version injects PEFT layers in place (peft_config exists
    # without a PeftModel wrapper); accept either upstream representation.
    core = pipe.vace.base_model.model if hasattr(pipe.vace, "base_model") else pipe.vace
    core.forward = MethodType(structured_vace_forward, core)
    pipe.model_fn = model_fn_wan_video
    return core


class _ConditionedVace:
    def __init__(self, model, tokens, mask, projection):
        self.model, self.tokens, self.mask, self.projection = model, tokens, mask, projection

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, x, *args, **kwargs):
        tokens, mask = self.tokens, self.mask
        if x.shape[0] != tokens.shape[0]:
            if x.shape[0] != 2 * tokens.shape[0]:
                raise ValueError("Unexpected CFG batch")
            tokens, mask = tokens.repeat(2, 1, 1), mask.repeat(2, 1, 1)
        return self.model(x, *args, structure_tokens=tokens, structure_mask=mask,
                          structure_projection=self.projection, **kwargs)


def model_fn_wan_video(*args, structure_tokens=None, structure_mask=None,
                       structure_projection=None, **kwargs):
    from diffsynth.pipelines.wan_video import model_fn_wan_video as official_model_fn
    if structure_tokens is not None:
        if kwargs.get("sliding_window_size") is not None or kwargs.get("use_unified_sequence_parallel"):
            raise ValueError("This fixed-clip adapter does not support temporal tiling or DDP")
        kwargs["vace"] = _ConditionedVace(kwargs["vace"], structure_tokens, structure_mask, structure_projection)
    return official_model_fn(*args, **kwargs)


def parameter_groups(pipe):
    adapter, lora = [], []
    for name, p in pipe.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("structure_adapter."):
            adapter.append((name, p))
        elif name.startswith("vace.") and ".vace_blocks." in name and any(
                f".{attn}.{target}.lora_" in name for attn in ("self_attn", "cross_attn") for target in ("q", "k", "v", "o")):
            lora.append((name, p))
        else:
            raise ValueError(f"Unauthorized trainable parameter: {name}")
    if sum(p.numel() for _, p in adapter) != 142929 or sum(p.numel() for _, p in lora) != 2949120:
        raise ValueError(f"Wrong parameter counts: Adapter={sum(p.numel() for _, p in adapter)}, LoRA={sum(p.numel() for _, p in lora)}")
    return adapter, lora


def attach_trainable_adapter(pipe, seed):
    from diffsynth.diffusion import DiffusionTrainingModule
    from adapter import MultiViewStructureAdapter
    torch.manual_seed(seed)
    DiffusionTrainingModule().switch_pipe_to_training_mode(
        pipe, trainable_models=None, lora_base_model="vace", lora_target_modules="q,k,v,o", lora_rank=8, task="sft")
    peft_config = pipe.vace.peft_config["default"]
    if peft_config.lora_alpha != 8 or peft_config.r != 8:
        raise ValueError("LoRA alpha/rank must both be 8")
    for p in pipe.vace.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    torch.manual_seed(seed + 1)
    pipe.structure_adapter = MultiViewStructureAdapter().to(device=pipe.device, dtype=torch.float32)
    pipe.structure_adapter.requires_grad_(True)
    pipe.vace.train()
    pipe.structure_adapter.train()
    core = install_structure_interface(pipe)
    parameter_groups(pipe)
    return core


def load_increments(pipe, adapter_path, lora_path, seed=20261110):
    """Reload the two separate increments onto the same original VACE base."""
    from safetensors.torch import load_file
    attach_trainable_adapter(pipe, seed)
    pipe.structure_adapter.load_state_dict(load_file(str(adapter_path)), strict=True)
    expected = {name.removeprefix("vace."): p for name, p in parameter_groups(pipe)[1]}
    state = load_file(str(lora_path))
    if set(state) != set(expected):
        raise ValueError("LoRA keys do not match this original VACE architecture")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name])
    pipe.requires_grad_(False)
    pipe.eval()
    return pipe
