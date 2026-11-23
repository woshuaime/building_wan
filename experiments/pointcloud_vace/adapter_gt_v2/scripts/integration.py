"""Fresh D-architecture increments and batch-safe, local VACE integration.

No vendor file is changed. Checkpoint policy is attached to the loaded blocks;
the official DiT forward and the original nonreentrant checkpoint helper run
unchanged. Adapter dimensions and initialization match the historical D design;
map sampling additionally respects this dataset's image-index pixel centers.
"""
import importlib.util
import sys
from pathlib import Path
from types import MethodType

import torch


def _architecture():
    name = "_pointcloud_adapter_gt_v2_architecture"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("adapter.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].MultiViewStructureAdapter


def _core(pipe):
    return pipe.vace.base_model.model if hasattr(pipe.vace, "base_model") else pipe.vace


def _checkpoint_policy(model, enabled, offload, *args, **kwargs):
    from diffsynth.core.gradient import gradient_checkpoint_forward
    policy = getattr(model, "_adapter_gt_v2_checkpoint", None)
    if policy is not None:
        enabled, offload = policy, False
    return gradient_checkpoint_forward(model, enabled, offload, *args, **kwargs)


def configure_checkpointing(pipe, mode):
    """Select full, none, or even-index blocks in both original block lists."""
    if mode not in ("full", "partial", "none"):
        raise ValueError(f"Unknown checkpointing mode: {mode}")
    import diffsynth.pipelines.wan_video as pipeline_module
    # This name is the dispatch call imported by the official model_fn. The
    # policy falls back to the original helper for every unrelated model.
    pipeline_module.gradient_checkpoint_forward = _checkpoint_policy
    selected = {}
    for name, blocks in (("dit", pipe.dit.blocks), ("vace", _core(pipe).vace_blocks)):
        selected[name] = []
        for index, block in enumerate(blocks):
            active = mode == "full" or (mode == "partial" and index % 2 == 0)
            block._adapter_gt_v2_checkpoint = active
            if active:
                selected[name].append(index)
    pipe.adapter_gt_v2_checkpointing = dict(mode=mode, blocks=selected,
                                          use_reentrant=False, cpu_offload=False)
    return pipe.adapter_gt_v2_checkpointing


def structured_vace_forward(self, x, vace_context, context, t_mod, freqs,
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
    for block in self.vace_blocks:
        c = _checkpoint_policy(block, use_gradient_checkpointing,
                               use_gradient_checkpointing_offload,
                               c, x, context, t_mod, freqs)
    return torch.unbind(c)[:-1]


class _ConditionedVace:
    def __init__(self, model, tokens, mask, projection):
        self.model, self.tokens, self.mask, self.projection = model, tokens, mask, projection

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, x, *args, **kwargs):
        if x.shape[0] != self.tokens.shape[0]:
            raise ValueError("Training structure batch must match DiT batch exactly")
        return self.model(x, *args, structure_tokens=self.tokens,
                          structure_mask=self.mask,
                          structure_projection=self.projection, **kwargs)


def model_fn_wan_video(*args, structure_tokens=None, structure_mask=None,
                       structure_projection=None, **kwargs):
    from diffsynth.pipelines.wan_video import model_fn_wan_video as official_model_fn
    if args:
        raise ValueError("Use explicit model_fn keyword arguments")
    latents, context, timestep = (kwargs[key] for key in ("latents", "context", "timestep"))
    batch = latents.shape[0]
    if context.shape[0] != batch or timestep.ndim != 1 or timestep.shape[0] != batch:
        raise ValueError("Latents, text and independently sampled timesteps must have the same batch")
    if kwargs.get("sliding_window_size") is not None or kwargs.get("use_unified_sequence_parallel"):
        raise ValueError("This single-GPU clip trainer does not support temporal tiling or DDP")
    if structure_tokens is not None:
        kwargs["vace"] = _ConditionedVace(kwargs["vace"], structure_tokens,
                                         structure_mask, structure_projection)
    return official_model_fn(**kwargs)


def parameter_groups(pipe):
    adapter, lora = [], []
    for name, parameter in pipe.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("structure_adapter."):
            adapter.append((name, parameter))
        elif name.startswith("vace.") and ".vace_blocks." in name and any(
                f".{attention}.{target}.lora_" in name
                for attention in ("self_attn", "cross_attn")
                for target in ("q", "k", "v", "o")):
            lora.append((name, parameter))
        else:
            raise ValueError(f"Unauthorized trainable parameter: {name}")
    sizes = (sum(p.numel() for _, p in adapter), sum(p.numel() for _, p in lora))
    extra = 256 * 64 if getattr(pipe.structure_adapter, "parallax_proj", None) is not None else 0
    if sizes != (142929 + extra, 2949120):
        raise ValueError(f"Wrong Adapter/LoRA parameter counts: {sizes}")
    return adapter, lora


def attach(pipe, seed, checkpointing="full"):
    """Create fresh FP32 increments on the original frozen BF16 VACE."""
    from diffsynth.diffusion import DiffusionTrainingModule
    torch.manual_seed(seed)
    DiffusionTrainingModule().switch_pipe_to_training_mode(
        pipe, trainable_models=None, lora_base_model="vace",
        lora_target_modules="q,k,v,o", lora_rank=8, task="sft")
    config = pipe.vace.peft_config["default"]
    if config.lora_alpha != 8 or config.r != 8 or config.lora_dropout != 0:
        raise ValueError("Expected zero-dropout rank8/alpha8 VACE LoRA")
    for parameter in pipe.vace.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    torch.manual_seed(seed + 1)
    pipe.structure_adapter = _architecture()().to(device=pipe.device, dtype=torch.float32)
    pipe.structure_adapter.requires_grad_(True)
    pipe.vace.train()
    pipe.structure_adapter.train()
    core = _core(pipe)
    core.forward = MethodType(structured_vace_forward, core)
    pipe.model_fn = model_fn_wan_video
    configure_checkpointing(pipe, checkpointing)
    parameter_groups(pipe)
    return core


def increments(pipe):
    adapter, lora = parameter_groups(pipe)
    return ({name.removeprefix("structure_adapter."): parameter.detach().cpu().contiguous().clone()
             for name, parameter in adapter},
            {name.removeprefix("vace."): parameter.detach().cpu().contiguous().clone()
             for name, parameter in lora})


def load_increments_state(pipe, adapter, lora):
    pipe.structure_adapter.load_state_dict(adapter, strict=True)
    expected = {name.removeprefix("vace."): parameter for name, parameter in parameter_groups(pipe)[1]}
    if set(lora) != set(expected):
        raise ValueError("LoRA keys differ from this run's official VACE architecture")
    with torch.no_grad():
        for name, parameter in expected.items():
            if lora[name].shape != parameter.shape:
                raise ValueError(f"LoRA shape mismatch: {name}")
            parameter.copy_(lora[name])


def attention_identity():
    """Report the actual installed attention dispatch without changing libraries."""
    import diffsynth.models.wan_video_dit as model
    if model.FLASH_ATTN_3_AVAILABLE:
        backend = "flash_attn_3"
    elif model.FLASH_ATTN_2_AVAILABLE:
        backend = "flash_attn_2"
    elif model.SAGE_ATTN_AVAILABLE:
        # SageAttention is an inference kernel; preserve full backward support
        # by selecting the built-in BF16 SDPA path for this training process.
        model.SAGE_ATTN_AVAILABLE = False
        backend = "torch_sdpa"
    else:
        backend = "torch_sdpa"
    return dict(dispatch=backend, torch_version=torch.__version__,
                flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
                memory_efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
                math_sdp_enabled=torch.backends.cuda.math_sdp_enabled())


def verify_batch(pipe, examples, context):
    """Compare real two-item inference/gradients with independent forwards.

This is a training-layout correctness check, not generated-video evaluation.
Each example contains input_latents, vace_context and batched geometry tensors.
The temporary nonzero deltas prevent zero initialization from hiding a broken
Adapter gradient chain. All trainable values and Torch RNG are restored.
"""
    import gc
    if len(examples) != 2:
        raise ValueError("Batch equivalence needs exactly two independent examples")
    adapter, lora = parameter_groups(pipe)
    named = adapter + lora
    original = increments(pipe)
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all()
    old_mode = pipe.adapter_gt_v2_checkpointing["mode"]
    result = dict(passed=False, test="independent_vs_combined_real_training_batch",
                  output_rtol=.03, output_atol=.03, gradient_relative_l2_limit=.05)
    try:
        configure_checkpointing(pipe, "full")
        perturb = torch.Generator(device="cpu").manual_seed(2026111501)
        with torch.no_grad():
            pipe.structure_adapter.structure_proj.weight.copy_(
                torch.randn(pipe.structure_adapter.structure_proj.weight.shape,
                            generator=perturb).to(pipe.device) * .01)
            for name, parameter in lora:
                if ".lora_B." in name:
                    parameter.copy_(torch.randn(parameter.shape, generator=perturb).to(pipe.device) * .001)
        inputs = []
        # The two samples deliberately use distinct scalar scheduler queries;
        # the installed scheduler's scalar argmin cannot accept vector times.
        for index, example in enumerate(examples):
            clean = example["input_latents"].to(device=pipe.device, dtype=pipe.torch_dtype)
            generator = torch.Generator(device="cpu").manual_seed(2026111510 + index)
            noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(
                device=pipe.device, dtype=pipe.torch_dtype)
            timestep = pipe.scheduler.timesteps[[167 + 564 * index]].to(
                device=pipe.device, dtype=pipe.torch_dtype)
            inputs.append(dict(
                latents=pipe.scheduler.add_noise(clean, noise, timestep),
                target=pipe.scheduler.training_target(clean, noise, timestep),
                time_weight=pipe.scheduler.training_weight(timestep).to(pipe.device),
                timestep=timestep,
                vace_context=example["vace_context"].to(device=pipe.device, dtype=pipe.torch_dtype),
                geometry={key: value.to(pipe.device) for key, value in example["geometry"].items()}))

        def run(items, reference_gradients=None):
            for _, parameter in named:
                parameter.grad = None
            g = {key: torch.cat([item["geometry"][key] for item in items], dim=0)
                 for key in items[0]["geometry"]}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = pipe.structure_adapter(g)
                prediction = pipe.model_fn(
                    **{name: getattr(pipe, name) for name in pipe.in_iteration_models},
                    latents=torch.cat([item["latents"] for item in items]),
                    timestep=torch.cat([item["timestep"] for item in items]),
                    context=context.to(pipe.device).expand(len(items), -1, -1).contiguous(),
                    vace_context=torch.cat([item["vace_context"] for item in items]),
                    vace_scale=1., use_gradient_checkpointing=True,
                    use_gradient_checkpointing_offload=False,
                    structure_tokens=tokens,
                    structure_mask=torch.ones((*tokens.shape[:2], 1), device=pipe.device),
                    structure_projection=pipe.structure_adapter.structure_proj)
                target = torch.cat([item["target"] for item in items])
                mse = (prediction.float() - target.float()).square().mean((1, 2, 3, 4))
                weights = torch.stack([item["time_weight"].reshape(()) for item in items])
                # Each independent pass contributes half of the two-item mean.
                loss = (mse * weights).sum() / 2
            loss.backward()
            output = prediction.detach().float().cpu()
            gradients = {name: None if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
                         for name, parameter in named}
            torch.cuda.synchronize()
            return output, gradients

        references, reference_gradients = [], {}
        for item in inputs:
            output, gradients = run([item])
            references.append(output)
            for name, value in gradients.items():
                if value is not None:
                    reference_gradients[name] = reference_gradients.get(name, 0) + value
        actual, actual_gradients = run(inputs)
        expected = torch.cat(references)
        difference = actual - expected
        output_l2 = float(difference.double().square().sum().sqrt() /
                          expected.double().square().sum().sqrt().clamp_min(1.e-12))
        outputs_close = torch.allclose(actual, expected, rtol=.03, atol=.03)
        result.update(output_shape=list(actual.shape), output_max_absolute=float(difference.abs().max()),
                      output_relative_l2=output_l2, outputs_close=outputs_close)
        groups = {}
        for group, parameters in (("adapter", adapter), ("lora", lora)):
            squared_error, squared_reference, maximum = 0., 0., 0.
            missing, nonfinite = [], []
            for name, _ in parameters:
                expected_grad, actual_grad = reference_gradients.get(name), actual_gradients.get(name)
                if expected_grad is None or actual_grad is None:
                    missing.append(name)
                    continue
                if not bool(torch.isfinite(actual_grad).all()):
                    nonfinite.append(name)
                diff = actual_grad.double() - expected_grad.double()
                squared_error += float(diff.square().sum())
                squared_reference += float(expected_grad.double().square().sum())
                maximum = max(maximum, float(diff.abs().max()))
            relative = (squared_error / max(squared_reference, 1.e-24)) ** .5
            groups[group] = dict(relative_l2=relative, max_absolute=maximum,
                                 reference_l2=squared_reference ** .5,
                                 missing=missing, nonfinite=nonfinite)
        upstream = sum(float(value.double().square().sum()) for name, value in actual_gradients.items()
                       if name.startswith("structure_adapter.") and ".structure_proj." not in name
                       and value is not None) ** .5
        result.update(gradient_groups=groups, adapter_upstream_gradient_l2=upstream,
                      timesteps=[float(item["timestep"][0]) for item in inputs])
        result["passed"] = bool(outputs_close and output_l2 <= .03 and upstream > 0 and all(
            values["relative_l2"] <= .05 and not values["missing"] and not values["nonfinite"]
            and values["reference_l2"] > 0 for values in groups.values()))
    except torch.cuda.OutOfMemoryError as exc:
        result.update(error=str(exc), cuda_oom=True, reason="batch2_correctness_check_exceeds_available_memory")
    finally:
        load_increments_state(pipe, *original)
        configure_checkpointing(pipe, old_mode)
        for _, parameter in named:
            parameter.grad = None
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state_all(rng_cuda)
        gc.collect()
        torch.cuda.empty_cache()
    return result
