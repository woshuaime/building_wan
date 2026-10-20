"""VACE-branch LoRA training for the two fixed roof-completion clips."""

import csv
import random
import time
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure_lora(pipe, settings):
    import torch
    from diffsynth.diffusion import DiffusionTrainingModule

    require(settings["lora_base_model"] == "vace", "Only VACE LoRA is supported")
    require(settings["lora_alpha"] == settings["lora_rank"],
            "Saved DiffSynth deltas require alpha/rank = 1")
    require(settings["lora_dropout"] == 0 and settings["lora_bias"] == "none",
            "Unexpected LoRA dropout or bias")
    require(not settings["lora_use_rslora"] and not settings["lora_use_dora"]
            and settings["lora_init_lora_weights"] is True, "Expected standard zero-delta LoRA initialization")
    require(settings["lora_parameter_dtype"] == "float32"
            and settings["backbone_dtype"] == "bfloat16", "Unexpected training precision")
    require(not pipe.vram_management_enabled, "Inference offload cannot be used during backward")
    targets = settings["lora_target_modules"]
    linear_names = [name for name, module in pipe.vace.named_modules()
                    if isinstance(module, torch.nn.Linear)]
    for target in targets:
        require(any(name == target or name.endswith("." + target) for name in linear_names),
                f"VACE LoRA target did not match: {target}")
    trainer = DiffusionTrainingModule()
    trainer.switch_pipe_to_training_mode(
        pipe, lora_base_model="vace", lora_target_modules=",".join(targets),
        lora_rank=settings["lora_rank"], task="sft",
    )
    adapter = pipe.vace.peft_config["default"]
    for attribute in ("lora_alpha", "lora_dropout", "bias", "use_rslora", "use_dora", "init_lora_weights"):
        key = attribute if attribute.startswith("lora_") else "lora_" + attribute
        require(getattr(adapter, attribute) == settings[key], f"PEFT default differs from config: {attribute}")
    pipe.scheduler.set_timesteps(settings["scheduler_training_steps"], training=True)
    pipe.vace.train()
    for parameter in pipe.vace.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
    named_parameters = trainable_lora_parameters(pipe)
    print(f"VACE LoRA: {len(named_parameters)} tensors / "
          f"{sum(parameter.numel() for _, parameter in named_parameters):,} parameters; "
          f"rank={settings['lora_rank']}, alpha={settings['lora_alpha']}, "
          "FP32 parameters, BF16 autocast; DiT frozen with backward enabled", flush=True)
    return named_parameters


def trainable_lora_parameters(pipe):
    named_parameters = [(name, parameter) for name, parameter in pipe.named_parameters()
                        if parameter.requires_grad]
    require(named_parameters, "No trainable VACE LoRA parameters")
    require(all(name.startswith("vace.") and
                (name.endswith(".lora_A.default.weight") or name.endswith(".lora_B.default.weight"))
                for name, _ in named_parameters), "Trainable parameter outside VACE LoRA")
    return named_parameters


def make_optimizer(pipe, settings):
    import torch

    parameters = [parameter for _, parameter in trainable_lora_parameters(pipe)]
    optimizer = torch.optim.AdamW(
        parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        betas=tuple(settings["adam_betas"]), eps=settings["adam_eps"],
    )
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    require(len(optimizer_parameters) == len(parameters)
            and {id(parameter) for parameter in optimizer_parameters} == {id(parameter) for parameter in parameters},
            "Optimizer must contain exactly the trainable VACE LoRA parameters")
    return optimizer


def encode_sample(pipe, sample, settings, context, vace_scale):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoUnit_InputVideoEmbedder, WanVideoUnit_VACE

    require(len(sample["gt"]) == len(sample["conditioning"]) == len(sample["model_mask"]),
            "Target, Damaged conditions and masks have different frame counts")
    with torch.no_grad():
        target = WanVideoUnit_InputVideoEmbedder().process(
            pipe, input_video=sample["gt"], noise=None, tiled=settings["tiled"],
            tile_size=tuple(settings["tile_size"]), tile_stride=tuple(settings["tile_stride"]),
            vace_reference_image=None, framewise_decoding=False,
            animate2_reference_image=None, animate2_reference_video=None,
        )
        condition = WanVideoUnit_VACE().process(
            pipe, vace_video=sample["conditioning"], vace_video_mask=sample["model_mask"],
            vace_reference_image=None, vace_scale=vace_scale,
            height=sample["conditioning"][0].height, width=sample["conditioning"][0].width,
            num_frames=len(sample["conditioning"]), tiled=settings["tiled"],
            tile_size=tuple(settings["tile_size"]), tile_stride=tuple(settings["tile_stride"]),
        )
    require(condition["vace_context"] is not None, "VACE conditioning was not constructed")
    require(condition["vace_context"].shape[1] == 96, "Expected VACE 32 latent + 64 mask channels")
    require(target["input_latents"].shape[2:] == condition["vace_context"].shape[2:],
            "GT target and VACE condition latent dimensions are not aligned")
    return dict(
        input_latents=target["input_latents"], context=context,
        vace_context=condition["vace_context"], vace_scale=condition["vace_scale"],
        use_gradient_checkpointing=settings["gradient_checkpointing"],
        use_gradient_checkpointing_offload=settings["gradient_checkpointing_offload"],
        min_timestep_boundary=settings["min_timestep_boundary"],
        max_timestep_boundary=settings["max_timestep_boundary"],
        cfg_scale=1, cfg_merge=False,
    )


def prepare_frozen_inputs(pipe, samples, config):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder

    print("Training preparation: encoding the shared positive text with frozen T5", flush=True)
    with torch.no_grad():
        context = WanVideoUnit_PromptEmbedder().process(
            pipe, prompt=config["inference"]["prompt"], positive=True,
        )["context"]
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    cached = []
    for sample in samples:
        print(f"Training preparation: {sample['pilot']} GT target / Damaged + ModelMask conditions",
              flush=True)
        cached.append((sample["pilot"], encode_sample(
            pipe, sample, config["training"], context, config["inference"]["vace_scale"],
        )))
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    print("Frozen T5/VAE tensors cached in memory; T5/VAE on CPU; "
          "no trainable VACE output cached", flush=True)
    return cached


def save_lora(pipe, output_dir, update):
    from safetensors.torch import save

    state = {name.removeprefix("vace."): parameter.detach().cpu().contiguous()
             for name, parameter in trainable_lora_parameters(pipe)}
    destination = Path(output_dir) / f"lora_update{update:04d}.safetensors"
    payload = save(state, metadata=dict(base_model="vace", optimizer_update=str(update),
                                       checkpoint_type="lora_delta_not_resume_state"))
    with destination.open("xb") as handle:
        handle.write(payload)
    print(f"Saved VACE LoRA update {update}: {destination}", flush=True)
    return destination


def train_updates(pipe, cached, settings, output_dir, on_update=None):
    import torch
    from diffsynth.diffusion import FlowMatchSFTLoss
    from tqdm import tqdm

    optimizer = make_optimizer(pipe, settings)
    named_parameters = trainable_lora_parameters(pipe)
    parameters = [parameter for _, parameter in named_parameters]
    actual_updates = 0
    parameters_changed = False
    gradient_check_passed = False
    recent_losses = []
    started = time.monotonic()
    device_type = torch.device(pipe.device).type
    try:
        with (Path(output_dir) / "loss.csv").open("x", newline="", encoding="utf-8") as loss_handle:
            writer = csv.DictWriter(loss_handle, fieldnames=["update", "sample", "loss", "lr"])
            writer.writeheader()
            loss_handle.flush()
            with tqdm(total=settings["max_updates"], desc="VACE LoRA training", unit="update",
                      dynamic_ncols=True) as progress:
                for update in range(1, settings["max_updates"] + 1):
                    pilot, inputs = cached[(update - 1) % len(cached)]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        loss = FlowMatchSFTLoss(pipe, **inputs)
                    require(loss.ndim == 0 and bool(torch.isfinite(loss)), f"Non-finite loss at update {update}")
                    require(loss.requires_grad, f"Gradient chain disconnected at update {update}")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, max_norm=settings["max_grad_norm"], error_if_nonfinite=True,
                    )
                    early_limit = min(5, settings["max_updates"])
                    early = update <= early_limit
                    watched_parameter = None
                    if early and float(gradient_norm) > 0:
                        watched_parameter = next(parameter for parameter in parameters
                                                 if parameter.grad is not None and bool(torch.any(parameter.grad != 0)))
                        previous_value = watched_parameter.detach().clone()
                    optimizer.step()
                    actual_updates = update
                    if watched_parameter is not None:
                        parameters_changed |= not torch.equal(watched_parameter.detach(), previous_value)
                        gradient_check_passed = True
                    loss_value = float(loss.detach())
                    learning_rate = optimizer.param_groups[0]["lr"]
                    writer.writerow(dict(update=update, sample=pilot, loss=loss_value, lr=learning_rate))
                    loss_handle.flush()
                    if early or update % 10 == 0 or update in settings["save_updates"]:
                        require(bool(torch.stack([torch.isfinite(parameter).all() for parameter in parameters]).all()),
                                f"Non-finite LoRA parameter after optimizer update {update}")
                    if update == early_limit:
                        require(gradient_check_passed and parameters_changed,
                                f"No effective LoRA gradient/weight change within the first {early_limit} updates")
                    if update in settings["save_updates"]:
                        save_lora(pipe, output_dir, update)
                    if on_update is not None:
                        on_update(update, loss_value, pilot)
                    recent_losses.append(loss_value)
                    progress.set_postfix(sample=pilot, loss=f"{loss_value:.6f}", lr=f"{learning_rate:.1e}",
                                         refresh=False)
                    progress.update(1)
                    if early:
                        progress.write(f"Update {update}: finite loss; gradient norm={float(gradient_norm):.6g}; "
                                       f"effective gradient seen={gradient_check_passed}; "
                                       f"LoRA changed={parameters_changed}; optimizer contains only VACE LoRA")
                    if update % 10 == 0:
                        progress.write(f"Update {update}/{settings['max_updates']}: mean loss "
                                       f"{sum(recent_losses) / len(recent_losses):.6f}; "
                                       f"elapsed {time.monotonic() - started:.1f}s")
                        recent_losses.clear()
    except BaseException:
        print(f"TRAINING FAILED: actual optimizer updates={actual_updates}; "
              "existing LoRA files retained; no retry", flush=True)
        raise
    return dict(actual_updates=actual_updates, gradient_check_passed=gradient_check_passed,
                parameters_changed=parameters_changed)


def run_training(config, samples, output_dir, model_paths, on_update=None):
    import numpy as np
    import torch
    from vace_p01 import load_local_vace

    settings = config["training"]
    require([sample["pilot"] for sample in samples] == ["P04", "P06"],
            "Training order must be P04 then P06")
    require(settings["max_updates"] == 300 and settings["save_updates"] == [100, 300],
            "Only the requested 300-update budget and 100/300 checkpoints are supported")
    require(settings["batch_size"] == settings["gradient_accumulation_steps"] == 1
            and settings["num_workers"] == 0, "Expected batch 1, accumulation 1, workers 0")
    require(settings["lr_schedule"] == "constant" and settings["gradient_checkpointing"],
            "Expected constant learning rate and gradient checkpointing")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.cuda.reset_peak_memory_stats()
    print("TRAINING: fresh VACE LoRA; 300 actual AdamW updates; P04/P06 alternating; "
          "GT supervision only", flush=True)
    pipe = load_local_vace(model_paths, memory_mode="full")
    configure_lora(pipe, settings)
    cached = prepare_frozen_inputs(pipe, samples, config)
    result = train_updates(pipe, cached, settings, output_dir, on_update=on_update)
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print("TRAINING UPDATES FINISHED (awaiting clean worker exit): "
          f"actual optimizer updates={result['actual_updates']}; "
          f"peak allocated={result['peak_allocated_bytes'] / 1024**3:.2f} GiB; "
          f"peak reserved={result['peak_reserved_bytes'] / 1024**3:.2f} GiB", flush=True)
    return result
