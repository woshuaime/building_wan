"""Run DiffSynth Wan training on one GPU or with multi-GPU DDP."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist


_COMPACT_SFT_SHARED_KEYS = (
    "input_latents",
    "use_gradient_checkpointing",
    "use_gradient_checkpointing_offload",
    "cfg_merge",
    "vace_scale",
    "max_timestep_boundary",
    "min_timestep_boundary",
)

_SHARED_TEXT_CONTEXT_FORMAT = "shared-text-context-bf16-v1"


def _compact_sft_cache(inputs, include_text_context=True):
    """Keep only values consumed by Wan FlowMatchSFTLoss and its DiT call."""
    if not isinstance(inputs, tuple) or len(inputs) != 3:
        raise TypeError("Expected Wan cache output to be a three-item tuple.")

    inputs_shared, inputs_posi, _inputs_nega = inputs
    if "input_latents" not in inputs_shared:
        raise KeyError("Wan cache output is missing input_latents.")
    if "context" not in inputs_posi:
        raise KeyError("Wan cache output is missing positive text context.")

    compact_shared = {
        key: inputs_shared[key]
        for key in _COMPACT_SFT_SHARED_KEYS
        if key in inputs_shared
    }
    compact_posi = (
        {"context": inputs_posi["context"]} if include_text_context else {}
    )

    # Training transfers floating cache tensors to the pipeline BF16 dtype before
    # using them. Storing that final dtype now avoids retaining duplicate FP32 data.
    for mapping in (compact_shared, compact_posi):
        for key, value in mapping.items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    value = value.to(dtype=torch.bfloat16)
                mapping[key] = value.detach().cpu().contiguous()

    # FlowMatchSFTLoss does not consume the negative-conditioning dictionary.
    return compact_shared, compact_posi, {}


def _bf16_cpu_context(inputs):
    """Extract the positive text context in its final cached representation."""
    if not isinstance(inputs, tuple) or len(inputs) != 3:
        raise TypeError("Expected Wan cache output to be a three-item tuple.")
    _inputs_shared, inputs_posi, _inputs_nega = inputs
    if "context" not in inputs_posi:
        raise KeyError("Wan cache output is missing positive text context.")
    context = inputs_posi["context"]
    if not isinstance(context, torch.Tensor):
        raise TypeError("Wan positive text context must be a tensor.")
    if context.is_floating_point():
        context = context.to(dtype=torch.bfloat16)
    return context.detach().cpu().contiguous()


def _load_shared_text_context(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid shared text context payload: {path}")
    if payload.get("format") != _SHARED_TEXT_CONTEXT_FORMAT:
        raise ValueError(
            f"Unsupported shared text context format in {path}: "
            f"{payload.get('format')!r}"
        )
    context = payload.get("context")
    if not isinstance(context, torch.Tensor):
        raise TypeError(f"Shared text context is not a tensor: {path}")
    if context.dtype != torch.bfloat16 or context.device.type != "cpu":
        raise ValueError(
            f"Shared text context must be CPU BF16, found "
            f"{context.device.type} {context.dtype}: {path}"
        )
    return context.contiguous()


def _save_shared_text_context(path: Path, context: torch.Tensor):
    """Atomically create or validate the single context shared by all samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _load_shared_text_context(path)
        if not torch.equal(existing, context):
            raise ValueError(
                "Cached samples do not share one text context; refusing to "
                f"reuse {path}."
            )
        return existing

    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(
            {"format": _SHARED_TEXT_CONTEXT_FORMAT, "context": context},
            temporary_path,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return context


class _SharedTextContextCacheLoader:
    """Load a latent-only sample and restore its shared positive context."""

    def __init__(self, shared_text_context_path: Path):
        self.context = _load_shared_text_context(shared_text_context_path)

    def __call__(self, data):
        inputs = torch.load(data, map_location="cpu", weights_only=False)
        if not isinstance(inputs, tuple) or len(inputs) != 3:
            raise TypeError(f"Invalid compact SFT cache sample: {data}")
        inputs_shared, inputs_posi, inputs_nega = inputs
        if not isinstance(inputs_posi, dict):
            raise TypeError(f"Invalid positive input mapping in cache sample: {data}")
        if "context" in inputs_posi:
            raise ValueError(
                f"Cache sample unexpectedly duplicates shared context: {data}"
            )
        inputs_posi = dict(inputs_posi)
        inputs_posi["context"] = self.context
        return inputs_shared, inputs_posi, inputs_nega


def _load_training_components(diffsynth_root: Path):
    root = str(diffsynth_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    import accelerate
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        InitProcessGroupKwargs,
    )
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import (
        ImageCropAndResize,
        LoadAudio,
        LoadVideo,
        ToAbsolutePath,
    )
    from diffsynth.diffusion import (
        ModelLogger,
        launch_data_process_task,
        launch_training_task,
    )
    from examples.wanvideo.model_training.train import WanTrainingModule, wan_parser

    return {
        "accelerate": accelerate,
        "DDPKwargs": DistributedDataParallelKwargs,
        "InitKwargs": InitProcessGroupKwargs,
        "UnifiedDataset": UnifiedDataset,
        "ImageCropAndResize": ImageCropAndResize,
        "LoadAudio": LoadAudio,
        "LoadVideo": LoadVideo,
        "ToAbsolutePath": ToAbsolutePath,
        "ModelLogger": ModelLogger,
        "launch_data_process_task": launch_data_process_task,
        "launch_training_task": launch_training_task,
        "WanTrainingModule": WanTrainingModule,
        "wan_parser": wan_parser,
    }


def _run_training(
    diffsynth_root: Path,
    training_arguments: list[str],
    backend: str,
    compact_sft_cache: bool = False,
    shared_text_context_path: Path | None = None,
):
    components = _load_training_components(diffsynth_root)
    args = components["wan_parser"]().parse_args(training_arguments)
    accelerator = components["accelerate"].Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            components["DDPKwargs"](
                find_unused_parameters=args.find_unused_parameters
            ),
            components["InitKwargs"](backend=backend),
        ],
    )
    UnifiedDataset = components["UnifiedDataset"]
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": components["ToAbsolutePath"](
                args.dataset_base_path
            )
            >> components["LoadVideo"](
                args.num_frames,
                4,
                1,
                frame_processor=components["ImageCropAndResize"](
                    512, 512, None, 16, 16
                ),
            ),
            "input_audio": components["ToAbsolutePath"](args.dataset_base_path)
            >> components["LoadAudio"](sr=16000),
            "wantodance_music_path": components["ToAbsolutePath"](
                args.dataset_base_path
            ),
        },
    )
    if compact_sft_cache:
        prompts = {item.get("prompt") for item in dataset.data}
        if (
            len(prompts) != 1
            or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
        ):
            raise ValueError(
                "Shared text context caching requires exactly one non-empty "
                "prompt value across the metadata."
            )
    if shared_text_context_path is not None and args.task in ("sft", "sft:train"):
        if not dataset.load_from_cache:
            raise ValueError(
                "--shared-text-context-path training requires a cached dataset."
            )
        dataset.cached_data_operator = _SharedTextContextCacheLoader(
            shared_text_context_path
        )
    base_model_class = components["WanTrainingModule"]

    class ProjectWanTrainingModule(base_model_class):
        def forward(self, data, inputs=None):
            outputs = super().forward(data, inputs=inputs)
            if compact_sft_cache:
                if self._project_shared_text_context is None:
                    context = _bf16_cpu_context(outputs)
                    self._project_shared_text_context = _save_shared_text_context(
                        shared_text_context_path,
                        context,
                    )
                return _compact_sft_cache(outputs, include_text_context=False)
            return outputs

    if compact_sft_cache and args.task != "sft:data_process":
        raise ValueError(
            "--compact-sft-cache is only valid with --task sft:data_process."
        )
    if compact_sft_cache and shared_text_context_path is None:
        raise ValueError(
            "--compact-sft-cache requires --shared-text-context-path."
        )

    model = ProjectWanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device=(
            "cpu"
            if args.initialize_model_on_cpu or args.enable_model_cpu_offload
            else accelerator.device
        ),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model._project_shared_text_context = None

    class DistributedMeanModelLogger(components["ModelLogger"]):
        def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
            loss = kwargs.get("loss")
            if isinstance(loss, torch.Tensor) and accelerator.num_processes > 1:
                gathered = accelerator.gather(loss.detach().float().reshape(1))
                kwargs["loss"] = gathered.mean()
            super().on_step_end(
                accelerator,
                model,
                save_steps,
                **kwargs,
            )
            cooldown_seconds = float(
                os.environ.get("WAN_STEP_COOLDOWN_SECONDS", "0")
            )
            if cooldown_seconds > 0:
                time.sleep(cooldown_seconds)

    model_logger = DistributedMeanModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
    )
    launcher_map = {
        "sft:data_process": components["launch_data_process_task"],
        "direct_distill:data_process": components["launch_data_process_task"],
        "sft": components["launch_training_task"],
        "sft:train": components["launch_training_task"],
        "direct_distill": components["launch_training_task"],
        "direct_distill:train": components["launch_training_task"],
    }
    launcher_map[args.task](
        accelerator,
        dataset,
        model,
        model_logger,
        args=args,
    )


def _worker(
    local_rank: int,
    world_size: int,
    backend: str,
    init_method: str,
    diffsynth_root: str,
    training_arguments: list[str],
    compact_sft_cache: bool,
    shared_text_context_path: str | None,
):
    os.environ.update(
        {
            "RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(local_rank),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=local_rank,
        world_size=world_size,
    )
    try:
        _run_training(
            Path(diffsynth_root),
            training_arguments,
            backend,
            compact_sft_cache,
            None if shared_text_context_path is None else Path(shared_text_context_path),
        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffsynth-root", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--compact-sft-cache", action="store_true")
    parser.add_argument("--shared-text-context-path", type=Path)
    launcher_args, training_arguments = parser.parse_known_args()
    if launcher_args.num_processes < 1:
        raise SystemExit("--num-processes must be at least 1.")
    if torch.cuda.device_count() < launcher_args.num_processes:
        raise SystemExit(
            f"Requested {launcher_args.num_processes} processes but only "
            f"{torch.cuda.device_count()} CUDA devices are visible."
        )

    if launcher_args.num_processes == 1:
        _run_training(
            launcher_args.diffsynth_root,
            training_arguments,
            launcher_args.backend,
            launcher_args.compact_sft_cache,
            launcher_args.shared_text_context_path,
        )
        return

    rendezvous = (
        Path(tempfile.gettempdir())
        / f"python_3d_scanner_wan_ddp_{uuid.uuid4().hex}"
    )
    rendezvous.unlink(missing_ok=True)
    try:
        torch.multiprocessing.spawn(
            _worker,
            args=(
                launcher_args.num_processes,
                launcher_args.backend,
                rendezvous.as_uri(),
                str(launcher_args.diffsynth_root.resolve()),
                training_arguments,
                launcher_args.compact_sft_cache,
                (
                    None
                    if launcher_args.shared_text_context_path is None
                    else str(launcher_args.shared_text_context_path.resolve())
                ),
            ),
            nprocs=launcher_args.num_processes,
            join=True,
        )
    finally:
        rendezvous.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
