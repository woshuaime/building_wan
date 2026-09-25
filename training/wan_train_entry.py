"""Run DiffSynth Wan training with a Windows-compatible DDP launcher."""

from __future__ import annotations

import argparse
import faulthandler
import os
import socket
import sys
import time
import traceback
from pathlib import Path

for _name, _value in {
    "PYTHONUNBUFFERED": "1",
    "PYTHONFAULTHANDLER": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUDA_MODULE_LOADING": "LAZY",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
}.items():
    os.environ.setdefault(_name, _value)
try:
    faulthandler.enable(all_threads=True)
except (RuntimeError, OSError):
    pass

import torch
import torch.distributed as dist


def _configure_runtime_environment(backend: str) -> None:
    """Keep native runtimes from creating an unbounded number of threads."""
    defaults = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    if backend == "nccl":
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("NCCL_DEBUG", "WARN")
    try:
        faulthandler.enable(all_threads=True)
    except (RuntimeError, OSError):
        pass


def resolve_backend(requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value == "auto":
        return "gloo" if sys.platform == "win32" else "nccl"
    if value not in {"gloo", "nccl"}:
        raise ValueError(f"Unsupported distributed backend: {requested}")
    if value == "nccl" and sys.platform == "win32":
        raise ValueError("NCCL is not supported by the Windows training launcher")
    return value


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


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
    model = components["WanTrainingModule"](
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
    rank: int | None = None,
):
    _configure_runtime_environment(backend)
    rank = int(os.environ.get("RANK", local_rank) if rank is None else rank)
    os.environ.update(
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(local_rank),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": str(os.environ.get("MASTER_ADDR", "127.0.0.1")),
            "MASTER_PORT": str(os.environ.get("MASTER_PORT", "29500")),
        }
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan training.")
    visible_devices = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= visible_devices:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for {visible_devices} visible CUDA devices."
        )
    torch.cuda.set_device(local_rank)
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method or "env://",
            rank=rank,
            world_size=world_size,
        )
        initialized_here = True
    try:
        _run_training(Path(diffsynth_root), training_arguments, backend)
        if initialized_here:
            dist.barrier()
    except BaseException:
        print(
            f"[rank {local_rank}] training process failed:\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffsynth-root", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    launcher_args, training_arguments = parser.parse_known_args()
    backend = resolve_backend(launcher_args.backend)

    # torchrun owns process creation on Linux. Avoid nesting a second spawn
    # tree around CUDA, which is a common source of native crashes.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        _worker(
            local_rank,
            world_size,
            backend,
            "env://",
            str(launcher_args.diffsynth_root.resolve()),
            training_arguments,
            rank=rank,
        )
        return

    if launcher_args.num_processes < 2:
        raise SystemExit("This entry point is intended for at least two GPUs.")
    if torch.cuda.device_count() < launcher_args.num_processes:
        raise SystemExit(
            f"Requested {launcher_args.num_processes} processes but only "
            f"{torch.cuda.device_count()} CUDA devices are visible."
        )

    if sys.platform != "win32":
        raise SystemExit(
            "Linux requires torchrun; invoke training through "
            "train_split_domain_lora.py or torchrun directly."
        )

    # Windows keeps the spawn launcher, but uses a fresh TCP rendezvous
    # instead of a reused file path that can be left behind after a crash.
    master_port = _free_local_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.multiprocessing.spawn(
        _worker,
        args=(
            launcher_args.num_processes,
            backend,
            f"tcp://127.0.0.1:{master_port}",
            str(launcher_args.diffsynth_root.resolve()),
            training_arguments,
        ),
        nprocs=launcher_args.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
