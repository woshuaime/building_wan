"""Run memory-safe two-stage Wan LoRA training with synchronized multi-GPU DDP."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from .experiment_config import (
        ConfigError,
        MODEL_FILES,
        MODEL_ID,
        json_ready,
        load_config,
        model_origin_spec,
        preflight,
        section,
    )
except ImportError:
    from experiment_config import (
        ConfigError,
        MODEL_FILES,
        MODEL_ID,
        json_ready,
        load_config,
        model_origin_spec,
        preflight,
        section,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAN_ENTRY = Path(__file__).resolve().with_name("wan_train_entry.py")


def resolve_backend(runtime: dict) -> str:
    """Use NCCL on Linux and Gloo on Windows unless explicitly overridden."""
    requested = str(runtime.get("distributed_backend", "auto")).strip().lower()
    if requested == "auto":
        return "gloo" if os.name == "nt" else "nccl"
    if requested not in {"gloo", "nccl"}:
        raise ConfigError(
            "runtime.distributed_backend must be auto, gloo, or nccl."
        )
    if requested == "nccl" and os.name == "nt":
        raise ConfigError("NCCL is not supported by the Windows launcher.")
    return requested


def training_environment(base: dict[str, str], backend: str) -> dict[str, str]:
    """Apply conservative process settings for CUDA and native runtimes."""
    environment = base.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_MODULE_LOADING": "LAZY",
            "TORCH_SHOW_CPP_STACKTRACES": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if backend == "nccl":
        environment.update(
            {
                "NCCL_ASYNC_ERROR_HANDLING": "1",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "NCCL_BLOCKING_WAIT": "1",
                "NCCL_DEBUG": "WARN",
            }
        )
    return environment


def normalize_gpu_ids(runtime: dict) -> list[int]:
    value = runtime.get("gpus", runtime.get("gpu", 0))
    values = value if isinstance(value, list) else [value]
    try:
        gpu_ids = [int(item) for item in values]
    except (TypeError, ValueError) as error:
        raise ConfigError("runtime.gpus must be a list of GPU indices.") from error
    if len(gpu_ids) < 2 or len(set(gpu_ids)) != len(gpu_ids):
        raise ConfigError("Split DDP training requires at least two unique GPUs.")
    if any(item < 0 for item in gpu_ids):
        raise ConfigError("GPU indices must be non-negative.")
    return gpu_ids


def expected_optimizer_steps(
    cached_samples: int,
    world_size: int,
    dataset_repeat: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
) -> int:
    batches_per_rank = math.ceil(cached_samples / world_size)
    updates_per_epoch = math.ceil(
        batches_per_rank * dataset_repeat / gradient_accumulation_steps
    )
    return updates_per_epoch * num_epochs


def _origin_spec(index: int) -> str:
    return f"{MODEL_ID}:{MODEL_FILES[index]}"


def _common_training_arguments(config, resolved) -> list[str]:
    training = section(config, "training")
    return [
        "--height",
        str(training["height"]),
        "--width",
        str(training["width"]),
        "--num_frames",
        str(training["num_frames"]),
        "--dataset_num_workers",
        str(training["dataset_num_workers"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--remove_prefix_in_ckpt",
        "pipe.dit.",
        "--lora_base_model",
        "dit",
        "--lora_target_modules",
        "q,k,v,o,ffn.0,ffn.2",
        "--lora_rank",
        str(training["lora_rank"]),
    ]


def build_cache_command(config, resolved) -> list[str]:
    runtime = section(config, "runtime")
    training = section(config, "training")
    train_script = (
        resolved["diffsynth_root"]
        / "examples"
        / "wanvideo"
        / "model_training"
        / "train.py"
    )
    command = [
        str(resolved["python"]),
        "-u",
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        "--dynamo_backend",
        "no",
        str(train_script),
        "--dataset_base_path",
        str(resolved["dataset_root"]),
        "--dataset_metadata_path",
        str(resolved["metadata_path"]),
        "--dataset_repeat",
        "1",
        "--model_id_with_origin_paths",
        model_origin_spec(),
        "--num_epochs",
        "1",
        "--output_path",
        str(resolved["cache_dir"]),
        "--offload_models",
        _origin_spec(0),
        "--task",
        "sft:data_process",
    ]
    command.extend(_common_training_arguments(config, resolved))
    return command


def build_train_command(config, resolved) -> list[str]:
    runtime = section(config, "runtime")
    training = section(config, "training")
    gpu_ids = normalize_gpu_ids(runtime)
    backend = resolve_backend(runtime)
    model_paths = local_model_paths(resolved)
    if os.name == "nt":
        launcher = [
            str(resolved["python"]),
            "-u",
            str(WAN_ENTRY),
            "--num-processes",
            str(len(gpu_ids)),
        ]
    else:
        launcher = [
            str(resolved["python"]),
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(len(gpu_ids)),
            str(WAN_ENTRY),
        ]
    command = launcher + [
        "--diffsynth-root",
        str(resolved["diffsynth_root"]),
        "--backend",
        backend,
        "--dataset_base_path",
        str(resolved["cache_dir"]),
        "--dataset_repeat",
        str(training["dataset_repeat"]),
        "--model_paths",
        json.dumps([str(path) for path in model_paths]),
        "--tokenizer_path",
        str(local_tokenizer_path(resolved)),
        "--num_epochs",
        str(training["num_epochs"]),
        "--save_steps",
        str(training["save_steps"]),
        "--gradient_accumulation_steps",
        str(training["gradient_accumulation_steps"]),
        "--output_path",
        str(resolved["training_output"]),
        "--offload_models",
        f"{model_paths[1]},{model_paths[2]}",
        "--task",
        "sft:train",
        "--use_gradient_checkpointing_offload",
        "--enable_csv_log",
    ]
    command.extend(_common_training_arguments(config, resolved))
    return command


def local_model_paths(resolved) -> tuple[Path, Path, Path]:
    root = resolved["diffsynth_root"] / "models"
    paths = (
        root
        / "Wan-AI"
        / "Wan2.1-T2V-1.3B"
        / "diffusion_pytorch_model.safetensors",
        root
        / "DiffSynth-Studio"
        / "Wan-Series-Converted-Safetensors"
        / "models_t5_umt5-xxl-enc-bf16.safetensors",
        root
        / "DiffSynth-Studio"
        / "Wan-Series-Converted-Safetensors"
        / "Wan2.1_VAE.safetensors",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ConfigError(f"Local model files are missing: {missing}")
    return paths


def local_tokenizer_path(resolved) -> Path:
    path = (
        resolved["diffsynth_root"]
        / "models"
        / "Wan-AI"
        / "Wan2.1-T2V-1.3B"
        / "google"
        / "umt5-xxl"
    )
    if not (path / "tokenizer.json").is_file():
        raise ConfigError(f"Local tokenizer is missing: {path}")
    return path


def resolve_split_paths(config, resolved):
    training = section(config, "training")
    cache_value = training.get("cache_dir")
    if not cache_value:
        raise ConfigError("training.cache_dir is required for split DDP training.")
    cache_dir = Path(str(cache_value)).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    resolved["cache_dir"] = cache_dir.resolve()
    return resolved


def write_manifest(config, config_path, resolved, commands):
    runtime = section(config, "runtime")
    training = section(config, "training")
    output = resolved["training_output"]
    output.mkdir(parents=True, exist_ok=True)
    cache_count = len(list(resolved["cache_dir"].rglob("*.pth")))
    gpu_ids = normalize_gpu_ids(runtime)
    payload = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "config_path": str(config_path),
        "experiment_name": config.get("experiment_name"),
        "strategy": "split_cache_ddp",
        "gpus": gpu_ids,
        "distributed_backend": resolve_backend(runtime),
        "source_sample_count": resolved["sample_count"],
        "cached_sample_count": cache_count,
        "expected_optimizer_steps": expected_optimizer_steps(
            cache_count,
            len(gpu_ids),
            int(training["dataset_repeat"]),
            int(training["num_epochs"]),
            int(training["gradient_accumulation_steps"]),
        )
        if cache_count
        else None,
        "commands": commands,
        "resolved_paths": json_ready(resolved),
    }
    (output / "launcher_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run(command, cwd: Path, env: dict):
    print(subprocess.list2cmdline(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("all", "cache", "train"), default="all")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    resolved = resolve_split_paths(config, preflight(config))
    runtime = section(config, "runtime")
    backend = resolve_backend(runtime)
    training = section(config, "training")
    gpu_ids = normalize_gpu_ids(runtime)
    commands = {
        "cache": build_cache_command(config, resolved),
        "train": build_train_command(config, resolved),
    }
    print(f"Configuration OK: {config.get('experiment_name', config_path.stem)}")
    print(f"Source samples: {resolved['sample_count']}")
    print(f"Synchronized GPUs: {gpu_ids}")
    print(f"Distributed backend: {backend}")
    print(f"Cache: {resolved['cache_dir']}")
    print(f"Training output: {resolved['training_output']}")
    for stage in ("cache", "train"):
        if args.stage in ("all", stage):
            print(f"{stage} command:")
            print(subprocess.list2cmdline(commands[stage]))
    if not args.run:
        print("DRY-RUN complete; add --run to execute.")
        return

    env = training_environment(os.environ, backend)
    if args.stage in ("all", "cache"):
        cached = list(resolved["cache_dir"].rglob("*.pth"))
        if len(cached) == resolved["sample_count"]:
            print(f"Cache already complete ({len(cached)} files); skipping cache stage.")
        elif cached and not args.allow_existing:
            raise ConfigError(
                f"Cache is partial ({len(cached)}/{resolved['sample_count']} files); "
                "use --allow-existing to resume it."
            )
        else:
            cache_env = env.copy()
            cache_env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
            _run(commands["cache"], resolved["diffsynth_root"], cache_env)

    cached = list(resolved["cache_dir"].rglob("*.pth"))
    if args.stage in ("all", "train"):
        if len(cached) != resolved["sample_count"]:
            raise ConfigError(
                f"Cache is incomplete ({len(cached)}/{resolved['sample_count']} files); "
                "refusing to start training."
            )
        existing = list(resolved["training_output"].glob("*.safetensors"))
        if existing and not args.allow_existing:
            raise ConfigError(
                "Training output already contains checkpoints; use --allow-existing."
            )
        train_env = env.copy()
        train_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        train_env["DIFFSYNTH_ROOT"] = str(resolved["diffsynth_root"])
        train_env["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
        cooldown_seconds = float(training.get("step_cooldown_seconds", 0.0))
        if cooldown_seconds < 0:
            raise ConfigError("training.step_cooldown_seconds must be non-negative.")
        train_env["WAN_STEP_COOLDOWN_SECONDS"] = str(cooldown_seconds)
        _run(commands["train"], PROJECT_ROOT, train_env)

    write_manifest(config, config_path, resolved, commands)


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Error: {error}") from error
