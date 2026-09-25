"""Launch a reproducible DiffSynth Wan2.1 domain-LoRA training run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from .experiment_config import (
        ConfigError,
        json_ready,
        load_config,
        model_origin_spec,
        preflight,
        section,
    )
except ImportError:  # Direct script execution.
    from experiment_config import (
        ConfigError,
        json_ready,
        load_config,
        model_origin_spec,
        preflight,
        section,
    )


def training_environment(base: dict[str, str]) -> dict[str, str]:
    """Keep the single-process path aligned with the DDP safety defaults."""
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
    return environment


def build_command(config, resolved):
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
        "--height",
        str(training["height"]),
        "--width",
        str(training["width"]),
        "--num_frames",
        str(training["num_frames"]),
        "--dataset_repeat",
        str(training["dataset_repeat"]),
        "--dataset_num_workers",
        str(training["dataset_num_workers"]),
        "--model_id_with_origin_paths",
        model_origin_spec(),
        "--learning_rate",
        str(training["learning_rate"]),
        "--num_epochs",
        str(training["num_epochs"]),
        "--save_steps",
        str(training["save_steps"]),
        "--gradient_accumulation_steps",
        str(training["gradient_accumulation_steps"]),
        "--remove_prefix_in_ckpt",
        "pipe.dit.",
        "--output_path",
        str(resolved["training_output"]),
        "--lora_base_model",
        "dit",
        "--lora_target_modules",
        "q,k,v,o,ffn.0,ffn.2",
        "--lora_rank",
        str(training["lora_rank"]),
        "--initialize_model_on_cpu",
        "--enable_model_cpu_offload",
        "--enable_optimizer_cpu_offload",
        "--use_gradient_checkpointing_offload",
        "--enable_csv_log",
    ]
    return command


def write_launch_manifest(config, config_path, resolved, command):
    output_dir = resolved["training_output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "config_path": str(config_path),
        "experiment_name": config.get("experiment_name"),
        "sample_count": resolved["sample_count"],
        "gpu": section(config, "runtime").get("gpu"),
        "command": command,
        "resolved_paths": json_ready(resolved),
    }
    (output_dir / "launcher_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="实验JSON配置文件")
    parser.add_argument(
        "--run",
        action="store_true",
        help="真正启动训练；省略时只进行检查并打印命令",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="允许在已有输出目录中再次训练",
    )
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    resolved = preflight(config)
    command = build_command(config, resolved)
    print(f"配置检查通过: {config.get('experiment_name', config_path.stem)}")
    print(f"样本数量: {resolved['sample_count']}")
    print(f"训练输出: {resolved['training_output']}")
    print("启动命令:")
    print(subprocess.list2cmdline(command))

    if not args.run:
        print("DRY-RUN完成；加 --run 才会真正训练。")
        return

    output_dir = resolved["training_output"]
    existing_weights = list(output_dir.glob("*.safetensors")) if output_dir.exists() else []
    if existing_weights and not args.allow_existing:
        names = ", ".join(path.name for path in existing_weights)
        raise ConfigError(
            f"输出目录已有权重({names})，为避免覆盖已停止。"
            "确需重跑时添加 --allow-existing。"
        )

    write_launch_manifest(config, config_path, resolved, command)
    env = training_environment(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(section(config, "runtime").get("gpu", 0))
    subprocess.run(
        command,
        cwd=resolved["diffsynth_root"],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"错误: {error}") from error
