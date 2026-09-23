"""Generate fair base/LoRA comparison videos with fixed inference settings."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .experiment_config import (
        ConfigError,
        MODEL_FILES,
        MODEL_ID,
        json_ready,
        load_config,
        preflight,
        section,
    )
except ImportError:  # Direct script execution.
    from experiment_config import (
        ConfigError,
        MODEL_FILES,
        MODEL_ID,
        json_ready,
        load_config,
        preflight,
        section,
    )


def safe_label(label):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    if not value:
        raise ConfigError(f"无效的验证标签: {label!r}")
    return value


def write_metadata(path, payload):
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def probe_video(path, expected_width, expected_height, expected_frames, expected_fps):
    import imageio

    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        decoded_frames = sum(1 for _ in reader)
    finally:
        reader.close()
    width, height = metadata.get("size", (None, None))
    fps = float(metadata.get("fps", 0.0))
    if (width, height) != (expected_width, expected_height):
        raise RuntimeError(
            f"视频尺寸不符: {(width, height)} != {(expected_width, expected_height)}"
        )
    if decoded_frames != expected_frames:
        raise RuntimeError(f"视频帧数不符: {decoded_frames} != {expected_frames}")
    if abs(fps - expected_fps) > 0.05:
        raise RuntimeError(f"视频FPS不符: {fps} != {expected_fps}")
    return {
        "codec": metadata.get("codec"),
        "width": width,
        "height": height,
        "fps": fps,
        "decoded_frames": decoded_frames,
        "duration_seconds": metadata.get("duration"),
    }


def launch_worker(config_path, config, resolved, selected, overwrite):
    command = [
        str(resolved["python"]),
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--worker",
    ]
    if selected:
        command.extend(["--only", ",".join(selected)])
    if overwrite:
        command.append("--overwrite")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(section(config, "runtime").get("gpu", 0))
    env["PYTHONUNBUFFERED"] = "1"
    print("验证启动命令:", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, env=env, check=True)


def run_worker(config, config_path, resolved, selected, overwrite):
    validation = section(config, "validation")
    output_dir = resolved["validation_output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "validation_metadata.json"

    available = {item["label"]: item for item in resolved["checkpoints"]}
    labels = selected or list(available)
    unknown = [label for label in labels if label not in available]
    if unknown:
        raise ConfigError(f"未知验证标签: {', '.join(unknown)}")
    for label in labels:
        output_path = output_dir / f"{safe_label(label)}.mp4"
        if output_path.exists() and not overwrite:
            raise ConfigError(f"验证结果已存在: {output_path}；覆盖时添加 --overwrite")

    os.chdir(resolved["diffsynth_root"])
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(
        resolved["diffsynth_root"] / "models"
    )
    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video

    if not torch.cuda.is_available():
        raise ConfigError("CUDA不可用，无法执行视频验证")
    reserve = float(validation.get("vram_reserved_gb", 2.0))
    total_vram = torch.cuda.mem_get_info("cuda")[1] / (1024**3)
    vram_limit = total_vram - reserve
    if vram_limit <= 0:
        raise ConfigError(f"显存预留值过大: 总显存 {total_vram:.2f} GB")

    enable_vram_management = bool(
        validation.get("enable_vram_management", True)
    )
    vram_config = (
        {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        if enable_vram_management
        else {}
    )
    if resolved["model_paths"] is None:
        model_configs = [
            ModelConfig(
                model_id=MODEL_ID,
                origin_file_pattern=pattern,
                **vram_config,
            )
            for pattern in MODEL_FILES
        ]
    else:
        model_configs = [
            ModelConfig(path=str(path), **vram_config)
            for path in resolved["model_paths"]
        ]
    tokenizer_config = (
        ModelConfig(
            model_id=MODEL_ID,
            origin_file_pattern="google/umt5-xxl/",
        )
        if resolved["tokenizer_path"] is None
        else ModelConfig(path=str(resolved["tokenizer_path"]))
    )
    print(f"加载基础模型，显存管理上限: {vram_limit:.2f} GB")
    pipeline_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device": "cuda",
        "model_configs": model_configs,
        "tokenizer_config": tokenizer_config,
    }
    if enable_vram_management:
        pipeline_kwargs["vram_limit"] = vram_limit
    pipe = WanVideoPipeline.from_pretrained(
        **pipeline_kwargs,
    )

    existing_payload = None
    if metadata_path.is_file():
        try:
            existing_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_payload = None
    preserved_results = []
    if existing_payload:
        preserved_results = [
            item
            for item in existing_payload.get("results", [])
            if item.get("label") not in labels
        ]

    payload = {
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "config_path": str(config_path),
        "experiment_name": config.get("experiment_name"),
        "device": torch.cuda.get_device_name(0),
        "visible_gpu_from_config": section(config, "runtime").get("gpu", 0),
        "prompt": validation["prompt"],
        "negative_prompt": validation["negative_prompt"],
        "seed": int(validation["seed"]),
        "height": int(validation["height"]),
        "width": int(validation["width"]),
        "num_frames": int(validation["num_frames"]),
        "num_inference_steps": int(validation["num_inference_steps"]),
        "cfg_scale": float(validation["cfg_scale"]),
        "fps": float(validation["fps"]),
        "quality": int(validation["quality"]),
        "lora_alpha": float(validation.get("lora_alpha", 1.0)),
        "enable_vram_management": enable_vram_management,
        "results": preserved_results,
    }
    write_metadata(metadata_path, payload)

    for label in labels:
        item = available[label]
        pipe.clear_lora(verbose=0)
        checkpoint = item["path"]
        if checkpoint is not None:
            pipe.load_lora(
                pipe.dit,
                str(checkpoint),
                alpha=float(validation.get("lora_alpha", 1.0)),
            )
        output_path = output_dir / f"{safe_label(label)}.mp4"
        print(f"生成 {label}: {output_path}")
        started = time.perf_counter()
        try:
            video = pipe(
                prompt=str(validation["prompt"]),
                negative_prompt=str(validation["negative_prompt"]),
                seed=int(validation["seed"]),
                height=int(validation["height"]),
                width=int(validation["width"]),
                num_frames=int(validation["num_frames"]),
                num_inference_steps=int(validation["num_inference_steps"]),
                cfg_scale=float(validation["cfg_scale"]),
                tiled=True,
            )
            save_video(
                video,
                str(output_path),
                fps=float(validation["fps"]),
                quality=int(validation["quality"]),
            )
            video_probe = probe_video(
                output_path,
                int(validation["width"]),
                int(validation["height"]),
                int(validation["num_frames"]),
                float(validation["fps"]),
            )
            result = {
                "label": label,
                "checkpoint": checkpoint,
                "output": output_path,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "status": "complete",
                "video_probe": video_probe,
            }
        except Exception as error:
            result = {
                "label": label,
                "checkpoint": checkpoint,
                "output": output_path,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "status": "failed",
                "error": repr(error),
            }
            payload["results"].append(result)
            order = {name: index for index, name in enumerate(available)}
            payload["results"].sort(
                key=lambda entry: order.get(entry.get("label"), len(order))
            )
            write_metadata(metadata_path, payload)
            raise
        payload["results"].append(result)
        order = {name: index for index, name in enumerate(available)}
        payload["results"].sort(
            key=lambda entry: order.get(entry.get("label"), len(order))
        )
        write_metadata(metadata_path, payload)
        torch.cuda.empty_cache()

    result_by_label = {item.get("label"): item for item in payload["results"]}
    if all(
        result_by_label.get(label, {}).get("status") == "complete"
        for label in available
    ):
        payload["finished_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    write_metadata(metadata_path, payload)
    print(f"验证完成: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="实验JSON配置文件")
    parser.add_argument("--run", action="store_true", help="真正生成验证视频")
    parser.add_argument("--only", help="只验证指定标签，逗号分隔")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有验证视频")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    resolved = preflight(config, check_checkpoints=True)
    selected = [item.strip() for item in args.only.split(",") if item.strip()] if args.only else []
    labels = selected or [item["label"] for item in resolved["checkpoints"]]
    print(f"配置检查通过，将比较: {', '.join(labels)}")
    print(f"验证输出: {resolved['validation_output']}")

    if args.worker:
        run_worker(config, config_path, resolved, selected, args.overwrite)
    elif args.run:
        launch_worker(config_path, config, resolved, selected, args.overwrite)
    else:
        print("DRY-RUN完成；加 --run 才会加载模型并生成视频。")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"错误: {error}") from error
