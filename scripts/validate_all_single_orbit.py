"""为单轨全部 checkpoint 生成 base/LoRA 对比视频。

一次 config 里放 base 加所有 step-*.safetensors，基础模型只加载一次，
每个标签只做 clear_lora + load_lora，因此可以连续生成全部视频。

用法：
    python scripts/validate_all_single_orbit.py                    # 只写 config 并预检
    python scripts/validate_all_single_orbit.py --run              # 实际生成视频
    python scripts/validate_all_single_orbit.py --run --only base,step-11200

建议放到 tmux 里跑，并在外层用 tee 记录日志：
    tmux new -s wan-val
    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_all_single_orbit.py --run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

STEP_PATTERN = re.compile(r"step-(\d+)\.safetensors")

# 固定推理参数，与 outputs/validation/9.28 的对比保持一致。
SEED = 20260924
NUM_FRAMES = 81
HEIGHT = 384
WIDTH = 384
NUM_INFERENCE_STEPS = 30
CFG_SCALE = 5.0
NEGATIVE_PROMPT = (
    "people, trees, cars, text, watermark, cluttered background, "
    "camera shake, flicker, deformed geometry, duplicated building, "
    "cropped object, low quality, blurry"
)


def split_info(path: Path) -> tuple[str, int, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    prompts = {row.get("prompt", "").strip() for row in rows}
    if not rows or len(prompts) != 1 or not next(iter(prompts)):
        raise ValueError(f"Expected one nonempty prompt in {path}")
    videos = [row.get("video", "").strip() for row in rows]
    if any(not video for video in videos) or len(set(videos)) != len(videos):
        raise ValueError(f"Missing or duplicate videos in {path}")
    return next(iter(prompts)), len(rows), hashlib.sha256(
        "\n".join(videos).encode("utf-8")
    ).hexdigest()


def training_is_running(root: Path) -> bool:
    result = subprocess.run(
        ["ps", "-eww", "-o", "args="], capture_output=True, text=True, check=True
    )
    return any(
        "wan_train_entry.py" in line and str(root) in line
        for line in result.stdout.splitlines()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="生成视频，否则只写 config 并预检")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有视频")
    parser.add_argument("--only", help="只验证指定标签，逗号分隔，例如 base,step-11200")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/building_wan_a6000_single_orbit_all"),
        help="权重目录；默认平铺后的 building_wan_a6000_single_orbit_all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/validation_all_single_orbit"),
        help="视频输出目录",
    )
    args = parser.parse_args()

    if args.overwrite and not args.run:
        parser.error("--overwrite requires --run")

    root = Path(__file__).resolve().parents[1]
    if args.run and training_is_running(root):
        raise RuntimeError("训练仍在运行，先等它结束再生成验证视频")

    checkpoint_dir = (root / args.checkpoint_dir).resolve()
    output_dir = (root / args.output_dir).resolve()

    found = []
    for path in checkpoint_dir.glob("step-*.safetensors"):
        if match := STEP_PATTERN.fullmatch(path.name):
            found.append((int(match.group(1)), path))
    if not found:
        raise FileNotFoundError(f"没有找到权重: {checkpoint_dir}")
    found.sort()
    print(f"找到 {len(found)} 个权重: step-{found[0][0]} ~ step-{found[-1][0]}")

    split_root = root / "configs/single_orbit_same_prompt_v1"
    val_metadata = split_root / "metadata_val.csv"
    manifest = json.loads((split_root / "split_manifest.json").read_text(encoding="utf-8"))

    prompt, val_count, val_hash = split_info(val_metadata)
    if val_count != manifest["counts"]["val"] or val_hash != manifest["split_video_sha256"]["val"]:
        raise ValueError("验证集清单与固定划分不一致")

    checkpoints = [{"label": "base", "path": None}]
    checkpoints += [{"label": f"step-{step}", "path": str(path)} for step, path in found]

    config = {
        "experiment_name": output_dir.name,
        "runtime": {
            "python": sys.executable,
            "diffsynth_root": str(root / "code/DiffSynth-Studio"),
            "gpu": 0,
            "model_paths": [
                str(root / "models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
                str(root / "models/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"),
                str(root / "models/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"),
            ],
            "tokenizer_path": str(root / "models/Wan2.1-T2V-1.3B/google/umt5-xxl"),
        },
        "dataset": {
            "root": str(root / "data/single_orbit_videos"),
            "metadata": str(val_metadata),
            "expected_count": val_count,
        },
        "training": {
            "output_dir": str(checkpoint_dir),
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
        },
        "validation": {
            "output_dir": str(output_dir),
            "checkpoints": checkpoints,
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": SEED,
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "cfg_scale": CFG_SCALE,
            "fps": 18,
            "quality": 5,
            "lora_alpha": 1.0,
            "vram_reserved_gb": 2.0,
            "enable_vram_management": True,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "validation_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise FileExistsError(
                f"{config_path} 已存在且内容不同，先确认或删掉它再跑"
            )
    else:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"已写入 {config_path}")

    command = [
        sys.executable,
        str(root / "code/Python_3D_Scanner/training/validate_domain_lora.py"),
        "--config",
        str(config_path),
    ]
    if args.run:
        command.append("--run")
        labels = [item.strip() for item in (args.only or "").split(",") if item.strip()]
        if labels:
            command += ["--only", ",".join(labels)]
        if args.overwrite:
            command.append("--overwrite")
    subprocess.run(command, check=True)
    if not args.run:
        print("预检通过。加上 --run 才会生成视频。")


if __name__ == "__main__":
    main()
