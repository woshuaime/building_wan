"""Prepare and optionally run the single-orbit A6000 base/LoRA comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


EXPERIMENT = "building_wan_a6000_single_orbit"
STEP_PATTERN = re.compile(r"step-(\d+)\.safetensors")


def split_info(path: Path) -> tuple[str, int, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    prompts = {row.get("prompt", "").strip() for row in rows}
    if not rows or len(prompts) != 1 or not next(iter(prompts)):
        raise ValueError(f"Expected one nonempty prompt in {path}")
    videos = [row.get("video", "").strip() for row in rows]
    if any(not video for video in videos) or len(set(videos)) != len(videos):
        raise ValueError(f"Missing or duplicate videos in {path}")
    video_hash = hashlib.sha256("\n".join(videos).encode("utf-8")).hexdigest()
    return next(iter(prompts)), len(rows), video_hash


def latest_checkpoint(directory: Path) -> tuple[int, Path]:
    checkpoints = [
        (int(match.group(1)), path)
        for path in directory.glob("step-*.safetensors")
        if (match := STEP_PATTERN.fullmatch(path.name))
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No LoRA checkpoint found in {directory}")
    return max(checkpoints)


def training_is_running(root: Path) -> bool:
    result = subprocess.run(
        ["ps", "-eww", "-o", "args="], capture_output=True, text=True, check=True
    )
    return any(
        "wan_train_entry.py" in line and str(root) in line
        for line in result.stdout.splitlines()
    )


def pending_labels(output: Path, labels: list[str], overwrite: bool) -> list[str]:
    metadata_path = output / "validation_metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Cannot read validation metadata: {metadata_path}") from error
        completed = {
            item.get("label"): item
            for item in payload.get("results", [])
            if item.get("status") == "complete"
        }
    else:
        completed = {}

    pending = []
    for label in labels:
        video = output / f"{label}.mp4"
        result = completed.get(label)
        if result and result.get("output") == str(video) and video.is_file() and video.stat().st_size > 0:
            continue
        if video.exists() and not overwrite:
            raise FileExistsError(
                f"Unverified video exists: {video}; rerun with --run --overwrite"
            )
        pending.append(label)
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="Generate videos after the training process exits"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate only unfinished existing videos"
    )
    args = parser.parse_args()
    if args.overwrite and not args.run:
        parser.error("--overwrite requires --run")

    root = Path(__file__).resolve().parents[1]
    if args.run and training_is_running(root):
        raise RuntimeError("Training is still running; wait before generating validation videos")
    split_root = root / "configs/single_orbit_same_prompt_v1"
    train_metadata = split_root / "metadata_train.csv"
    val_metadata = split_root / "metadata_val.csv"
    manifest = json.loads((split_root / "split_manifest.json").read_text(encoding="utf-8"))
    prompt, train_count, train_hash = split_info(train_metadata)
    val_prompt, val_count, val_hash = split_info(val_metadata)
    if val_prompt != prompt or prompt != manifest["prompt"]:
        raise ValueError("Training and validation prompts differ")
    for split, count, video_hash in (
        ("train", train_count, train_hash),
        ("val", val_count, val_hash),
    ):
        if count != manifest["counts"][split] or video_hash != manifest["split_video_sha256"][split]:
            raise ValueError(f"{split} metadata differs from the fixed split manifest")

    step, checkpoint = latest_checkpoint(root / "checkpoints" / EXPERIMENT)
    output = root / "outputs/validation" / EXPERIMENT / f"step-{step}"
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "validation_config.json"

    config = {
        "experiment_name": EXPERIMENT,
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
            "output_dir": str(checkpoint.parent),
            "height": 384,
            "width": 384,
            "num_frames": 81,
        },
        "validation": {
            "output_dir": str(output),
            "checkpoints": [
                {"label": "base", "path": None},
                {"label": f"step-{step}", "path": str(checkpoint)},
            ],
            "prompt": prompt,
            "negative_prompt": (
                "people, trees, cars, text, watermark, cluttered background, "
                "camera shake, flicker, deformed geometry, duplicated building, "
                "cropped object, low quality, blurry"
            ),
            "seed": 20260924,
            "height": 384,
            "width": 384,
            "num_frames": 81,
            "num_inference_steps": 30,
            "cfg_scale": 5.0,
            "fps": 18,
            "quality": 5,
            "lora_alpha": 1.0,
            "vram_reserved_gb": 2.0,
            "enable_vram_management": True,
        },
    }

    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise FileExistsError(f"Validation config already differs: {config_path}")
    else:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    command = [
        sys.executable,
        str(root / "code/Python_3D_Scanner/training/validate_domain_lora.py"),
        "--config",
        str(config_path),
    ]
    subprocess.run(command, check=True)
    print(f"Comparing base against step-{step}; output: {output}", flush=True)
    if args.run:
        labels = ["base", f"step-{step}"]
        pending = pending_labels(output, labels, args.overwrite)
        if not pending:
            print("Both validation videos are already complete.")
            return
        run_command = [*command, "--run", "--only", ",".join(pending)]
        if args.overwrite:
            run_command.append("--overwrite")
        subprocess.run(run_command, check=True)


if __name__ == "__main__":
    main()
