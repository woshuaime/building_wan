"""Shared configuration and validation helpers for domain LoRA experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
MODEL_FILES = (
    "diffusion_pytorch_model*.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"无法读取配置文件 {path}: {error}") from error
    if not isinstance(config, dict):
        raise ConfigError("配置文件根节点必须是 JSON 对象")
    return config, path


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"缺少配置段: {name}")
    return value

def resolve_path(value: str, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ConfigError(f"{label}不存在: {path}")


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ConfigError(f"{label}不存在: {path}")


def read_metadata(metadata_path: Path, dataset_root: Path) -> list[dict[str, str]]:
    require_file(metadata_path, "数据清单")
    rows: list[dict[str, str]] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"video", "prompt"}.issubset(reader.fieldnames):
            raise ConfigError("数据清单必须包含 video 和 prompt 两列")
        for line_number, row in enumerate(reader, start=2):
            video = (row.get("video") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if not video or not prompt:
                raise ConfigError(f"数据清单第 {line_number} 行缺少 video 或 prompt")
            video_path = (dataset_root / video).resolve()
            try:
                video_path.relative_to(dataset_root.resolve())
            except ValueError as error:
                raise ConfigError(f"视频路径越界: {video}") from error
            if not video_path.is_file():
                raise ConfigError(f"视频不存在: {video_path}")
            rows.append({"video": video, "prompt": prompt})
    if not rows:
        raise ConfigError("数据清单中没有样本")
    return rows


def validate_shape(height: int, width: int, num_frames: int) -> None:
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ConfigError("height 和 width 必须是正数且能被16整除")
    if num_frames <= 0 or (num_frames - 1) % 4:
        raise ConfigError("num_frames 必须满足 4n+1，例如17、33、49或81")


def preflight(config: dict[str, Any], *, check_checkpoints: bool = False) -> dict[str, Any]:
    runtime = section(config, "runtime")
    dataset = section(config, "dataset")
    training = section(config, "training")
    validation = section(config, "validation")

    python_path = resolve_path(str(runtime.get("python", "")))
    diffsynth_root = resolve_path(str(runtime.get("diffsynth_root", "")))
    dataset_root = resolve_path(str(dataset.get("root", "")))
    metadata_path = resolve_path(str(dataset.get("metadata", "")))
    training_output = resolve_path(str(training.get("output_dir", "")))
    validation_output = resolve_path(str(validation.get("output_dir", "")))

    require_file(python_path, "训练环境 Python")
    require_directory(diffsynth_root, "DiffSynth目录")
    require_file(
        diffsynth_root / "examples" / "wanvideo" / "model_training" / "train.py",
        "DiffSynth训练入口",
    )
    require_directory(dataset_root, "视频数据目录")
    rows = read_metadata(metadata_path, dataset_root)

    expected_count = dataset.get("expected_count")
    if expected_count is not None and len(rows) != int(expected_count):
        raise ConfigError(
            f"样本数不符: 实际 {len(rows)}，配置期望 {expected_count}"
        )

    validate_shape(
        int(training["height"]),
        int(training["width"]),
        int(training["num_frames"]),
    )
    validate_shape(
        int(validation["height"]),
        int(validation["width"]),
        int(validation["num_frames"]),
    )

    checkpoints: list[dict[str, Any]] = []
    for item in validation.get("checkpoints", []):
        if not isinstance(item, dict) or not item.get("label"):
            raise ConfigError("validation.checkpoints 中每项都必须包含 label")
        checkpoint = {"label": str(item["label"]), "path": None}
        if item.get("path"):
            checkpoint["path"] = resolve_path(str(item["path"]))
            if check_checkpoints:
                require_file(checkpoint["path"], f"权重 {checkpoint['label']}")
        checkpoints.append(checkpoint)
    if not checkpoints or checkpoints[0]["path"] is not None:
        raise ConfigError("checkpoints 第一项必须是 path=null 的基础模型")

    return {
        "python": python_path,
        "diffsynth_root": diffsynth_root,
        "dataset_root": dataset_root,
        "metadata_path": metadata_path,
        "sample_count": len(rows),
        "training_output": training_output,
        "validation_output": validation_output,
        "checkpoints": checkpoints,
    }


def model_origin_spec() -> str:
    return ",".join(f"{MODEL_ID}:{pattern}" for pattern in MODEL_FILES)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value
