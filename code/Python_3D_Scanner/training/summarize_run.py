"""Summarize training loss, checkpoints, settings, and validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone

try:
    from .experiment_config import (
        ConfigError,
        json_ready,
        load_config,
        preflight,
        resolve_path,
        section,
    )
except ImportError:  # Direct script execution.
    from experiment_config import (
        ConfigError,
        json_ready,
        load_config,
        preflight,
        resolve_path,
        section,
    )


def load_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_losses(path):
    if not path.is_file():
        return []
    values = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("key") != "loss":
                continue
            value = float(row["value"])
            if not math.isfinite(value):
                raise ConfigError(f"loss中出现非有限值: {value}")
            values.append({"step": int(row["step"]), "value": value})
    return values


def loss_summary(values):
    if not values:
        return {"count": 0}
    losses = [item["value"] for item in values]
    window = min(5, len(losses))
    return {
        "count": len(losses),
        "first": losses[0],
        "last": losses[-1],
        "minimum": min(losses),
        "maximum": max(losses),
        "mean": sum(losses) / len(losses),
        "first_window_mean": sum(losses[:window]) / window,
        "last_window_mean": sum(losses[-window:]) / window,
        "window_size": window,
    }


def probe_validation_videos(validation_metadata):
    if not validation_metadata:
        return []
    try:
        import imageio
    except ImportError as error:
        return [{"valid": False, "error": f"无法导入imageio: {error}"}]

    expected_size = (
        int(validation_metadata["width"]),
        int(validation_metadata["height"]),
    )
    expected_frames = int(validation_metadata["num_frames"])
    expected_fps = float(validation_metadata.get("fps", 18.0))
    probes = []
    for result in validation_metadata.get("results", []):
        path = result.get("output")
        probe = {"label": result.get("label"), "path": path, "valid": False}
        try:
            reader = imageio.get_reader(str(path))
            try:
                metadata = reader.get_meta_data()
                decoded_frames = sum(1 for _ in reader)
            finally:
                reader.close()
            size = tuple(metadata.get("size", (None, None)))
            fps = float(metadata.get("fps", 0.0))
            probe.update(
                {
                    "codec": metadata.get("codec"),
                    "width": size[0],
                    "height": size[1],
                    "fps": fps,
                    "decoded_frames": decoded_frames,
                    "valid": (
                        size == expected_size
                        and decoded_frames == expected_frames
                        and abs(fps - expected_fps) <= 0.05
                    ),
                }
            )
        except Exception as error:
            probe["error"] = repr(error)
        probes.append(probe)
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="实验JSON配置文件")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    resolved = preflight(config, check_checkpoints=True)
    training_dir = resolved["training_output"]
    validation_dir = resolved["validation_output"]
    losses = load_losses(training_dir / "loss.csv")
    checkpoints = [
        {
            "name": path.name,
            "path": path,
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(training_dir.glob("*.safetensors"))
    ]
    validation_metadata = load_json(validation_dir / "validation_metadata.json")
    video_probes = probe_validation_videos(validation_metadata)
    expected_labels = {item["label"] for item in resolved["checkpoints"]}
    completed_labels = {
        item.get("label")
        for item in (validation_metadata or {}).get("results", [])
        if item.get("status") == "complete"
    }
    valid_probe_labels = {
        item.get("label") for item in video_probes if item.get("valid")
    }
    review_value = section(config, "validation").get("review_path")
    manual_review = load_json(resolve_path(review_value)) if review_value else None
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_name": config.get("experiment_name"),
        "config_path": config_path,
        "sample_count": resolved["sample_count"],
        "training_args": load_json(training_dir / "training_args.json"),
        "loss": loss_summary(losses),
        "checkpoints": checkpoints,
        "validation": validation_metadata,
        "validation_video_probes": video_probes,
        "manual_review": manual_review,
        "status": {
            "training_artifacts_present": bool(checkpoints and losses),
            "validation_complete": bool(
                validation_metadata
                and completed_labels == expected_labels
                and valid_probe_labels == expected_labels
            ),
        },
    }
    validation_dir.mkdir(parents=True, exist_ok=True)
    summary_path = validation_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"实验: {payload['experiment_name']}")
    print(f"样本: {payload['sample_count']}")
    print(f"loss步数: {payload['loss']['count']}")
    print(f"checkpoint: {len(checkpoints)}")
    print(f"验证完成: {payload['status']['validation_complete']}")
    print(f"汇总文件: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as error:
        raise SystemExit(f"错误: {error}") from error
