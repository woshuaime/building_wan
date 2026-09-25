"""Audit every artifact required by the long-running experiment goal."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = Path(r"D:\code\DiffSynth-Studio")
OUTPUT = PROJECT_ROOT / "outputs" / "experiment_completion_audit.json"


def csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def file_count(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.rglob(pattern)) if path.exists() else 0


def check(name: str, actual, expected, path: Path | None = None) -> dict:
    return {
        "name": name,
        "status": "complete" if actual == expected else "pending",
        "actual": actual,
        "expected": expected,
        "path": str(path) if path else None,
    }


def exists_check(name: str, path: Path) -> dict:
    return check(name, path.is_file(), True, path)


def main() -> int:
    split_root = PROJECT_ROOT / "training" / "data_splits" / "building_v1"
    train_clips = PROJECT_ROOT / "outputs" / "temporal_clips_train_all"
    val_clips = PROJECT_ROOT / "outputs" / "temporal_clips_val_all"
    test_clips = PROJECT_ROOT / "outputs" / "temporal_clips_test_all"
    formal_cache = (
        DIFFSYNTH_ROOT / "models" / "cache" / "building_train_1912_all_256_combined"
    )
    formal_train = (
        DIFFSYNTH_ROOT / "models" / "train" / "building_train_1912_all_256_cooled_lora"
    )
    formal_validation = (
        PROJECT_ROOT / "outputs" / "validation" / "building_train_1912_all_256_cooled"
    )
    balanced_train = (
        DIFFSYNTH_ROOT / "models" / "train" / "building_train_500_balanced_384_cooled_lora"
    )
    balanced_validation = (
        PROJECT_ROOT / "outputs" / "validation" / "building_train_500_balanced_384_cooled"
    )

    checks = [
        check("split_train_buildings", csv_rows(split_root / "metadata_train.csv"), 1912, split_root / "metadata_train.csv"),
        check("split_val_buildings", csv_rows(split_root / "metadata_val.csv"), 239, split_root / "metadata_val.csv"),
        check("split_test_buildings", csv_rows(split_root / "metadata_test.csv"), 239, split_root / "metadata_test.csv"),
        check("train_temporal_clips", csv_rows(train_clips / "metadata.csv"), 9560, train_clips / "metadata.csv"),
        check("val_temporal_clips", csv_rows(val_clips / "metadata.csv"), 1195, val_clips / "metadata.csv"),
        check("test_temporal_clips", csv_rows(test_clips / "metadata.csv"), 1195, test_clips / "metadata.csv"),
        exists_check(
            "dual_smoke_256_checkpoint",
            DIFFSYNTH_ROOT / "models" / "train" / "building_dual_3060_smoke_lossavg_lora" / "step-3.safetensors",
        ),
        check(
            "dual_smoke_256_loss_rows",
            csv_rows(DIFFSYNTH_ROOT / "models" / "train" / "building_dual_3060_smoke_lossavg_lora" / "loss.csv"),
            3,
        ),
        exists_check(
            "dual_smoke_384_checkpoint",
            DIFFSYNTH_ROOT / "models" / "train" / "building_dual_3060_smoke_384_lora" / "step-3.safetensors",
        ),
        check(
            "dual_smoke_384_loss_rows",
            csv_rows(DIFFSYNTH_ROOT / "models" / "train" / "building_dual_3060_smoke_384_lora" / "loss.csv"),
            3,
        ),
        check("balanced_500_loss_rows", csv_rows(balanced_train / "loss.csv"), 250, balanced_train / "loss.csv"),
        check("balanced_500_checkpoints", file_count(balanced_train, "step-*.safetensors"), 5, balanced_train),
        check("balanced_500_validation_videos", file_count(balanced_validation, "*.mp4"), 6, balanced_validation),
        exists_check("balanced_500_diagnostics", balanced_validation / "video_diagnostics.json"),
        check("formal_cache_files", file_count(formal_cache, "*.pth"), 9560, formal_cache),
        check("formal_loss_rows", csv_rows(formal_train / "loss.csv"), 4780, formal_train / "loss.csv"),
    ]

    for step in (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4780):
        checks.append(
            exists_check(
                f"formal_checkpoint_step_{step}",
                formal_train / f"step-{step}.safetensors",
            )
        )
    checks.extend(
        [
            check("formal_validation_videos", file_count(formal_validation, "*.mp4"), 7, formal_validation),
            exists_check("formal_validation_diagnostics", formal_validation / "video_diagnostics.json"),
            exists_check("formal_checkpoint_selection", formal_validation / "checkpoint_selection.json"),
            exists_check("loss_curve_asset", PROJECT_ROOT / "outputs" / "report_assets" / "loss_curves.png"),
            exists_check("gpu_telemetry_summary", PROJECT_ROOT / "outputs" / "report_assets" / "gpu_telemetry_summary.json"),
        ]
    )

    reports = list((PROJECT_ROOT / "reports").glob("Wan2.1建筑视频LoRA双卡实验报告_*.docx")) if (PROJECT_ROOT / "reports").exists() else []
    renders = list((PROJECT_ROOT / "outputs" / "report_render").glob("page-*.png")) if (PROJECT_ROOT / "outputs" / "report_render").exists() else []
    checks.append(check("final_word_report", len(reports), 1, reports[0] if len(reports) == 1 else PROJECT_ROOT / "reports"))
    checks.append(
        {
            "name": "word_render_pages",
            "status": "complete" if renders else "pending",
            "actual": len(renders),
            "expected": ">=1",
            "path": str(PROJECT_ROOT / "outputs" / "report_render"),
        }
    )

    complete = sum(item["status"] == "complete" for item in checks)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "complete": complete == len(checks),
        "summary": {"complete": complete, "total": len(checks), "pending": len(checks) - complete},
        "checks": checks,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0 if payload["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
