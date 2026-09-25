"""Run the long formal cache/train/validation pipeline with resumable stage checks."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"D:\scj\envs\diffsynth\python.exe")
TRAIN_RUNNER = PROJECT_ROOT / "training" / "train_split_domain_lora.py"
VALIDATOR = PROJECT_ROOT / "training" / "validate_domain_lora.py"
ANALYZER = PROJECT_ROOT / "training" / "analyze_validation_videos.py"
SUMMARIZER = PROJECT_ROOT / "training" / "summarize_run.py"

CACHE_ROOT = Path(
    r"D:\code\DiffSynth-Studio\models\cache\building_train_1912_all_256_combined"
)
PRESERVED_CACHE = CACHE_ROOT / "half0"
HALF0_REMAINING_CACHE = CACHE_ROOT / "half0_remaining"
HALF1_CACHE = CACHE_ROOT / "half1"
TRAIN_OUTPUT = Path(
    r"D:\code\DiffSynth-Studio\models\train\building_train_1912_all_256_cooled_lora"
)
VALIDATION_OUTPUT = (
    PROJECT_ROOT / "outputs" / "validation" / "building_train_1912_all_256_cooled"
)

HALF0_REMAINING_CONFIG = (
    PROJECT_ROOT / "training" / "configs" / "cache_4726_half0_remaining_256.json"
)
HALF1_CONFIG = PROJECT_ROOT / "training" / "configs" / "cache_4780_half1_256.json"
FORMAL_CONFIG = PROJECT_ROOT / "training" / "configs" / "train_1912_all_256_cooled.json"

STATUS_PATH = PROJECT_ROOT / "outputs" / "formal_pipeline_status.json"
TELEMETRY_PATH = PROJECT_ROOT / "outputs" / "formal_gpu_telemetry.csv"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pth_count(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.pth")) if path.exists() else 0


def write_status(stage: str, state: str, **extra) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": now(), "stage": stage, "state": state, **extra}
    STATUS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"STATUS {stage}: {state} {extra}", flush=True)


def telemetry_loop(stage: str, stop: threading.Event) -> None:
    new_file = not TELEMETRY_PATH.exists()
    with TELEMETRY_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(
                [
                    "timestamp",
                    "stage",
                    "gpu",
                    "temperature_c",
                    "utilization_pct",
                    "memory_mib",
                    "power_w",
                ]
            )
        while not stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            stamp = now()
            for line in result.stdout.splitlines():
                values = [item.strip() for item in line.split(",")]
                if len(values) == 5:
                    writer.writerow([stamp, stage, *values])
            handle.flush()
            stop.wait(10.0)


def run_stage(stage: str, command: list[str]) -> None:
    write_status(stage, "running", command=subprocess.list2cmdline(command))
    stop = threading.Event()
    thread = threading.Thread(target=telemetry_loop, args=(stage, stop), daemon=True)
    thread.start()
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    finally:
        stop.set()
        thread.join(timeout=15)
    write_status(stage, "completed")


def cache_command(config: Path, *, allow_existing: bool = False) -> list[str]:
    command = [
        str(PYTHON),
        str(TRAIN_RUNNER),
        "--config",
        str(config),
        "--stage",
        "cache",
        "--run",
    ]
    if allow_existing:
        command.append("--allow-existing")
    return command


def loss_rows() -> int:
    path = TRAIN_OUTPUT / "loss.csv"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    try:
        preserved = pth_count(PRESERVED_CACHE)
        if preserved != 54:
            raise RuntimeError(f"Expected 54 preserved cache files, found {preserved}.")

        remaining = pth_count(HALF0_REMAINING_CACHE)
        if remaining != 4726:
            run_stage(
                "cache_half0_remaining",
                cache_command(
                    HALF0_REMAINING_CONFIG,
                    allow_existing=bool(remaining),
                ),
            )
        if pth_count(HALF0_REMAINING_CACHE) != 4726:
            raise RuntimeError("half0_remaining cache count is not 4726 after caching.")

        half1 = pth_count(HALF1_CACHE)
        if half1 != 4780:
            run_stage(
                "cache_half1",
                cache_command(HALF1_CONFIG, allow_existing=bool(half1)),
            )
        if pth_count(HALF1_CACHE) != 4780:
            raise RuntimeError("half1 cache count is not 4780 after caching.")

        total_cache = pth_count(CACHE_ROOT)
        if total_cache != 9560:
            raise RuntimeError(f"Expected 9560 total cache files, found {total_cache}.")
        write_status("cache_complete", "completed", total_cache=total_cache)

        final_checkpoint = TRAIN_OUTPUT / "step-4780.safetensors"
        rows = loss_rows()
        if not (final_checkpoint.is_file() and rows == 4780):
            if any(TRAIN_OUTPUT.glob("*.safetensors")) or rows:
                raise RuntimeError(
                    f"Formal training output is partial (loss rows={rows}); manual audit required."
                )
            run_stage(
                "formal_train",
                [
                    str(PYTHON),
                    str(TRAIN_RUNNER),
                    "--config",
                    str(FORMAL_CONFIG),
                    "--stage",
                    "train",
                    "--run",
                ],
            )
        if not final_checkpoint.is_file() or loss_rows() != 4780:
            raise RuntimeError("Formal training did not produce 4780 rows and final checkpoint.")

        expected_videos = {
            "base.mp4",
            "step-500.mp4",
            "step-1000.mp4",
            "step-2000.mp4",
            "step-3000.mp4",
            "step-4000.mp4",
            "step-4780.mp4",
        }
        present_videos = (
            {item.name for item in VALIDATION_OUTPUT.glob("*.mp4")}
            if VALIDATION_OUTPUT.exists()
            else set()
        )
        if not expected_videos.issubset(present_videos):
            if present_videos:
                raise RuntimeError(
                    f"Formal validation is partial ({len(present_videos)}/7); manual audit required."
                )
            run_stage(
                "formal_validation",
                [str(PYTHON), str(VALIDATOR), "--config", str(FORMAL_CONFIG), "--run"],
            )

        run_stage(
            "formal_analysis",
            [str(PYTHON), str(ANALYZER), str(VALIDATION_OUTPUT)],
        )
        run_stage(
            "formal_summary",
            [str(PYTHON), str(SUMMARIZER), "--config", str(FORMAL_CONFIG)],
        )
        write_status(
            "pipeline",
            "completed",
            total_cache=pth_count(CACHE_ROOT),
            loss_rows=loss_rows(),
            validation_videos=len(list(VALIDATION_OUTPUT.glob("*.mp4"))),
        )
        return 0
    except Exception as error:
        write_status("pipeline", "failed", error=f"{type(error).__name__}: {error}")
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
