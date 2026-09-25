"""Record a compact formal-pipeline snapshot every 30 minutes."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(
    r"D:\code\DiffSynth-Studio\models\cache\building_train_1912_all_256_combined"
)
TRAIN_OUTPUT = Path(
    r"D:\code\DiffSynth-Studio\models\train\building_train_1912_all_256_cooled_lora"
)
VALIDATION_OUTPUT = (
    PROJECT_ROOT / "outputs" / "validation" / "building_train_1912_all_256_cooled"
)
PIPELINE_STATUS = PROJECT_ROOT / "outputs" / "formal_pipeline_status.json"
LATEST_PATH = PROJECT_ROOT / "outputs" / "formal_30min_monitor_latest.json"
HISTORY_PATH = PROJECT_ROOT / "outputs" / "formal_30min_monitor.jsonl"
INTERVAL_SECONDS = 30 * 60


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def loss_rows() -> int:
    path = TRAIN_OUTPUT / "loss.csv"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def gpu_snapshot() -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 5:
            rows.append(
                {
                    "gpu": int(values[0]),
                    "utilization_pct": float(values[1]),
                    "memory_mib": float(values[2]),
                    "temperature_c": float(values[3]),
                    "power_w": float(values[4]),
                }
            )
    return rows


def read_pipeline_status() -> dict:
    if not PIPELINE_STATUS.is_file():
        return {"stage": "unknown", "state": "missing"}
    return json.loads(PIPELINE_STATUS.read_text(encoding="utf-8"))


def take_snapshot() -> dict:
    status = read_pipeline_status()
    return {
        "timestamp": now(),
        "pipeline_stage": status.get("stage"),
        "pipeline_state": status.get("state"),
        "cache_files": sum(1 for _ in CACHE_ROOT.rglob("*.pth")),
        "loss_rows": loss_rows(),
        "checkpoints": len(list(TRAIN_OUTPUT.glob("step-*.safetensors"))),
        "validation_videos": len(list(VALIDATION_OUTPUT.glob("*.mp4"))),
        "gpus": gpu_snapshot(),
    }


def main() -> None:
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    while True:
        snapshot = take_snapshot()
        encoded = json.dumps(snapshot, ensure_ascii=False)
        LATEST_PATH.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with HISTORY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        print(encoded, flush=True)
        if (
            snapshot["pipeline_stage"] == "pipeline"
            and snapshot["pipeline_state"] in {"completed", "failed"}
        ):
            return
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
