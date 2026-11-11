"""Small shared utilities for the six-call reference-guidance experiment."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
TRAIN_SOURCE = ROOT / "experiments/pointcloud_vace/adapter_d/scripts"
for directory in (TRAIN_SOURCE, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
PYTHON = "/home/shi/miniconda3/envs/scj/bin/python"
CASE = ROOT / "data/pointcloud_vace/微调任务/cases/wall_13697"
TRAIN_RUN = ROOT / "outputs/pointcloud_vace/adapter_d/run_20261110_050724"
OLD_PREDICTION = ROOT / "outputs/pointcloud_vace/adapter_d_eval/run_20261110_150749/cases/wall_13697/raw_vace.mp4"


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def event(run, stage, **extra):
    record = dict(time=now(), stage=stage, pid=os.getpid(), **extra)
    write(Path(run) / "status.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def circular_indices(start):
    if not 0 <= start < 80:
        raise ValueError("The orbit start must be in 0..79")
    return [(start + i) % 80 for i in range(17)]


def decode(path, width=832, height=480):
    import numpy as np
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-vsync", "0", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def observed_case():
    # Reuse the existing point filtering, camera normalization and source parser.
    path = ROOT / "experiments/pointcloud_vace/adapter_d_eval/scripts/common.py"
    spec = importlib.util.spec_from_file_location("d_eval_observed_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.observed_points(CASE)


def output_path(run, group, seed):
    return Path(run) / f"group_{group}" / f"seed_{seed}.mp4"


def result_path(run, group, seed):
    return Path(run) / "logs" / f"{group}_{seed}.done.json"
