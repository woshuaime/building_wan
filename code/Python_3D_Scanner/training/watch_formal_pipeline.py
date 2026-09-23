"""Watch the detached formal pipeline and retry recoverable exits."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"D:\scj\envs\diffsynth\python.exe")
PIPELINE = PROJECT_ROOT / "training" / "run_formal_pipeline.py"
STATUS = PROJECT_ROOT / "outputs" / "formal_pipeline_status.json"
WATCH_STATUS = PROJECT_ROOT / "outputs" / "formal_watchdog_status.json"
STDOUT = PROJECT_ROOT / "outputs" / "formal_pipeline_stdout.log"
STDERR = PROJECT_ROOT / "outputs" / "formal_pipeline_stderr.log"
INITIAL_PID = 25304
MAX_RETRIES = 2


def process_exists(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return str(pid) in result.stdout


def pipeline_completed() -> bool:
    if not STATUS.is_file():
        return False
    try:
        payload = json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("stage") == "pipeline" and payload.get("state") == "completed"


def write_status(state: str, **extra) -> None:
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "state": state,
        **extra,
    }
    WATCH_STATUS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    pid = INITIAL_PID
    retries = 0
    write_status("watching", pid=pid, retries=retries)
    while True:
        while process_exists(pid):
            time.sleep(30)
        if pipeline_completed():
            write_status("completed", pid=pid, retries=retries)
            return 0
        if retries >= MAX_RETRIES:
            write_status("retry_limit_reached", pid=pid, retries=retries)
            return 1
        retries += 1
        time.sleep(20)
        with STDOUT.open("a", encoding="utf-8") as stdout, STDERR.open(
            "a", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                [str(PYTHON), "-u", str(PIPELINE)],
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
            )
        pid = process.pid
        write_status("restarted", pid=pid, retries=retries)


if __name__ == "__main__":
    raise SystemExit(main())
