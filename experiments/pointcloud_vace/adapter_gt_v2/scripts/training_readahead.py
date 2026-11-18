"""Compare bounded filesystem read-ahead during the running job, using completed formal steps."""
import fcntl
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
import time

import data
from common import now, sha, write_json
from fullframe_cache import entry

LOOKAHEAD = 4
BLOCK = 32
COUNT = 25


def samples(run, end):
    rows = [json.loads(line) for line in (run / "loss.jsonl").read_text().splitlines()]
    rows = [r for r in rows if end - COUNT <= r["samples_consumed"] <= end]
    if [r["samples_consumed"] for r in rows] != list(range(end - COUNT, end + 1)):
        raise ValueError("Non-contiguous formal sample history")
    wall = [b["elapsed_training_seconds"] - a["elapsed_training_seconds"] for a, b in zip(rows, rows[1:])]
    return dict(first=end-COUNT+1, last=end, mean=mean(wall), median=median(wall), seconds=wall)


def warm(path, buffer):
    with path.open("rb", buffering=0) as handle:
        while handle.readinto(buffer):
            pass


def run_reader(run):
    cache = json.loads((run / "cache_complete.json").read_text())
    if sha(run / "schedule.json") != cache["schedule_sha256"]:
        raise ValueError("Changed training schedule")
    rows = json.loads((run / "schedule.json").read_text())
    latents = Path(cache["cache_dir"])
    geometry = data.directories(run)[1] / "geometry"
    output = run / "training_speed" / "readahead.json"
    decision = json.loads(output.read_text()) if output.exists() else {}
    if decision.get("retained") is False:
        return
    buffer = bytearray(256 * 1024)
    start = None
    next_index = None
    last_cursor = None
    while True:
        if (run / "training_complete.json").exists() or (run / "failure.json").exists():
            return
        state = json.loads((run / "progress.json").read_text())
        if state.get("stage") != "training" or (run / "stop.request").exists():
            next_index = start = last_cursor = None
            time.sleep(2)
            continue
        cursor = state["samples_consumed"]
        if last_cursor is not None and cursor < last_cursor:
            start = next_index = None
        last_cursor = cursor
        if start is None:
            start = cursor
        if next_index is None:
            next_index = cursor + 1
        if not decision.get("retained") and cursor >= start + 2*BLOCK:
            on, off = samples(run, start + BLOCK), samples(run, start + 2*BLOCK)
            keep = on["mean"] < .95 * off["mean"] and on["median"] < .95 * off["median"]
            decision = dict(time=now(), retained=keep, ahead=LOOKAHEAD, enabled=on, disabled=off,
                            source_sha256=sha(Path(__file__)), schedule_sha256=cache["schedule_sha256"])
            write_json(output, decision)
            if not keep:
                return
        enabled = decision.get("retained") or cursor < start + BLOCK
        if enabled:
            for i in range(max(next_index, cursor+1), min(cursor+LOOKAHEAD+1, len(rows))):
                row = rows[i]
                key = data.cache_key(run, dict(row, frame_ids=[]))
                warm(entry(latents, row), buffer)
                warm(geometry / key[:2] / (key + ".npz"), buffer)
                next_index = i+1
        time.sleep(.3)


if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve()
    with (run / "training_speed" / "readahead.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_reader(run)
