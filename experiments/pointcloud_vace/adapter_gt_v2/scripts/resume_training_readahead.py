"""Keep the selected CPU file reader available across a full-frame training resume."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from common import PYTHON, ROOT
from run import session_name


def attach(run, resume):
    state = run / "training_speed"
    if not json.loads((state / "readahead.json").read_text())["retained"]:
        raise ValueError("Filesystem read-ahead was not selected for this run")
    if (run / "training_complete.json").exists():
        return
    script = Path(__file__).resolve()
    command = [PYTHON, "-B", str(script.with_name("run_training_speed.py")), "launch", str(run)]
    if resume:
        subprocess.run(command, cwd=ROOT, check=True)
    target = run / "resume.sh"
    temp = target.with_name(f"resume.{os.getpid()}.tmp")
    temp.write_text("#!/usr/bin/env bash\nexec " + shlex.join([PYTHON, "-B", str(script), "resume", str(run)]) + "\n")
    temp.chmod(0o755)
    os.replace(temp, target)
    session = session_name(run)
    with (state / "readahead.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)
    command = ["env", "CUDA_VISIBLE_DEVICES=", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
               PYTHON, "-B", "-u", str(script.with_name("training_readahead.py")), str(run)]
    shell = shlex.join(command) + " >> " + shlex.quote(str(state / "readahead.log")) + " 2>&1"
    subprocess.run(["tmux", "new-window", "-d", "-t", session, "-n", "read-ahead", "-c", str(ROOT), shell], check=True)


if __name__ == "__main__":
    mode, run = sys.argv[1:]
    if mode not in ("resume", "attach"):
        raise SystemExit("Expected resume or attach")
    attach(Path(run).resolve(), mode == "resume")
