"""Run-specific tmux launch, orderly stop, resume and read-only status."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from common import ROOT, PYTHON, now, write_json, runtime_env


def session_name(run):
    return "adapter_gt_v2_" + run.name.removeprefix("run_")


def running(run):
    with (run / "run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def launch(run, resume=False):
    if not (run / "config.json").exists():
        raise ValueError(f"Missing config.json in {run}")
    if (run / "training_complete.json").exists():
        print(json.dumps(dict(run=str(run), status="completed", restarted=False), ensure_ascii=False))
        return
    if running(run):
        print(json.dumps(dict(run=str(run), status="already_running", tmux_session=session_name(run)), ensure_ascii=False))
        return
    if (run / "run.pid").exists() and not resume:
        raise ValueError("This run was launched before; use its explicit resume command")
    if resume:
        (run / "stop.request").unlink(missing_ok=True)
        # Preserve the old exit/failure evidence instead of allowing stale files
        # to masquerade as this invocation's outcome.
        stamp = str(time.time_ns())
        for name in ("exit.json", "failure.json", "stopped.json"):
            path = run / name
            if path.exists():
                (run / "invocation_history").mkdir(exist_ok=True)
                path.rename(run / "invocation_history" / f"{stamp}_{name}")
    session = session_name(run)
    command = [str(PYTHON), "-u", "-B", str(Path(__file__).with_name("pipeline.py")), str(run)]
    if resume:
        command.append("--resume")
    try:
        env = runtime_env(run)
    except TypeError:
        env = runtime_env()
    # Bash receives arguments through shlex.join.  No user-controlled text is
    # interpolated as executable shell syntax, and there is no manual hold gate.
    # A pre-existing tmux server retains its original environment.  Put all
    # experiment variables in the command itself rather than relying on the
    # environment of the tmux client to update that server.
    names = ("TMPDIR", "TMP", "TEMP", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME",
             "CUDA_CACHE_PATH", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "TRITON_CACHE_DIR",
             "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES",
             "DIFFSYNTH_SKIP_DOWNLOAD", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
             "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    explicit_command = ["env"] + [name + "=" + env[name] for name in names if name in env] + command
    shell = ("set -o pipefail\n" + shlex.join(explicit_command) + " 2>&1 | tee -a " +
             shlex.quote(str(run / "train.log")) +
             "\nresult=${PIPESTATUS[0]}\nprintf '\\nPipeline exited with code %s. See progress.json.\\n' \"$result\"\n"
             "exec bash --noprofile --norc")
    shell_command = shlex.join(["bash", "--noprofile", "--norc", "-c", shell])
    exists = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
    if exists:
        subprocess.run(["tmux", "new-window", "-t", session, "-n", "resume", "-c", str(ROOT), shell_command],
                       env=env, check=True)
    else:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "training", "-c", str(ROOT), shell_command],
                       env=env, check=True)
    # Initial launch receipt only; never wait for caching, autotuning or an update.
    time.sleep(0.3)
    pane_pid = subprocess.check_output(["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"], text=True).splitlines()[-1]
    pid_file = run / "run.pid"
    record = dict(time=now(), run=str(run), tmux_session=session,
                  pane_pid=int(pane_pid), supervisor_pid=int(pid_file.read_text().strip()) if pid_file.exists() else None,
                  resume=resume, independent_of_codex_and_ssh=True,
                  attach_command=shlex.join(["tmux", "attach", "-t", session]),
                  stop_command=shlex.join([str(PYTHON), str(Path(__file__).resolve()), "stop", str(run)]),
                  resume_command=shlex.join([str(PYTHON), str(Path(__file__).resolve()), "resume", str(run)]),
                  log=str(run / "train.log"), no_automatic_validation=True)
    write_json(run / "launch.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare-rgb", "prepare-parallax", "launch", "resume", "stop", "status"))
    parser.add_argument("run", type=Path)
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    if arguments.mode in ("prepare-rgb", "prepare-parallax"):
        from rgb_recipe import prepare
        prepare(run, native=arguments.mode == "prepare-parallax")
        print(f"已准备三轮正式训练及八栋验证 · {run}")
        return
    if not run.is_dir():
        parser.error(f"Run directory does not exist: {run}")
    from rgb_recipe import OUT, PARALLAX_OUT
    if run.parent in (OUT, PARALLAX_OUT):
        from rgb_queue import dispatch
        if not (run / "config.json").is_file():
            parser.error("请先完成 prepare-rgb")
        dispatch(arguments.mode, run)
        return
    if arguments.mode in ("launch", "resume"):
        launch(run, resume=arguments.mode == "resume")
    elif arguments.mode == "stop":
        write_json(run / "stop.request", dict(time=now(), requested_by_pid=os.getpid(),
                   action="save at complete optimizer boundary and stop supervisor without restart"))
        print(json.dumps(dict(run=str(run), stop_requested=True, active=running(run),
                              message="The worker will save and stop at a complete update boundary."), ensure_ascii=False))
    else:
        progress_path = run / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(stage="not_started")
        progress.update(supervisor_active=running(run), tmux_session=session_name(run))
        print(json.dumps(progress, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
