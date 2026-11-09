"""Independent tmux owner and existing exclusive A6000 lock, with bounded startup OOM fallbacks."""
import argparse
import fcntl
import shlex
import subprocess
import sys
import time
import traceback
from common import *


def supervise(run, resume):
    sys.path.insert(0, str(ROOT / "scripts"))
    from vace_p01 import gpu_lock, check_old_training
    from vace_p01_environment import _nvml_snapshot
    code = 1
    began = time.monotonic()
    try:
        with (run / "run.lock").open("a") as own_lock, gpu_lock():
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _nvml_snapshot()["compute_processes"]:
                raise ValueError("GPU occupied; no competing job was started")
            check_old_training()
            if (run / "attempt_started.json").exists() and not resume:
                raise ValueError("This formal job was already launched")
            if not resume:
                write_json(run / "attempt_started.json", dict(time=now(), supervisor_pid=os.getpid()))
            (run / "run.pid").write_text(str(os.getpid()) + "\n")
            attempts = range(3) if not resume else [json.loads((run / "runtime_config.json").read_text())["attempt"]]
            for attempt in attempts:
                command = [PYTHON, "-u", "-B", str(Path(__file__).with_name("train.py")), str(run), "--attempt", str(attempt)]
                if resume:
                    command.append("--resume")
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL)
                (run / "worker.pid").write_text(str(process.pid) + "\n")
                write_json(run / "attempts" / f"attempt_{attempt}_launch.json", dict(time=now(), pid=process.pid, command=command))
                print(f"WORKER pid={process.pid} attempt={attempt} tmux={os.environ.get('TMUX')}", flush=True)
                code = process.wait()
                write_json(run / "attempts" / f"attempt_{attempt}_exit.json", dict(time=now(), returncode=code))
                if code == 0 or code != 42 or (run / "first_update.json").exists() or resume:
                    break
                print(f"STARTUP OOM attempt={attempt}; using next authorized memory configuration", flush=True)
    except BaseException:
        print(traceback.format_exc(), flush=True)
    write_json(run / "exit.json", dict(time=now(), returncode=code, elapsed_seconds=time.monotonic() - began,
                                      training_completed=(run / "training_complete.json").exists()))
    return code


def launch(run, resume, hold=False):
    if (run / "first_update.json").exists() and not resume:
        raise ValueError("Existing successful training launch; use its status, do not launch again")
    session = "apc2mesh_D_" + run.name.removeprefix("run_")
    result = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
    if result.returncode == 0:
        print(json.dumps(dict(existing_session=session, run=str(run)), ensure_ascii=False))
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", DIFFSYNTH_SKIP_DOWNLOAD="True",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4")
    cmd = [PYTHON, "-u", "-B", str(Path(__file__).resolve()), "supervise", str(run)]
    if resume:
        cmd.append("--resume")
    explicit_env = [f"{name}={env[name]}" for name in ("CUDA_VISIBLE_DEVICES", "PYTHONUNBUFFERED",
                    "DIFFSYNTH_SKIP_DOWNLOAD", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM",
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")]
    shell_command = "exec env " + shlex.join(explicit_env + cmd) + " >> " + shlex.quote(str(run / "run.log")) + " 2>&1"
    if hold:
        # One-time handoff gate lets the caller record the real tmux identity in
        # project docs/Git before releasing GPU work. No dependency remains after release.
        shell_command = "while [ ! -f " + shlex.quote(str(run / "start.ready")) + "]; do sleep 1; done\n" + shell_command
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), shell_command], env=env, check=True)
    pane = subprocess.check_output(["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"], text=True).strip()
    record = dict(time=now(), tmux_session=session, supervisor_pid=int(pane), run=str(run),
                  log=str(run / "run.log"), command=cmd, independent_of_codex_and_ssh=True,
                  held_for_startup_record=hold)
    write_json(run / "launch.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "supervise"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    if args.mode == "launch":
        launch(args.run.resolve(), args.resume, args.hold)
    else:
        sys.exit(supervise(args.run.resolve(), args.resume))
