"""One detached tmux session for the V2 RGB continuation and its final validation."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback

from common import PYTHON, ROOT, now, runtime_env, write_json
from pipeline import acquire_gpu, read_json
from rgb_dashboard import dashboard, panel
from run import running

HERE = Path(__file__).resolve().parent


def session_name(run):
    return "v2_rgb_" + run.name.removeprefix("run_")


def status(run):
    value = read_json(run / "progress.json", {"stage": "prepared"})
    train = read_json(run / "train_progress.json", {})
    validation = read_json(run / "validation_progress.json", {})
    return dict(**{"run": str(run), "session": session_name(run), "active": running(run)},
                queue=value, train=train, validation=validation)


def mark(run, stage, **fields):
    write_json(run / "progress.json", dict(stage=stage, time=now(), owner=os.getpid(), **fields))


def require_active(run):
    if (run / "stop.request").is_file():
        raise InterruptedError("已收到暂停请求")


def invoke(run, owner, name, *options):
    require_active(run)
    command = [PYTHON, "-Bu", str(HERE / name), "--run", str(run), *options]
    with (run / "train.log").open("a", buffering=1) as log:
        print(now(), shlex.join(command), file=log, flush=True)
        # Keep the run lease alive if the supervisor is unexpectedly terminated.
        child = subprocess.Popen(command, cwd=ROOT, env=runtime_env(run), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, pass_fds=(owner.fileno(),),
                                 start_new_session=True)
        code = child.wait()
        print(now(), "worker exit", code, file=log)
    if code == 75:
        raise InterruptedError("已保存进度并暂停")
    if code:
        raise RuntimeError(f"{name} 退出码 {code}，见 train.log")


def supervise(run):
    with (run / "run.lock").open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: (run / "stop.request").touch())
        if (run / "validation_complete.json").is_file():
            return
        finished = threading.Event()
        ui = threading.Thread(target=dashboard, args=(run,finished), daemon=True)
        ui.start()
        try:
            if not (run / "validation_rendered.json").is_file():
                mark(run, "waiting_for_gpu")
                # Owned lock spans the transition. The finished training child releases
                # all CUDA memory before the separate validation child loads the model.
                with acquire_gpu(run):
                    if not (run / "training_complete.json").is_file():
                        mark(run, "training")
                        invoke(run, owner, "train_rgb.py")
                    if not (run / "training_complete.json").is_file():
                        raise RuntimeError("训练进程退出但缺少完成记录")
                    mark(run, "validation")
                    invoke(run, owner, "validate_rgb.py", "--phase", "render")
            if not (run / "validation_rendered.json").is_file():
                raise RuntimeError("缺少八栋视频完成记录")
            mark(run, "reporting")
            invoke(run, owner, "validate_rgb.py", "--phase", "report")
            if not (run / "validation_complete.json").is_file():
                raise RuntimeError("缺少逐栋报告完成记录")
            mark(run, "complete")
        except InterruptedError as error:
            mark(run, "paused", reason=str(error))
        except BaseException as error:
            with (run / "train.log").open("a") as log:
                traceback.print_exc(file=log)
            mark(run, "failed", reason=str(error))
            raise
        finally:
            finished.set()
            ui.join()


def launch(run, resume):
    with (run / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if running(run) or (run / "validation_complete.json").is_file():
            print(json.dumps(status(run), ensure_ascii=False))
            return
        if not resume and (run / "launch.json").is_file():
            raise ValueError("使用此 run 下的 resume.sh 继续")
        (run / "stop.request").unlink(missing_ok=True)
        session = session_name(run)
        command = "exec " + shlex.join([PYTHON, "-Bu", str(Path(__file__).resolve()), str(run)])
        if subprocess.run(["tmux", "has-session", "-t", "="+session], capture_output=True).returncode == 0:
            args = ["new-window", "-t", session, "-n", "续跑"]
        else:
            args = ["new-session", "-d", "-s", session, "-n", "训练"]
        pane, owner_pid = subprocess.check_output(
            ["tmux", *args, "-P", "-F", "#{pane_id} #{pane_pid}", "-c", str(ROOT), command],
            text=True).split()
        with contextlib.suppress(subprocess.CalledProcessError):
            subprocess.run(["tmux","set-option","-wt",pane,"remain-on-exit","on"],check=True,
                           stderr=subprocess.DEVNULL)
        log_path = run / "train.log"
        log_path.touch(exist_ok=True)
        follow = "exec " + shlex.join(["tail", "--pid=" + owner_pid, "-n", "80", "-F", "--", str(log_path)])
        result = subprocess.run(["tmux", "split-window", "-v", "-d", "-t", pane, "-p", "57",
                                 "-c", str(ROOT), follow], capture_output=True, text=True)
        if result.returncode:
            print(f"训练窗口已打开。日志分栏未能打开：{result.stderr.strip()}，日志在 {log_path}", file=sys.stderr)
        write_json(run / "launch.json", dict(time=now(), session=session, pane=pane, resume=resume))
        # Serialize simultaneous clients until the tmux worker has claimed its lease.
        for _ in range(30):
            if running(run):
                break
            time.sleep(.05)
        print(json.dumps(status(run), ensure_ascii=False))


def dispatch(mode, run):
    if mode in ("launch", "resume"):
        launch(run, resume=mode=="resume")
    elif mode == "stop":
        (run / "stop.request").touch()
        print("已请求暂停。训练在一次完整更新后保存，验证保留已完成的建筑。")
    else:
        print(json.dumps(status(run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    supervise(Path(sys.argv[1]).resolve())
