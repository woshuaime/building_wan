"""One detached tmux session for the V2 RGB continuation and its final validation."""
import contextlib
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

from common import PYTHON, ROOT, now, runtime_env, write_json
from pipeline import acquire_gpu, read_json
from rgb_recipe import TOTAL
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


def bar(value, total, size):
    count = max(0, min(size, int(value * size / max(1, total))))
    return "[" + "█" * count + "░" * (size-count) + f"] {value}/{total}"


def panel(run):
    stage = read_json(run / "progress.json", {}).get("stage", "prepared")
    train = read_json(run / "train_progress.json", {})
    val = read_json(run / "validation_progress.json", {})
    updates = int(train.get("updates", 0))
    cached = int(train.get("cached", train.get("cache", 0)))
    width = max(9, min(30, shutil.get_terminal_size((80, 24)).columns - 43))
    labels = {"prepared":"配置就绪", "waiting_for_gpu":"等待 GPU 空闲", "training":"正式训练",
              "validation":"八栋验证", "reporting":"逐栋报告", "paused":"已暂停", "failed":"运行异常",
              "complete":"完成"}
    lines = ["V2 结构与点纹训练   " + labels.get(stage, stage),
             "窗口编码  " + bar(cached, TOTAL, width),
             "正式更新  " + bar(updates, TOTAL, width),
             "验证成片  " + bar(val.get("completed", 0), 8, width)]
    if stage == "training":
        lines.append(f"当前 {train.get('stage','加载模型')}  ·  建筑 {train.get('building','…')}  ·  "
                     f"RGB 辅助更新 {train.get('aux_updates',0)}")
        rate = train.get("seconds_per_update")
        if rate and updates:
            eta = str(datetime.timedelta(seconds=int(rate * (TOTAL - updates))))
            lines.append(f"近期 {rate:.2f} 秒/更新  ·  训练预计剩余 {eta}")
        lines.append(f"已保存到第 {train.get('durable_updates',0)} 步")
    if stage in ("validation", "reporting"):
        lines.append(f"建筑 {val.get('building','…')}  {val.get('stage','准备')}  " +
                     bar(val.get("step",0),val.get("steps",30),width))
        if val.get("scored") is not None:
            lines.append("逐栋检查  " + bar(val["scored"],8,width))
        if val.get("seconds_per_step"):
            lines.append(f"采样 {val['seconds_per_step']:.2f} 秒/步  ·  "
                         f"预计剩余 {val.get('eta_seconds','…')} 秒")
    if stage == "failed":
        lines.append(str(read_json(run / "progress.json").get("reason","")))
    return "\n".join(lines)


def dashboard(run, finished):
    prior = None
    while True:
        frame = panel(run)
        if frame != prior:
            print("\033[H\033[2J" + frame, flush=True)
            prior = frame
        if finished.wait(.8):
            print("\033[H\033[2J" + panel(run), flush=True)
            return


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
        pane = subprocess.check_output(["tmux", *args, "-P", "-F", "#{pane_id}",
                                        "-c", str(ROOT), command], text=True).strip()
        with contextlib.suppress(subprocess.CalledProcessError):
            subprocess.run(["tmux","set-option","-wt",pane,"remain-on-exit","on"],check=True,
                           stderr=subprocess.DEVNULL)
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
