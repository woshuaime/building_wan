"""CPU preparation, manual GPU execution and resumable single-owner runs."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from structure_data import EPOCHS, MAX_EPOCHS, OUTPUTS, ROOT, TRAIN_COUNT, now, prepare

sys.path.append(str(ROOT / "scripts"))


class Paused(InterruptedError):
    pass


class Progress:
    def __init__(self, run, stage, **fields):
        self.run = run
        self.fields = dict(stage=stage, pid=os.getpid(), **fields)

    def update(self, announce=False, **fields):
        self.fields.update(fields, time=now())
        # Atomic publication without forcing the mechanical disk to flush every update.
        path = self.run / "progress.json"
        staging = path.with_suffix(".tmp")
        staging.write_text(json.dumps(self.fields, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(staging, path)
        if announce:
            print(json.dumps(self.fields, ensure_ascii=False), flush=True)

    def stop(self):
        if (self.run / "stop.request").exists():
            raise Paused("已收到暂停请求")


@contextmanager
def run_lock(run):
    with (run / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("该 run 已有进程操作") from error
        yield handle


def status(run):
    path = run / "progress.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        pid = payload["pid"]
        process = Path(f"/proc/{pid}/cmdline")
        try:
            command = os.fsdecode(process.read_bytes())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            command = ""
        payload["process_running"] = (str(run) in command or
                                      (str(run.relative_to(ROOT)) in command and "run.py" in command))
    else:
        payload = dict(stage="prepared" if (run / "prepared.json").is_file() else "unprepared",
                       process_running=False, updates=0, total=EPOCHS * TRAIN_COUNT)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def request_pause(run):
    with (run / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            path = run / "progress.json"
            info = json.loads(path.read_text()) if path.is_file() else {"stage": "prepared"}
            if info["stage"] in ("starting", "training", "epoch_review", "validation", "encoding", "scoring"):
                (run / "stop.request").touch()
                print("已请求暂停 · 当前更新会保存完整状态后退出，验证在采样步边界暂停。")
                return
        print("该 run 当前没有训练或验证需要暂停。")


def prepare_commands(run):
    python = shlex.quote(sys.executable)
    script = shlex.quote(str(Path(__file__).resolve()))
    path = shlex.quote(str(run))
    return "\n".join(f"{python} -Bu {script} {option} --run {path}"
                     for option in ("start", "status", "pause", "resume", "resume --epochs 5", "validate"))


def gpu_action(args, run, owner):
    from common import nvml_snapshot
    from pipeline import external_processes
    from vace_p01 import gpu_lock

    if not (run / "prepared.json").is_file():
        raise ValueError("先执行 prepare 完成 CPU 数据准备")
    latest = run / "checkpoints/latest.pt"
    if args.action == "start" and latest.exists():
        raise ValueError("该 run 已有更新记录，请使用 resume")
    if args.action == "validate" and not (run / f"training_{args.epochs}.json").is_file():
        raise ValueError(f"第 {args.epochs} 轮训练完成记录尚未发布")
    if args.action == "resume" and args.epochs == MAX_EPOCHS and not (run / "training_3.json").is_file():
        raise ValueError("续到第 5 轮前需完成默认的 3 轮")
    if args.action == "resume" and (run / f"training_{args.epochs}.json").is_file():
        print(f"第 {args.epochs} 轮已完成。若要接着训练第 4–5 轮，请使用 resume --epochs 5。"
              if args.epochs == EPOCHS else "第 5 轮已完成。")
        return
    runtime = run / ".work/runtime"
    for key, folder in (("HF_HOME", "hf"), ("TORCH_HOME", "torch"), ("XDG_CACHE_HOME", "xdg"),
                        ("CUDA_CACHE_PATH", "cuda"), ("TRITON_CACHE_DIR", "triton"),
                        ("NUMBA_CACHE_DIR", "numba")):
        location = runtime / folder
        location.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(location)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="True",
                      TOKENIZERS_PARALLELISM="false")

    @contextmanager
    def gpu_available():
        with gpu_lock(ROOT) as gpu_handle:
            snapshot = nvml_snapshot()
            other = external_processes(snapshot)
            if other:
                raise RuntimeError(f"GPU 正由其他进程使用（PID {', '.join(str(p['pid']) for p in other)}）")
            yield snapshot, gpu_handle

    (run / "stop.request").unlink(missing_ok=True)
    progress = Progress(run, "starting", action=args.action, target_epochs=args.epochs)
    progress.update(announce=True)

    def stop_handler(_signum, _frame):
        (run / "stop.request").touch()

    previous_handlers = {kind: signal.signal(kind, stop_handler) for kind in (signal.SIGTERM, signal.SIGINT)}
    try:
        if args.action == "validate":
            from evaluation import rendered, review
            if not rendered(run, args.epochs):
                with gpu_available() as (_, lease):
                    # The CUDA context exits with the child. Colleagues can use
                    # the card while the parent decodes and scores the videos.
                    descriptors = (owner.fileno(), lease.fileno())
                    command = [sys.executable, "-Bu", str(Path(__file__).with_name("evaluation.py")),
                               str(run), str(args.epochs), *map(str, descriptors)]
                    result = subprocess.run(command, pass_fds=descriptors, start_new_session=True)
                progress.fields.update(json.loads((run / "progress.json").read_text()), pid=os.getpid())
                if result.returncode == 75:
                    raise Paused("验证已在已提交成片处暂停")
                if result.returncode:
                    raise RuntimeError(f"采样进程退出 {result.returncode}，已核验成片可继续复用")
            review(run, args.epochs, progress)
        else:
            with gpu_available() as (snapshot, _):
                from train import train
                train(run, args.epochs, progress, snapshot)
    except Paused as error:
        progress.update(announce=True, stage="paused", reason=str(error))
    except BaseException as error:
        progress.update(announce=True, stage="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        for kind, handler in previous_handlers.items():
            signal.signal(kind, handler)


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "start", "status", "pause", "resume", "validate"):
        sub = actions.add_parser(name)
        sub.add_argument("--run", required=True, type=Path)
        if name == "prepare":
            sub.add_argument("--workers", type=int, default=4, choices=range(1, 9), metavar="1-8")
        if name in ("start", "resume", "validate"):
            sub.set_defaults(epochs=EPOCHS)
        if name in ("resume", "validate"):
            sub.add_argument("--epochs", choices=(EPOCHS, MAX_EPOCHS), default=EPOCHS, type=int)
    return parser.parse_args()


def main():
    args = parse()
    run = args.run.resolve()
    if run.parent != OUTPUTS or not run.name.startswith("run_"):
        raise ValueError(f"--run must be a run_* directory directly under {OUTPUTS}")
    if args.action == "status":
        status(run)
        return
    if args.action == "prepare":
        run.mkdir(parents=True, exist_ok=True)
    if not run.is_dir():
        raise FileNotFoundError(run)
    if args.action == "pause":
        request_pause(run)
        return
    with run_lock(run) as owner:
        if args.action == "prepare":
            if (run / "prepared.json").exists():
                from structure_data import PreparedData
                PreparedData(run)
            else:
                prepare(run, args.workers)
            print(prepare_commands(run))
        else:
            gpu_action(args, run, owner)


if __name__ == "__main__":
    main()
