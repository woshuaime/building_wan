"""Visible, recorded execution and real progress hooks for the P01 baseline."""

import codecs
import contextlib
import functools
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class _MirrorStream:
    def __init__(self, terminal, log, lock):
        self.terminal = terminal
        self.log = log
        self.lock = lock

    def write(self, text):
        with self.lock:
            self.log.write(text)
            self.log.flush()
            self.terminal.write(text)
            self.terminal.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.log.flush()
            self.terminal.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", None) or "utf-8"

    @property
    def errors(self):
        return getattr(self.terminal, "errors", None) or "replace"


@contextlib.contextmanager
def run_log(output):
    path = Path(output) / "run.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with path.open("x", encoding="utf-8", buffering=1) as log:
        lock = threading.RLock()
        sys.stdout = _MirrorStream(original_stdout, log, lock)
        sys.stderr = _MirrorStream(original_stderr, log, lock)
        try:
            yield path
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr


def _stage(output, stage, detail, fields, visible):
    output = Path(output)
    timestamp = datetime.now(timezone.utc).isoformat()
    state_path = output / "stage_state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    completed = fields.get("completed", False)
    event = dict(fields, timestamp_utc=timestamp, stage=stage, detail=detail, completed=completed)
    state = dict(previous)
    state.update(event)
    state.update(last_stage=stage, updated_at_utc=timestamp)
    state.setdefault("created_at_utc", timestamp)
    state.setdefault("last_completed_stage", None)
    state.setdefault("last_completed_at_utc", None)
    if previous.get("stage") != stage or previous.get("completed"):
        state["stage_started_at_utc"] = timestamp
    if completed:
        state.update(last_completed_stage=stage, last_completed_at_utc=timestamp)
    descriptor, temporary = tempfile.mkstemp(prefix=".stage_state_", suffix=".json", dir=output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    with (output / "stage_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
    if visible:
        print(f"[{timestamp}] {stage}: {detail}" + (" [completed]" if completed else ""), flush=True)


def record_stage(output, stage, detail="", **fields):
    _stage(output, stage, detail, fields, visible=True)


def _terminate_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_visible_process(command, environment, lock_fd, output):
    output = Path(output)
    interrupted = False
    requested_signal = None
    original_handlers = {}
    process = None

    def request_termination(signum, frame):
        nonlocal requested_signal
        requested_signal = signum

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGHUP):
                original_handlers[signum] = signal.signal(signum, request_termination)
        with (output / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                command, env=environment, cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                bufsize=0, start_new_session=True, pass_fds=(lock_fd,),
            )
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            termination_deadline = None
            try:
                while selector.get_map() or process.poll() is None:
                    try:
                        if requested_signal is not None and not interrupted:
                            interrupted = True
                            print(f"Worker termination requested: {signal.Signals(requested_signal).name}", flush=True)
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                            termination_deadline = time.monotonic() + 15
                        if termination_deadline is not None and process.poll() is None \
                                and time.monotonic() >= termination_deadline:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            termination_deadline = None
                        for key, events in selector.select(timeout=0.1):
                            content = os.read(key.fileobj.fileno(), 65536)
                            if not content:
                                selector.unregister(key.fileobj)
                                continue
                            log.write(content)
                            log.flush()
                            sys.stdout.write(decoder.decode(content))
                            sys.stdout.flush()
                    except KeyboardInterrupt:
                        requested_signal = signal.SIGINT
                sys.stdout.write(decoder.decode(b"", final=True))
                sys.stdout.flush()
                return process.wait(), interrupted
            except BaseException:
                _terminate_group(process)
                raise
            finally:
                selector.close()
                process.stdout.close()
    except BaseException:
        if process is not None:
            _terminate_group(process)
        raise
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)


def inference_progress(output, total=30):
    if not isinstance(total, int) or total <= 0:
        raise ValueError("Progress total must be a positive integer")

    def progress(timesteps):
        from tqdm import tqdm

        if hasattr(timesteps, "__len__") and len(timesteps) != total:
            raise ValueError("Actual scheduler timestep count differs from the frozen total")
        completed = 0
        record_stage(output, "denoising", "Actual scheduler iteration started",
                     completed_steps=0, total_steps=total, completed=False)
        bar = tqdm(total=total, desc="VACE Inference", file=sys.stdout, disable=False,
                   dynamic_ncols=True, mininterval=0, unit="step")
        try:
            for timestep in timesteps:
                if completed >= total:
                    raise ValueError("Scheduler yielded too many timesteps")
                yield timestep
                completed += 1
                bar.update(1)
                _stage(output, "denoising", f"Completed actual step {completed}/{total}",
                       dict(completed_steps=completed, total_steps=total, completed=False), visible=False)
            if completed != total:
                raise ValueError("Scheduler ended before all frozen steps completed")
        finally:
            bar.close()
        record_stage(output, "denoising", "All actual scheduler iterations completed",
                     completed_steps=completed, total_steps=total, completed=True)

    return progress


@contextlib.contextmanager
def vae_stages(output, vae):
    originals = {}
    invocations = {"encode": 0, "decode": 0}

    def wrap(name, original):
        @functools.wraps(original)
        def invoke(*args, **kwargs):
            invocations[name] += 1
            detail = f"VAE {name} invocation {invocations[name]}"
            record_stage(output, "vae_" + name, detail, completed=False)
            result = original(*args, **kwargs)
            record_stage(output, "vae_" + name, detail, completed=True)
            return result

        return invoke

    try:
        for name in ("encode", "decode"):
            original = getattr(vae, name)
            originals[name] = (name in vars(vae), vars(vae).get(name))
            setattr(vae, name, wrap(name, original))
        yield
    finally:
        for name, (had_override, original) in originals.items():
            if had_override:
                setattr(vae, name, original)
            else:
                delattr(vae, name)
