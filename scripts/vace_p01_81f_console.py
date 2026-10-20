"""File-free console progress and foreground process supervision for VACE 81F."""

import contextlib
import functools
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class ConsoleState:
    def __init__(self, total_steps=30, event_sink=None):
        if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps <= 0:
            raise ValueError("Progress total must be a positive integer")
        self.total_steps = total_steps
        self.event_sink = event_sink
        self.started_at = time.monotonic()
        self.last_stage = None
        self.last_completed_stage = None
        self.completed_steps = 0
        self.completed = False

    @property
    def elapsed(self):
        return time.monotonic() - self.started_at

    def _emit(self, detail=""):
        if self.event_sink is not None:
            self.event_sink({
                "stage": self.last_stage,
                "detail": detail,
                "elapsed": self.elapsed,
                "completed": self.completed,
                "completed_steps": self.completed_steps,
                "total_steps": self.total_steps,
                "last_completed_stage": self.last_completed_stage,
            })

    def stage(self, name, detail="", completed=False):
        self.last_stage = name
        self.completed = completed
        if completed:
            self.last_completed_stage = name
        status = "completed" if completed else "started"
        print(f"[{self.elapsed:.1f}s] {name}: {detail} [{status}]", flush=True)
        self._emit(detail)

    def progress(self, timesteps):
        from tqdm import tqdm

        if hasattr(timesteps, "__len__") and len(timesteps) != self.total_steps:
            raise ValueError("Actual scheduler timestep count differs from the frozen total")
        self.completed_steps = 0
        self.stage("denoising", "Actual scheduler iteration started")
        bar = tqdm(total=self.total_steps, desc="VACE Inference", file=sys.stdout,
                   disable=False, dynamic_ncols=True, mininterval=0, unit="step")
        try:
            for timestep in timesteps:
                if self.completed_steps >= self.total_steps:
                    raise ValueError("Scheduler yielded too many timesteps")
                yield timestep
                self.completed_steps += 1
                bar.update(1)
                self._emit(f"Completed actual step {self.completed_steps}/{self.total_steps}")
            if self.completed_steps != self.total_steps:
                raise ValueError("Scheduler ended before all frozen steps completed")
        finally:
            bar.close()
        self.stage("denoising", "All actual scheduler iterations completed", completed=True)

    @contextlib.contextmanager
    def vae_stages(self, vae):
        originals = {}
        invocations = {"encode": 0, "decode": 0}

        def wrap(name, original):
            @functools.wraps(original)
            def invoke(*args, **kwargs):
                invocations[name] += 1
                detail = f"VAE {name} invocation {invocations[name]}"
                self.stage("vae_" + name, detail)
                result = original(*args, **kwargs)
                self.stage("vae_" + name, detail, completed=True)
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


def _signal_group(process, signum):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def _terminate_group(process):
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait()


def visible_worker(command, env, lock_fd, event_callback=None):
    interrupted = False
    requested_signal = None
    original_handlers = {}
    process = None
    read_fd = None
    write_fd = None
    selector = selectors.DefaultSelector()
    buffered = bytearray()

    def request_termination(signum, frame):
        nonlocal requested_signal
        requested_signal = signum

    def dispatch_lines():
        while b"\n" in buffered:
            content, separator, remaining = buffered.partition(b"\n")
            buffered[:] = remaining
            event = json.loads(content)
            if not isinstance(event, dict):
                raise ValueError("Worker status event must be an object")
            event_callback(event)
        if len(buffered) > 1048576:
            raise ValueError("Worker status event exceeded its size limit")

    def receive_events():
        while selector.get_map():
            try:
                content = os.read(read_fd, 65536)
            except BlockingIOError:
                break
            if not content:
                selector.unregister(read_fd)
                break
            buffered.extend(content)
            dispatch_lines()

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                original_handlers[signum] = signal.signal(signum, request_termination)
        environment = dict(env)
        inherited = [lock_fd]
        environment.pop("WAN_VACE81_STATUS_FD", None)
        if event_callback is not None:
            read_fd, write_fd = os.pipe()
            os.set_blocking(read_fd, False)
            selector.register(read_fd, selectors.EVENT_READ)
            inherited.append(write_fd)
            environment["WAN_VACE81_STATUS_FD"] = str(write_fd)
        process = subprocess.Popen(
            command, env=environment, cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=tuple(inherited),
        )
        if write_fd is not None:
            os.close(write_fd)
            write_fd = None
        termination_deadline = None
        while True:
            if requested_signal is not None and not interrupted:
                interrupted = True
                print(f"Worker termination requested: {signal.Signals(requested_signal).name}", flush=True)
                _signal_group(process, signal.SIGTERM)
                termination_deadline = time.monotonic() + 15
            if termination_deadline is not None and process.poll() is None \
                    and time.monotonic() >= termination_deadline:
                _signal_group(process, signal.SIGKILL)
                termination_deadline = None
            if selector.get_map():
                if selector.select(timeout=0.1):
                    receive_events()
            else:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            returncode = process.poll()
            if returncode is not None:
                if selector.get_map():
                    receive_events()
                if buffered:
                    raise ValueError("Worker status event ended without a newline")
                return returncode, interrupted
    except BaseException:
        if process is not None:
            _terminate_group(process)
        raise
    finally:
        selector.close()
        for descriptor in (read_fd, write_fd):
            if descriptor is not None:
                os.close(descriptor)
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
