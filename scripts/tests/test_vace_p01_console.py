"""CPU subprocess and progress checks; no models or GPU are used."""

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01_console as console


class TimedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.writes = []

    def write(self, value):
        self.writes.append((time.monotonic(), value))
        return super().write(value)


class ConsoleTests(unittest.TestCase):
    def test_run_log_mirrors_unbuffered_stdout_stderr_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            terminal, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(terminal), contextlib.redirect_stderr(errors):
                with console.run_log(output):
                    self.assertFalse(sys.stdout.isatty())
                    sys.stdout.write("progress\r")
                    self.assertIn(b"progress\r", (output / "run.log").read_bytes())
                    sys.stderr.write("error\n")
                self.assertIs(sys.stdout, terminal)
                self.assertIs(sys.stderr, errors)
            self.assertEqual(terminal.getvalue(), "progress\r")
            self.assertEqual(errors.getvalue(), "error\n")
            with (output / "run.log").open(newline="") as stream:
                self.assertEqual(stream.read(), "progress\rerror\n")
            with self.assertRaises(FileExistsError):
                with console.run_log(output):
                    self.fail("Existing logs must not be overwritten")

    def test_run_log_restores_streams_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal = io.StringIO()
            with contextlib.redirect_stdout(terminal):
                with self.assertRaisesRegex(RuntimeError, "fixture"):
                    with console.run_log(Path(directory)):
                        raise RuntimeError("fixture")
                self.assertIs(sys.stdout, terminal)

    def test_stage_state_keeps_last_completed_and_appends_real_events(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            console.record_stage(output, "model_loading", "started", completed=False)
            console.record_stage(output, "model_loading", "loaded", completed=True)
            console.record_stage(output, "conditioning", "started", completed=False)
            state = json.loads((output / "stage_state.json").read_text())
            events = [json.loads(line) for line in (output / "stage_events.jsonl").read_text().splitlines()]
            self.assertEqual(state["last_stage"], "conditioning")
            self.assertEqual(state["last_completed_stage"], "model_loading")
            self.assertEqual(state["last_completed_at_utc"], events[1]["timestamp_utc"])
            self.assertEqual(len(events), 3)
            self.assertTrue(state["updated_at_utc"].endswith("+00:00"))
            self.assertFalse(list(output.glob(".stage_state_*")))

    def test_visible_child_streams_without_newlines_and_keeps_real_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            terminal = TimedStream()
            program = ("import os,time;os.write(1,b'first\\r');time.sleep(0.3);"
                       "os.write(2,bytes([228,184]));time.sleep(0.05);"
                       "os.write(1,bytes([173]));raise SystemExit(7)")
            with (output / "lock").open("a") as lock, contextlib.redirect_stdout(terminal):
                with console.run_log(output):
                    returncode, interrupted = console.run_visible_process(
                        [sys.executable, "-u", "-B", "-c", program], os.environ.copy(), lock.fileno(), output)
            finished = time.monotonic()
            self.assertEqual(returncode, 7)
            self.assertFalse(interrupted)
            self.assertEqual(terminal.getvalue(), "first\r中")
            first_write = next(timestamp for timestamp, content in terminal.writes if "first" in content)
            self.assertGreater(finished - first_write, 0.2)
            self.assertEqual((output / "worker.log").read_bytes(), "first\r中".encode())
            self.assertEqual((output / "run.log").read_bytes(), "first\r中".encode())

    def test_child_native_signal_exit_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            with (output / "lock").open("a") as lock:
                code, interrupted = console.run_visible_process(
                    [sys.executable, "-B", "-c", "import os,signal;os.kill(os.getpid(),signal.SIGTERM)"],
                    os.environ.copy(), lock.fileno(), output)
            self.assertEqual(code, -signal.SIGTERM)
            self.assertFalse(interrupted)

    def test_parent_termination_only_stops_its_worker_and_restores_handler(self):
        original = signal.getsignal(signal.SIGTERM)
        unrelated = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(30)"])
        timer = threading.Timer(0.5, os.kill, args=(os.getpid(), signal.SIGTERM))
        try:
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                output = Path(directory)
                with (output / "lock").open("a") as lock:
                    timer.start()
                    code, interrupted = console.run_visible_process(
                        [sys.executable, "-u", "-B", "-c", "import time;print('ready',flush=True);time.sleep(30)"],
                        os.environ.copy(), lock.fileno(), output)
                self.assertEqual(code, -signal.SIGTERM)
                self.assertTrue(interrupted)
                self.assertIn(b"ready", (output / "worker.log").read_bytes())
                self.assertIsNone(unrelated.poll())
                self.assertEqual(signal.getsignal(signal.SIGTERM), original)
        finally:
            timer.cancel()
            timer.join()
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_worker_log_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "worker.log").write_bytes(b"original")
            with (output / "lock").open("a") as lock:
                with self.assertRaises(FileExistsError):
                    console.run_visible_process(["not-executed"], os.environ.copy(), lock.fileno(), output)
            self.assertEqual((output / "worker.log").read_bytes(), b"original")

    def test_keyboard_interrupt_stops_only_own_worker_and_retains_log(self):
        timer = threading.Timer(0.4, os.kill, args=(os.getpid(), signal.SIGINT))
        try:
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                output = Path(directory)
                with (output / "lock").open("a") as lock:
                    timer.start()
                    code, interrupted = console.run_visible_process(
                        [sys.executable, "-u", "-B", "-c", "import time;print('running',flush=True);time.sleep(30)"],
                        os.environ.copy(), lock.fileno(), output)
                self.assertEqual(code, -signal.SIGTERM)
                self.assertTrue(interrupted)
                self.assertIn(b"running", (output / "worker.log").read_bytes())
        finally:
            timer.cancel()
            timer.join()

    def test_vae_restores_preexisting_instance_method_override(self):
        class FixtureVAE:
            def encode(self):
                return "class"

            def decode(self):
                return "decoded"

        fixture = FixtureVAE()
        override = lambda: "instance"
        fixture.encode = override
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with console.vae_stages(Path(directory), fixture):
                self.assertEqual(fixture.encode(), "instance")
                self.assertEqual(fixture.decode(), "decoded")
        self.assertIs(fixture.encode, override)
        self.assertNotIn("decode", vars(fixture))

    def test_progress_counts_only_after_real_iteration_body_returns(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            iterator = console.inference_progress(output, total=2)([10, 5])
            self.assertEqual(next(iterator), 10)
            state = json.loads((output / "stage_state.json").read_text())
            self.assertEqual(state["completed_steps"], 0)
            self.assertEqual(next(iterator), 5)
            state = json.loads((output / "stage_state.json").read_text())
            self.assertEqual(state["completed_steps"], 1)
            self.assertIsNone(state["last_completed_stage"])
            with self.assertRaises(StopIteration):
                next(iterator)
            state = json.loads((output / "stage_state.json").read_text())
            self.assertEqual(state["completed_steps"], 2)
            self.assertEqual(state["last_completed_stage"], "denoising")

    def test_interrupted_progress_does_not_invent_completed_step(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            iterator = console.inference_progress(output, total=2)([10, 5])
            next(iterator)
            iterator.close()
            state = json.loads((output / "stage_state.json").read_text())
            self.assertEqual(state["completed_steps"], 0)
            self.assertFalse(state["completed"])

    def test_progress_refuses_wrong_scheduler_length(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "timestep count"):
                list(console.inference_progress(Path(directory), total=2)([1]))
            self.assertFalse((Path(directory) / "stage_state.json").exists())

    def test_vae_stage_wrappers_keep_results_and_restore_original_methods(self):
        class FixtureVAE:
            def encode(self, value, device):
                return value + device

            def decode(self, value):
                raise RuntimeError("real decode failure fixture")

        fixture = FixtureVAE()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "real decode failure"):
                with console.vae_stages(output, fixture):
                    self.assertEqual(fixture.encode("frame", device="cpu"), "framecpu")
                    fixture.decode("latent")
            self.assertNotIn("encode", vars(fixture))
            self.assertNotIn("decode", vars(fixture))
            state = json.loads((output / "stage_state.json").read_text())
            self.assertEqual(state["last_stage"], "vae_decode")
            self.assertEqual(state["last_completed_stage"], "vae_encode")
            self.assertFalse(state["completed"])

    def test_console_module_does_not_import_torch(self):
        program = ("import sys;sys.path.insert(0,'scripts');import vace_p01_console;"
                   "assert 'torch' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", program],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
