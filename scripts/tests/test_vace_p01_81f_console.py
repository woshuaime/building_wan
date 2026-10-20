"""CPU-only checks for file-free progress and foreground subprocess supervision."""

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01_81f_console as console


@contextlib.contextmanager
def inherited_lock():
    read_fd, write_fd = os.pipe()
    try:
        yield read_fd
    finally:
        os.close(read_fd)
        os.close(write_fd)


class ConsoleTests(unittest.TestCase):
    def test_stage_and_progress_keep_memory_state_without_files(self):
        events = []
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as terminal:
            before = set(Path(directory).iterdir())
            state = console.ConsoleState(total_steps=2, event_sink=events.append)
            state.stage("model_loading", "verified", completed=True)
            iterator = state.progress([10, 5])
            self.assertEqual(next(iterator), 10)
            self.assertEqual(state.completed_steps, 0)
            self.assertEqual(state.last_completed_stage, "model_loading")
            self.assertEqual(next(iterator), 5)
            self.assertEqual(state.completed_steps, 1)
            with self.assertRaises(StopIteration):
                next(iterator)
            self.assertEqual(state.completed_steps, 2)
            self.assertEqual(state.last_completed_stage, "denoising")
            self.assertTrue(state.completed)
            self.assertEqual(set(Path(directory).iterdir()), before)
        self.assertIn("VACE Inference", terminal.getvalue())
        self.assertEqual(events[-1]["completed_steps"], 2)
        self.assertGreaterEqual(events[-1]["elapsed"], 0)
        self.assertEqual(events[2]["completed_steps"], 1)

    def test_interrupted_iteration_never_invents_completion(self):
        with contextlib.redirect_stdout(io.StringIO()):
            state = console.ConsoleState(total_steps=2)
            iterator = state.progress([10, 5])
            next(iterator)
            iterator.close()
        self.assertEqual(state.completed_steps, 0)
        self.assertFalse(state.completed)
        self.assertIsNone(state.last_completed_stage)

    def test_wrong_scheduler_length_and_unsized_short_or_long_iterators_fail(self):
        for timesteps, reason in (([1], "timestep count"), (iter([1]), "before"),
                                  (iter([1, 2, 3]), "too many")):
            with self.subTest(reason=reason), contextlib.redirect_stdout(io.StringIO()):
                state = console.ConsoleState(total_steps=2)
                with self.assertRaisesRegex(ValueError, reason):
                    list(state.progress(timesteps))
                self.assertFalse(state.completed)

    def test_invalid_step_totals_fail(self):
        for total in (0, -1, True, 1.5):
            with self.subTest(total=total), self.assertRaises(ValueError):
                console.ConsoleState(total_steps=total)

    def test_vae_real_calls_and_overrides_are_restored_after_failure(self):
        class FixtureVAE:
            def encode(self, value):
                return value

            def decode(self, value):
                raise RuntimeError("real decode failure")

        fixture = FixtureVAE()
        override = lambda value: value + " encoded"
        fixture.encode = override
        with contextlib.redirect_stdout(io.StringIO()):
            state = console.ConsoleState()
            with self.assertRaisesRegex(RuntimeError, "real decode failure"):
                with state.vae_stages(fixture):
                    self.assertEqual(fixture.encode("frame"), "frame encoded")
                    fixture.decode("latent")
        self.assertIs(fixture.encode, override)
        self.assertNotIn("decode", vars(fixture))
        self.assertEqual(state.last_stage, "vae_decode")
        self.assertEqual(state.last_completed_stage, "vae_encode")
        self.assertFalse(state.completed)

    def test_broken_status_sink_is_not_silently_ignored(self):
        def broken_sink(event):
            raise BrokenPipeError("status reader is gone")

        with contextlib.redirect_stdout(io.StringIO()):
            state = console.ConsoleState(event_sink=broken_sink)
            with self.assertRaisesRegex(BrokenPipeError, "reader is gone"):
                state.stage("model_loading")

    def test_real_worker_exit_code_and_lock_fd_without_output_files(self):
        program = "import os,sys;os.fstat(int(sys.argv[1]));assert sys.stdin.read()=='';raise SystemExit(7)"
        with tempfile.TemporaryDirectory() as directory, inherited_lock() as lock_fd:
            before = set(Path(directory).iterdir())
            code, interrupted = console.visible_worker(
                [sys.executable, "-B", "-c", program, str(lock_fd)], os.environ.copy(), lock_fd)
            self.assertEqual((code, interrupted), (7, False))
            self.assertEqual(set(Path(directory).iterdir()), before)

    def test_native_worker_signal_preserves_real_code_and_final_status_event(self):
        events = []
        program = ("import os,signal;descriptor=int(os.environ['WAN_VACE81_STATUS_FD']);"
                   "os.write(descriptor,b'{\"stage\":\"denoising\",\"completed_steps\":2}\\n');"
                   "os.kill(os.getpid(),signal.SIGTERM)")
        with inherited_lock() as lock_fd:
            code, interrupted = console.visible_worker(
                [sys.executable, "-B", "-c", program], os.environ.copy(), lock_fd,
                event_callback=events.append)
        self.assertEqual((code, interrupted), (-signal.SIGTERM, False))
        self.assertEqual(events, [{"stage": "denoising", "completed_steps": 2}])

    def test_worker_stdout_stderr_are_inherited_and_live_without_newlines(self):
        wrapper = (
            "import os,sys;sys.path.insert(0,'scripts');"
            "from vace_p01_81f_console import visible_worker;"
            "read_fd,write_fd=os.pipe();"
            "program=\"import os,time;os.write(1,b'first\\\\r');time.sleep(0.3);os.write(2,b'second')\";"
            "code,interrupted=visible_worker([sys.executable,'-B','-c',program],os.environ.copy(),read_fd);"
            "os.close(read_fd);os.close(write_fd);raise SystemExit(code)"
        )
        process = subprocess.Popen([sys.executable, "-u", "-B", "-c", wrapper],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertEqual(process.stdout.read(6), b"first\r")
            self.assertIsNone(process.poll())
            remaining_stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(remaining_stdout, b"")
            self.assertEqual(stderr, b"second")
            self.assertEqual(process.returncode, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()

    def test_signals_stop_only_own_child_and_restore_parent_handlers(self):
        unrelated = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(30)"])
        try:
            for requested in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                original = signal.getsignal(requested)
                timer = threading.Timer(0.3, os.kill, args=(os.getpid(), requested))
                try:
                    with self.subTest(signal=requested), inherited_lock() as lock_fd, \
                            contextlib.redirect_stdout(io.StringIO()):
                        timer.start()
                        code, interrupted = console.visible_worker(
                            [sys.executable, "-B", "-c", "import time;time.sleep(30)"],
                            os.environ.copy(), lock_fd)
                        self.assertEqual(code, -signal.SIGTERM)
                        self.assertTrue(interrupted)
                        self.assertEqual(signal.getsignal(requested), original)
                        self.assertIsNone(unrelated.poll())
                finally:
                    timer.cancel()
                    timer.join()
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_status_callback_failure_terminates_worker_and_propagates(self):
        def reject_event(event):
            raise RuntimeError("status consumer failed")

        program = ("import os,time;os.write(int(os.environ['WAN_VACE81_STATUS_FD']),"
                   "b'{\"stage\":\"loading\"}\\n');time.sleep(30)")
        with inherited_lock() as lock_fd, self.assertRaisesRegex(RuntimeError, "consumer failed"):
            console.visible_worker([sys.executable, "-B", "-c", program], os.environ.copy(),
                                   lock_fd, event_callback=reject_event)

    def test_failed_spawn_closes_status_fds_and_restores_handlers(self):
        before = set(os.listdir("/proc/self/fd"))
        original = signal.getsignal(signal.SIGHUP)
        with inherited_lock() as lock_fd, self.assertRaises(FileNotFoundError):
            console.visible_worker(["/nonexistent/vace81-test-worker"], os.environ.copy(),
                                   lock_fd, event_callback=lambda event: None)
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)
        self.assertEqual(signal.getsignal(signal.SIGHUP), original)

    def test_module_import_does_not_import_torch(self):
        program = ("import sys;sys.path.insert(0,'scripts');import vace_p01_81f_console;"
                   "assert 'torch' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
