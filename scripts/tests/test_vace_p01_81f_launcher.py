"""CPU command-routing fixtures; no tmux server or GPU process is started."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "run_vace_p01_465_81f.sh"


class LauncherTests(unittest.TestCase):
    def execute_fixture(self, arguments, existing_tmux=False):
        with tempfile.TemporaryDirectory(prefix="vace81_shell_cpu_test_") as directory:
            root = Path(directory)
            python = root / "fixture_python"
            python.write_text('#!/bin/sh\nprintf "CPU_TEST_PYTHON"\nfor argument do printf " <%s>" "$argument"; done\nprintf "\\n"\n')
            python.chmod(0o700)
            tmux = root / "tmux"
            tmux.write_text('#!/bin/sh\nif [ "$1" = display-message ]; then printf "CPU_OWN_SESSION\\n"; '
                            'else printf "CPU_TEST_TMUX"; for argument do printf " <%s>" "$argument"; done; printf "\\n"; fi\n')
            tmux.chmod(0o700)
            environment = os.environ.copy()
            environment.update(WAN_PYTHON=str(python), PATH=str(root) + os.pathsep + environment.get("PATH", ""))
            environment.pop("TMUX", None)
            if existing_tmux:
                environment["TMUX"] = "/tmp/synthetic_cpu_tmux,1,0"
            return subprocess.run(["bash", str(LAUNCHER), *arguments], env=environment,
                                  capture_output=True, text=True, timeout=10)

    def test_default_routes_exactly_one_run_to_explicit_python(self):
        result = self.execute_fixture([])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<-u> <-B> <scripts/vace_p01_81f.py> <--run>", result.stdout)
        self.assertEqual(result.stdout.count("CPU_TEST_PYTHON"), 1)
        self.assertNotIn("CPU_TEST_TMUX", result.stdout)

    def test_dry_run_never_routes_to_inference(self):
        result = self.execute_fixture(["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<--dry-run>", result.stdout)
        self.assertNotIn("<--run>", result.stdout)

    def test_tmux_starts_or_attaches_visible_session_without_running_in_test(self):
        result = self.execute_fixture(["--tmux"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPU_TEST_TMUX <new-session> <-A> <-s> <vace_p01_81f>", result.stdout)
        self.assertIn(str(LAUNCHER), result.stdout)
        self.assertNotIn("CPU_TEST_PYTHON", result.stdout)

    def test_existing_tmux_is_reused_without_nesting(self):
        result = self.execute_fixture(["--tmux"], existing_tmux=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tmux attach-session -t CPU_OWN_SESSION", result.stdout)
        self.assertIn("<--run>", result.stdout)
        self.assertNotIn("CPU_TEST_TMUX", result.stdout)

    def test_unknown_arguments_fail_without_starting_anything(self):
        result = self.execute_fixture(["--ru"])
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("CPU_TEST_", result.stdout)


if __name__ == "__main__":
    unittest.main()
