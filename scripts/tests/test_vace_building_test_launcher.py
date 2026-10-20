"""CPU shell-routing fixtures; neither tmux sessions nor GPU models are started."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "run_vace_building_test.sh"


class BatchLauncherTests(unittest.TestCase):
    def execute_fixture(self, arguments, existing_tmux=False):
        with tempfile.TemporaryDirectory(prefix="vace_batch_shell_cpu_test_", dir="/tmp") as directory:
            root = Path(directory)
            python = root / "fixture_python"
            python.write_text('#!/bin/sh\nprintf "CPU_TEST_PYTHON"\nfor argument do printf " <%s>" "$argument"; done\nprintf "\\n"\n')
            python.chmod(0o700)
            tmux = root / "tmux"
            tmux.write_text('#!/bin/sh\nif [ "$1" = display-message ]; then printf "CPU_BATCH_SESSION\\n"; '
                            'else printf "CPU_TEST_TMUX"; for argument do printf " <%s>" "$argument"; done; printf "\\n"; fi\n')
            tmux.chmod(0o700)
            environment = os.environ.copy()
            environment.update(WAN_PYTHON=str(python), PATH=str(root) + os.pathsep + environment.get("PATH", ""))
            environment.pop("TMUX", None)
            if existing_tmux:
                environment["TMUX"] = "/tmp/synthetic_batch_cpu_tmux,1,0"
            return subprocess.run(["bash", str(LAUNCHER), *arguments], env=environment,
                                  capture_output=True, text=True, timeout=10)

    def test_default_routes_only_to_the_remaining_seven_sample_batch(self):
        result = self.execute_fixture([])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<-u> <-B> <scripts/vace_building_test.py> <--run>", result.stdout)
        self.assertEqual(result.stdout.count("CPU_TEST_PYTHON"), 1)
        self.assertNotIn("scripts/vace_p01_81f.py", result.stdout)

    def test_dry_run_never_routes_to_inference_or_tmux(self):
        result = self.execute_fixture(["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<--dry-run>", result.stdout)
        self.assertNotIn("<--run>", result.stdout)
        self.assertNotIn("CPU_TEST_TMUX", result.stdout)

    def test_prepare_never_routes_to_inference_or_tmux(self):
        result = self.execute_fixture(["--prepare"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<--prepare>", result.stdout)
        self.assertNotIn("<--run>", result.stdout)
        self.assertNotIn("CPU_TEST_TMUX", result.stdout)

    def test_tmux_uses_the_visible_named_batch_session(self):
        result = self.execute_fixture(["--tmux"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPU_TEST_TMUX <new-session> <-A> <-s> <vace_building_test>", result.stdout)
        self.assertIn(str(LAUNCHER), result.stdout)
        self.assertNotIn("CPU_TEST_PYTHON", result.stdout)

    def test_current_tmux_is_reused_without_starting_a_nested_session(self):
        result = self.execute_fixture(["--tmux"], existing_tmux=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tmux attach-session -t CPU_BATCH_SESSION", result.stdout)
        self.assertIn("<--run>", result.stdout)
        self.assertNotIn("CPU_TEST_TMUX", result.stdout)

    def test_unknown_or_combined_modes_fail_before_starting_anything(self):
        for arguments in (["--ru"], ["--prepare", "--tmux"], ["--run"]):
            with self.subTest(arguments=arguments):
                result = self.execute_fixture(arguments)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("CPU_TEST_", result.stdout)

    def test_help_explains_P01_reuse_without_starting_anything(self):
        result = self.execute_fixture(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Reuse P01", result.stdout)
        self.assertNotIn("CPU_TEST_", result.stdout)


if __name__ == "__main__":
    unittest.main()
