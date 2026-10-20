"""CPU shell routing for explicit resume; never launch a real tmux or model."""

import unittest

import test_vace_building_test_launcher as launcher_fixtures


class ResumeLauncherTests(unittest.TestCase):
    def execute_fixture(self, arguments, existing_tmux=False):
        return launcher_fixtures.BatchLauncherTests.execute_fixture(self, arguments, existing_tmux=existing_tmux)

    def test_resume_dry_run_preserves_flag_and_never_starts_GPU(self):
        for arguments in (["--resume", "--dry-run"], ["--dry-run", "--resume"]):
            with self.subTest(arguments=arguments):
                result = self.execute_fixture(arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("<--dry-run> <--resume>", result.stdout)
                self.assertNotIn("<--run>", result.stdout)
                self.assertNotIn("CPU_TEST_TMUX", result.stdout)

    def test_resume_tmux_preserves_flag_in_new_visible_session(self):
        result = self.execute_fixture(["--resume", "--tmux"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPU_TEST_TMUX <new-session> <-A> <-s> <vace_building_test>", result.stdout)
        self.assertIn(" --resume;", result.stdout)
        self.assertNotIn("CPU_TEST_PYTHON", result.stdout)

    def test_resume_reuses_current_tmux_without_losing_flag(self):
        result = self.execute_fixture(["--resume", "--tmux"], existing_tmux=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<--run> <--resume>", result.stdout)
        self.assertNotIn("CPU_TEST_TMUX", result.stdout)

    def test_resume_foreground_routes_to_authorized_resume_only(self):
        result = self.execute_fixture(["--resume"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<--run> <--resume>", result.stdout)

    def test_duplicate_resume_or_incompatible_modes_start_nothing(self):
        for arguments in (["--resume", "--resume"], ["--resume", "--dry-run", "--tmux"]):
            with self.subTest(arguments=arguments):
                result = self.execute_fixture(arguments)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("CPU_TEST_", result.stdout)


if __name__ == "__main__":
    unittest.main()
