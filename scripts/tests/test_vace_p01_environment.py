"""CPU mocks for the P01 environment gate; no CUDA probe is executed."""

import sys
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01_environment


class NvmlProcessTests(unittest.TestCase):
    def test_process_entries_keep_pid_and_memory(self):
        def query(handle, count, entries):
            count._obj.value = 2
            entries[0].pid = 456
            entries[0].usedGpuMemory = 1024
            entries[1].pid = 789
            entries[1].usedGpuMemory = 2 ** 64 - 1
            return 0

        library = SimpleNamespace(nvmlDeviceGetComputeRunningProcesses_v3=query)
        self.assertEqual(vace_p01_environment._nvml_processes(library, None, "Compute"),
                         [dict(pid=456, used_bytes=1024), dict(pid=789, used_bytes=None)])

    def test_insufficient_capacity_retries_real_query(self):
        capacities = []

        def query(handle, count, entries):
            capacities.append(count._obj.value)
            if len(capacities) == 1:
                count._obj.value = 48
                return 7
            count._obj.value = 1
            entries[0].pid = 123
            return 0

        library = SimpleNamespace(nvmlDeviceGetGraphicsRunningProcesses_v3=query)
        self.assertEqual(vace_p01_environment._nvml_processes(library, None, "Graphics"),
                         [dict(pid=123, used_bytes=0)])
        self.assertEqual(capacities, [32, 64])

    def test_unknown_status_is_not_treated_as_empty_gpu(self):
        library = SimpleNamespace(nvmlDeviceGetComputeRunningProcesses_v3=lambda *args: 3)
        with self.assertRaisesRegex(ValueError, "status 3"):
            vace_p01_environment._nvml_processes(library, None, "Compute")

    def test_missing_api_is_explicit_failure(self):
        with self.assertRaisesRegex(ValueError, "missing API"):
            vace_p01_environment._nvml_processes(SimpleNamespace(), None, "Compute")

    def test_unstable_list_is_explicit_failure(self):
        library = SimpleNamespace(nvmlDeviceGetComputeRunningProcesses_v3=lambda *args: 7)
        with self.assertRaisesRegex(ValueError, "changed repeatedly"):
            vace_p01_environment._nvml_processes(library, None, "Compute")

    def test_invalid_count_or_pid_is_explicit_failure(self):
        def oversized(handle, count, entries):
            count._obj.value = 33
            return 0

        library = SimpleNamespace(nvmlDeviceGetComputeRunningProcesses_v3=oversized)
        with self.assertRaisesRegex(ValueError, "count 33"):
            vace_p01_environment._nvml_processes(library, None, "Compute")

        def invalid_pid(handle, count, entries):
            count._obj.value = 1
            return 0

        library.nvmlDeviceGetComputeRunningProcesses_v3 = invalid_pid
        with self.assertRaisesRegex(ValueError, "invalid process ID"):
            vace_p01_environment._nvml_processes(library, None, "Compute")


class ExclusivityTests(unittest.TestCase):
    def test_only_current_pid_may_be_ignored(self):
        snapshot = dict(compute_processes=[dict(pid=123, used_bytes=1024)],
                        graphics_processes=[dict(pid=123, used_bytes=1)])
        vace_p01_environment._require_exclusive_gpu(snapshot, 123)
        snapshot["graphics_processes"].append(dict(pid=456, used_bytes=0))
        with self.assertRaisesRegex(ValueError, "456"):
            vace_p01_environment._require_exclusive_gpu(snapshot, 123)

    def test_external_compute_pid_is_refused_even_with_unknown_memory(self):
        with self.assertRaisesRegex(ValueError, "will not stop or ignore"):
            vace_p01_environment._require_exclusive_gpu(
                dict(compute_processes=[dict(pid=456, used_bytes=None)], graphics_processes=[]), 123)

    def test_missing_nvml_library_is_explicit_failure(self):
        with patch("ctypes.util.find_library", return_value=None):
            with self.assertRaisesRegex(ValueError, "cannot reliably check"):
                vace_p01_environment._nvml_snapshot()

    def test_nvml_initialization_failure_is_explicit(self):
        library = SimpleNamespace(nvmlInit_v2=lambda: 9)
        with patch("ctypes.util.find_library", return_value="mock-nvml"), patch("ctypes.CDLL", return_value=library):
            with self.assertRaisesRegex(ValueError, "nvmlInit_v2 failed with status 9"):
                vace_p01_environment._nvml_snapshot()

    def test_nvml_shutdown_runs_after_query_failure(self):
        stopped = []
        library = SimpleNamespace(nvmlInit_v2=lambda: 0, nvmlDeviceGetCount_v2=lambda *args: 3,
                                  nvmlShutdown=lambda: stopped.append(True) or 0)
        with patch("ctypes.util.find_library", return_value="mock-nvml"), patch("ctypes.CDLL", return_value=library):
            with self.assertRaisesRegex(ValueError, "nvmlDeviceGetCount_v2"):
                vace_p01_environment._nvml_snapshot()
        self.assertEqual(stopped, [True])

    def test_checked_api_missing_is_value_error(self):
        with self.assertRaisesRegex(ValueError, "Cannot reliably query"):
            vace_p01_environment._checked_nvml(SimpleNamespace(), "missing")


class EnvironmentTests(unittest.TestCase):
    def test_current_scj_cuda_probe_returns_report_without_nvidia_smi(self):
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                               is_bf16_supported=lambda: True, mem_get_info=lambda index: (40, 48),
                               get_device_name=lambda index: "NVIDIA RTX A6000")
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.8"), __version__="mock-torch")
        snapshot = dict(compute_processes=[], graphics_processes=[])
        with patch.object(vace_p01_environment, "EXPECTED_PYTHON", Path(sys.executable)), \
                patch.object(vace_p01_environment, "_nvml_snapshot", return_value=snapshot), \
                patch("shutil.which", side_effect=lambda name: f"/mock/{name}" if name != "nvidia-smi" else None), \
                patch("subprocess.run", return_value=SimpleNamespace(stdout="ffmpeg mock\n")), \
                patch("importlib.metadata.version", side_effect=lambda name: "2.1.5" if name == "diffsynth" else "mock-version"), \
                patch("importlib.util.find_spec", return_value=SimpleNamespace(
                    origin=str(vace_p01_environment.ROOT / "code/DiffSynth-Studio/diffsynth/__init__.py"))), \
                patch.dict(sys.modules, torch=torch):
            report = vace_p01_environment.check_environment()
        self.assertEqual(report["cuda"]["free_bytes"], 40)
        self.assertEqual(report["python"], sys.executable)
        self.assertEqual(report["project_root"], str(vace_p01_environment.ROOT))
        self.assertIsNone(report["nvidia_smi"])
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["GPU_inference_started"])
        self.assertEqual(report["ffprobe"]["path"], "/mock/ffprobe")
        self.assertIn("sentencepiece", report["packages"])

    def test_missing_ffprobe_stops_before_torch_probe(self):
        snapshot = dict(compute_processes=[], graphics_processes=[])
        with patch.object(vace_p01_environment, "EXPECTED_PYTHON", Path(sys.executable)), \
                patch.object(vace_p01_environment, "_nvml_snapshot", return_value=snapshot), \
                patch("shutil.which", side_effect=lambda name: "/mock/ffmpeg" if name == "ffmpeg" else None), \
                patch("subprocess.run", return_value=SimpleNamespace(stdout="ffmpeg mock\n")), \
                patch.dict(sys.modules, torch=None):
            with self.assertRaisesRegex(ValueError, "ffprobe is unavailable"):
                vace_p01_environment.check_environment()

    def test_broken_ffprobe_stops_before_torch_probe(self):
        snapshot = dict(compute_processes=[], graphics_processes=[])

        def run(arguments, **kwargs):
            if arguments[0].endswith("ffprobe"):
                raise subprocess.CalledProcessError(1, arguments)
            return SimpleNamespace(stdout="ffmpeg mock\n")

        with patch.object(vace_p01_environment, "EXPECTED_PYTHON", Path(sys.executable)), \
                patch.object(vace_p01_environment, "_nvml_snapshot", return_value=snapshot), \
                patch("shutil.which", side_effect=lambda name: f"/mock/{name}"), \
                patch("subprocess.run", side_effect=run), patch.dict(sys.modules, torch=None):
            with self.assertRaisesRegex(ValueError, "ffprobe cannot be executed"):
                vace_p01_environment.check_environment()

    def test_wrong_diffsynth_import_path_stops_before_torch_probe(self):
        snapshot = dict(compute_processes=[], graphics_processes=[])
        with patch.object(vace_p01_environment, "EXPECTED_PYTHON", Path(sys.executable)), \
                patch.object(vace_p01_environment, "_nvml_snapshot", return_value=snapshot), \
                patch("shutil.which", side_effect=lambda name: f"/mock/{name}"), \
                patch("subprocess.run", return_value=SimpleNamespace(stdout="video tool mock\n")), \
                patch("importlib.metadata.version", return_value="2.1.5"), \
                patch("importlib.util.find_spec", return_value=SimpleNamespace(origin="/wrong/diffsynth/__init__.py")), \
                patch.dict(sys.modules, torch=None):
            with self.assertRaisesRegex(ValueError, "import path"):
                vace_p01_environment.check_environment()

    def test_external_pid_stops_before_torch_probe(self):
        snapshot = dict(compute_processes=[dict(pid=999999, used_bytes=1)], graphics_processes=[])
        with patch.object(vace_p01_environment, "EXPECTED_PYTHON", Path(sys.executable)), \
                patch.object(vace_p01_environment, "_nvml_snapshot", return_value=snapshot), \
                patch("os.getpid", return_value=123), patch.dict(sys.modules, torch=None):
            with self.assertRaisesRegex(ValueError, "999999"):
                vace_p01_environment.check_environment()

    def test_wrong_interpreter_is_refused_before_hardware_probe(self):
        with patch.object(vace_p01_environment, "EXPECTED_PYTHON", Path("/wrong/python")), \
                patch.object(vace_p01_environment, "_nvml_snapshot") as probe:
            with self.assertRaisesRegex(ValueError, "Expected scj Python"):
                vace_p01_environment.check_environment()
        probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
