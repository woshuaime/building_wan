"""Read-only environment and exclusive-GPU gates for the P01 experiment."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path("/home/shi/miniconda3/envs/scj/bin/python")


def _checked_nvml(library, name, *arguments):
    import ctypes

    try:
        status = getattr(library, name)(*arguments)
    except (AttributeError, OSError, TypeError, ctypes.ArgumentError) as exc:
        raise ValueError(f"Cannot reliably query NVML {name}: {exc}") from exc
    if status != 0:
        raise ValueError(f"NVML {name} failed with status {status}; GPU availability is unknown")


def _nvml_processes(library, handle, kind):
    import ctypes

    class Process(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                    ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]

    name = f"nvmlDeviceGet{kind}RunningProcesses_v3"
    try:
        query = getattr(library, name)
    except AttributeError as exc:
        raise ValueError(f"Cannot reliably query NVML {name}: missing API") from exc
    capacity = 32
    for attempt in range(3):
        count = ctypes.c_uint(capacity)
        entries = (Process * capacity)()
        try:
            status = query(handle, ctypes.byref(count), entries)
        except (OSError, TypeError, ctypes.ArgumentError) as exc:
            raise ValueError(f"Cannot reliably query NVML {name}: {exc}") from exc
        if status == 7:
            capacity = max(capacity * 2, count.value + 16)
            if capacity > 4096:
                break
            continue
        if status != 0 or count.value > capacity:
            raise ValueError(f"NVML {name} failed with status {status}, count {count.value}; GPU availability is unknown")
        processes = []
        for index in range(count.value):
            process = entries[index]
            if process.pid <= 0:
                raise ValueError(f"NVML {name} returned an invalid process ID")
            memory = process.usedGpuMemory
            processes.append(dict(pid=int(process.pid), used_bytes=None if memory == 2 ** 64 - 1 else int(memory)))
        return processes
    raise ValueError(f"NVML {name} process list changed repeatedly or exceeded the safe capacity")


def _nvml_snapshot():
    import ctypes
    import ctypes.util

    library_name = ctypes.util.find_library("nvidia-ml")
    if not library_name:
        raise ValueError("NVML library is unavailable; cannot reliably check other GPU processes")
    try:
        library = ctypes.CDLL(library_name)
    except OSError as exc:
        raise ValueError(f"Cannot load NVML library: {exc}") from exc
    _checked_nvml(library, "nvmlInit_v2")
    try:
        count = ctypes.c_uint()
        _checked_nvml(library, "nvmlDeviceGetCount_v2", ctypes.byref(count))
        if count.value != 1:
            raise ValueError(f"P01 requires exactly one NVML GPU, found {count.value}")
        handle = ctypes.c_void_p()
        _checked_nvml(library, "nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(0), ctypes.byref(handle))
        name = ctypes.create_string_buffer(128)
        driver = ctypes.create_string_buffer(128)
        _checked_nvml(library, "nvmlDeviceGetName", handle, name, ctypes.c_uint(128))
        _checked_nvml(library, "nvmlSystemGetDriverVersion", driver, ctypes.c_uint(128))
        return dict(library=library_name, device_count=count.value, device_name=name.value.decode(),
                    driver_version=driver.value.decode(),
                    compute_processes=_nvml_processes(library, handle, "Compute"),
                    graphics_processes=_nvml_processes(library, handle, "Graphics"))
    finally:
        _checked_nvml(library, "nvmlShutdown")


def _require_exclusive_gpu(snapshot, current_pid):
    external = []
    for kind in ("compute", "graphics"):
        for process in snapshot[f"{kind}_processes"]:
            if process["pid"] != current_pid:
                external.append(dict(kind=kind, **process))
    if external:
        raise ValueError(f"Other GPU processes are running; P01 will not stop or ignore them: {external}")


def check_environment():
    import importlib.metadata
    import importlib.util
    import os
    import shutil
    import subprocess
    import sys

    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError(f"Expected scj Python {EXPECTED_PYTHON}, got {sys.executable}")
    nvml = _nvml_snapshot()
    _require_exclusive_gpu(nvml, os.getpid())
    video_tools = {}
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if not executable:
            raise ValueError(f"{name} is unavailable; required video evaluation cannot be checked")
        try:
            probe = subprocess.run([executable, "-version"], capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"{name} cannot be executed: {exc}") from exc
        if not probe.stdout.strip():
            raise ValueError(f"{name} returned no version information")
        video_tools[name] = dict(path=executable, version=probe.stdout.splitlines()[0])
    try:
        packages = {name: importlib.metadata.version(name) for name in
                    ("torch", "diffsynth", "safetensors", "Pillow", "numpy", "tqdm", "imageio", "imageio-ffmpeg",
                     "transformers", "accelerate", "einops", "sentencepiece")}
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"Required P01 dependency is missing: {exc}") from exc
    if packages["diffsynth"] != "2.1.5":
        raise ValueError(f"Expected DiffSynth 2.1.5, got {packages['diffsynth']}")
    specification = importlib.util.find_spec("diffsynth")
    expected_diffsynth = (ROOT / "code/DiffSynth-Studio/diffsynth/__init__.py").resolve()
    if specification is None or specification.origin is None or Path(specification.origin).resolve() != expected_diffsynth:
        raise ValueError("DiffSynth import path is not the project's third-party vendor directory")
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise ValueError(f"PyTorch cannot be imported: {exc}") from exc

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("P01 requires exactly one available CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The CUDA GPU does not support the frozen BF16 configuration")
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        cuda = dict(device_count=1, device_name=torch.cuda.get_device_name(0), bfloat16_supported=True,
                    runtime=torch.version.cuda, free_bytes=free_bytes, total_bytes=total_bytes)
    except RuntimeError as exc:
        raise ValueError(f"CUDA environment probe failed: {exc}") from exc
    return dict(python=sys.executable, python_version=sys.version, project_root=str(ROOT), cwd=os.getcwd(),
                packages=packages, torch_version=torch.__version__, cuda=cuda, nvml=nvml,
                current_pid=os.getpid(), diffsynth_import_path=str(expected_diffsynth), **video_tools,
                nvidia_smi=shutil.which("nvidia-smi"), model_loaded=False, GPU_inference_started=False)
