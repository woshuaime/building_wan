"""Shared, project-local runtime utilities for the GT-video V2 experiment."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
PYTHON = "/home/shi/miniconda3/envs/scj/bin/python"
CACHE = ROOT / "cache/pointcloud_vace/adapter_gt_v2"
PROMPT = ("A static building point cloud made of individually distinguishable sparse neutral gray dots on a uniform light gray background. "
          "The complete roof and walls are represented by separate dots with visible gaps between them. "
          "The camera moves smoothly around the building while the three-dimensional arrangement of the dots remains stationary. "
          "The point size, neutral gray appearance, and camera motion match the input point cloud video.")


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    path = Path(path).resolve()
    s = path.stat()
    return dict(path=str(path), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def event(run, stage, **extra):
    path = Path(run) / "progress.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(time=now(), stage=stage, pid=os.getpid(), **extra)
    write_json(path, previous)
    print(json.dumps(previous, ensure_ascii=False), flush=True)
    return previous


def nvml_snapshot():
    import ctypes
    import sys
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    from vace_p01_environment import _nvml_snapshot
    info = _nvml_snapshot()
    lib = ctypes.CDLL("libnvidia-ml.so.1")
    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
    if lib.nvmlInit_v2() != 0:
        raise RuntimeError("NVML initialization failed")
    try:
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
            raise RuntimeError("NVML device lookup failed")
        mem = Memory()
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
            raise RuntimeError("NVML memory query failed")
        info.update(total_bytes=mem.total, used_bytes=mem.used, free_bytes=mem.free,
                    memory_total_bytes=mem.total, memory_used_bytes=mem.used)
        return info
    finally:
        lib.nvmlShutdown()


def runtime_env(run=None):
    base = CACHE / "runtime"
    mapping = dict(TMPDIR="tmp", TMP="tmp", TEMP="tmp", HF_HOME="huggingface",
                   TORCH_HOME="torch", XDG_CACHE_HOME="xdg", CUDA_CACHE_PATH="cuda",
                   MPLCONFIGDIR="matplotlib", NUMBA_CACHE_DIR="numba", TRITON_CACHE_DIR="triton")
    env = dict(os.environ)
    for key, folder in mapping.items():
        path = base / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0",
               DIFFSYNTH_SKIP_DOWNLOAD="True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4")
    return env
