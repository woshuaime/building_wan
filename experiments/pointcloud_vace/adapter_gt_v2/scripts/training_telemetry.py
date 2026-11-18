"""Per-update reporting for an already locked single-GPU training worker."""
import atexit
import ctypes
import json
import os
from pathlib import Path

import common
import worker


class _Memory(ctypes.Structure):
    _fields_ = [(field, ctypes.c_ulonglong) for field in ("total", "free", "used")]


class _DeviceMemory:
    def __init__(self):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.handle = ctypes.c_void_p()
        self.check(self.library.nvmlInit_v2())
        try:
            self.check(self.library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))
        except BaseException:
            self.library.nvmlShutdown()
            raise
        atexit.register(self.library.nvmlShutdown)

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"NVML memory sampling failed ({status})")

    def snapshot(self):
        memory = _Memory()
        self.check(self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)))
        return dict(total_bytes=memory.total, used_bytes=memory.used, free_bytes=memory.free)


def install_telemetry(directory):
    """Only replace the two live display files without a durable write barrier."""
    directory = Path(directory).resolve()
    display = {directory / "progress.json", directory / "current_batch.json"}
    durable_write = common.write_json
    sampler = _DeviceMemory()

    def write(path, value):
        path = Path(path)
        if path not in display:
            return durable_write(path, value)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)

    worker.nvml_snapshot = sampler.snapshot
    common.write_json = worker.write_json = write
