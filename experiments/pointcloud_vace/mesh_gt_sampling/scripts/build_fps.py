"""Build only the official PyTorch3D FPS operator against the existing PyTorch."""
import os
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[4]
DATA = PROJECT / "data/pointcloud_vace"
RUNTIME = DATA / "mesh_gt_runtime"
DEPS = RUNTIME / "deps"
UPSTREAM = DATA / "third_party/pytorch3d"


def configure():
    for name, subdirectory in {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "PIP_CACHE_DIR": "pip_cache",
        "TORCH_EXTENSIONS_DIR": "build", "CUDA_CACHE_PATH": "cuda_cache", "XDG_CACHE_HOME": "cache",
        "PYTHONPYCACHEPREFIX": "pycache",
    }.items():
        path = RUNTIME / subdirectory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.4"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ["MAX_JOBS"] = "2"
    os.environ["PATH"] = str(DEPS / "bin") + ":/usr/local/cuda-12.4/bin:" + os.environ["PATH"]
    sys.path.insert(0, str(DEPS))


def build():
    configure()
    from torch.utils.cpp_extension import load
    package = DEPS / "pytorch3d"
    (package / "ops").mkdir(parents=True, exist_ok=True)
    for relative in ("__init__.py", "ops/sample_farthest_points.py", "ops/utils.py", "ops/knn.py"):
        shutil.copyfile(UPSTREAM / "pytorch3d" / relative, package / relative)
    shutil.copyfile(UPSTREAM / "LICENSE", package / "LICENSE")
    (package / "ops/__init__.py").write_text(
        '"""FPS-only installation; other PyTorch3D operators are not built."""\n'
        'from .sample_farthest_points import sample_farthest_points\n', encoding="utf-8")
    directory = RUNTIME / "build/pytorch3d_fps"
    directory.mkdir(parents=True, exist_ok=True)
    binding = directory / "binding.cpp"
    binding.write_text(
        '#include "sample_farthest_points/sample_farthest_points.h"\n'
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
        '  m.def("sample_farthest_points", &FarthestPointSampling);\n'
        '}\n', encoding="utf-8")
    source = UPSTREAM / "pytorch3d/csrc"
    module = load(name="_C", sources=[str(binding),
        str(source / "sample_farthest_points/sample_farthest_points_cpu.cpp"),
        str(source / "sample_farthest_points/sample_farthest_points.cu")],
        extra_include_paths=[str(source)], extra_cflags=["-O3", "-DWITH_CUDA"],
        extra_cuda_cflags=["-O3", "-DWITH_CUDA"], build_directory=str(directory), verbose=True)
    shutil.copyfile(module.__file__, package / Path(module.__file__).name)
    print("Built official PyTorch3D FPS:", package, flush=True)


if __name__ == "__main__":
    build()
