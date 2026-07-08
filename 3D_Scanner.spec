# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for 3D Scanner.

Build with the project runtime environment:
    D:\scj\envs\ad\python.exe -m PyInstaller 3D_Scanner.spec --clean --noconfirm

Output:
    dist/3D_Scanner/3D_Scanner.exe
"""

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    collect_all,
    copy_metadata,
)

block_cipher = None

try:
    SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
except NameError:
    SPEC_DIR = os.path.dirname(os.path.abspath(SPECPATH))

ROOT_DIR = SPEC_DIR
SRC_DIR = os.path.join(ROOT_DIR, "src")
ENTRY = os.path.join(SRC_DIR, "main.py")


def safe_collect(func, package, *args, **kwargs):
    try:
        return func(package, *args, **kwargs)
    except Exception:
        return []


def drop_conflicting_windows_dlls(entries):
    # Old ICU DLLs collected through transitive packages shadow Windows ICU and
    # break PySide6 QtWidgets with WinError 127 on Windows 10.
    blocked = {"icudt58.dll", "icuuc.dll"}
    filtered = []
    for entry in entries:
        names = []
        for value in entry[:2]:
            if isinstance(value, str):
                names.append(os.path.basename(value).lower())
        if any(name in blocked for name in names):
            continue
        filtered.append(entry)
    return filtered


hiddenimports = [
    # Project packages
    "ui",
    "ui.tk_widget",
    "core",
    "core.obj_loader",
    "core.track",
    "core.mesh_solid",
    "utils",
    "utils.screenshot",
    "utils.model_folder",

    # Qt runtime
    "shiboken6",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",

    # 3D/render stack
    "pyvista",
    "pyvistaqt",
    "qtpy",
    "vtk",

    # Open3D is imported at module load by core.obj_loader.
    "open3d",
    "open3d.cpu.pybind",
]
hiddenimports += safe_collect(collect_submodules, "vtkmodules")

# 使用 collect_all 收集 PySide6 和 shiboken6 的所有内容
pyside6_data, pyside6_binaries, pyside6_hiddenimports = safe_collect(collect_all, "PySide6")
shiboken6_data, shiboken6_binaries, shiboken6_hiddenimports = safe_collect(collect_all, "shiboken6")

datas = []
for package in (
    "pyvista",
    "pyvistaqt",
    "qtpy",
    "open3d",
):
    datas += safe_collect(collect_data_files, package)
    datas += safe_collect(copy_metadata, package)

# 添加 PySide6 和 shiboken6 的资源文件
datas += pyside6_data or []
datas += shiboken6_data or []

binaries = []
for package in ("vtkmodules", "open3d"):
    binaries += safe_collect(collect_dynamic_libs, package)

# 添加 PySide6 和 shiboken6 的二进制文件
binaries += pyside6_binaries or []
binaries += shiboken6_binaries or []
binaries = drop_conflicting_windows_dlls(binaries)

# 添加 PySide6 和 shiboken6 的隐藏导入
hiddenimports += pyside6_hiddenimports or []
hiddenimports += shiboken6_hiddenimports or []


a = Analysis(
    [ENTRY],
    pathex=[ROOT_DIR, SRC_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(ROOT_DIR, "runtime_hooks", "pyside6_dll_paths.py")],
    excludes=[
        # Heavy ML/web packages are not used by this desktop scanner.
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "tensorboard",
        "diffusers",
        "transformers",
        "accelerate",
        "dash",
        "flask",
        "jupyter",
        "notebook",
        "IPython",
        "cv2",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
a.binaries = drop_conflicting_windows_dlls(a.binaries)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="3D_Scanner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="3D_Scanner",
)
