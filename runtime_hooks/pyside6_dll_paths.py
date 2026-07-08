"""Ensure bundled PySide6/Shiboken DLLs are visible before Qt modules import."""

import os
import sys


def _prepend_path(path):
    if not path or not os.path.isdir(path):
        return
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except OSError:
            pass
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if not any(os.path.normcase(p) == os.path.normcase(path) for p in parts):
        os.environ["PATH"] = path + os.pathsep + current


_base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
for _rel in ("PySide6", "shiboken6"):
    _prepend_path(os.path.join(_base, _rel))
