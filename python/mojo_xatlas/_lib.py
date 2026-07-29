"""ctypes access to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "xatlas.mojo")
LIBRARY = os.environ.get("MOJO_XATLAS_LIB", os.path.join(ROOT, "dist", "libmojo-xatlas.so"))

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mxa_segment": ([I, I, I, F, I, I, I, I, I, I, I, I, I], I),
    "mxa_project": ([I, I, I, I, I, I], None),
    "mxa_bounds": ([I, I, I, I, I], None),
    "mxa_pack": ([I, I, I, I, F, I, I, I, I, I, I, I, I, I, I], None),
    "mxa_transform": ([I, I, I, I, I, I, I, F, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if not force and os.path.exists(LIBRARY) and os.path.getmtime(LIBRARY) >= os.path.getmtime(SOURCE):
        return LIBRARY
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run `pixi run build` first")
    os.makedirs(os.path.dirname(LIBRARY), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SOURCE, "-o", LIBRARY],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIBRARY):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIBRARY


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.float32), np.dtype(np.int64)):
        raise TypeError(f"unsupported FFI buffer dtype: {array.dtype}")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    return int(array.ctypes.data)
