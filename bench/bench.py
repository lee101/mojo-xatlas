"""End-to-end mojo-xatlas benchmarks against xatlas 0.0.11."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_xatlas as mx  # noqa: E402
import xatlas  # noqa: E402


def grid(size: int, wave: float = 0.0):
    axis = np.linspace(-1.0, 1.0, size + 1, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    z = wave * np.sin(x * 4.0) * np.cos(y * 3.0)
    vertices = np.ascontiguousarray(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
    row = np.arange(size, dtype=np.uint32)[:, None] * (size + 1)
    col = np.arange(size, dtype=np.uint32)[None, :]
    a = row + col
    b = a + 1
    d = a + size + 1
    c = d + 1
    faces = np.ascontiguousarray(
        np.stack((np.stack((a, b, c), axis=2), np.stack((a, c, d), axis=2)), axis=2)
        .reshape(-1, 3)
        .astype(np.uint32)
    )
    return vertices, faces, np.ascontiguousarray(vertices[:, :2])


def unwrap(module, vertices, faces):
    atlas = module.Atlas()
    atlas.add_mesh(vertices, faces)
    atlas.generate()
    return atlas


def repack(module, uvs, faces):
    atlas = module.Atlas()
    atlas.add_uv_mesh(uvs, faces)
    atlas.generate()
    return atlas


def timeit(function, repeats=3):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    flat_v, flat_f, flat_uv = grid(180)
    wave_v, wave_f, _ = grid(140, 0.08)
    cases = [
        (
            "Planar unwrap, 64,800 triangles",
            lambda: unwrap(mx, flat_v, flat_f),
            lambda: unwrap(xatlas, flat_v, flat_f),
        ),
        (
            "Curved unwrap, 39,200 triangles",
            lambda: unwrap(mx, wave_v, wave_f),
            lambda: unwrap(xatlas, wave_v, wave_f),
        ),
        (
            "UV repack, 64,800 triangles",
            lambda: repack(mx, flat_uv, flat_f),
            lambda: repack(xatlas, flat_uv, flat_f),
        ),
    ]
    unwrap(mx, flat_v[:4], np.array([[0, 1, 2]], dtype=np.uint32))
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| Case | mojo-xatlas | xatlas 0.0.11 | upstream / Mojo | Result |")
    print("|---|---:|---:|---:|---|")
    for name, ours, upstream in cases:
        mojo_seconds = timeit(ours)
        upstream_seconds = timeit(upstream)
        ratio = upstream_seconds / mojo_seconds
        result = "faster" if ratio > 1.0 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
