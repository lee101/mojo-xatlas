# mojo-xatlas

`mojo-xatlas` is a standalone Mojo implementation of the compute-heavy core of
the [`xatlas`](https://github.com/jpcy/xatlas) Python API: triangle-chart
segmentation, chart parameterization, and texture-atlas packing. It is useful for
lightmap and texture-painting UVs on meshes that can be represented well by
piecewise-planar charts.

This is an independent implementation, not a binding to the C++ xatlas library.
The Python package is named `mojo_xatlas`, so existing covered code can use:

```python
import mojo_xatlas as xatlas
```

## Covered API

The following xatlas 0.0.11 interfaces are implemented:

- `parametrize(positions, indices, normals=None, uvs=None)`
- `Atlas`, including `add_mesh`, `add_uv_mesh`, `generate`, indexing, mesh and
  chart queries, vertex assignments, utilization, and optional chart images
- `ChartOptions`, `PackOptions`, `Chart`, and `ChartType`
- `export(path, positions, indices=None, uvs=None, normals=None)`
- Multiple input meshes packed into one atlas
- Connected-component and dihedral chart segmentation
- Planar projection with vertex duplication only at chart seams
- Deterministic rotation-aware shelf packing, padding, bilinear borders,
  block alignment, explicit texel density, target resolution, and chart-size
  limiting
- Repacking existing UV meshes and honoring input UVs when
  `use_input_mesh_uvs` is enabled

The options use upstream names and defaults. `normal_deviation_weight`,
`max_cost`, and `use_input_mesh_uvs` affect segmentation or parameterization.
`padding`, `bilinear`, `resolution`, `texels_per_unit`, `max_chart_size`,
`blockAlign`, `rotate_charts`, and `create_image` affect packing.

## Not covered

This port deliberately does not yet implement xatlas's LSCM, orthographic, or
piecewise parameterizers; iterative proxy fitting; texture and normal seam
metrics; brute-force packing; arbitrary convex-hull rotation; or multi-page
atlases. The corresponding option fields remain present for source
compatibility, but `max_chart_area`, `max_boundary_length`, the seam and shape
weights, `max_iterations`, `fix_winding`, `bruteForce`, and
`rotate_charts_to_axis` are not applied. Supplied normals are shape-checked but
not currently used for segmentation. `atlas_count` is therefore zero before
generation and one after a non-empty generation.

The `create_image` result is a deterministic chart-rectangle debug image, not
xatlas's triangle-accurate padding raster. Highly curved charts will have more
distortion than upstream xatlas because this port uses planar projection. The
tests compare topology, numerical surface metrics, option defaults, dtypes,
packing scale, and error behavior against the real conda-forge
`xatlas-python` 0.0.11 package.

## Install

Install the pinned Mojo nightly, NumPy, pytest, and upstream xatlas reference
package:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-xatlas.so`.

## Usage

This example is also exercised by the test suite:

```python
import numpy as np
import mojo_xatlas as xatlas

vertices = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    dtype=np.float32,
)
faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

assert vmapping.shape == (4,)
assert indices.shape == (2, 3)
assert uvs.shape == (4, 2)
assert np.all((uvs > 0) & (uvs < 1))
```

Run it from the environment with:

```bash
pixi run python example.py
```

For multiple meshes or custom packing:

```python
atlas = xatlas.Atlas()
atlas.add_mesh(vertices, faces)

pack = xatlas.PackOptions()
pack.resolution = 512
pack.padding = 2
atlas.generate(pack_options=pack)

vmapping, indices, uvs = atlas[0]
print(atlas.width, atlas.height, atlas.utilization)
```

## Verification

```bash
pixi run build
pixi run test
pixi run bench
```

There are 39 tests. They use published xatlas behavior through the real
`xatlas-python` package and analytic plane and cube vectors.

## Benchmarks

Measured on 2026-08-27 on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic. Each value is the best of three end-to-end runs after loading
the Mojo library. The ratio is upstream time divided by Mojo time.

| Case | mojo-xatlas | xatlas 0.0.11 | upstream / Mojo | Result |
|---|---:|---:|---:|---|
| Planar unwrap, 64,800 triangles | 13.61 ms | 183.19 ms | 13.46x | faster |
| Curved unwrap, 39,200 triangles | 8.28 ms | 616.97 ms | 74.54x | faster |
| UV repack, 64,800 triangles | 8.09 ms | 222.73 ms | 27.53x | faster |

The curved case has a particularly large advantage because this port computes
piecewise-planar charts while upstream runs its more sophisticated
parameterization pipeline; the outputs satisfy the covered contract but do not
offer identical parameterization quality.

No GPU path is provided. These kernels do relatively little arithmetic per
input byte, while topology sorting and pointer-heavy edge traversal dominate;
this implementation therefore stays on the CPU.

## How it works

Python converts positions and UVs to C-contiguous row-major `float64` arrays and
indices and scratch buffers to C-contiguous `int64` arrays. NumPy owns every
allocation. Buffer addresses cross `ctypes` as 64-bit integers and the single
Mojo compilation unit reconstructs mutable pointers with
`AnyOrigin[mut=True]`.

Segmentation computes triangle normals, inserts undirected edges into an
open-addressed hash table, and joins adjacent faces with union-find when their
dihedral angle satisfies the chart threshold. Edge, chart-normal, and
union-find initialization use SIMD with scalar tails. UV-only connectivity
skips geometric normal work. Each chart receives an area-weighted normal and a
stable orthonormal 2D basis. Packing estimates a texel density from projected
chart area unless one is supplied, places sorted rectangles into deterministic
shelves, then transforms chart-local coordinates directly into normalized
atlas UVs. Single-chart topology uses a linear dense remap without sorting;
multi-chart topology uses a one-dimensional integer key. Mojo writes final
`float32` UVs directly into NumPy-owned buffers.
