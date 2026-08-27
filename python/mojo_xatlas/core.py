"""xatlas-compatible mesh unwrapping API backed by Mojo kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
import operator
from pathlib import Path

import numpy as np

from ._lib import addr, lib


class ChartType(IntEnum):
    Planar = 0
    Ortho = 1
    LSCM = 2
    Piecewise = 3
    Invalid = 4


@dataclass
class ChartOptions:
    max_chart_area: float = 0.0
    max_boundary_length: float = 0.0
    normal_deviation_weight: float = 2.0
    roundness_weight: float = np.float32(0.01).item()
    straightness_weight: float = 6.0
    normal_seam_weight: float = 4.0
    texture_seam_weight: float = 0.5
    max_cost: float = 2.0
    max_iterations: int = 1
    use_input_mesh_uvs: bool = False
    fix_winding: bool = False


@dataclass
class PackOptions:
    max_chart_size: int = 0
    padding: int = 0
    texels_per_unit: float = 0.0
    resolution: int = 0
    bilinear: bool = True
    blockAlign: bool = False
    bruteForce: bool = False
    create_image: bool = False
    rotate_charts_to_axis: bool = True
    rotate_charts: bool = True


@dataclass(frozen=True)
class Chart:
    faces: np.ndarray
    atlas_index: int
    type: ChartType
    material: int


@dataclass
class _InputMesh:
    positions: np.ndarray | None
    indices: np.ndarray
    normals: np.ndarray | None = None
    uvs: np.ndarray | None = None
    face_materials: np.ndarray | None = None
    uv_only: bool = False


@dataclass
class _MeshWork:
    mapping: np.ndarray
    indices: np.ndarray
    raw_uvs: np.ndarray
    vertex_charts: np.ndarray
    face_charts: np.ndarray
    bounds: np.ndarray
    charts: list[Chart]
    packed_uvs: np.ndarray | None = None
    global_chart_offset: int = 0


def _array(
    value,
    dtype,
    name: str,
    width: int,
    rows: int | None = None,
    *,
    copy: bool | None = None,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=dtype, order="C", copy=copy)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} array contains values incompatible with {dtype}") from error
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} array expected to be Nx{width}.")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(
            f"{name} array has invalid number of elements in the first dimension "
            f"(expected {rows}, got {array.shape[0]})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} array must contain only finite values.")
    return array


def _integer_array(
    value,
    dtype,
    name: str,
    width: int,
    rows: int | None = None,
) -> np.ndarray:
    original = np.asarray(value)
    if original.ndim != 2 or original.shape[1] != width:
        raise ValueError(f"{name} array expected to be Nx{width}.")
    if rows is not None and original.shape[0] != rows:
        raise ValueError(
            f"{name} array has invalid number of elements in the first dimension "
            f"(expected {rows}, got {original.shape[0]})"
        )
    limits = np.iinfo(dtype)
    if original.dtype.kind in "iu":
        if original.size and (
            (original.dtype.kind == "i" and int(original.min()) < limits.min)
            or int(original.max()) > limits.max
        ):
            raise ValueError(f"{name} array contains a value outside the {dtype} range.")
    elif original.dtype.kind == "f":
        exact = original.astype(np.float64, copy=False)
        upper = float(limits.max) + 1.0
        if (
            not np.all(np.isfinite(exact))
            or not np.all(exact == np.trunc(exact))
            or (exact.size and (exact.min() < limits.min or exact.max() >= upper))
        ):
            raise ValueError(f"{name} array must contain finite integer values in range.")
    else:
        try:
            for item in original.flat:
                integer = operator.index(item)
                if integer < limits.min or integer > limits.max:
                    raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} array must contain integer values in the {dtype} range."
            ) from error
    try:
        return np.ascontiguousarray(original, dtype=dtype)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{name} array contains a value outside the {dtype} range."
        ) from error


def _nonnegative_int(value, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a non-negative integer.") from error
    if result < 0 or result > np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be a non-negative 64-bit integer.")
    return result


def _finite_float(value, name: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = "non-negative " if nonnegative else ""
        raise ValueError(f"{name} must be a finite {qualifier}number.")
    return result


def _validate_options(chart_options: ChartOptions, pack_options: PackOptions) -> None:
    for name in (
        "max_chart_area",
        "max_boundary_length",
        "normal_deviation_weight",
        "roundness_weight",
        "straightness_weight",
        "normal_seam_weight",
        "texture_seam_weight",
        "max_cost",
    ):
        _finite_float(getattr(chart_options, name), f"chart_options.{name}")
    _nonnegative_int(chart_options.max_iterations, "chart_options.max_iterations")
    for name in ("max_chart_size", "padding", "resolution"):
        _nonnegative_int(getattr(pack_options, name), f"pack_options.{name}")
    _finite_float(
        pack_options.texels_per_unit,
        "pack_options.texels_per_unit",
        nonnegative=True,
    )


def _check_indices(indices: np.ndarray, vertex_count: int) -> None:
    if indices.size and (indices.min() < 0 or indices.max() >= vertex_count):
        raise RuntimeError("Adding mesh failed: an index is out of range")


def _edge_capacity(face_count: int) -> int:
    capacity = 8
    while capacity < max(8, face_count * 4):
        capacity <<= 1
    return capacity


def _segment(positions: np.ndarray, indices: np.ndarray, cosine: float):
    face_count = len(indices)
    labels = np.empty(face_count, dtype=np.int64)
    needs_normals = cosine > -1.0
    face_normals = np.empty(
        (face_count if needs_normals else 1, 3), dtype=np.float64
    )
    chart_normals = np.empty(
        (max(face_count, 1) if needs_normals else 1, 3), dtype=np.float64
    )
    parent = np.empty(face_count, dtype=np.int64)
    roots = np.empty(face_count, dtype=np.int64)
    capacity = _edge_capacity(face_count)
    edge_u = np.empty(capacity, dtype=np.int64)
    edge_v = np.empty(capacity, dtype=np.int64)
    edge_face = np.empty(capacity, dtype=np.int64)
    chart_count = lib().mxa_segment(
        addr(positions),
        addr(indices),
        face_count,
        cosine,
        addr(labels),
        addr(face_normals),
        addr(chart_normals),
        addr(parent),
        addr(roots),
        addr(edge_u),
        addr(edge_v),
        addr(edge_face),
        capacity,
    )
    if not needs_normals:
        return labels, np.empty((0, 3), dtype=np.float64)
    return labels, chart_normals[:chart_count].copy()


def _split_materials(labels: np.ndarray, materials: np.ndarray | None) -> np.ndarray:
    if materials is None:
        return labels
    pairs = np.column_stack((labels, materials))
    _, refined = np.unique(pairs, axis=0, return_inverse=True)
    return np.ascontiguousarray(refined, dtype=np.int64)


def _topology(
    mesh: _InputMesh, chart_options: ChartOptions
) -> tuple[np.ndarray, np.ndarray]:
    if mesh.uv_only or (chart_options.use_input_mesh_uvs and mesh.uvs is not None):
        labels, normals = _segment(mesh.uvs, mesh.indices, -1.0)
    else:
        angle = np.clip(
            60.0
            - 5.0 * float(chart_options.normal_deviation_weight)
            + 2.0 * (float(chart_options.max_cost) - 2.0),
            5.0,
            85.0,
        )
        labels, normals = _segment(
            mesh.positions, mesh.indices, math.cos(math.radians(float(angle)))
        )
    refined = _split_materials(labels, mesh.face_materials)
    if not np.array_equal(refined, labels):
        chart_count = int(refined.max()) + 1
        if len(normals):
            new_normals = np.zeros((chart_count, 3), dtype=np.float64)
            for chart in range(chart_count):
                old = labels[np.flatnonzero(refined == chart)[0]]
                new_normals[chart] = normals[old]
            normals = new_normals
    return refined, normals


def _prepare_mesh(mesh: _InputMesh, chart_options: ChartOptions) -> _MeshWork:
    if len(mesh.indices) == 0:
        return _MeshWork(
            mapping=np.empty(0, dtype=np.int64),
            indices=np.empty((0, 3), dtype=np.int64),
            raw_uvs=np.empty((0, 2), dtype=np.float64),
            vertex_charts=np.empty(0, dtype=np.int64),
            face_charts=np.empty(0, dtype=np.int64),
            bounds=np.empty((0, 4), dtype=np.float64),
            charts=[],
        )
    face_charts, chart_normals = _topology(mesh, chart_options)
    chart_count = int(face_charts.max()) + 1
    vertex_count = len(mesh.uvs) if mesh.uv_only else len(mesh.positions)
    if chart_count == 1:
        used = np.zeros(vertex_count, dtype=bool)
        used[mesh.indices] = True
        mapping = np.ascontiguousarray(np.flatnonzero(used), dtype=np.int64)
        vertex_charts = np.zeros(len(mapping), dtype=np.int64)
        if len(mapping) == vertex_count:
            new_indices = mesh.indices
        else:
            remap = np.cumsum(used, dtype=np.int64) - 1
            new_indices = np.ascontiguousarray(remap[mesh.indices], dtype=np.int64)
    else:
        corner_keys = np.repeat(face_charts, 3)
        corner_keys *= vertex_count
        corner_keys += mesh.indices.reshape(-1)
        unique_keys, inverse = np.unique(corner_keys, return_inverse=True)
        mapping = np.ascontiguousarray(unique_keys % vertex_count, dtype=np.int64)
        vertex_charts = np.ascontiguousarray(unique_keys // vertex_count, dtype=np.int64)
        new_indices = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int64)
    if mesh.uv_only or (chart_options.use_input_mesh_uvs and mesh.uvs is not None):
        raw_uvs = np.ascontiguousarray(mesh.uvs[mapping], dtype=np.float64)
    else:
        raw_uvs = np.empty((len(mapping), 2), dtype=np.float64)
        lib().mxa_project(
            addr(mesh.positions),
            addr(mapping),
            addr(vertex_charts),
            addr(chart_normals),
            len(mapping),
            addr(raw_uvs),
        )
    chart_bounds = np.empty((chart_count, 4), dtype=np.float64)
    lib().mxa_bounds(
        addr(raw_uvs),
        addr(vertex_charts),
        len(mapping),
        chart_count,
        addr(chart_bounds),
    )
    charts = []
    for chart in range(chart_count):
        faces = np.flatnonzero(face_charts == chart).astype(np.uint32)
        material = 0
        if mesh.face_materials is not None and len(faces):
            material = int(mesh.face_materials[faces[0]])
        charts.append(Chart(faces, 0, ChartType.Planar, material))
    return _MeshWork(
        mapping=mapping,
        indices=new_indices,
        raw_uvs=raw_uvs,
        vertex_charts=vertex_charts,
        face_charts=face_charts,
        bounds=chart_bounds,
        charts=charts,
    )


class Atlas:
    def __init__(self):
        self._meshes: list[_InputMesh] = []
        self._results: list[_MeshWork] = []
        self._generated = False
        self._width = 0
        self._height = 0
        self._texels_per_unit = 0.0
        self._utilization = 0.0
        self._image: np.ndarray | None = None
        self._rects: tuple[np.ndarray, ...] | None = None

    def add_mesh(self, positions, indices, normals=None, uvs=None):
        positions_array = _array(positions, np.float64, "Position", 3, copy=True)
        indices_array = _integer_array(indices, np.int64, "Index", 3)
        if len(positions_array) and not len(indices_array):
            raise RuntimeError("Adding mesh failed: Invalid index count")
        _check_indices(indices_array, len(positions_array))
        normals_array = (
            None
            if normals is None
            else _array(
                normals,
                np.float64,
                "Normal",
                3,
                len(positions_array),
                copy=True,
            )
        )
        uvs_array = (
            None
            if uvs is None
            else _array(
                uvs,
                np.float64,
                "Texture coordinate",
                2,
                len(positions_array),
                copy=True,
            )
        )
        self._meshes.append(
            _InputMesh(
                positions_array,
                indices_array,
                normals_array,
                uvs_array,
            )
        )
        self._generated = False

    def add_uv_mesh(self, uvs, indices, face_materials=None):
        uvs_array = _array(uvs, np.float64, "Texture coordinate", 2, copy=True)
        indices_array = _integer_array(indices, np.int64, "Index", 3)
        if len(uvs_array) and not len(indices_array):
            raise RuntimeError("Adding mesh failed: Invalid index count")
        _check_indices(indices_array, len(uvs_array))
        material_array = None
        if face_materials is not None:
            material_array = _integer_array(
                face_materials,
                np.int64,
                "Face material ID",
                1,
                len(indices_array),
            )
            material_array = np.ascontiguousarray(material_array[:, 0])
        self._meshes.append(
            _InputMesh(
                None,
                indices_array,
                uvs=uvs_array,
                face_materials=material_array,
                uv_only=True,
            )
        )
        self._generated = False

    def generate(
        self,
        chart_options=ChartOptions(),
        pack_options=PackOptions(),
        verbose=False,
    ):
        if not isinstance(chart_options, ChartOptions):
            raise TypeError("chart_options must be a ChartOptions instance")
        if not isinstance(pack_options, PackOptions):
            raise TypeError("pack_options must be a PackOptions instance")
        _validate_options(chart_options, pack_options)
        if not self._meshes:
            self._results = []
            self._generated = True
            return
        self._results = [_prepare_mesh(mesh, chart_options) for mesh in self._meshes]
        total_charts = sum(len(result.charts) for result in self._results)
        if total_charts == 0:
            for result in self._results:
                result.packed_uvs = np.empty((0, 2), dtype=np.float32)
            self._width = self._height = 0
            self._texels_per_unit = self._utilization = 0.0
            self._rects = None
            self._image = None
            self._generated = True
            return
        all_bounds = np.empty((total_charts, 4), dtype=np.float64)
        offset = 0
        for result in self._results:
            count = len(result.charts)
            result.global_chart_offset = offset
            all_bounds[offset : offset + count] = result.bounds
            offset += count
        widths = np.ascontiguousarray(all_bounds[:, 2] - all_bounds[:, 0])
        heights = np.ascontiguousarray(all_bounds[:, 3] - all_bounds[:, 1])
        target = int(pack_options.resolution) or 1024
        content_area = float(np.sum(np.maximum(widths * heights, 1e-12)))
        if pack_options.texels_per_unit > 0:
            texels_per_unit = float(pack_options.texels_per_unit)
        else:
            texels_per_unit = math.sqrt(0.8 * target * target / content_area)
        if pack_options.max_chart_size > 0:
            largest = float(np.max(np.maximum(widths, heights)))
            if largest * texels_per_unit > pack_options.max_chart_size:
                texels_per_unit = pack_options.max_chart_size / largest
        margin = max(0, int(pack_options.padding)) + int(bool(pack_options.bilinear))
        estimated_heights = np.ceil(heights * texels_per_unit) + 2 * margin
        estimated_widths = np.ceil(widths * texels_per_unit) + 2 * margin
        order = np.ascontiguousarray(
            np.argsort(-np.maximum(estimated_heights, estimated_widths), kind="stable"),
            dtype=np.int64,
        )
        xs = np.empty(total_charts, dtype=np.int64)
        ys = np.empty(total_charts, dtype=np.int64)
        rotated = np.empty(total_charts, dtype=np.int64)
        pixel_widths = np.empty(total_charts, dtype=np.int64)
        pixel_heights = np.empty(total_charts, dtype=np.int64)
        size_result = np.empty(2, dtype=np.int64)
        lib().mxa_pack(
            addr(widths),
            addr(heights),
            addr(order),
            total_charts,
            texels_per_unit,
            margin,
            max(1, int(math.ceil(target * 1.25))),
            int(bool(pack_options.rotate_charts)),
            int(bool(pack_options.blockAlign)),
            addr(xs),
            addr(ys),
            addr(rotated),
            addr(pixel_widths),
            addr(pixel_heights),
            addr(size_result),
        )
        self._width = max(1, int(size_result[0]))
        self._height = max(1, int(size_result[1]))
        self._texels_per_unit = texels_per_unit
        content_pixels = np.sum(
            np.ceil(widths * texels_per_unit) * np.ceil(heights * texels_per_unit)
        )
        self._utilization = float(
            np.clip(content_pixels / (self._width * self._height), 0.0, 1.0)
        )
        for result in self._results:
            if result.global_chart_offset:
                global_charts = np.ascontiguousarray(
                    result.vertex_charts + result.global_chart_offset,
                    dtype=np.int64,
                )
            else:
                global_charts = result.vertex_charts
            packed = np.empty(result.raw_uvs.shape, dtype=np.float32)
            lib().mxa_transform(
                addr(result.raw_uvs),
                addr(global_charts),
                addr(all_bounds),
                addr(xs),
                addr(ys),
                addr(rotated),
                len(result.mapping),
                texels_per_unit,
                margin,
                self._width,
                self._height,
                addr(packed),
            )
            result.packed_uvs = packed
        self._rects = (xs, ys, pixel_widths, pixel_heights)
        self._image = self._make_image(margin) if pack_options.create_image else None
        self._generated = True
        if verbose:
            print("--- Generated Atlas ---")
            print(f"Utilization: {self._utilization * 100:.6f}%")
            print(f"Charts: {total_charts}")
            print(f"Size: {self._width}x{self._height}")
            print()

    def _make_image(self, margin: int) -> np.ndarray:
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        xs, ys, widths, heights = self._rects
        for chart in range(len(xs)):
            x0, y0 = int(xs[chart]), int(ys[chart])
            x1, y1 = x0 + int(widths[chart]), y0 + int(heights[chart])
            image[y0:y1, x0:x1] = (0, 0, 255)
            seed = np.random.default_rng(chart)
            color = ((seed.integers(0, 255, 3) + 192) * 0.5).astype(np.uint8)
            image[y0 + margin : y1 - margin, x0 + margin : x1 - margin] = color
        return image

    def _mesh(self, mesh_index: int) -> _MeshWork:
        if not self._generated:
            raise IndexError(
                f"Mesh index {mesh_index} out of bounds for atlas with 0 meshes."
            )
        if mesh_index < 0 or mesh_index >= len(self._results):
            raise IndexError(
                f"Mesh index {mesh_index} out of bounds for atlas with "
                f"{len(self._results)} meshes."
            )
        return self._results[mesh_index]

    def get_mesh(self, mesh_index):
        result = self._mesh(int(mesh_index))
        return (
            result.mapping.astype(np.uint32),
            result.indices.astype(np.uint32),
            result.packed_uvs.copy(),
        )

    def get_mesh_vertex_assignment(self, mesh_index):
        result = self._mesh(int(mesh_index))
        return (
            np.zeros(len(result.mapping), dtype=np.uint32),
            result.vertex_charts.astype(np.uint32),
        )

    def get_mesh_chart_count(self, mesh_index):
        return len(self._mesh(int(mesh_index)).charts)

    def get_mesh_chart(self, mesh_index, chart_index):
        charts = self._mesh(int(mesh_index)).charts
        chart_index = int(chart_index)
        if chart_index < 0 or chart_index >= len(charts):
            raise IndexError(
                f"Chart index {chart_index} out of bounds for mesh with "
                f"{len(charts)} charts."
            )
        return charts[chart_index]

    def get_utilization(self, atlas_index):
        if int(atlas_index) != 0 or not self._generated or not self._results:
            raise IndexError(f"Atlas index {atlas_index} out of bounds.")
        return self._utilization

    def get_chart_image(self, atlas_index):
        if int(atlas_index) != 0 or not self._generated or not self._results:
            raise IndexError(f"Atlas index {atlas_index} out of bounds.")
        if self._image is None:
            raise RuntimeError("The atlas does not have an image.")
        return self._image.copy()

    @property
    def atlas_count(self):
        return int(self._generated and self.chart_count > 0)

    @property
    def mesh_count(self):
        return len(self._meshes)

    @property
    def chart_count(self):
        return sum(len(result.charts) for result in self._results)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def texels_per_unit(self):
        return self._texels_per_unit

    @property
    def utilization(self):
        return self.get_utilization(0)

    @property
    def chart_image(self):
        return self.get_chart_image(0)

    def __len__(self):
        return self.mesh_count

    def __getitem__(self, mesh_index):
        return self.get_mesh(mesh_index)


def parametrize(positions, indices, normals=None, uvs=None):
    atlas = Atlas()
    atlas.add_mesh(positions, indices, normals, uvs)
    atlas.generate()
    return atlas[0]


def export(path, positions, indices=None, uvs=None, normals=None):
    positions_array = _array(positions, np.float32, "Position", 3)
    indices_array = (
        None
        if indices is None
        else _integer_array(indices, np.uint32, "Index", 3)
    )
    if indices_array is not None:
        _check_indices(indices_array, len(positions_array))
    normals_array = (
        None
        if normals is None
        else _array(normals, np.float32, "Normal", 3, len(positions_array))
    )
    uvs_array = (
        None
        if uvs is None
        else _array(uvs, np.float32, "Texture coordinates", 2, len(positions_array))
    )
    target = Path(path)
    try:
        handle = target.open("w", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Cannot open path {path}") from error
    with handle:
        for x, y, z in positions_array:
            handle.write(f"v {x:g} {y:g} {z:g}\n")
        if normals_array is not None:
            for x, y, z in normals_array:
                handle.write(f"vn {x:g} {y:g} {z:g}\n")
        if uvs_array is not None:
            for u, v in uvs_array:
                handle.write(f"vt {u:g} {v:g}\n")
        if indices_array is not None:
            for face in indices_array:
                values = []
                for index in face + 1:
                    if normals_array is not None and uvs_array is not None:
                        values.append(f"{index}/{index}/{index}")
                    elif normals_array is not None:
                        values.append(f"{index}//{index}")
                    elif uvs_array is not None:
                        values.append(f"{index}/{index}")
                    else:
                        values.append(str(index))
                handle.write("f " + " ".join(values) + "\n")
