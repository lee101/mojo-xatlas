"""Numerical and behavioral parity checks against xatlas 0.0.11."""

from pathlib import Path

import numpy as np
import pytest
import xatlas

import mojo_xatlas as mx


PLANE_VERTICES = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
)
PLANE_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
CUBE_VERTICES = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float32,
)
CUBE_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.uint32,
)


def generate(module, vertices, faces, chart_options=None, pack_options=None):
    atlas = module.Atlas()
    atlas.add_mesh(vertices, faces)
    kwargs = {}
    if chart_options is not None:
        kwargs["chart_options"] = chart_options
    if pack_options is not None:
        kwargs["pack_options"] = pack_options
    atlas.generate(**kwargs)
    return atlas


def local_pixel_edges(atlas, result, face):
    _, indices, uvs = result
    triangle = indices[face]
    points = uvs[triangle] * np.array([atlas.width, atlas.height])
    return np.sort(
        [
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[0] - points[2]),
        ]
    ) / atlas.texels_per_unit


def test_option_defaults_match_upstream():
    ours_chart, ref_chart = mx.ChartOptions(), xatlas.ChartOptions()
    for name in (
        "max_chart_area",
        "max_boundary_length",
        "normal_deviation_weight",
        "roundness_weight",
        "straightness_weight",
        "normal_seam_weight",
        "texture_seam_weight",
        "max_cost",
        "max_iterations",
        "use_input_mesh_uvs",
        "fix_winding",
    ):
        assert getattr(ours_chart, name) == pytest.approx(getattr(ref_chart, name))
    ours_pack, ref_pack = mx.PackOptions(), xatlas.PackOptions()
    for name in (
        "max_chart_size",
        "padding",
        "texels_per_unit",
        "resolution",
        "bilinear",
        "blockAlign",
        "bruteForce",
        "create_image",
        "rotate_charts_to_axis",
        "rotate_charts",
    ):
        assert getattr(ours_pack, name) == getattr(ref_pack, name)


def test_plane_parametrize_matches_upstream_topology_and_shape():
    ours = mx.parametrize(PLANE_VERTICES, PLANE_FACES)
    ref = xatlas.parametrize(PLANE_VERTICES, PLANE_FACES)
    for got, expected in zip(ours, ref):
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
    assert np.array_equal(ours[0], ref[0])
    assert np.array_equal(ours[1], ref[1])


def test_plane_parameterization_is_numerically_congruent_to_upstream():
    ours_atlas = generate(mx, PLANE_VERTICES, PLANE_FACES)
    ref_atlas = generate(xatlas, PLANE_VERTICES, PLANE_FACES)
    ours = ours_atlas[0]
    ref = ref_atlas[0]
    ours_points = ours[2] * [ours_atlas.width, ours_atlas.height]
    ref_points = ref[2] * [ref_atlas.width, ref_atlas.height]
    ours_distances = np.sort(
        np.linalg.norm(ours_points[:, None] - ours_points[None, :], axis=2).ravel()
    )
    ref_distances = np.sort(
        np.linalg.norm(ref_points[:, None] - ref_points[None, :], axis=2).ravel()
    )
    assert np.allclose(
        ours_distances / ours_distances[-1],
        ref_distances / ref_distances[-1],
        atol=2e-3,
    )


def test_cube_segmentation_matches_upstream():
    ours = generate(mx, CUBE_VERTICES, CUBE_FACES)
    ref = generate(xatlas, CUBE_VERTICES, CUBE_FACES)
    assert ours.chart_count == ref.chart_count == 6
    assert ours.get_mesh_chart_count(0) == ref.get_mesh_chart_count(0) == 6
    assert len(ours[0][0]) == len(ref[0][0]) == 24
    ours_sizes = sorted(len(ours.get_mesh_chart(0, i).faces) for i in range(6))
    ref_sizes = sorted(len(ref.get_mesh_chart(0, i).faces) for i in range(6))
    assert ours_sizes == ref_sizes == [2] * 6


def test_cube_projection_preserves_surface_metric():
    atlas = generate(mx, CUBE_VERTICES, CUBE_FACES)
    result = atlas[0]
    for face in range(len(CUBE_FACES)):
        got = local_pixel_edges(atlas, result, face)
        source = CUBE_VERTICES[CUBE_FACES[face]]
        expected = np.sort(
            [
                np.linalg.norm(source[1] - source[0]),
                np.linalg.norm(source[2] - source[1]),
                np.linalg.norm(source[0] - source[2]),
            ]
        )
        assert np.allclose(got, expected, rtol=5e-3)


def test_results_have_upstream_dtypes_and_uv_range():
    mapping, indices, uvs = mx.parametrize(CUBE_VERTICES, CUBE_FACES)
    assert mapping.dtype == np.uint32
    assert indices.dtype == np.uint32
    assert uvs.dtype == np.float32
    assert np.all((uvs > 0.0) & (uvs < 1.0))
    assert indices.min() == 0
    assert indices.max() == len(mapping) - 1


def test_triangle_winding_and_area_are_valid():
    mapping, indices, uvs = mx.parametrize(CUBE_VERTICES, CUBE_FACES)
    triangles = uvs[indices]
    left = triangles[:, 1] - triangles[:, 0]
    right = triangles[:, 2] - triangles[:, 0]
    signed_area = left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
    assert np.all(np.abs(signed_area) > 1e-8)
    assert len(mapping) == 24


def test_uv_mesh_repacking_matches_upstream_shapes():
    ours, ref = mx.Atlas(), xatlas.Atlas()
    uvs = PLANE_VERTICES[:, :2]
    ours.add_uv_mesh(uvs, PLANE_FACES)
    ref.add_uv_mesh(uvs, PLANE_FACES)
    ours.generate()
    ref.generate()
    assert ours.chart_count == ref.chart_count == 1
    assert ours[0][0].shape == ref[0][0].shape == (4,)
    assert ours[0][1].shape == ref[0][1].shape == (2, 3)
    assert np.allclose(local_pixel_edges(ours, ours[0], 0), [1, 1, np.sqrt(2)], rtol=5e-3)


def test_use_input_mesh_uvs_preserves_input_metric():
    options = mx.ChartOptions()
    options.use_input_mesh_uvs = True
    stretched = PLANE_VERTICES[:, :2] * [3.0, 0.5]
    atlas = mx.Atlas()
    atlas.add_mesh(PLANE_VERTICES, PLANE_FACES, uvs=stretched)
    atlas.generate(chart_options=options)
    edges = local_pixel_edges(atlas, atlas[0], 0)
    assert np.allclose(edges, np.sort([0.5, 3.0, np.hypot(3.0, 0.5)]), rtol=5e-3)


def test_chart_threshold_options_affect_dihedral_segmentation():
    angle = np.deg2rad(55.0)
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, np.cos(angle), np.sin(angle)]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    default = generate(mx, vertices, faces)
    permissive = mx.ChartOptions()
    permissive.normal_deviation_weight = 0
    by_weight = generate(mx, vertices, faces, chart_options=permissive)
    permissive = mx.ChartOptions()
    permissive.max_cost = 7
    by_cost = generate(mx, vertices, faces, chart_options=permissive)
    assert default.chart_count == 2
    assert by_weight.chart_count == by_cost.chart_count == 1


def test_uv_face_materials_split_charts_and_set_metadata():
    atlas = mx.Atlas()
    atlas.add_uv_mesh(
        PLANE_VERTICES[:, :2],
        PLANE_FACES,
        face_materials=np.array([[4], [9]], dtype=np.int32),
    )
    atlas.generate()
    assert atlas.chart_count == 2
    assert {atlas.get_mesh_chart(0, i).material for i in range(2)} == {4, 9}


def test_disconnected_components_become_distinct_charts():
    vertices = np.vstack((PLANE_VERTICES, PLANE_VERTICES + [3, 0, 0]))
    faces = np.vstack((PLANE_FACES, PLANE_FACES + 4))
    atlas = generate(mx, vertices, faces)
    ref = generate(xatlas, vertices, faces)
    assert atlas.chart_count == ref.chart_count == 2


def test_multiple_meshes_share_one_atlas():
    atlas = mx.Atlas()
    atlas.add_mesh(PLANE_VERTICES, PLANE_FACES)
    atlas.add_mesh(CUBE_VERTICES, CUBE_FACES)
    atlas.generate()
    assert atlas.mesh_count == len(atlas) == 2
    assert atlas.atlas_count == 1
    assert atlas.chart_count == 7
    assert atlas[0][1].shape == (2, 3)
    assert atlas[1][1].shape == (12, 3)


def test_vertex_assignment_and_chart_metadata():
    atlas = generate(mx, CUBE_VERTICES, CUBE_FACES)
    atlas_indices, chart_indices = atlas.get_mesh_vertex_assignment(0)
    assert atlas_indices.dtype == chart_indices.dtype == np.uint32
    assert np.array_equal(np.unique(atlas_indices), [0])
    assert np.array_equal(np.unique(chart_indices), np.arange(6))
    chart = atlas.get_mesh_chart(0, 0)
    assert chart.type == mx.ChartType.Planar
    assert chart.atlas_index == 0
    assert chart.material == 0
    assert chart.faces.dtype == np.uint32


def test_pack_resolution_tracks_upstream_scale():
    ours_pack, ref_pack = mx.PackOptions(), xatlas.PackOptions()
    ours_pack.resolution = ref_pack.resolution = 128
    ours = generate(mx, CUBE_VERTICES, CUBE_FACES, pack_options=ours_pack)
    ref = generate(xatlas, CUBE_VERTICES, CUBE_FACES, pack_options=ref_pack)
    assert 0.6 <= ours.texels_per_unit / ref.texels_per_unit <= 1.4
    assert 0.6 <= (ours.width * ours.height) / (ref.width * ref.height) <= 1.6


def test_explicit_texel_density_controls_surface_scale():
    pack = mx.PackOptions()
    pack.texels_per_unit = 32.0
    atlas = generate(mx, PLANE_VERTICES, PLANE_FACES, pack_options=pack)
    assert atlas.texels_per_unit == 32.0
    assert np.allclose(local_pixel_edges(atlas, atlas[0], 0), [1, 1, np.sqrt(2)], rtol=2e-2)


def test_padding_and_bilinear_leave_border():
    pack = mx.PackOptions()
    pack.padding = 4
    atlas = generate(mx, PLANE_VERTICES, PLANE_FACES, pack_options=pack)
    uvs = atlas[0][2] * [atlas.width, atlas.height]
    assert uvs.min() >= 5.0
    assert np.min([atlas.width - uvs[:, 0].max(), atlas.height - uvs[:, 1].max()]) >= 4.0
    no_bilinear = mx.PackOptions()
    no_bilinear.bilinear = False
    compact = generate(mx, PLANE_VERTICES, PLANE_FACES, pack_options=no_bilinear)
    default = generate(mx, PLANE_VERTICES, PLANE_FACES)
    assert compact.width < default.width
    assert compact.height < default.height


def test_chart_image_behavior_matches_upstream():
    atlas = generate(mx, PLANE_VERTICES, PLANE_FACES)
    with pytest.raises(RuntimeError, match="does not have an image"):
        _ = atlas.chart_image
    pack = mx.PackOptions()
    pack.create_image = True
    atlas = generate(mx, PLANE_VERTICES, PLANE_FACES, pack_options=pack)
    image = atlas.chart_image
    assert image.shape == (atlas.height, atlas.width, 3)
    assert image.dtype == np.uint8
    assert image.max() > 0


def test_verbose_output_matches_upstream_labels(capsys):
    atlas = mx.Atlas()
    atlas.add_mesh(PLANE_VERTICES, PLANE_FACES)
    atlas.generate(verbose=True)
    output = capsys.readouterr().out
    assert "Generated Atlas" in output
    assert "Utilization:" in output
    assert "Charts: 1" in output
    assert "Size:" in output


@pytest.mark.parametrize(
    "positions,indices,error",
    [
        (np.ones(3), PLANE_FACES, "Nx3"),
        (np.ones((2, 2)), PLANE_FACES, "Nx3"),
        (PLANE_VERTICES, np.ones(3), "Nx3"),
        (PLANE_VERTICES, np.ones((2, 2)), "Nx3"),
    ],
)
def test_shape_errors_match_upstream(positions, indices, error):
    with pytest.raises(ValueError, match=error):
        mx.Atlas().add_mesh(positions, indices)
    with pytest.raises(ValueError, match=error):
        xatlas.Atlas().add_mesh(positions, indices)


def test_out_of_range_index_error_matches_upstream():
    faces = PLANE_FACES.copy()
    faces[0, 0] = 99
    with pytest.raises(RuntimeError, match="out of range"):
        mx.Atlas().add_mesh(PLANE_VERTICES, faces)
    with pytest.raises(RuntimeError, match="out of range"):
        xatlas.Atlas().add_mesh(PLANE_VERTICES, faces)


@pytest.mark.parametrize(
    "indices",
    [
        np.array([[0.0, 1.5, 2.0]]),
        np.array([[0.0, np.nan, 2.0]]),
        np.array([[0, 1, 2**63]], dtype=object),
    ],
)
def test_indices_reject_silent_narrowing(indices):
    with pytest.raises(ValueError, match="integer|range"):
        mx.Atlas().add_mesh(PLANE_VERTICES, indices)


def test_nonfinite_geometry_and_options_are_rejected():
    vertices = PLANE_VERTICES.copy()
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mx.Atlas().add_mesh(vertices, PLANE_FACES)
    atlas = mx.Atlas()
    atlas.add_mesh(PLANE_VERTICES, PLANE_FACES)
    pack = mx.PackOptions()
    pack.texels_per_unit = np.inf
    with pytest.raises(ValueError, match="finite"):
        atlas.generate(pack_options=pack)
    pack.texels_per_unit = 0
    pack.padding = 1.5
    with pytest.raises(ValueError, match="integer"):
        atlas.generate(pack_options=pack)


def test_input_buffers_are_copied_and_kept_alive():
    vertices = PLANE_VERTICES.copy()
    faces = PLANE_FACES.copy()
    atlas = mx.Atlas()
    atlas.add_mesh(vertices, faces)
    vertices[:] = np.nan
    faces[:] = 99
    del vertices, faces
    atlas.generate()
    assert atlas.chart_count == 1
    assert atlas[0][1].shape == (2, 3)


def test_empty_mesh_behavior_matches_upstream():
    vertices = np.empty((0, 3), dtype=np.float32)
    faces = np.empty((0, 3), dtype=np.uint32)
    ours, ref = mx.Atlas(), xatlas.Atlas()
    ours.add_mesh(vertices, faces)
    ref.add_mesh(vertices, faces)
    ours.generate()
    ref.generate()
    assert ours.mesh_count == ref.mesh_count == 1
    assert ours.chart_count == ref.chart_count == 0
    assert ours.atlas_count == ref.atlas_count == 0
    assert ours.width == ref.width == 0
    assert ours[0][0].shape == ref[0][0].shape == (0,)


def test_query_bounds_errors():
    atlas = mx.Atlas()
    with pytest.raises(IndexError, match="out of bounds"):
        atlas.get_mesh(0)
    atlas.add_mesh(PLANE_VERTICES, PLANE_FACES)
    atlas.generate()
    with pytest.raises(IndexError, match="out of bounds"):
        atlas.get_mesh(1)
    with pytest.raises(IndexError, match="out of bounds"):
        atlas.get_mesh_chart(0, 1)
    with pytest.raises(IndexError, match="out of bounds"):
        atlas.get_utilization(1)


def test_export_writes_upstream_compatible_obj(tmp_path: Path):
    target = tmp_path / "plane.obj"
    mx.export(target, PLANE_VERTICES, PLANE_FACES, PLANE_VERTICES[:, :2])
    lines = target.read_text().splitlines()
    assert sum(line.startswith("v ") for line in lines) == 4
    assert sum(line.startswith("vt ") for line in lines) == 4
    assert sum(line.startswith("f ") for line in lines) == 2
    assert lines[-1] == "f 1/1 3/3 4/4"


def test_export_supports_normals_and_rejects_bad_indices(tmp_path: Path):
    target = tmp_path / "normals.obj"
    normals = np.tile([0, 0, 1], (len(PLANE_VERTICES), 1))
    mx.export(target, PLANE_VERTICES, PLANE_FACES, normals=normals)
    lines = target.read_text().splitlines()
    assert sum(line.startswith("vn ") for line in lines) == 4
    assert lines[-1] == "f 1//1 3//3 4//4"
    with pytest.raises(RuntimeError, match="out of range"):
        mx.export(target, PLANE_VERTICES, np.array([[0, 1, 99]]))


def test_parametrize_is_deterministic():
    first = mx.parametrize(CUBE_VERTICES, CUBE_FACES)
    second = mx.parametrize(CUBE_VERTICES, CUBE_FACES)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_simd_tail_with_nine_disconnected_charts():
    vertices = np.vstack(
        [PLANE_VERTICES[:3] + [offset * 3, 0, 0] for offset in range(9)]
    )
    faces = np.arange(27, dtype=np.uint32).reshape(-1, 3)
    atlas = generate(mx, vertices, faces)
    _, indices, uvs = atlas[0]
    triangles = uvs[indices]
    left = triangles[:, 1] - triangles[:, 0]
    right = triangles[:, 2] - triangles[:, 0]
    areas = np.abs(left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0])
    assert atlas.chart_count == 9
    assert np.all(areas > 1e-8)


def test_single_chart_topology_omits_unreferenced_vertices():
    vertices = np.vstack((PLANE_VERTICES, [[9.0, 9.0, 9.0]]))
    atlas = generate(mx, vertices, PLANE_FACES)
    mapping, indices, _ = atlas[0]
    assert np.array_equal(mapping, np.arange(4, dtype=np.uint32))
    assert np.array_equal(indices, PLANE_FACES)


def test_claimed_packing_controls_affect_output():
    rectangular = PLANE_VERTICES * [1.0, 3.0, 1.0]
    base = generate(mx, rectangular, PLANE_FACES)
    no_rotate_options = mx.PackOptions()
    no_rotate_options.rotate_charts = False
    no_rotate = generate(
        mx, rectangular, PLANE_FACES, pack_options=no_rotate_options
    )
    assert (base.width, base.height) != (no_rotate.width, no_rotate.height)

    cube_base = generate(mx, CUBE_VERTICES, CUBE_FACES)
    limited_options = mx.PackOptions()
    limited_options.max_chart_size = 32
    limited = generate(mx, CUBE_VERTICES, CUBE_FACES, pack_options=limited_options)
    assert limited.texels_per_unit < cube_base.texels_per_unit

    aligned_options = mx.PackOptions()
    aligned_options.blockAlign = True
    aligned = generate(mx, CUBE_VERTICES, CUBE_FACES, pack_options=aligned_options)
    xs, ys, widths, heights = aligned._rects
    assert np.all(widths % 4 == 0)
    assert np.all(heights % 4 == 0)


@pytest.mark.parametrize("size", [255, 257])
def test_segmentation_across_parallel_threshold(size):
    axis = np.linspace(-1.0, 1.0, size + 1, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    vertices = np.ascontiguousarray(
        np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    )
    row = np.arange(size, dtype=np.int64)[:, None] * (size + 1)
    col = np.arange(size, dtype=np.int64)[None, :]
    a = row + col
    b = a + 1
    d = a + size + 1
    c = d + 1
    faces = np.ascontiguousarray(
        np.stack(
            (np.stack((a, b, c), axis=2), np.stack((a, c, d), axis=2)),
            axis=2,
        ).reshape(-1, 3)
    )
    atlas = generate(mx, vertices, faces)
    mapping, new_indices, _ = atlas[0]
    assert atlas.chart_count == 1
    assert len(mapping) == len(vertices)
    assert new_indices.shape == faces.shape
