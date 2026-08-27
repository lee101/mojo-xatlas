"""Mesh chart segmentation, planar parameterization, and atlas packing kernels."""

from std.math import ceil, iota, sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def clear_edges_range(
    edge_u: IPtr,
    edge_v: IPtr,
    edge_face: IPtr,
    start: Int,
    end: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var fill = SIMD[DType.int64, W](-1)
    var i = start
    while i + W <= end:
        edge_u.store(i, fill)
        edge_v.store(i, fill)
        edge_face.store(i, fill)
        i += W
    while i < end:
        edge_u[i] = -1
        edge_v[i] = -1
        edge_face[i] = -1
        i += 1


def clear_float_range(values: FPtr, end: Int):
    comptime W = simdwidthof[DType.float64]()
    var fill = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= end:
        values.store(i, fill)
        i += W
    while i < end:
        values[i] = 0.0
        i += 1


def compute_face_normal(
    positions: FPtr,
    indices: IPtr,
    face: Int,
    face_normals: FPtr,
):
    var ia = Int(indices[face * 3])
    var ib = Int(indices[face * 3 + 1])
    var ic = Int(indices[face * 3 + 2])
    var ax = positions[ia * 3]
    var ay = positions[ia * 3 + 1]
    var az = positions[ia * 3 + 2]
    var ux = positions[ib * 3] - ax
    var uy = positions[ib * 3 + 1] - ay
    var uz = positions[ib * 3 + 2] - az
    var vx = positions[ic * 3] - ax
    var vy = positions[ic * 3 + 1] - ay
    var vz = positions[ic * 3 + 2] - az
    var nx = uy * vz - uz * vy
    var ny = uz * vx - ux * vz
    var nz = ux * vy - uy * vx
    var length = sqrt(nx * nx + ny * ny + nz * nz)
    if length > 0.0:
        nx /= length
        ny /= length
        nz /= length
    face_normals[face * 3] = nx
    face_normals[face * 3 + 1] = ny
    face_normals[face * 3 + 2] = nz


def find_root(parent: IPtr, value: Int) -> Int:
    var root = value
    while Int(parent[root]) != root:
        root = Int(parent[root])
    var current = value
    while Int(parent[current]) != current:
        var next_value = Int(parent[current])
        parent[current] = Int64(root)
        current = next_value
    return root


def join(parent: IPtr, left: Int, right: Int):
    var a = find_root(parent, left)
    var b = find_root(parent, right)
    if a == b:
        return
    if a < b:
        parent[b] = Int64(a)
    else:
        parent[a] = Int64(b)


def segment(
    positions: FPtr,
    indices: IPtr,
    face_count: Int,
    cosine_threshold: Float64,
    labels: IPtr,
    face_normals: FPtr,
    chart_normals: FPtr,
    parent: IPtr,
    root_chart: IPtr,
    edge_u: IPtr,
    edge_v: IPtr,
    edge_face: IPtr,
    edge_capacity: Int,
) -> Int:
    clear_edges_range(edge_u, edge_v, edge_face, 0, edge_capacity)

    comptime W = simdwidthof[DType.float64]()
    var empty = SIMD[DType.int64, W](-1)
    var f = 0
    while f + W <= face_count:
        parent.store[alignment=1](f, iota[DType.int64, W](Int64(f)))
        root_chart.store[alignment=1](f, empty)
        f += W
    while f < face_count:
        parent[f] = Int64(f)
        root_chart[f] = -1
        f += 1
    if cosine_threshold > -1.0:
        for face in range(face_count):
            compute_face_normal(positions, indices, face, face_normals)

    var mask = edge_capacity - 1
    for f in range(face_count):
        for e in range(3):
            var u = Int(indices[f * 3 + e])
            var v = Int(indices[f * 3 + (e + 1) % 3])
            if u > v:
                var swap_value = u
                u = v
                v = swap_value
            var slot = (u * 1000003 + v * 9176) & mask
            while True:
                if Int(edge_u[slot]) == -1:
                    edge_u[slot] = Int64(u)
                    edge_v[slot] = Int64(v)
                    edge_face[slot] = Int64(f)
                    break
                if Int(edge_u[slot]) == u and Int(edge_v[slot]) == v:
                    var other = Int(edge_face[slot])
                    if cosine_threshold <= -1.0:
                        join(parent, f, other)
                    else:
                        var dot = (
                            face_normals[f * 3] * face_normals[other * 3]
                            + face_normals[f * 3 + 1]
                            * face_normals[other * 3 + 1]
                            + face_normals[f * 3 + 2]
                            * face_normals[other * 3 + 2]
                        )
                        if dot >= cosine_threshold:
                            join(parent, f, other)
                    break
                slot = (slot + 1) & mask

    var chart_count = 0
    for f in range(face_count):
        var root = find_root(parent, f)
        if Int(root_chart[root]) == -1:
            root_chart[root] = Int64(chart_count)
            chart_count += 1
        labels[f] = root_chart[root]
    if cosine_threshold <= -1.0:
        return chart_count
    clear_float_range(chart_normals, chart_count * 3)
    for f in range(face_count):
        var chart = Int(labels[f])
        var ia = Int(indices[f * 3])
        var ib = Int(indices[f * 3 + 1])
        var ic = Int(indices[f * 3 + 2])
        var ux = positions[ib * 3] - positions[ia * 3]
        var uy = positions[ib * 3 + 1] - positions[ia * 3 + 1]
        var uz = positions[ib * 3 + 2] - positions[ia * 3 + 2]
        var vx = positions[ic * 3] - positions[ia * 3]
        var vy = positions[ic * 3 + 1] - positions[ia * 3 + 1]
        var vz = positions[ic * 3 + 2] - positions[ia * 3 + 2]
        chart_normals[chart * 3] += uy * vz - uz * vy
        chart_normals[chart * 3 + 1] += uz * vx - ux * vz
        chart_normals[chart * 3 + 2] += ux * vy - uy * vx
    for chart in range(chart_count):
        var nx = chart_normals[chart * 3]
        var ny = chart_normals[chart * 3 + 1]
        var nz = chart_normals[chart * 3 + 2]
        var length = sqrt(nx * nx + ny * ny + nz * nz)
        if length > 0.0:
            chart_normals[chart * 3] = nx / length
            chart_normals[chart * 3 + 1] = ny / length
            chart_normals[chart * 3 + 2] = nz / length
        else:
            chart_normals[chart * 3 + 2] = 1.0
    return chart_count


def project(
    positions: FPtr,
    mapping: IPtr,
    vertex_charts: IPtr,
    chart_normals: FPtr,
    vertex_count: Int,
    raw_uvs: FPtr,
):
    for i in range(vertex_count):
        var source = Int(mapping[i])
        var chart = Int(vertex_charts[i])
        var nx = chart_normals[chart * 3]
        var ny = chart_normals[chart * 3 + 1]
        var nz = chart_normals[chart * 3 + 2]
        var rx = 0.0
        var ry = 0.0
        var rz = 1.0
        if nz > 0.9 or nz < -0.9:
            ry = 1.0
            rz = 0.0
        var tx = ry * nz - rz * ny
        var ty = rz * nx - rx * nz
        var tz = rx * ny - ry * nx
        var tlength = sqrt(tx * tx + ty * ty + tz * tz)
        tx /= tlength
        ty /= tlength
        tz /= tlength
        var bx = ny * tz - nz * ty
        var by = nz * tx - nx * tz
        var bz = nx * ty - ny * tx
        var px = positions[source * 3]
        var py = positions[source * 3 + 1]
        var pz = positions[source * 3 + 2]
        raw_uvs[i * 2] = px * tx + py * ty + pz * tz
        raw_uvs[i * 2 + 1] = px * bx + py * by + pz * bz


def bounds(
    raw_uvs: FPtr,
    vertex_charts: IPtr,
    vertex_count: Int,
    chart_count: Int,
    chart_bounds: FPtr,
):
    for chart in range(chart_count):
        chart_bounds[chart * 4] = 1.0e300
        chart_bounds[chart * 4 + 1] = 1.0e300
        chart_bounds[chart * 4 + 2] = -1.0e300
        chart_bounds[chart * 4 + 3] = -1.0e300
    for i in range(vertex_count):
        var chart = Int(vertex_charts[i])
        var u = raw_uvs[i * 2]
        var v = raw_uvs[i * 2 + 1]
        if u < chart_bounds[chart * 4]:
            chart_bounds[chart * 4] = u
        if v < chart_bounds[chart * 4 + 1]:
            chart_bounds[chart * 4 + 1] = v
        if u > chart_bounds[chart * 4 + 2]:
            chart_bounds[chart * 4 + 2] = u
        if v > chart_bounds[chart * 4 + 3]:
            chart_bounds[chart * 4 + 3] = v


def pack(
    widths: FPtr,
    heights: FPtr,
    order: IPtr,
    chart_count: Int,
    texels_per_unit: Float64,
    margin: Int,
    target_width: Int,
    rotate_charts: Int,
    block_align: Int,
    xs: IPtr,
    ys: IPtr,
    rotated: IPtr,
    pixel_widths: IPtr,
    pixel_heights: IPtr,
    size_result: IPtr,
):
    var cursor_x = 0
    var cursor_y = 0
    var row_height = 0
    var used_width = 0
    for item in range(chart_count):
        var chart = Int(order[item])
        var content_w = max(1, Int(ceil(widths[chart] * texels_per_unit)))
        var content_h = max(1, Int(ceil(heights[chart] * texels_per_unit)))
        var turn = 0
        if rotate_charts != 0 and content_h > content_w:
            var swap_size = content_w
            content_w = content_h
            content_h = swap_size
            turn = 1
        var rect_w = content_w + margin * 2
        var rect_h = content_h + margin * 2
        if block_align != 0:
            rect_w = ((rect_w + 3) // 4) * 4
            rect_h = ((rect_h + 3) // 4) * 4
        if cursor_x > 0 and cursor_x + rect_w > target_width:
            cursor_y += row_height
            cursor_x = 0
            row_height = 0
        xs[chart] = Int64(cursor_x)
        ys[chart] = Int64(cursor_y)
        rotated[chart] = Int64(turn)
        pixel_widths[chart] = Int64(rect_w)
        pixel_heights[chart] = Int64(rect_h)
        cursor_x += rect_w
        row_height = max(row_height, rect_h)
        used_width = max(used_width, cursor_x)
    size_result[0] = Int64(used_width)
    size_result[1] = Int64(cursor_y + row_height)


def transform(
    raw_uvs: FPtr,
    vertex_charts: IPtr,
    chart_bounds: FPtr,
    xs: IPtr,
    ys: IPtr,
    rotated: IPtr,
    vertex_count: Int,
    texels_per_unit: Float64,
    margin: Int,
    atlas_width: Int,
    atlas_height: Int,
    packed_uvs: F32Ptr,
):
    for i in range(vertex_count):
        var chart = Int(vertex_charts[i])
        var local_u = raw_uvs[i * 2] - chart_bounds[chart * 4]
        var local_v = raw_uvs[i * 2 + 1] - chart_bounds[chart * 4 + 1]
        var px = Float64(Int(xs[chart]) + margin) + 0.5
        var py = Float64(Int(ys[chart]) + margin) + 0.5
        if Int(rotated[chart]) == 0:
            px += local_u * texels_per_unit
            py += local_v * texels_per_unit
        else:
            px += local_v * texels_per_unit
            py += (
                chart_bounds[chart * 4 + 2] - chart_bounds[chart * 4] - local_u
            ) * texels_per_unit
        packed_uvs[i * 2] = Float32(px / Float64(atlas_width))
        packed_uvs[i * 2 + 1] = Float32(py / Float64(atlas_height))


@export("mxa_segment")
def mxa_segment(
    positions: Int,
    indices: Int,
    face_count: Int,
    cosine_threshold: Float64,
    labels: Int,
    face_normals: Int,
    chart_normals: Int,
    parent: Int,
    root_chart: Int,
    edge_u: Int,
    edge_v: Int,
    edge_face: Int,
    edge_capacity: Int,
) abi("C") -> Int:
    return segment(
        FPtr(unsafe_from_address=positions),
        IPtr(unsafe_from_address=indices),
        face_count,
        cosine_threshold,
        IPtr(unsafe_from_address=labels),
        FPtr(unsafe_from_address=face_normals),
        FPtr(unsafe_from_address=chart_normals),
        IPtr(unsafe_from_address=parent),
        IPtr(unsafe_from_address=root_chart),
        IPtr(unsafe_from_address=edge_u),
        IPtr(unsafe_from_address=edge_v),
        IPtr(unsafe_from_address=edge_face),
        edge_capacity,
    )


@export("mxa_project")
def mxa_project(
    positions: Int,
    mapping: Int,
    vertex_charts: Int,
    chart_normals: Int,
    vertex_count: Int,
    raw_uvs: Int,
) abi("C"):
    project(
        FPtr(unsafe_from_address=positions),
        IPtr(unsafe_from_address=mapping),
        IPtr(unsafe_from_address=vertex_charts),
        FPtr(unsafe_from_address=chart_normals),
        vertex_count,
        FPtr(unsafe_from_address=raw_uvs),
    )


@export("mxa_bounds")
def mxa_bounds(
    raw_uvs: Int,
    vertex_charts: Int,
    vertex_count: Int,
    chart_count: Int,
    chart_bounds: Int,
) abi("C"):
    bounds(
        FPtr(unsafe_from_address=raw_uvs),
        IPtr(unsafe_from_address=vertex_charts),
        vertex_count,
        chart_count,
        FPtr(unsafe_from_address=chart_bounds),
    )


@export("mxa_pack")
def mxa_pack(
    widths: Int,
    heights: Int,
    order: Int,
    chart_count: Int,
    texels_per_unit: Float64,
    margin: Int,
    target_width: Int,
    rotate_charts: Int,
    block_align: Int,
    xs: Int,
    ys: Int,
    rotated: Int,
    pixel_widths: Int,
    pixel_heights: Int,
    size_result: Int,
) abi("C"):
    pack(
        FPtr(unsafe_from_address=widths),
        FPtr(unsafe_from_address=heights),
        IPtr(unsafe_from_address=order),
        chart_count,
        texels_per_unit,
        margin,
        target_width,
        rotate_charts,
        block_align,
        IPtr(unsafe_from_address=xs),
        IPtr(unsafe_from_address=ys),
        IPtr(unsafe_from_address=rotated),
        IPtr(unsafe_from_address=pixel_widths),
        IPtr(unsafe_from_address=pixel_heights),
        IPtr(unsafe_from_address=size_result),
    )


@export("mxa_transform")
def mxa_transform(
    raw_uvs: Int,
    vertex_charts: Int,
    chart_bounds: Int,
    xs: Int,
    ys: Int,
    rotated: Int,
    vertex_count: Int,
    texels_per_unit: Float64,
    margin: Int,
    atlas_width: Int,
    atlas_height: Int,
    packed_uvs: Int,
) abi("C"):
    transform(
        FPtr(unsafe_from_address=raw_uvs),
        IPtr(unsafe_from_address=vertex_charts),
        FPtr(unsafe_from_address=chart_bounds),
        IPtr(unsafe_from_address=xs),
        IPtr(unsafe_from_address=ys),
        IPtr(unsafe_from_address=rotated),
        vertex_count,
        texels_per_unit,
        margin,
        atlas_width,
        atlas_height,
        F32Ptr(unsafe_from_address=packed_uvs),
    )
