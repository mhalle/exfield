"""Export exfield meshes to VTK ``.vtu`` files.

Two modes, one exporter:

* **Bezier (default)** — every element becomes a VTK Bezier cell
  (``vtkBezierCurve``/``Quadrilateral``/``Hexahedron``) whose control
  points are the exact Bernstein coefficients of the element geometry
  (see ``exfield.bounds``). The written geometry is mathematically
  identical to the scaffold: no resolution parameter, no cracks.
  Verified against VTK's own cell evaluation at ~1e-16.
* **Tessellated** (``tessellate=N``) — linear line/quad/hex cells on a
  per-element xi lattice, for consumers without higher-order support
  (some Slicer filters, vtk.js, quick-look tools).

This module targets the *format*, not the library: ``.vtu`` is XML +
base64, written directly, so exfield stays NumPy-only. The ``vtk``
package appears only in dev-time tests as an ordering/read-back oracle
(run those via ``uv run --with vtk pytest tests/test_vtu.py``).

VTK's higher-order point ordering (corners, then edges, then faces,
then body — with defined edge directions) is transcribed from
``vtkHigherOrderHexahedron::PointIndexFromIJK`` and pinned against
tables dumped from VTK 9.6 in ``tests/data/vtk_ordering_tables.json``.
Getting an edge direction wrong is silently-warped geometry; do not
edit :func:`point_index_from_ijk` without those tests.

Face-inherited elements (the 1-D/2-D elements that appear only in
``Faces:`` lists) export exactly: restricting a Bezier lattice to an
axis-aligned face is a slice of the parent's control lattice.
"""

import base64
import struct
import warnings
import zlib
from xml.sax.saxutils import quoteattr

import numpy as np

from .bounds import element_control_points
from .errors import EvaluationError
from .evaluate import Evaluator
from .mesh import Field, Mesh

# VTK cell type ids
VTK_VERTEX = 1
VTK_LINE = 3
VTK_QUAD = 9
VTK_HEXAHEDRON = 12
VTK_BEZIER_CURVE = 75
VTK_BEZIER_QUADRILATERAL = 77
VTK_BEZIER_HEXAHEDRON = 79


# ------------------------------------------------- VTK point ordering


def point_index_from_ijk(ijk, orders):
    """VTK higher-order point id for lattice position ``ijk``.

    Transcribed from VTK's ``vtkHigherOrderCurve``/``Quadrilateral``/
    ``Hexahedron`` ``PointIndexFromIJK``; pinned against tables dumped
    from VTK 9.6.2 (tests/data/vtk_ordering_tables.json).
    """
    dim = len(orders)
    if dim == 1:
        i, = ijk
        o, = orders
        if i == 0:
            return 0
        if i == o:
            return 1
        return i + 1
    if dim == 2:
        i, j = ijk
        o1, o2 = orders
        ibdy = i == 0 or i == o1
        jbdy = j == 0 or j == o2
        if ibdy and jbdy:  # corner
            return (2 if j else 1) if i else (3 if j else 0)
        offset = 4
        if jbdy:  # on an i-axis edge
            return (i - 1) + (o1 - 1 + o2 - 1 if j else 0) + offset
        if ibdy:  # on a j-axis edge
            return (j - 1) + ((o1 - 1) if i else 2 * (o1 - 1) + (o2 - 1)) \
                + offset
        offset += 2 * (o1 - 1 + o2 - 1)
        return offset + (i - 1) + (o1 - 1) * (j - 1)
    i, j, k = ijk
    o1, o2, o3 = orders
    ibdy = i == 0 or i == o1
    jbdy = j == 0 or j == o2
    kbdy = k == 0 or k == o3
    nbdy = int(ibdy) + int(jbdy) + int(kbdy)
    if nbdy == 3:  # corner
        return ((2 if j else 1) if i else (3 if j else 0)) + (4 if k else 0)
    offset = 8
    if nbdy == 2:  # edge
        if not ibdy:  # i-axis edge
            return (i - 1) + (o1 - 1 + o2 - 1 if j else 0) \
                + (2 * (o1 - 1 + o2 - 1) if k else 0) + offset
        if not jbdy:  # j-axis edge
            return (j - 1) + ((o1 - 1) if i else 2 * (o1 - 1) + (o2 - 1)) \
                + (2 * (o1 - 1 + o2 - 1) if k else 0) + offset
        # k-axis edge
        offset += 4 * (o1 - 1) + 4 * (o2 - 1)
        return (k - 1) + (o3 - 1) * ((2 if j else 1) if i else
                                     (3 if j else 0)) + offset
    offset += 4 * (o1 - 1 + o2 - 1 + o3 - 1)
    if nbdy == 1:  # face
        if ibdy:  # i-normal face
            return (j - 1) + (o2 - 1) * (k - 1) \
                + ((o2 - 1) * (o3 - 1) if i else 0) + offset
        offset += 2 * (o2 - 1) * (o3 - 1)
        if jbdy:  # j-normal face
            return (i - 1) + (o1 - 1) * (k - 1) \
                + ((o3 - 1) * (o1 - 1) if j else 0) + offset
        offset += 2 * (o3 - 1) * (o1 - 1)
        # k-normal face
        return (i - 1) + (o1 - 1) * (j - 1) \
            + ((o1 - 1) * (o2 - 1) if k else 0) + offset
    # body
    offset += 2 * ((o2 - 1) * (o3 - 1) + (o3 - 1) * (o1 - 1)
                   + (o1 - 1) * (o2 - 1))
    return offset + (i - 1) + (o1 - 1) * ((j - 1) + (o2 - 1) * (k - 1))


_BEZIER_TYPE = {1: VTK_BEZIER_CURVE, 2: VTK_BEZIER_QUADRILATERAL,
                3: VTK_BEZIER_HEXAHEDRON}
_LINEAR_TYPE = {1: VTK_LINE, 2: VTK_QUAD, 3: VTK_HEXAHEDRON}


# ------------------------------------------------- control lattices


def _control_lattice(evaluator, element_id):
    """Bernstein control lattice of the evaluator's field on an element.

    Returns ``(values, orders)`` with lattice index ``m = p1 + n1*(p2 +
    n2*p3)`` (xi1 fastest). Face-inherited elements are exact: with our
    axis-aligned face maps, the restriction of a Bezier lattice to a
    face is a slice (with axis permutation) of the parent's lattice.
    """
    dimension, def_eid, A, b = evaluator._resolve(element_id)
    P, basis = evaluator.element_parameters(def_eid, dimension)
    if basis is None:
        raise EvaluationError(
            f"Cannot export constant field on element {element_id}")
    ctrl = element_control_points(P, basis)
    orders = list(basis.orders)
    if A is None:
        return ctrl, orders
    D = len(orders)
    n = [o + 1 for o in orders]
    d = A.shape[1]
    child_orders = []
    for c in range(d):
        child_orders.append(orders[int(np.argmax(A[:, c]))])
    child_n = [o + 1 for o in child_orders]
    total = int(np.prod(child_n))
    out = np.empty((total, ctrl.shape[1]))
    for m2 in range(total):
        rem = m2
        p = []
        for cn in child_n:
            p.append(rem % cn)
            rem //= cn
        idx = [0] * D
        for a in range(D):
            if A[a].any():
                idx[a] = p[int(np.argmax(A[a]))]
            else:
                idx[a] = 0 if b[a] < 0.5 else orders[a]
        m = idx[0]
        if D >= 2:
            m += n[0] * idx[1]
        if D >= 3:
            m += n[0] * n[1] * idx[2]
        out[m2] = ctrl[m]
    return out, child_orders


# ------------------------------------------------------ point dedup


class _PointPool:
    """Shared point list with tolerance-based dedup.

    The dedup key includes any attached point-data values, so coincident
    points carrying *different* field values (seams, discontinuities)
    are kept separate rather than silently merged. ``quanta`` is a
    per-component vector — scaled geometry components and unscaled
    field components live at different magnitudes and must not share a
    quantum.
    """

    def __init__(self, quanta):
        self.quanta = np.asarray(quanta, dtype=float)
        self.points = []
        self.data = []           # per point: concatenated extra values
        self._index = {}

    def add(self, xyz, extra):
        key_vec = np.concatenate([xyz, extra]) if len(extra) else xyz
        key = tuple(np.round(
            key_vec / self.quanta[:len(key_vec)]).astype(np.int64))
        idx = self._index.get(key)
        if idx is None:
            idx = len(self.points)
            self._index[key] = idx
            # Copy, don't alias: the tessellation caller passes a row
            # view of a whole-element lattice, so keeping the view would
            # both expose us to later mutation of the caller's array and
            # pin every element lattice — including the rows dedup threw
            # away — alive for the length of the export.
            self.points.append(np.array(xyz, dtype=float))
            self.data.append(np.array(extra, dtype=float))
        return idx


# ------------------------------------------------------- XML writer


def _b64(array):
    raw = np.ascontiguousarray(array).tobytes()
    payload = zlib.compress(raw)
    header = struct.pack("<QQQQ", 1, len(raw), len(raw), len(payload))
    return (base64.b64encode(header).decode("ascii")
            + base64.b64encode(payload).decode("ascii"))


_VTK_TYPE_NAME = {np.dtype(np.float64): "Float64",
                  np.dtype(np.int64): "Int64",
                  np.dtype(np.int32): "Int32",
                  np.dtype(np.uint8): "UInt8"}


def _data_array(name, array, n_components=1):
    array = np.ascontiguousarray(array)
    type_name = _VTK_TYPE_NAME[array.dtype]
    return (f'<DataArray type="{type_name}" Name={quoteattr(name)} '
            f'NumberOfComponents="{n_components}" format="binary">\n'
            f"{_b64(array)}\n</DataArray>\n")


def _points_array(points):
    """Geometry as the (n, 3) array VTK's ``Points`` requires.

    That array is *always* 3-component. exfield coordinate fields are
    not — a 2-D scaffold carries 2-component coordinates — and writing
    those straight into a ``NumberOfComponents="3"`` array does not
    fail: VTK reads consecutive (x, y) pairs as (x, y, z) triples, so
    the point count silently drops by a third and every coordinate
    lands on the wrong axis. Pad the missing components with zeros (the
    convention VTK consumers expect for planar data) and refuse more
    than three, which the format cannot represent at all.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 3))
    if array.ndim != 2:
        raise ValueError(
            f"points must be (n, n_components), got shape {array.shape}")
    n_comp = array.shape[1]
    if n_comp > 3:
        raise ValueError(
            f"Cannot write a {n_comp}-component field as VTK geometry: a "
            f".vtu Points array holds exactly 3 components. Export a "
            f"coordinate field with 3 components or fewer.")
    if n_comp == 3:
        return array
    padded = np.zeros((len(array), 3))
    padded[:, :n_comp] = array
    return padded


# VTU file-format version and higher-order point ordering.
#
# A .vtu declaring version <= 2.0 is read as using the OLD higher-order
# hexahedron point ordering, in which the interior points of the two
# xi3 edges on the xi2=max face (VTK edges 10 and 11) are swapped
# relative to PointIndexFromIJK. VTK's reader silently applies that
# swap to such files, so a correctly ordered tricubic lattice comes
# back with those edges exchanged -- visible only on cells whose xi3
# axis carries interior control points (degree >= 2 in xi3 together
# with a hexahedral cell). We write the corrected ordering, so any
# grid containing higher-order cells must declare >= 2.1. Grids of
# purely linear cells are unaffected and stay at 1.0 for maximum
# consumer compatibility.
_VTU_VERSION_LINEAR = "1.0"
_VTU_VERSION_HIGHER_ORDER = "2.1"


def _write_vtu(path, points, connectivity, offsets, types, cell_data,
               point_data, degrees):
    version = (_VTU_VERSION_HIGHER_ORDER if degrees is not None
               else _VTU_VERSION_LINEAR)
    points = _points_array(points)
    out = []
    out.append(f'<VTKFile type="UnstructuredGrid" version="{version}" '
               'byte_order="LittleEndian" header_type="UInt64" '
               'compressor="vtkZLibDataCompressor">\n')
    out.append("<UnstructuredGrid>\n")
    out.append(f'<Piece NumberOfPoints="{len(points)}" '
               f'NumberOfCells="{len(types)}">\n')
    out.append("<Points>\n")
    out.append(_data_array("Points", points, n_components=3))
    out.append("</Points>\n")
    out.append("<Cells>\n")
    out.append(_data_array("connectivity",
                           np.asarray(connectivity, dtype=np.int64)))
    out.append(_data_array("offsets", np.asarray(offsets, dtype=np.int64)))
    out.append(_data_array("types", np.asarray(types, dtype=np.uint8)))
    out.append("</Cells>\n")
    attrs = ""
    if degrees is not None:
        attrs = ' HigherOrderDegrees="HigherOrderDegrees"'
    out.append(f"<CellData{attrs}>\n")
    if degrees is not None:
        out.append(_data_array("HigherOrderDegrees",
                               np.asarray(degrees, dtype=np.int32),
                               n_components=3))
    for name, array in cell_data.items():
        array = np.asarray(array)
        if array.dtype != np.uint8:
            array = array.astype(np.int64)
        out.append(_data_array(name, array))
    out.append("</CellData>\n")
    out.append("<PointData>\n")
    for name, array in point_data.items():
        array = np.asarray(array, dtype=np.float64)
        n_comp = 1 if array.ndim == 1 else array.shape[1]
        out.append(_data_array(name, array, n_components=n_comp))
    out.append("</PointData>\n")
    out.append("</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")
    with open(path, "w", encoding="ascii") as fh:
        fh.write("".join(out))


# ---------------------------------------------------------- exporter


def _quiet_evaluator(field, dimension):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Evaluator(field, dimension=dimension)


def _resolve_field(mesh, field):
    if field is None:
        return mesh.coordinates
    if isinstance(field, str):
        return mesh.fields[field]
    if isinstance(field, Field):
        return field
    raise TypeError(f"field must be a name or Field, not {type(field)}")


def export_vtu(mesh, path, field=None, dimension=None, element_ids=None,
               groups=True, extra_fields=(), tessellate=None, dedup=True,
               scale=1.0):
    """Export one element-mesh dimension of a Mesh to a ``.vtu`` file.

    Parameters
    ----------
    mesh : exfield Mesh.
    path : output .vtu path.
    field : geometry field (name/Field; default ``mesh.coordinates``).
    dimension : element dimension to export (default: highest present).
        Export other dimensions as separate files — mixed-dimension
        grids confuse many consumers.
    element_ids : subset of elements (default all in that dimension).
    groups : write one 0/1 cell-data array per annotation group (True),
        or an explicit list of group names.
    extra_fields : real-valued fields to attach as point data. In
        Bezier mode a field must share the geometry's basis per element
        and is then attached via its own Bernstein control values —
        which VTK interpolates with the cell's shape functions, so
        probing the field downstream is EXACT. In tessellate mode
        fields are evaluated at the lattice points.
    tessellate : None for exact Bezier cells; an int N for linear cells
        with N segments along each cubic axis (linear axes keep 1
        segment); or a per-axis tuple.
    dedup : share coincident points across cells (tolerance-keyed,
        including attached field values). Makes tessellated surfaces
        watertight.
    scale : multiply geometry coordinates (e.g. 1e-3 for µm -> mm).

    Returns a summary dict (points, cells, path, mode).
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("export_vtu takes an exfield Mesh")
    if scale == 0:
        raise ValueError("scale=0 would collapse all geometry to the "
                         "origin; pass the unit conversion you mean")
    geometry = _resolve_field(mesh, field)
    if dimension is None:
        dimension = mesh.highest_dimension
        if dimension == 0:
            raise ValueError("Mesh has no elements")
    element_mesh = mesh.element_meshes.get(dimension)
    if element_mesh is None or len(element_mesh) == 0:
        raise ValueError(f"No mesh{dimension}d elements to export")
    ids = sorted(element_ids) if element_ids is not None \
        else sorted(element_mesh.elements)
    evaluator = _quiet_evaluator(geometry, dimension)
    extra = [(f.name, _quiet_evaluator(f, dimension))
             for f in (_resolve_field(mesh, x) for x in extra_fields)]

    # An extra field must cover every element being exported. VTK point
    # data is one array over the whole grid, so a field defined on only
    # some elements (ordinary in EX — templates differ element to
    # element) has no representation. Check up front: otherwise the
    # first uncovered element aborts the export mid-write with a bare
    # "not defined on element N" from the resolver, after the caller has
    # already paid for everything before it. Resolves are cached, so the
    # main loop pays nothing for this pass.
    for name, f_ev in extra:
        missing = []
        for eid in ids:
            try:
                f_ev._resolve(eid)
            except EvaluationError:
                missing.append(eid)
                if len(missing) > 3:
                    break
        if missing:
            shown = ", ".join(str(e) for e in missing[:3])
            more = "" if len(missing) <= 3 else ", ..."
            raise ValueError(
                f"extra field {name!r} is not defined on element(s) "
                f"{shown}{more} of mesh{dimension}d, which are being "
                f"exported. VTK point data must cover every cell: pass "
                f"element_ids= restricted to where the field is defined, "
                f"or drop it from extra_fields.")

    # Per-component dedup quanta from a cheap bbox probe. The dedup key
    # concatenates SCALED geometry with UNSCALED extra-field values, so
    # each half needs a quantum in its own units: one shared quantum
    # either merges real field seams at large scale or overflows the
    # int64 key at small scale.
    probe = ids[:: max(1, len(ids) // 20)][:20]
    span = 0.0
    extra_spans = [0.0] * len(extra)
    for eid in probe:
        try:
            x = evaluator.evaluate(eid, np.full(dimension, 0.5))
        except EvaluationError:
            continue
        span = max(span, float(np.abs(x).max()))
        for i, (_name, f_ev) in enumerate(extra):
            v = f_ev.evaluate(eid, np.full(dimension, 0.5))
            extra_spans[i] = max(extra_spans[i], float(np.abs(v).max()))
    rel = 1e-9 if dedup else 1e-16
    quanta = [np.full(geometry.number_of_components,
                      (span or 1.0) * abs(scale) * rel)]
    for (_name, f_ev), s in zip(extra, extra_spans):
        quanta.append(np.full(f_ev.field.number_of_components,
                              (s or 1.0) * rel))
    pool = _PointPool(np.concatenate(quanta))

    connectivity = []
    offsets = []
    types = []
    degrees = [] if tessellate is None else None
    exported_ids = []
    skipped = []

    if tessellate is None:
        for eid in ids:
            try:
                ctrl, orders = _control_lattice(evaluator, eid)
            except EvaluationError:
                skipped.append(eid)
                continue
            extra_lattices = []
            for name, f_ev in extra:
                f_ctrl, f_orders = _control_lattice(f_ev, eid)
                if f_orders != orders:
                    raise ValueError(
                        f"extra field {name!r} has different basis orders "
                        f"{f_orders} vs geometry {orders} on element {eid};"
                        f" cannot attach as Bezier point data")
                extra_lattices.append(f_ctrl)
            n = [o + 1 for o in orders]
            npts = int(np.prod(n))
            cell_ids = [0] * npts
            for m in range(npts):
                rem = m
                ijk = []
                for c in n:
                    ijk.append(rem % c)
                    rem //= c
                vtk_slot = point_index_from_ijk(tuple(ijk), tuple(orders))
                extras = (np.concatenate([lat[m] for lat in extra_lattices])
                          if extra_lattices else np.empty(0))
                cell_ids[vtk_slot] = pool.add(ctrl[m] * scale, extras)
            connectivity.extend(cell_ids)
            offsets.append(len(connectivity))
            types.append(_BEZIER_TYPE[dimension])
            degrees.append(list(orders) + [1] * (3 - dimension))
            exported_ids.append(eid)
    else:
        for eid in ids:
            try:
                dimension_, def_eid, _A, _b = evaluator._resolve(eid)
                _P, basis = evaluator.element_parameters(def_eid, dimension_)
                orders = None
                if _A is None:
                    orders = list(basis.orders)
                else:
                    orders = [basis.orders[int(np.argmax(_A[:, c]))]
                              for c in range(_A.shape[1])]
            except EvaluationError:
                skipped.append(eid)
                continue
            if isinstance(tessellate, int):
                res = [tessellate if o > 1 else 1 for o in orders]
            else:
                res = list(tessellate)
            axes = [np.linspace(0.0, 1.0, r + 1) for r in res]
            nn = [r + 1 for r in res]
            total = int(np.prod(nn))
            # lattice with xi1 fastest: m = i + nn0*(j + nn1*k)
            lattice = np.empty((total, dimension))
            for m in range(total):
                rem = m
                for axis_index in range(dimension):
                    lattice[m, axis_index] = axes[axis_index][
                        rem % nn[axis_index]]
                    rem //= nn[axis_index]
            xyz = evaluator.evaluate(eid, lattice) * scale
            extra_values = [f_ev.evaluate(eid, lattice)
                            for _name, f_ev in extra]
            point_ids = np.empty(len(lattice), dtype=np.int64)
            for m in range(len(lattice)):
                extras = (np.concatenate([v[m] for v in extra_values])
                          if extra_values else np.empty(0))
                point_ids[m] = pool.add(xyz[m], extras)

            def lat(i, j=0, k=0):
                return point_ids[i + nn[0] * (j + (nn[1] * k
                                                   if dimension >= 2 else 0))]

            if dimension == 1:
                for i in range(res[0]):
                    connectivity.extend([lat(i), lat(i + 1)])
                    offsets.append(len(connectivity))
                    types.append(VTK_LINE)
                    exported_ids.append(eid)
            elif dimension == 2:
                for j in range(res[1]):
                    for i in range(res[0]):
                        connectivity.extend([
                            lat(i, j), lat(i + 1, j),
                            lat(i + 1, j + 1), lat(i, j + 1)])
                        offsets.append(len(connectivity))
                        types.append(VTK_QUAD)
                        exported_ids.append(eid)
            else:
                for k in range(res[2]):
                    for j in range(res[1]):
                        for i in range(res[0]):
                            connectivity.extend([
                                lat(i, j, k), lat(i + 1, j, k),
                                lat(i + 1, j + 1, k), lat(i, j + 1, k),
                                lat(i, j, k + 1), lat(i + 1, j, k + 1),
                                lat(i + 1, j + 1, k + 1),
                                lat(i, j + 1, k + 1)])
                            offsets.append(len(connectivity))
                            types.append(VTK_HEXAHEDRON)
                            exported_ids.append(eid)

    # cell data
    cell_data = {"element_id": np.asarray(exported_ids, dtype=np.int64)}
    if groups:
        names = sorted(mesh.groups) if groups is True else list(groups)
        exported_set = np.asarray(exported_ids)
        for name in names:
            group = mesh.groups.get(name)
            if group is None:
                continue
            members = group.element_ids(dimension)
            if not members:
                continue
            member_array = np.isin(exported_set,
                                   sorted(members)).astype(np.uint8)
            cell_data[name] = member_array

    # point data
    point_data = {}
    if extra:
        stacked = np.array(pool.data)
        col = 0
        for name, f_ev in extra:
            n_comp = f_ev.field.number_of_components
            point_data[name] = stacked[:, col:col + n_comp]
            col += n_comp

    _write_vtu(path, pool.points, connectivity, offsets, types,
               cell_data, point_data, degrees)
    return {"path": path, "points": len(pool.points), "cells": len(types),
            "mode": "bezier" if tessellate is None else "tessellated",
            "skipped_elements": skipped}


def export_markers_vtu(mesh, path, host_field="coordinates",
                       location_field="marker_location",
                       name_field="marker_name", scale=1.0):
    """Export embedded marker points as VTK vertex cells.

    Marker world positions are evaluated from their stored
    ``(element, xi)`` locations on the host field. Names cannot be
    stored in a .vtu (no string arrays); they are written as a
    ``marker_number`` point array and returned as an ordered list.
    """
    location = mesh.fields.get(location_field)
    if location is None:
        raise ValueError(f"No field {location_field!r} in mesh")
    evaluator = _quiet_evaluator(_resolve_field(mesh, host_field),
                                 location.host_mesh_dimension)
    names = []
    points = []
    for nodeset in mesh.nodesets.values():
        for node in nodeset:
            if location_field not in node.fields:
                continue
            element_id, xi = node.fields[location_field][0]
            if element_id is None:
                continue
            points.append(evaluator.evaluate(element_id, xi) * scale)
            name = node.fields.get(name_field)
            names.append(name[0] if name else str(node.identifier))
    connectivity = list(range(len(points)))
    offsets = list(range(1, len(points) + 1))
    types = [VTK_VERTEX] * len(points)
    point_data = {"marker_number": np.arange(len(points), dtype=np.float64)}
    _write_vtu(path, points, connectivity, offsets, types, {}, point_data,
               None)
    return names
