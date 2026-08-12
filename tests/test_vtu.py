"""VTK export tests.

Pure tests run always: the point-ordering transcription is pinned
against tables dumped from VTK 9.6.2 (tests/data/
vtk_ordering_tables.json), the written XML is validated structurally,
and tessellated points are cross-checked against exfield evaluation.

Oracle tests (class TestAgainstVTK) need the vtk package and skip
otherwise — run them via:  uv run --with vtk pytest tests/test_vtu.py
"""

import base64
import json
import math
import os
import struct
import warnings
import zlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import exfield
from exfield.vtu import _control_lattice, point_index_from_ijk

DATA = os.path.join(os.path.dirname(__file__), "data")


# ------------------------------------------- synthetic tricubic block
#
# The vagus corpus (the only exact-Bezier golden data) has no element
# with interior control points along xi3: its boxes are cubic*linear*
# linear. That makes VTK's higher-order hexahedron xi3-edge slots
# structurally unexercisable, and it hid a real corruption of every
# tricubic cell. This block is the minimal fixture that reaches them:
# two curved tricubic-Hermite hexes sharing a face, hence sharing four
# cubic-cubic edges that carry interior control points in both
# directions.


def _block_warp(u, v, w):
    return (u + 0.30 * v * w,
            v + 0.40 * math.sin(1.1 * u) + 0.25 * w * w,
            w + 0.35 * u * v - 0.20 * math.cos(0.9 * v))


def _block_warp_derivatives(u, v, w):
    return ((1.0, 0.44 * math.cos(1.1 * u), 0.35 * v),
            (0.30 * w, 1.0, 0.35 * u + 0.18 * math.sin(0.9 * v)),
            (0.30 * v, 0.50 * w, 1.0))


def tricubic_block_exf(n_elements=2):
    """EX text: a 1-D row of ``n_elements`` curved tricubic hexes."""
    nodes = {}
    for k in range(2):
        for j in range(2):
            for i in range(n_elements + 1):
                nodes[(i, j, k)] = len(nodes) + 1
    lines = ["EX Version: 2", "Region: /", "!#nodeset nodes",
             "Shape. Dimension=0", "#Fields=1",
             "1) coordinates, coordinate, rectangular cartesian, real,"
             " #Components=3"]
    lines += [f" {c}. #Values=4 (value,d/ds1,d/ds2,d/ds3)" for c in "xyz"]
    for (i, j, k), n in sorted(nodes.items(), key=lambda kv: kv[1]):
        u, v, w = float(i), float(j), float(k)
        value = _block_warp(u, v, w)
        du, dv, dw = _block_warp_derivatives(u, v, w)
        lines.append(f"Node: {n}")
        lines += ["  " + "  ".join(f"{x: .15e}" for x in
                                   (value[c], du[c], dv[c], dw[c]))
                  for c in range(3)]
    lines += ["!#mesh mesh3d, dimension=3, nodeset=nodes",
              "Shape. Dimension=3, line*line*line",
              "#Scale factor sets=0", "#Nodes=8", "#Fields=1",
              "1) coordinates, coordinate, rectangular cartesian, real,"
              " #Components=3"]
    for c in "xyz":
        lines += [f" {c}. c.Hermite*c.Hermite*c.Hermite, no modify,"
                  " standard node based.", "  #Nodes=8"]
        for local in range(1, 9):
            lines += [f"  {local}. #Values=3",
                      "   Value labels: value d/ds1 d/ds2",
                      "  0. #Values=1", "   Value labels: zero",
                      f"  {local}. #Values=1", "   Value labels: d/ds3",
                      "  0. #Values=3", "   Value labels: zero zero zero"]
    for e in range(n_elements):
        ids = [nodes[(e + di, dj, dk)]
               for dk in (0, 1) for dj in (0, 1) for di in (0, 1)]
        lines += [f"Element: {e + 1}", " Nodes:",
                  " " + " ".join(str(x) for x in ids)]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def tricubic_block(tmp_path_factory):
    path = tmp_path_factory.mktemp("block") / "tricubic_block.exf"
    path.write_text(tricubic_block_exf(2))
    return exfield.load(str(path))


@pytest.fixture(scope="module")
def tricubic_single(tmp_path_factory):
    """One isolated curved tricubic hex — no shared faces, no pooling."""
    path = tmp_path_factory.mktemp("hex") / "tricubic_single.exf"
    path.write_text(tricubic_block_exf(1))
    return exfield.load(str(path))


# ------------------------------------------------------ ordering pin


class TestPointOrdering:
    def test_matches_vtk_dumped_tables(self):
        """The transcription of PointIndexFromIJK must match VTK 9.6.2
        exactly for every degree combination in use. Do not weaken."""
        with open(os.path.join(DATA, "vtk_ordering_tables.json")) as fh:
            tables = json.load(fh)
        checked = 0
        for name, table in tables.items():
            kind, orders_repr = name.split("_", 1)
            orders = tuple(json.loads(orders_repr))
            for key, expected in table.items():
                ijk = tuple(int(v) for v in key.split(","))
                assert point_index_from_ijk(ijk, orders) == expected, \
                    (name, ijk)
                checked += 1
        assert checked > 100

    def test_bijective(self):
        for orders in ((3,), (3, 3), (3, 1, 1), (3, 3, 3), (2, 3, 1)):
            n = [o + 1 for o in orders]
            total = int(np.prod(n))
            seen = set()
            for m in range(total):
                rem = m
                ijk = []
                for c in n:
                    ijk.append(rem % c)
                    rem //= c
                seen.add(point_index_from_ijk(tuple(ijk), orders))
            assert seen == set(range(total)), orders

    def test_curve_convention(self):
        # endpoints first, then interior in ascending order
        assert [point_index_from_ijk((i,), (3,)) for i in range(4)] \
            == [0, 2, 3, 1]


# ------------------------------------------------------ file structure


def _read_data_array(elem):
    text = elem.text.strip()
    # split the separately-base64'd header (4 x UInt64) from payload
    header_b64_len = ((32 + 2) // 3) * 4
    header = base64.b64decode(text[:header_b64_len])
    _nblocks, raw_len, _last, _clen = struct.unpack("<QQQQ", header)
    payload = zlib.decompress(base64.b64decode(text[header_b64_len:]))
    assert len(payload) == raw_len
    dtype = {"Float64": np.float64, "Int64": np.int64,
             "Int32": np.int32, "UInt8": np.uint8}[elem.get("type")]
    values = np.frombuffer(payload, dtype=dtype)
    n_comp = int(elem.get("NumberOfComponents", "1"))
    return values.reshape(-1, n_comp) if n_comp > 1 else values


def _parse_vtu(path):
    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    arrays = {}
    for section in ("Points", "Cells", "CellData", "PointData"):
        sec = piece.find(section)
        if sec is None:
            continue
        for da in sec.findall("DataArray"):
            arrays[(section, da.get("Name"))] = _read_data_array(da)
    return piece, arrays


def _file_version(path):
    root = ET.parse(path).getroot()
    return tuple(int(v) for v in root.get("version").split("."))


@pytest.fixture(scope="module")
def exported(vagus_mesh, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("vtu") / "vagus3d.vtu")
    summary = exfield.export_vtu(
        vagus_mesh, path,
        extra_fields=["vagus coordinates", "straight coordinates"])
    return vagus_mesh, path, summary


class TestBezierExport:

    def test_summary_and_structure(self, exported):
        mesh, path, summary = exported
        assert summary["mode"] == "bezier"
        assert summary["cells"] == len(mesh.mesh3d)
        assert summary["skipped_elements"] == []
        piece, arrays = _parse_vtu(path)
        assert int(piece.get("NumberOfCells")) == summary["cells"]
        assert int(piece.get("NumberOfPoints")) == summary["points"]
        types = arrays[("Cells", "types")]
        assert set(types) == {79}                     # VTK_BEZIER_HEXAHEDRON
        degrees = arrays[("CellData", "HigherOrderDegrees")]
        assert degrees.shape[1] == 3
        assert list(degrees[0]) == [3, 1, 1]          # vagus box elements
        conn = arrays[("Cells", "connectivity")]
        offsets = arrays[("Cells", "offsets")]
        assert offsets[-1] == len(conn)
        assert np.all(np.diff(offsets) == 16)         # 4x2x2 lattices

    def test_higher_order_files_declare_corrected_ordering(self, exported,
                                                           tmp_path):
        """A .vtu declaring version <= 2.0 is read as using the OLD
        higher-order hexahedron ordering, in which the interior points
        of VTK edges 10 and 11 (the two xi3 edges on the xi2=max face)
        are swapped. VTK applies that swap on read, so declaring 1.0 —
        as this writer used to — silently corrupted every hexahedral
        cell with interior control points along xi3. Do not lower.
        """
        _mesh, path, _summary = exported
        assert _file_version(path) >= (2, 1)

    def test_tricubic_block_shares_a_face(self, tricubic_block, tmp_path):
        path = str(tmp_path / "block.vtu")
        summary = exfield.export_vtu(tricubic_block, path, dimension=3,
                                     groups=False)
        _piece, arrays = _parse_vtu(path)
        assert list(arrays[("CellData", "HigherOrderDegrees")][0]) \
            == [3, 3, 3]
        # 2 x 64 lattice points, the 4x4 shared face pooled once
        assert summary["points"] == 2 * 64 - 16
        assert _file_version(path) >= (2, 1)

    def test_tricubic_control_points_land_in_ordered_slots(self,
                                                           tricubic_block,
                                                           tmp_path):
        """Every lattice slot of a tricubic hex holds the control point
        the ordering says it should — including the xi3 edges at xi2=max
        (VTK edges 10 and 11), which only exist when two cubic axes meet.

        This is the *pure* counterpart to the TestAgainstVTK readback
        tests. ``test_matches_vtk_dumped_tables`` pins
        :func:`point_index_from_ijk` in isolation and the readback tests
        pin the file VTK actually reads, but vtk is not a dependency, so
        those skip on a default checkout. Without this test, mis-wiring
        the writer's use of the ordering — as opposed to the ordering
        itself — leaves a green ``pytest`` run.
        """
        path = str(tmp_path / "slots.vtu")
        exfield.export_vtu(tricubic_block, path, dimension=3, groups=False)
        _piece, arrays = _parse_vtu(path)
        points = arrays[("Points", "Points")]
        conn = arrays[("Cells", "connectivity")]
        offsets = arrays[("Cells", "offsets")]
        element_ids = arrays[("CellData", "element_id")]
        degrees = arrays[("CellData", "HigherOrderDegrees")]
        # the m -> ijk decomposition below is written out for a 4x4x4
        # lattice rather than re-derived, so pin the degrees first
        assert [list(d) for d in degrees] == [[3, 3, 3]] * 2
        ev = tricubic_block.evaluator("coordinates", dimension=3)
        start = 0
        for c, eid in enumerate(element_ids):
            cell = points[conn[start:offsets[c]]]
            start = offsets[c]
            ctrl, orders = _control_lattice(ev, int(eid))
            assert len(cell) == 64
            for m in range(64):
                ijk = (m % 4, (m // 4) % 4, m // 16)
                slot = point_index_from_ijk(ijk, tuple(orders))
                assert np.allclose(cell[slot], ctrl[m], atol=1e-12), \
                    (eid, ijk, slot)

    def test_control_points_shared_between_cells(self, exported):
        _mesh, _path, summary = exported
        # 88 cells x 16 points = 1408 naive; adjacency must share
        assert summary["points"] < 88 * 16

    def test_group_and_element_arrays(self, exported):
        mesh, path, summary = exported
        piece, arrays = _parse_vtu(path)
        element_ids = arrays[("CellData", "element_id")]
        assert sorted(element_ids) == sorted(mesh.mesh3d.elements)
        lv = arrays[("CellData", "left vagus nerve")]
        members = mesh.groups["left vagus nerve"].element_ids(3)
        assert lv.sum() == len(members)

    def test_extra_field_point_data(self, exported):
        mesh, path, summary = exported
        _piece, arrays = _parse_vtu(path)
        material = arrays[("PointData", "vagus coordinates")]
        assert material.shape == (summary["points"], 3)
        # corner control values equal the field value at the corner
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = mesh.evaluator("coordinates", dimension=3)
            mat = mesh.evaluator("vagus coordinates", dimension=3)
        points = arrays[("Points", "Points")]
        eid = sorted(mesh.mesh3d.elements)[0]
        corner_xyz = ev.evaluate(eid, [0.0, 0.0, 0.0])
        idx = int(np.argmin(np.linalg.norm(points - corner_xyz, axis=1)))
        assert np.allclose(points[idx], corner_xyz, atol=1e-6)
        assert np.allclose(material[idx],
                           mat.evaluate(eid, [0.0, 0.0, 0.0]), atol=1e-9)

    def test_scale(self, vagus_mesh, tmp_path):
        path = str(tmp_path / "scaled.vtu")
        exfield.export_vtu(vagus_mesh, path, dimension=3, groups=False,
                           scale=1e-3)
        _piece, arrays = _parse_vtu(path)
        span = np.ptp(arrays[("Points", "Points")], axis=0)
        assert 100 < span.max() < 1000    # ~500 mm nerve, was ~5e5 µm

    def test_dedup_is_scale_invariant(self, cube_mesh, tmp_path):
        """Scale must change units, not topology. The dedup quantum was
        computed from unscaled coordinates while scaled ones went into
        the pool: scale=1e-12 collapsed a cube's 64 control points to 1."""
        counts = {}
        for scale in (1.0, 1e-3, 1e-12):
            summary = exfield.export_vtu(
                cube_mesh, str(tmp_path / f"cube_{scale:g}.vtu"),
                groups=False, scale=scale)
            counts[scale] = summary["points"]
        assert len(set(counts.values())) == 1, counts

    def test_extra_field_seam_survives_scale(self, vagus_mesh, tmp_path):
        """The dedup key mixes SCALED geometry with UNSCALED field
        values; a single shared quantum merged real field seams at
        large scale and overflowed the int64 key at small scale. Point
        counts with an attached field must match at every scale."""
        counts = {}
        for scale in (1.0, 1e4, 1e-3, 1e-12):
            summary = exfield.export_vtu(
                vagus_mesh, str(tmp_path / f"vs_{scale:g}.vtu"),
                dimension=3, groups=False, scale=scale,
                extra_fields=["vagus coordinates"])
            counts[scale] = summary["points"]
        assert len(set(counts.values())) == 1, counts

    def test_dedup_false_never_merges(self, cube_mesh, tmp_path):
        """dedup=False must keep every lattice point distinct at any
        scale — the near-zero quantum used to overflow int64 and merge
        everything through a NaN-cast sentinel key."""
        import warnings as _w
        for scale in (1.0, 1e-9):
            with _w.catch_warnings():
                _w.simplefilter("error", RuntimeWarning)
                summary = exfield.export_vtu(
                    cube_mesh, str(tmp_path / f"nd_{scale:g}.vtu"),
                    groups=False, dedup=False, scale=scale)
            assert summary["points"] == 64, (scale, summary["points"])

    def test_scale_zero_refused(self, cube_mesh, tmp_path):
        with pytest.raises(ValueError, match="scale=0"):
            exfield.export_vtu(cube_mesh, str(tmp_path / "z.vtu"),
                               scale=0.0)

    def test_inherited_2d_export(self, vagus_mesh, tmp_path):
        """2-D epineurium (serendipity) plus face-inherited quads."""
        path = str(tmp_path / "vagus2d.vtu")
        summary = exfield.export_vtu(vagus_mesh, path, dimension=2,
                                     groups=["epineurium"])
        assert summary["cells"] == len(vagus_mesh.mesh2d)
        _piece, arrays = _parse_vtu(path)
        assert set(arrays[("Cells", "types")]) == {77}  # BEZIER_QUAD
        degrees = arrays[("CellData", "HigherOrderDegrees")]
        assert {tuple(d) for d in degrees} <= {(3, 3, 1), (3, 1, 1),
                                               (1, 3, 1), (1, 1, 1)}

    def test_inherited_1d_matches_evaluation(self, vagus_mesh, tmp_path):
        """Face-inherited line elements export exact Bezier curves:
        the curve's control polygon evaluates to the element geometry."""
        path = str(tmp_path / "vagus1d.vtu")
        trunk = sorted(
            vagus_mesh.groups["orientation anterior"].element_ids(1))
        summary = exfield.export_vtu(vagus_mesh, path, dimension=1,
                                     element_ids=trunk, groups=False)
        assert summary["cells"] == 50
        _piece, arrays = _parse_vtu(path)
        points = arrays[("Points", "Points")]
        conn = arrays[("Cells", "connectivity")]
        offsets = arrays[("Cells", "offsets")]
        element_ids = arrays[("CellData", "element_id")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=1)
        start = 0
        for c, eid in enumerate(element_ids[:5]):
            cell_conn = conn[start:offsets[c]]
            start = offsets[c]
            ctrl = points[cell_conn]      # VTK order: end0, end1, interior
            b = np.array([ctrl[0], ctrl[2], ctrl[3], ctrl[1]])
            for t in (0.0, 0.3, 0.7, 1.0):
                # de Casteljau
                q = b.copy()
                for _ in range(3):
                    q = (1 - t) * q[:-1] + t * q[1:]
                assert np.allclose(q[0], ev.evaluate(int(eid), [t]),
                                   atol=1e-6), (eid, t)


class TestTessellatedExport:
    def test_points_lie_on_geometry(self, vagus_mesh, tmp_path):
        path = str(tmp_path / "tess3d.vtu")
        summary = exfield.export_vtu(vagus_mesh, path, dimension=3,
                                     groups=False, tessellate=4)
        _piece, arrays = _parse_vtu(path)
        points = arrays[("Points", "Points")]
        types = arrays[("Cells", "types")]
        assert set(types) == {12}                       # VTK_HEXAHEDRON
        # 88 elements x 4 segments (cubic axis) x 1 x 1
        assert summary["cells"] == 88 * 4
        assert ("CellData", "HigherOrderDegrees") not in arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=3)
        # every lattice corner of element eid at xi=(0.25,1,0) etc must
        # be an exported point
        eid = sorted(vagus_mesh.mesh3d.elements)[3]
        for xi in ([0.25, 0.0, 1.0], [0.5, 1.0, 0.0], [1.0, 1.0, 1.0]):
            x = ev.evaluate(eid, xi)
            d = np.linalg.norm(points - x, axis=1).min()
            assert d < 1e-6, xi

    def test_watertight_shared_lattice(self, vagus_mesh, tmp_path):
        path = str(tmp_path / "tess2d.vtu")
        summary = exfield.export_vtu(vagus_mesh, path, dimension=2,
                                     groups=False, tessellate=2)
        piece, arrays = _parse_vtu(path)
        conn = arrays[("Cells", "connectivity")]
        # shared points: total refs must exceed unique points comfortably
        assert len(conn) > 1.5 * summary["points"]

    def test_extra_fields_tessellated(self, vagus_mesh, tmp_path):
        path = str(tmp_path / "tess1d.vtu")
        trunk = sorted(
            vagus_mesh.groups["orientation anterior"].element_ids(1))
        exfield.export_vtu(vagus_mesh, path, dimension=1,
                           element_ids=trunk, groups=False, tessellate=8,
                           extra_fields=["vagus coordinates"])
        _piece, arrays = _parse_vtu(path)
        material = arrays[("PointData", "vagus coordinates")]
        # material 3rd component spans [0, 1] along the trunk
        assert material[:, 2].min() == pytest.approx(0.0, abs=1e-6)
        assert material[:, 2].max() == pytest.approx(1.0, abs=1e-3)


class TestMarkers:
    def test_marker_export(self, vagus_mesh, tmp_path):
        path = str(tmp_path / "markers.vtu")
        names = exfield.export_markers_vtu(vagus_mesh, path)
        assert names
        assert any("jugular" in n or "nodose" in n for n in names)
        _piece, arrays = _parse_vtu(path)
        points = arrays[("Points", "Points")]
        assert points.shape == (len(names), 3)


# ------------------------------------------------- live VTK oracle


def _readback_residual(vtk, mesh, path, seed, samples=40):
    """Export ``mesh``, read it back through VTK, and return the worst
    relative disagreement between VTK's cell interpolation and ours."""
    exfield.export_vtu(mesh, path, dimension=3, groups=False)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetNumberOfCells() == len(mesh.mesh3d)
    ev = mesh.evaluator("coordinates", dimension=3)
    element_ids = grid.GetCellData().GetArray("element_id")
    rng = np.random.default_rng(seed)
    worst = 0.0
    for c in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(c)
        assert cell.GetNumberOfPoints() == 64
        eid = int(element_ids.GetTuple1(c))
        for _ in range(samples):
            xi = rng.random(3)
            x = [0.0, 0.0, 0.0]
            weights = [0.0] * cell.GetNumberOfPoints()
            cell.EvaluateLocation(vtk.reference(0), list(xi), x, weights)
            ours = ev.evaluate(eid, xi)
            scale = max(1.0, float(np.abs(ours).max()))
            worst = max(worst,
                        float(np.abs(np.array(x) - ours).max()) / scale)
    return worst


class TestAgainstVTK:
    """Needs the vtk package: uv run --with vtk pytest tests/test_vtu.py"""

    @pytest.fixture
    def vtk(self):
        return pytest.importorskip("vtk")

    def test_ordering_against_live_vtk(self, vtk):
        for orders in ((3, 1, 1), (3, 3, 3), (2, 3, 1)):
            n = [o + 1 for o in orders]
            for k in range(n[2]):
                for j in range(n[1]):
                    for i in range(n[0]):
                        expected = \
                            vtk.vtkHigherOrderHexahedron.PointIndexFromIJK(
                                i, j, k, list(orders))
                        assert point_index_from_ijk((i, j, k), orders) \
                            == expected, (orders, i, j, k)

    def test_readback_and_evaluate(self, vtk, vagus_mesh, tmp_path):
        """VTK reads the written file and evaluates the same geometry."""
        path = str(tmp_path / "roundtrip.vtu")
        exfield.export_vtu(vagus_mesh, path, dimension=3, groups=False,
                           extra_fields=["vagus coordinates"])
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(path)
        reader.Update()
        grid = reader.GetOutput()
        assert grid.GetNumberOfCells() == len(vagus_mesh.mesh3d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=3)
            mat = vagus_mesh.evaluator("vagus coordinates", dimension=3)
        element_ids = grid.GetCellData().GetArray("element_id")
        material = grid.GetPointData().GetArray("vagus coordinates")
        rng = np.random.default_rng(4)
        worst = worst_m = 0.0
        for c in range(0, grid.GetNumberOfCells(), 9):
            cell = grid.GetCell(c)
            eid = int(element_ids.GetTuple1(c))
            for _ in range(5):
                xi = rng.random(3)
                x = [0.0, 0.0, 0.0]
                weights = [0.0] * cell.GetNumberOfPoints()
                cell.EvaluateLocation(vtk.reference(0), list(xi), x,
                                      weights)
                ours = ev.evaluate(eid, xi)
                scale = max(1.0, np.abs(ours).max())
                worst = max(worst,
                            float(np.abs(np.array(x) - ours).max()) / scale)
                # point-data interpolation with the same weights must
                # reproduce the material field exactly
                interp = np.zeros(3)
                for p in range(cell.GetNumberOfPoints()):
                    pid = cell.GetPointIds().GetId(p)
                    interp += weights[p] * np.array(
                        material.GetTuple3(pid))
                worst_m = max(worst_m, float(np.abs(
                    interp - mat.evaluate(eid, xi)).max()))
        assert worst < 1e-9
        assert worst_m < 1e-9

    def test_tricubic_readback_and_evaluate(self, vtk, tricubic_block,
                                            tmp_path):
        """Regression: full tricubic hexes sharing cubic-cubic edges.

        The vagus golden mesh is cubic*linear*linear, so its xi3 edges
        carry no interior control points and VTK's legacy higher-order
        hexahedron edge swap was invisible there. Every cell of this
        block reaches those slots; before the file-version fix the two
        xi3 edges on the xi2=max face came back exchanged and this
        residual was ~0.1 in a unit-sized geometry.
        """
        worst = _readback_residual(vtk, tricubic_block,
                                   str(tmp_path / "tricubic_roundtrip.vtu"),
                                   seed=11)
        assert worst < 1e-9, worst

    def test_single_tricubic_hex_readback(self, vtk, tricubic_single,
                                          tmp_path):
        """The same regression on one isolated hex, which is how the
        edge swap was originally reproduced.

        ``test_tricubic_readback_and_evaluate`` pins a two-hex block, so
        every cell there has a pooled shared face. Pinning a lone cell
        keeps the reproducer honest: the ordering must be right per
        cell, not merely self-consistent across a shared face.
        """
        worst = _readback_residual(vtk, tricubic_single,
                                   str(tmp_path / "single_roundtrip.vtu"),
                                   seed=5)
        assert worst < 1e-9, worst


# ------------------------------------------------ non-3-component geometry
#
# Every fixture above is 3-component, which is what let a 2-component
# coordinate field write a NumberOfComponents="3" Points array holding
# 2-component data: VTK reads consecutive (x, y) pairs as (x, y, z)
# triples, silently dropping a third of the points and rotating the rest
# onto the wrong axes. Planar scaffolds are ordinary, so this is
# reachable, and it fails without raising anything.


def _planar_exf(n_components=2):
    """A one-element bilinear square whose coordinate field has
    ``n_components`` components (extra components are zero)."""
    corners = [(0.0, 0.0), (3.0, 0.0), (0.0, 5.0), (3.0, 5.0)]
    names = ["x", "y", "z", "w"][:n_components]
    declaration = ("1) coordinates, coordinate, rectangular cartesian, "
                   f"real, #Components={n_components}")
    node_component = "\n".join(f" {n}.  #Values=1 (value)" for n in names)
    element_component = "\n".join(
        f" {n}. l.Lagrange*l.Lagrange, no modify, standard node based.\n"
        "  #Nodes=4\n"
        + "".join(f"  {i}. #Values=1\n   Value labels: value\n"
                  for i in range(1, 5)).rstrip("\n")
        for n in names)
    nodes = "\n".join(
        f"Node: {i}\n"
        + "\n".join(f" {(list(c) + [0.0, 0.0])[k]}"
                    for k in range(n_components))
        for i, c in enumerate(corners, start=1))
    return f"""EX Version: 3
Region: /
!#nodeset nodes
Define node template: node1
Shape. Dimension=0
#Fields=1
{declaration}
{node_component}
Node template: node1
{nodes}
!#mesh mesh2d, dimension=2, face mesh=mesh1d, nodeset=nodes
Define element template: element1
Shape. Dimension=2, line*line
#Scale factor sets=0
#Nodes=4
#Fields=1
{declaration}
{element_component}
Element template: element1
Element: 1
 Nodes:
 1 2 3 4
"""


class TestNonThreeComponentGeometry:
    def test_two_component_field_pads_to_z_zero(self, tmp_path):
        mesh = exfield.loads(_planar_exf(2))
        path = str(tmp_path / "planar.vtu")
        info = exfield.export_vtu(mesh, path)
        piece, arrays = _parse_vtu(path)
        points = arrays[("Points", "Points")]
        # declared point count and payload must agree; z padded with 0
        assert points.shape == (info["points"], 3)
        assert int(piece.get("NumberOfPoints")) == len(points)
        assert np.all(points[:, 2] == 0.0)
        assert {tuple(p[:2]) for p in points} == {
            (0.0, 0.0), (3.0, 0.0), (0.0, 5.0), (3.0, 5.0)}

    def test_two_component_tessellated_pads_too(self, tmp_path):
        mesh = exfield.loads(_planar_exf(2))
        path = str(tmp_path / "planar_tess.vtu")
        info = exfield.export_vtu(mesh, path, tessellate=2)
        piece, arrays = _parse_vtu(path)
        points = arrays[("Points", "Points")]
        assert points.shape == (info["points"], 3)
        assert int(piece.get("NumberOfPoints")) == len(points)
        assert np.all(points[:, 2] == 0.0)

    def test_three_component_geometry_is_unchanged(self, tmp_path):
        mesh = exfield.loads(_planar_exf(3))
        path = str(tmp_path / "planar3.vtu")
        info = exfield.export_vtu(mesh, path)
        _piece, arrays = _parse_vtu(path)
        assert arrays[("Points", "Points")].shape == (info["points"], 3)

    def test_more_than_three_components_refused(self, tmp_path):
        mesh = exfield.loads(_planar_exf(4))
        with pytest.raises(ValueError, match="3 components"):
            exfield.export_vtu(mesh, str(tmp_path / "four.vtu"))
