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
import os
import struct
import warnings
import zlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import exfield
from exfield.vtu import point_index_from_ijk

DATA = os.path.join(os.path.dirname(__file__), "data")


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
