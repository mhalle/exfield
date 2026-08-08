"""Golden tests against Zinc 4.2.1 on a real fitted vagus scaffold
(SPARC dataset 426, sub-f001 left).

The reference JSON is produced by ``tests/golden_zinc.py`` run under a
python with cmlibs.zinc installed; exfield itself never needs Zinc.
Covers field values and first derivatives at fixed (element, xi) on all
three mesh dimensions — including face-inherited 1-D/2-D elements — and
per-group arclength integrals at Zinc's own quadrature order.
"""

import json
import warnings

import numpy as np
import pytest

import exfield

from conftest import GOLDEN_JSON

pytestmark = pytest.mark.skipif(
    not __import__("os").path.exists(GOLDEN_JSON),
    reason="golden data not generated")


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_JSON) as fh:
        return json.load(fh)


def _evaluator(mesh, field, dim, cache={}):
    key = (id(mesh), field, dim)
    if key not in cache:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cache[key] = exfield.Evaluator(mesh.fields[field], dimension=dim)
    return cache[key]


def test_values_and_derivatives_match_zinc(golden, vagus_mesh):
    n = 0
    for dim_s, mesh_data in golden["meshes"].items():
        dim = int(dim_s)
        for s in mesh_data["samples"]:
            if s["field"] not in vagus_mesh.fields:
                continue
            ev = _evaluator(vagus_mesh, s["field"], dim)
            v = ev.evaluate(s["element"], s["xi"])
            ref = np.array(s["values"])
            scale = max(1.0, np.abs(ref).max())
            assert np.abs(v - ref).max() / scale < 1e-12, \
                (s["field"], dim, s["element"], s["xi"])
            if "derivatives" in s:
                d = ev.evaluate_derivatives(s["element"], s["xi"])
                refd = np.array(s["derivatives"])
                scale = max(1.0, np.abs(refd).max())
                assert np.abs(d - refd).max() / scale < 1e-10
            n += 1
    assert n > 1000


def test_group_membership_matches_zinc(golden, vagus_mesh):
    for gname, ginfo in golden["groups"].items():
        group = vagus_mesh.groups[gname]
        for dim in (1, 2, 3):
            key = f"mesh{dim}d"
            if key in ginfo:
                assert len(group.element_ids(dim)) == ginfo[key], gname
        if "nodes" in ginfo:
            assert len(group.node_ids("nodes")) == ginfo["nodes"], gname


def test_group_arclength_matches_zinc_gauss4(golden, vagus_mesh,
                                             vagus_coordinates_1d):
    """Per-element Gauss-4 integration reproduces Zinc's
    FieldMeshIntegral to near machine precision (order='zinc'
    bug-compatibility mode)."""
    ev = vagus_coordinates_1d
    gx, gw = np.polynomial.legendre.leggauss(
        exfield.ArclengthTable.ZINC_ORDER)
    gx = 0.5 * (gx + 1.0)
    gw = 0.5 * gw

    def element_length(eid):
        return sum(w * np.linalg.norm(ev.evaluate_derivatives(eid, [x])[0])
                   for x, w in zip(gx, gw))

    n = 0
    for gname, ginfo in golden["groups"].items():
        ref = ginfo.get("arclength_gauss4")
        if ref is None:
            continue
        ours = sum(element_length(e)
                   for e in vagus_mesh.groups[gname].element_ids(1))
        assert ours == pytest.approx(ref, rel=1e-10), gname
        n += 1
    assert n >= 50


def test_group_area_and_volume_match_zinc_gauss4(golden, vagus_mesh):
    """Surface areas (2-D groups) and volumes (3-D groups) reproduce
    Zinc's FieldMeshIntegral via the Gram-determinant Jacobian."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev2 = vagus_mesh.evaluator("coordinates", dimension=2)
        ev3 = vagus_mesh.evaluator("coordinates", dimension=3)
    n = 0
    for gname, ginfo in golden["groups"].items():
        group = vagus_mesh.groups[gname]
        if "area_gauss4" in ginfo:
            ours = exfield.integrate(
                ev2, element_ids=sorted(group.element_ids(2)), order="zinc")
            assert ours == pytest.approx(ginfo["area_gauss4"],
                                         rel=1e-10), gname
            n += 1
        if "volume_gauss4" in ginfo:
            ours = exfield.integrate(
                ev3, element_ids=sorted(group.element_ids(3)), order="zinc")
            assert ours == pytest.approx(ginfo["volume_gauss4"],
                                         rel=1e-10), gname
            n += 1
    assert n >= 100


def test_trunk_length_matches_prototype_measurement(vagus_mesh,
                                                    vagus_coordinates_1d):
    """The f001-left trunk ('orientation anterior', 50 elements) measures
    491.9 mm — the number documented in the scaffold-extraction analysis.
    The 'left vagus nerve' 1-D group is NOT the centreline (it includes
    circumferential rings); integrating it gives thousands of mm."""
    trunk = sorted(vagus_mesh.groups["orientation anterior"].element_ids(1))
    assert len(trunk) == 50
    table = exfield.ArclengthTable.build(vagus_coordinates_1d,
                                         element_ids=trunk)
    assert table.total / 1000.0 == pytest.approx(491.9, abs=0.05)
