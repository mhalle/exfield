"""Writer tests: roundtrip fidelity and output stability.

The strongest writer validation is external and part of CI-by-hand:
Zinc 4.2.1 reads the rewritten f001 vagus scaffold and evaluates all
29,520 golden samples bit-identically to the original (see
tests/golden_zinc.py). These tests cover what runs without Zinc.
"""

import warnings

import numpy as np
import pytest

import exfield


def _roundtrip(mesh):
    return exfield.loads(exfield.dumps(mesh))


def _compare_evaluation(m1, m2, field_name, dimension, samples=20):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e1 = exfield.Evaluator(m1.fields[field_name], dimension=dimension)
        e2 = exfield.Evaluator(m2.fields[field_name], dimension=dimension)
    rng = np.random.default_rng(7)
    ids = sorted(m1.mesh(dimension).elements)
    for _ in range(samples):
        eid = ids[rng.integers(len(ids))]
        xi = rng.random(dimension)
        try:
            v1 = e1.evaluate(eid, xi)
        except exfield.EvaluationError:
            continue
        v2 = e2.evaluate(eid, xi)
        assert np.allclose(v1, v2, rtol=1e-12, atol=1e-9), (eid, xi)


class TestRoundtrip:
    def test_cube(self, cube_mesh):
        m2 = _roundtrip(cube_mesh)
        assert len(m2.nodes) == len(cube_mesh.nodes)
        assert len(m2.mesh(3)) == len(cube_mesh.mesh(3))
        assert m2.mesh(3)[1].faces == cube_mesh.mesh(3)[1].faces
        _compare_evaluation(cube_mesh, m2, "notcoordinates", 3)

    def test_write_is_byte_stable(self, cube_mesh):
        text = exfield.dumps(cube_mesh)
        assert exfield.dumps(exfield.loads(text)) == text

    def test_vagus_scaffold(self, vagus_mesh):
        m2 = _roundtrip(vagus_mesh)
        assert len(m2.nodes) == len(vagus_mesh.nodes)
        for d in (1, 2, 3):
            assert len(m2.mesh(d)) == len(vagus_mesh.mesh(d))
        assert set(m2.groups) == set(vagus_mesh.groups)
        for g in vagus_mesh.groups:
            for d in (1, 2, 3):
                assert (m2.groups[g].element_ids(d)
                        == vagus_mesh.groups[g].element_ids(d)), g
        for dim in (1, 2, 3):
            _compare_evaluation(vagus_mesh, m2, "coordinates", dim,
                                samples=10)
        # marker data survives
        markers1 = {n.identifier: n.fields["marker_name"][0]
                    for n in vagus_mesh.nodes if "marker_name" in n.fields}
        markers2 = {n.identifier: n.fields["marker_name"][0]
                    for n in m2.nodes if "marker_name" in n.fields}
        assert markers1 == markers2

    def test_vagus_write_is_byte_stable(self, vagus_mesh):
        text = exfield.dumps(vagus_mesh)
        assert exfield.dumps(exfield.loads(text)) == text


class TestFormat:
    def test_header_and_number_format(self, cube_mesh):
        text = exfield.dumps(cube_mesh)
        lines = text.splitlines()
        assert lines[0] == "EX Version: 3"
        assert lines[1] == "Region: /"
        # Zinc real format: space + 22-char right-justified %22.15e
        assert " 0.000000000000000e+00" in text
        assert "\t" not in text

    def test_quoting(self):
        assert exfield.exwriter.make_valid_token("plain") == "plain"
        assert exfield.exwriter.make_valid_token("has space") == '"has space"'
        assert exfield.exwriter.make_valid_token('q"q') == '"q\\"q"'
        assert exfield.exwriter.make_valid_token("a,b") == '"a,b"'

    def test_scale_factors_wrap_at_five(self, vagus_mesh):
        text = exfield.dumps(vagus_mesh)
        in_sf = False
        for line in text.splitlines():
            if line == " Scale factors:":
                in_sf = True
                continue
            if in_sf:
                if line.startswith(" ") and "e" in line:
                    assert len(line.split()) <= 5
                else:
                    in_sf = False


class TestZeroValueGuard:
    def test_warns_on_empty_node_field_template(self):
        """Zinc segfaults on '#Values=0 ()' general node fields; the
        writer must warn (found via the atlas-proto compiler)."""
        import warnings as w
        import numpy as np
        from exfield.mesh import Field, Mesh, NodeFieldTemplate
        mesh = Mesh("/")
        f = Field("marker_name", "field", "string", 1)
        f.mesh = mesh
        mesh.fields["marker_name"] = f
        node = mesh.nodes.get_or_create(1)
        node.fields["marker_name"] = ["hello"]
        node.templates["marker_name"] = [NodeFieldTemplate()]  # empty!
        with pytest.warns(UserWarning, match="Zinc crashes"):
            exfield.dumps(mesh)
        # and a proper template does not warn
        nft = NodeFieldTemplate()
        nft.set_value_number_of_versions("value", 1)
        node.templates["marker_name"] = [nft]
        with w.catch_warnings():
            w.simplefilter("error")
            exfield.dumps(mesh)
