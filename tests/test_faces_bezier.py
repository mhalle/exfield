"""Face inheritance mapping and Hermite->Bezier conversion tests."""

import numpy as np
import pytest

import exfield
from exfield.faces import face_to_element_map


class TestFaceMaps:
    def test_face_axis_order(self):
        # faces in order xi1=0, xi1=1, xi2=0, xi2=1, xi3=0, xi3=1
        for f, (axis, value) in enumerate(
                [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (2, 0.0),
                 (2, 1.0)]):
            A, b = face_to_element_map(3, f)
            assert b[axis] == value
            assert A[axis].sum() == 0.0

    def test_cyclic_mapping_is_not_lexicographic(self):
        """The xi2 faces of a cube map face (u, v) -> (xi3, xi1), not
        (xi1, xi3). Getting this wrong is silent."""
        A, _ = face_to_element_map(3, 2)   # xi2=0 face
        u = np.array([0.3, 0.9])
        xi = A @ u
        assert xi[2] == pytest.approx(0.3)   # face u -> xi3
        assert xi[0] == pytest.approx(0.9)   # face v -> xi1
        # xi1 and xi3 faces are lexicographic
        A1, _ = face_to_element_map(3, 0)
        assert (A1 @ u)[1] == pytest.approx(0.3)
        assert (A1 @ u)[2] == pytest.approx(0.9)
        A5, _ = face_to_element_map(3, 4)
        assert (A5 @ u)[0] == pytest.approx(0.3)
        assert (A5 @ u)[1] == pytest.approx(0.9)

    def test_inherited_evaluation_continuity(self, vagus_mesh):
        """A face-inherited 2-D element evaluates continuously with its
        3-D parent across the shared face (spot check)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev2 = exfield.Evaluator(vagus_mesh.fields["coordinates"],
                                    dimension=2)
            ev3 = exfield.Evaluator(vagus_mesh.fields["coordinates"],
                                    dimension=3)
        parent = next(e for e in vagus_mesh.element_mesh(3).elements.values()
                      if e.faces and e.faces[0] is not None)
        fid = parent.faces[0]     # xi1=0 face
        v_face = ev2.evaluate(fid, [0.3, 0.7])
        v_parent = ev3.evaluate(parent.identifier, [0.0, 0.3, 0.7])
        assert np.allclose(v_face, v_parent, rtol=1e-12)


class TestBezier:
    def test_conversion_is_exact(self):
        rng = np.random.default_rng(3)
        p0, d0, p1, d1 = rng.random((4, 3))
        ctrl = exfield.hermite_to_bezier_1d(p0, d0, p1, d1)
        for t in (0.0, 0.2, 0.5, 0.9, 1.0):
            hermite = ((2 * t**3 - 3 * t**2 + 1) * p0
                       + (t**3 - 2 * t**2 + t) * d0
                       + (-2 * t**3 + 3 * t**2) * p1
                       + (t**3 - t**2) * d1)
            # de Casteljau
            b = ctrl.copy()
            for level in range(3):
                b = (1 - t) * b[:-1] + t * b[1:]
            assert np.allclose(b[0], hermite, atol=1e-14)

    def test_matrix_matches_function(self):
        M = exfield.hermite_to_bezier_matrix()
        p0, d0, p1, d1 = 1.0, 2.0, 3.0, -1.0
        ctrl = M @ np.array([p0, d0, p1, d1])
        expected = exfield.hermite_to_bezier_1d([p0], [d0], [p1], [d1])[:, 0]
        assert np.allclose(ctrl, expected)
