"""Analytic tests for mesh integrals and coordinate conversions."""

import warnings

import numpy as np
import pytest

import exfield
from exfield.coordinates import to_rectangular_cartesian


class TestIntegrals:
    def test_box_volume_and_area_exact(self, cube_mesh):
        """cube.exf is a 1.5 x 1 x 1 box (tricubic Hermite with affine
        geometry): volume 1.5, total face area 8."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev3 = exfield.Evaluator(cube_mesh.fields["notcoordinates"],
                                    dimension=3)
        assert exfield.integrate(ev3) == pytest.approx(1.5, rel=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev2 = exfield.Evaluator(cube_mesh.fields["notcoordinates"],
                                    dimension=2)
        # faces inherit geometry from the volume element
        area = exfield.integrate(ev2)
        assert area == pytest.approx(2 * (1.5 + 1.5 + 1.0), rel=1e-12)

    def test_integrand(self, cube_mesh):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev3 = exfield.Evaluator(cube_mesh.fields["notcoordinates"],
                                    dimension=3)
        # integral of x over the box = Lx^2/2 * Ly * Lz = 1.125
        result = exfield.integrate(ev3, integrand=lambda v: v[:, 0])
        assert result == pytest.approx(1.5 ** 2 / 2, rel=1e-12)

    def test_1d_measure_matches_arclength(self, vagus_mesh):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=1)
        trunk = sorted(
            vagus_mesh.groups["orientation anterior"].element_ids(1))
        table = exfield.ArclengthTable.build(ev, element_ids=trunk)
        total = exfield.integrate(ev, element_ids=trunk)
        assert total == pytest.approx(table.total, rel=1e-9)


class TestCoordinateConversion:
    def test_cylindrical(self):
        xyz = to_rectangular_cartesian([2.0, np.pi / 2, 5.0],
                                       "cylindrical polar")
        assert np.allclose(xyz, [0.0, 2.0, 5.0], atol=1e-12)

    def test_spherical(self):
        # x = r cos(phi) cos(theta), y = r cos(phi) sin(theta),
        # z = r sin(phi)  (Zinc's convention)
        xyz = to_rectangular_cartesian([1.0, 0.0, np.pi / 2],
                                       "spherical polar")
        assert np.allclose(xyz, [0.0, 0.0, 1.0], atol=1e-12)

    def test_prolate_spheroidal(self):
        # x = f cosh(l) cos(mu), y = f sinh(l) sin(mu) cos(th),
        # z = f sinh(l) sin(mu) sin(th)
        f, lam, mu, th = 35.0, 0.7, 1.1, 0.4
        xyz = to_rectangular_cartesian([lam, mu, th],
                                       "prolate spheroidal", focus=f)
        assert xyz[0] == pytest.approx(f * np.cosh(lam) * np.cos(mu))
        assert xyz[1] == pytest.approx(
            f * np.sinh(lam) * np.sin(mu) * np.cos(th))
        assert xyz[2] == pytest.approx(
            f * np.sinh(lam) * np.sin(mu) * np.sin(th))

    def test_oblate_spheroidal(self):
        f, lam, mu, th = 2.0, 0.5, 0.8, 1.2
        xyz = to_rectangular_cartesian([lam, mu, th],
                                       "oblate spheroidal", focus=f)
        assert xyz[0] == pytest.approx(
            f * np.cosh(lam) * np.cos(mu) * np.sin(th))
        assert xyz[1] == pytest.approx(f * np.sinh(lam) * np.sin(mu))
        assert xyz[2] == pytest.approx(
            f * np.cosh(lam) * np.cos(mu) * np.cos(th))

    def test_focus_required(self):
        with pytest.raises(ValueError, match="focus"):
            to_rectangular_cartesian([1, 1, 1], "prolate spheroidal")

    def test_batch(self):
        v = np.array([[1.0, 0.0, 0.0], [2.0, np.pi, 1.0]])
        out = to_rectangular_cartesian(v, "cylindrical polar")
        assert out.shape == (2, 3)
        assert np.allclose(out[1], [-2.0, 0.0, 1.0], atol=1e-12)


class TestNonCartesianGuard:
    def test_evaluator_warns(self):
        text = (
            "EX Version: 2\nRegion: /\n!#nodeset nodes\n"
            "Shape. Dimension=0\n#Fields=1\n"
            "1) coordinates, coordinate, cylindrical polar, real, "
            "#Components=3\n"
            " r. #Values=1 (value)\n theta. #Values=1 (value)\n"
            " z. #Values=1 (value)\n"
            "Node: 1\n 2.0\n 1.0\n 5.0\nNode: 2\n 3.0\n 2.0\n 6.0\n"
            "!#mesh mesh1d, dimension=1, nodeset=nodes\n"
            "Shape. Dimension=1, line\n"
            "#Scale factor sets=0\n#Nodes=2\n#Fields=1\n"
            "1) coordinates, coordinate, cylindrical polar, real, "
            "#Components=3\n"
            " r. l.Lagrange, no modify, standard node based.\n"
            "  #Nodes=2\n  1. #Values=1\n   Value labels: value\n"
            "  2. #Values=1\n   Value labels: value\n"
            " theta. l.Lagrange, no modify, standard node based.\n"
            "  #Nodes=2\n  1. #Values=1\n   Value labels: value\n"
            "  2. #Values=1\n   Value labels: value\n"
            " z. l.Lagrange, no modify, standard node based.\n"
            "  #Nodes=2\n  1. #Values=1\n   Value labels: value\n"
            "  2. #Values=1\n   Value labels: value\n"
            "Element: 1\n Nodes:\n 1 2\n")
        mesh = exfield.loads(text)
        with pytest.warns(UserWarning, match="cylindrical polar"):
            exfield.Evaluator(mesh.fields["coordinates"], dimension=1)
