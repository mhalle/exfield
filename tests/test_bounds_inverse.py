"""Tests for rigorous element bounds and branch-and-bound inverse
mapping."""

import warnings

import numpy as np
import pytest

import exfield
from exfield.bounds import (aabb_distance_squared, element_aabb,
                            monomial_to_bernstein)
from exfield.basis import parse_basis


class TestBernsteinBounds:
    def test_monomial_to_bernstein_roundtrip(self):
        # Bernstein control values of x^2 on [0,1] are [0, 0, 1] for
        # order 2: b2 = x^2 exactly
        M = monomial_to_bernstein((2,))
        c = M @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(c, [0.0, 0.0, 1.0])
        # constant 1 -> all control values 1 (partition of unity)
        c = M @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(c, [1.0, 1.0, 1.0])

    @pytest.mark.parametrize("desc", [
        "c.Hermite", "c.Hermite*c.Hermite*c.Hermite",
        "c.Hermite*l.Lagrange*l.Lagrange",
        "c.HermiteSerendipity(2)*c.HermiteSerendipity",
    ])
    def test_aabb_is_conservative(self, desc):
        """The Bezier control-net box must contain the geometry for any
        random parameters (convex hull property)."""
        basis = parse_basis(desc)
        rng = np.random.default_rng(11)
        for _ in range(5):
            P = rng.normal(size=(basis.number_of_functions, 3))
            mins, maxs = element_aabb(P, basis)
            xis = rng.random((400, basis.dimension))
            pts = basis.evaluate(xis) @ P
            assert np.all(pts >= mins - 1e-9)
            assert np.all(pts <= maxs + 1e-9)

    def test_aabb_distance(self):
        mins = np.array([[0.0, 0.0], [10.0, 10.0]])
        maxs = np.array([[1.0, 1.0], [11.0, 11.0]])
        d2 = aabb_distance_squared(np.array([0.5, 0.5]), mins, maxs)
        assert d2[0] == 0.0
        assert d2[1] == pytest.approx(2 * 9.5 ** 2)


class TestBranchAndBound:
    def test_round_trips_exact_on_real_scaffold(self, vagus_mesh):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=3)
        ids = sorted(vagus_mesh.mesh3d.elements)
        rng = np.random.default_rng(0)
        for _ in range(40):
            eid = ids[rng.integers(len(ids))]
            xi = rng.random(3)
            target = ev.evaluate(eid, xi)
            loc = exfield.find_location(ev, target, element_ids="all")
            assert loc.residual < 1e-6, (eid, xi, loc)

    def test_pruning_matches_restricted_search(self, vagus_mesh):
        """Branch-and-bound over all elements must find the same point
        as a search restricted to the known answer's element."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=3)
        ids = sorted(vagus_mesh.mesh3d.elements)
        rng = np.random.default_rng(38)   # includes the folded-nerve case
        for _ in range(10):
            eid = ids[rng.integers(len(ids))]
            xi = rng.random(3)
            target = ev.evaluate(eid, xi)
            full = exfield.find_location(ev, target, element_ids="all")
            restricted = exfield.find_location(ev, target,
                                               element_ids=[eid])
            assert full.residual <= restricted.residual + 1e-9

    def test_fused_value_and_jacobian(self, vagus_mesh):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=3)
        eid = sorted(vagus_mesh.mesh3d.elements)[5]
        xis = np.random.default_rng(2).random((30, 3))
        x, J = ev.value_and_jacobian(eid, xis)
        assert np.allclose(x, ev.evaluate(eid, xis), atol=1e-13)
        assert np.allclose(J, ev.evaluate_derivatives(eid, xis),
                           atol=1e-13)
        x1, J1 = ev.value_and_jacobian(eid, xis[0])
        assert x1.shape == (3,) and J1.shape == (3, 3)
