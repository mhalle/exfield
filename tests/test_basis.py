"""Basis regression tests. Function ordering is the #1 source of
silently-wrong geometry; these pin Zinc's contract exactly."""

import numpy as np
import pytest

from exfield.basis import parse_basis
from exfield.errors import UnsupportedExFeature


def test_cubic_hermite_1d_functions():
    b = parse_basis("c.Hermite")
    assert b.number_of_functions == 4
    assert b.number_of_nodes == 2
    # order: n1 value, n1 d/ds1, n2 value, n2 d/ds1
    assert b.function_node == [0, 0, 1, 1]
    assert b.function_derivatives == [(0,), (1,), (0,), (1,)]
    x = 0.3
    phi = b.evaluate([x])
    assert phi[0] == pytest.approx(1 - 3 * x**2 + 2 * x**3)
    assert phi[1] == pytest.approx(x - 2 * x**2 + x**3)
    assert phi[2] == pytest.approx(3 * x**2 - 2 * x**3)
    assert phi[3] == pytest.approx(-x**2 + x**3)
    # derivative dofs are unit at their end: phi2'(0)=1, phi4'(1)=1
    d0 = b.evaluate([0.0], (1,))
    d1 = b.evaluate([1.0], (1,))
    assert d0[1] == pytest.approx(1.0)
    assert d1[3] == pytest.approx(1.0)


def test_tricubic_hermite_per_node_label_order():
    """Per node: value, d/ds1, d/ds2, d2/ds1ds2, d/ds3, d2/ds1ds3,
    d2/ds2ds3, d3/ds1ds2ds3 — the derivative bitmask with xi1 fastest."""
    b = parse_basis("c.Hermite*c.Hermite*c.Hermite")
    assert b.number_of_functions == 64
    assert b.number_of_nodes == 8
    expected = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    assert b.function_derivatives[:8] == expected
    assert b.function_node[:8] == [0] * 8
    assert b.function_node[8:16] == [1] * 8


def test_nodes_ordered_xi1_fastest():
    b = parse_basis("l.Lagrange*l.Lagrange*l.Lagrange")
    # node n at corner (i1,i2,i3): value function is 1 at that corner
    for node in range(8):
        i = (node & 1, (node >> 1) & 1, (node >> 2) & 1)
        phi = b.evaluate([float(i[0]), float(i[1]), float(i[2])])
        assert phi[node] == pytest.approx(1.0)
        assert phi.sum() == pytest.approx(1.0)


def test_quadratic_and_cubic_lagrange_match_zinc_matrices():
    q = parse_basis("q.Lagrange")
    x = 0.7
    phi = q.evaluate([x])
    assert phi[0] == pytest.approx(1 - 3 * x + 2 * x * x)
    assert phi[1] == pytest.approx(4 * x - 4 * x * x)
    assert phi[2] == pytest.approx(-x + 2 * x * x)
    c = parse_basis("c.Lagrange")
    phi = c.evaluate([x])
    assert phi[0] == pytest.approx(1 - 5.5 * x + 9 * x**2 - 4.5 * x**3)
    assert phi[1] == pytest.approx(9 * x - 22.5 * x**2 + 13.5 * x**3)
    assert phi[2] == pytest.approx(-4.5 * x + 18 * x**2 - 13.5 * x**3)
    assert phi[3] == pytest.approx(x - 4.5 * x**2 + 4.5 * x**3)


def test_partition_of_unity():
    for desc in ("c.Hermite*l.Lagrange*l.Lagrange",
                 "c.Hermite*c.Hermite",
                 "q.Lagrange*c.Lagrange"):
        b = parse_basis(desc)
        rng = np.random.default_rng(42)
        for _ in range(5):
            xi = rng.random(b.dimension)
            phi = b.evaluate(xi)
            values = [phi[f] for f in range(b.number_of_functions)
                      if b.function_derivatives[f] ==
                      (0,) * b.dimension]
            assert sum(values) == pytest.approx(1.0)


def test_serendipity_2d_structure():
    b = parse_basis("c.HermiteSerendipity(2)*c.HermiteSerendipity")
    assert b.number_of_functions == 12
    assert b.number_of_nodes == 4
    assert b.function_derivatives[:3] == [(0, 0), (1, 0), (0, 1)]
    # partition of unity over the 4 value functions
    xi = np.array([0.3, 0.8])
    phi = b.evaluate(xi)
    assert phi[0] + phi[3] + phi[6] + phi[9] == pytest.approx(1.0)
    # value functions are interpolatory at corners
    assert b.evaluate([0.0, 0.0])[0] == pytest.approx(1.0)
    assert b.evaluate([1.0, 1.0])[9] == pytest.approx(1.0)
    # no cross-derivative dof exists
    assert (1, 1) not in b.function_derivatives


def test_serendipity_3d_structure():
    b = parse_basis(
        "c.HermiteSerendipity(2;3)*c.HermiteSerendipity*c.HermiteSerendipity")
    assert b.number_of_functions == 32
    assert b.number_of_nodes == 8
    values = [f for f, d in enumerate(b.function_derivatives)
              if d == (0, 0, 0)]
    phi = b.evaluate([0.2, 0.6, 0.9])
    assert sum(phi[f] for f in values) == pytest.approx(1.0)


def test_unsupported_bases_decline():
    for desc in ("l.simplex(2)*l.simplex", "polygon(5;2)*polygon",
                 "LagrangeHermite",
                 "c.HermiteSerendipity(2)*c.HermiteSerendipity*l.Lagrange"):
        with pytest.raises(UnsupportedExFeature):
            parse_basis(desc)


def test_derivative_matches_finite_difference():
    b = parse_basis("c.Hermite*q.Lagrange")
    xi = np.array([0.4, 0.6])
    h = 1e-6
    d = b.evaluate_derivatives(xi)
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        fd = (b.evaluate(xi + e) - b.evaluate(xi - e)) / (2 * h)
        assert np.allclose(d[k], fd, atol=1e-6)
