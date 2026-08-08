"""Rigorous per-element bounding boxes via Bernstein control points.

Every basis exfield supports is polynomial with a known blending matrix,
so an element's geometry has exact monomial coefficients
``C = B.T @ P``. Re-expressing those in the Bernstein basis gives a
Bezier control net whose convex hull contains the geometry (the convex
hull property), so the control points' min/max is a *mathematically
guaranteed* axis-aligned bounding box — no sampling, no fudge tolerance.

This is what makes branch-and-bound nearest-point search exact: an
element whose box is farther than the best distance found so far
provably cannot contain the nearest point. (Zinc's equivalent,
``FeMeshFieldRanges``, samples the field and pads by 1% of the mesh
range; it treats elements as black boxes and cannot get a rigorous
bound.)

Monomial index order matches ``exfield.basis``: xi1 power fastest.
"""

import numpy as np
from functools import reduce

from math import comb


def _bernstein_to_monomial(order):
    """T with T[j, i] = coefficient of x^j in Bernstein b_i^order(x)."""
    T = np.zeros((order + 1, order + 1))
    for i in range(order + 1):
        # b_i(x) = C(n,i) x^i (1-x)^(n-i) = sum_k C(n,i) C(n-i,k) (-1)^k x^(i+k)
        for k in range(order - i + 1):
            T[i + k, i] = comb(order, i) * comb(order - i, k) * (-1.0) ** k
    return T


_MONO_TO_BERNSTEIN_1D = {
    order: np.linalg.inv(_bernstein_to_monomial(order))
    for order in (0, 1, 2, 3)
}


def monomial_to_bernstein(orders):
    """Change-of-basis matrix from tensor monomial coefficients (xi1
    power fastest) to Bernstein control values (same index layout)."""
    mats = [_MONO_TO_BERNSTEIN_1D[o] for o in orders]
    # index m = p1 + (o1+1)*(p2 + ...): xi1 fastest -> last xi outermost
    return reduce(np.kron, reversed(mats))


def element_control_points(P, basis):
    """Bezier control points of an element's field: (n_monomials,
    n_components). Their min/max per component is a rigorous AABB."""
    C_mono = basis.B.T @ P                       # monomial coefficients
    M = monomial_to_bernstein(basis.orders)
    return M @ C_mono


def element_aabb(P, basis):
    """(mins, maxs) arrays over components — guaranteed to contain the
    element geometry."""
    ctrl = element_control_points(P, basis)
    return ctrl.min(axis=0), ctrl.max(axis=0)


def aabb_distance_squared(point, mins, maxs):
    """Squared distance from point(s) to AABB(s), 0 inside.

    ``mins``/``maxs`` may be (n_comp,) or stacked (n_el, n_comp);
    ``point`` is (n_comp,). Vectorised over elements."""
    d = np.maximum(mins - point, 0.0) + np.maximum(point - maxs, 0.0)
    return np.sum(d * d, axis=-1)
