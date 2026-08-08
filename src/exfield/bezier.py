"""Cubic Hermite <-> Bezier basis conversion.

The conversion is an exact change of basis — the two bases span the same
cubic space. BUT the round trip loses structure: Bezier control points
make C1 continuity across element boundaries *emergent* rather than
enforced, bake derivative versions at bifurcations into separate control
points, and absorb scale factors into coefficients. Fine for evaluation
and display (e.g. exporting to SVG/VTK splines); wrong for anything that
edits geometry — edit the Hermite parameters and re-convert instead.
"""

import numpy as np

#: Matrix M such that [b0 b1 b2 b3] = M @ [p0 d0 p1 d1] per component,
#: for a cubic Hermite segment with values p and xi-derivatives d.
HERMITE_TO_BEZIER = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0 / 3.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -1.0 / 3.0],
    [0.0, 0.0, 1.0, 0.0],
])

BEZIER_TO_HERMITE = np.linalg.inv(HERMITE_TO_BEZIER)


def hermite_to_bezier_1d(p0, d0, p1, d1):
    """Control points of the cubic Bezier equal to a Hermite segment.

    Parameters are the end values and end xi-derivatives (n-vectors).
    Returns an array (4, n): [p0, p0 + d0/3, p1 - d1/3, p1].

    See the module docstring: exact for evaluation/display; do not edit
    geometry in Bezier form.
    """
    p0 = np.asarray(p0, dtype=float)
    d0 = np.asarray(d0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    return np.stack([p0, p0 + d0 / 3.0, p1 - d1 / 3.0, p1])


def hermite_to_bezier_matrix():
    """The 4x4 change-of-basis matrix (copy)."""
    return HERMITE_TO_BEZIER.copy()
