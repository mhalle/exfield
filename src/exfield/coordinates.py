"""Coordinate system conversions to rectangular cartesian.

Formulas verbatim from Zinc's ``general/geometry.cpp``:

* cylindrical polar ``(r, theta, z)``:
  ``x = r cos(theta); y = r sin(theta); z = z``
* spherical polar ``(r, theta, phi)``:
  ``x = r cos(phi) cos(theta); y = r cos(phi) sin(theta); z = r sin(phi)``
* prolate spheroidal ``(lambda, mu, theta)`` with focus ``f``:
  ``x = f cosh(l) cos(mu); y = f sinh(l) sin(mu) cos(th);
  z = f sinh(l) sin(mu) sin(th)``
* oblate spheroidal ``(lambda, mu, theta)`` with focus ``f``:
  ``x = f cosh(l) cos(mu) sin(th); y = f sinh(l) sin(mu);
  z = f cosh(l) cos(mu) cos(th)``

Conversion is deliberately explicit, not automatic: an ``Evaluator``
over a non-cartesian field *warns* and returns the raw components,
because silently transforming values is its own hazard. Convert with::

    xyz = to_rectangular_cartesian(values, field.coordinate_system,
                                   focus=field.focus)
"""

import numpy as np

CONVERTIBLE = ("rectangular cartesian", "cylindrical polar",
               "spherical polar", "prolate spheroidal",
               "oblate spheroidal")


def to_rectangular_cartesian(values, coordinate_system, focus=None):
    """Convert coordinate values to rectangular cartesian.

    ``values``: shape (3,) or (n, 3) in the declared system's component
    order (as stored in the EX file). Returns the same shape in x, y, z.
    ``focus`` is required for prolate/oblate spheroidal (the EX field
    header carries it as ``focus=``).
    """
    v = np.asarray(values, dtype=float)
    single = v.ndim == 1
    v = np.atleast_2d(v)
    if v.shape[1] != 3:
        raise ValueError(f"Expected 3 components, got {v.shape[1]}")
    cs = (coordinate_system or "rectangular cartesian").lower()
    if cs == "rectangular cartesian":
        out = v.copy()
    elif cs == "cylindrical polar":
        r, theta, z = v[:, 0], v[:, 1], v[:, 2]
        out = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    elif cs == "spherical polar":
        r, theta, phi = v[:, 0], v[:, 1], v[:, 2]
        out = np.stack([r * np.cos(phi) * np.cos(theta),
                        r * np.cos(phi) * np.sin(theta),
                        r * np.sin(phi)], axis=1)
    elif cs == "prolate spheroidal":
        if focus is None:
            raise ValueError("prolate spheroidal conversion requires focus")
        lam, mu, theta = v[:, 0], v[:, 1], v[:, 2]
        a = focus * np.sinh(lam) * np.sin(mu)
        out = np.stack([focus * np.cosh(lam) * np.cos(mu),
                        a * np.cos(theta),
                        a * np.sin(theta)], axis=1)
    elif cs == "oblate spheroidal":
        if focus is None:
            raise ValueError("oblate spheroidal conversion requires focus")
        lam, mu, theta = v[:, 0], v[:, 1], v[:, 2]
        a = focus * np.cosh(lam) * np.cos(mu)
        out = np.stack([a * np.sin(theta),
                        focus * np.sinh(lam) * np.sin(mu),
                        a * np.cos(theta)], axis=1)
    else:
        raise ValueError(
            f"Cannot convert from coordinate system {coordinate_system!r}")
    return out[0] if single else out
