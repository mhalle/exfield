"""Inverse mapping: world coordinates -> (element, xi).

Gauss-Newton iteration on the squared distance, seeded from a sample
grid, with xi clipped to [0, 1]^d.

Hazards this API is shaped around (EXFIELD_GOTCHAS.md §3):

* On branching structures, "nearest point on the mesh" is the wrong
  question — a branch running alongside the trunk is frequently nearer
  than the trunk. ``element_ids`` is required for 1-D meshes with branch
  points; it is not an optimisation.
* xi is clipped, so a point beyond the end of a chain projects to the
  endpoint and returns a plausible-looking address with a large
  residual. The result carries an explicit ``boundary`` flag.
* Newton with multi-candidate seeding can still find a local minimum;
  the result exposes ``runner_up_residual`` so ambiguity is visible.
"""

from dataclasses import dataclass

import numpy as np

from .evaluate import Evaluator, endpoint_keys


@dataclass
class Location:
    """Result of an inverse mapping for one point."""
    element_id: int
    xi: np.ndarray
    residual: float          # distance from point to mapped location
    boundary: bool           # True if xi is pinned to the element boundary
                             # with the minimum outside — the point
                             # projects past the mesh edge
    runner_up_residual: float  # best residual found in any *other*
                               # element (inf if none searched); when
                               # close to `residual`, the address is
                               # ambiguous

    @property
    def ambiguous(self):
        # <= so an exact tie (both residuals 0.0 at a shared endpoint)
        # is flagged; strict < would report 0 < 0 as unambiguous
        return self.runner_up_residual <= 2.0 * self.residual


def _has_branch_points(evaluator):
    """Geometric branch detection: any 1-D element endpoint shared by
    more than two elements. Works for face-inherited elements."""
    ends = endpoint_keys(evaluator, sorted(evaluator.element_mesh.elements))
    node_use = {}
    for a, b in ends.values():
        for n in (a, b):
            node_use[n] = node_use.get(n, 0) + 1
            if node_use[n] > 2:
                return True
    return False


def find_location(evaluator, point, element_ids=None, n_sample=12,
                  n_candidates=4, tol=1e-10, max_iterations=50):
    """Find the (element, xi) closest to a world point.

    Parameters
    ----------
    evaluator : Evaluator (or Field/Mesh, converted).
    point : world coordinates, shape (n_components,).
    element_ids : elements to search. REQUIRED when a 1-D mesh has branch
        points — pass the chain/subset you mean, or ``"all"`` to insist
        on searching everything (and accept that a nearby branch may
        win).
    n_sample : seed samples per xi direction per element.
    n_candidates : how many best seeds to polish with Gauss-Newton.

    Returns
    -------
    Location
    """
    if not isinstance(evaluator, Evaluator):
        evaluator = Evaluator(evaluator)
    point = np.asarray(point, dtype=float)
    mesh = evaluator.element_mesh
    if element_ids is None or element_ids == "all":
        if (element_ids is None and evaluator.dimension == 1
                and len(mesh) > 1 and _has_branch_points(evaluator)):
            raise ValueError(
                "This 1-D mesh has branch points; 'nearest point on the "
                "mesh' is ambiguous because a branch running alongside "
                "the trunk is frequently nearer than the trunk. Pass "
                "element_ids= selecting the elements you mean (or "
                "element_ids='all' to insist).")
        ids = [e.identifier for e in mesh]
    else:
        ids = list(element_ids)
    if not ids:
        raise ValueError("No elements to search")
    dim = evaluator.dimension

    # Branch-and-bound over rigorous per-element bounding boxes (Bezier
    # control net convex hull; see exfield.bounds). Elements are polished
    # in ascending box-distance order and the sweep stops as soon as the
    # next box is farther than the best point found — which is exact:
    # a pruned element provably cannot contain a nearer point. This is
    # the same strategy as Zinc's FeMeshFieldRanges pruning, but with
    # guaranteed rather than sampled-and-padded boxes.
    bounds = [evaluator.element_bounds(eid) for eid in ids]
    mins = np.array([b[0] for b in bounds])
    maxs = np.array([b[1] for b in bounds])
    d = np.maximum(mins - point, 0.0) + np.maximum(point - maxs, 0.0)
    box_d2 = np.einsum("ij,ij->i", d, d)
    order = np.argsort(box_d2, kind="stable")

    axis = np.linspace(0.0, 1.0, n_sample)
    grid = np.stack(np.meshgrid(*([axis] * dim), indexing="ij"),
                    axis=-1).reshape(-1, dim)
    n_keep = max(1, n_candidates)

    best = None            # (eid, dist, xi, pinned)
    runner_up = np.inf     # best distance in any element other than best
    pruned_lower = np.inf  # lower bound on distance to first pruned element
    for idx in order:
        # Strict > : boxes at EXACTLY the best distance are polished,
        # not pruned. A pruned box distance is only a lower bound — a
        # box touching the query point (box distance 0) may hold a far
        # point, and pruning it would report runner_up_residual 0.0 for
        # an element that is not actually close. Polishing resolves
        # both that false tie and the true shared-endpoint tie exactly.
        if best is not None and box_d2[idx] > best[1] ** 2:
            pruned_lower = float(np.sqrt(box_d2[idx]))
            break
        eid = ids[idx]
        # multi-start polish within this element (all starts iterated
        # together as one batch)
        x = evaluator.evaluate(eid, grid)
        d2 = np.einsum("ij,ij->i", x - point, x - point)
        seeds = grid[np.argsort(d2)[:n_keep]]
        dist, xi, pinned = _gauss_newton_multi(
            evaluator, eid, point, seeds, tol, max_iterations)
        if best is None or dist < best[1]:
            if best is not None:
                runner_up = min(runner_up, best[1])
            best = (eid, dist, xi, pinned)
        else:
            runner_up = min(runner_up, dist)
    eid, dist, xi, pinned = best
    # Every element whose box came within `dist` (inclusive) was
    # polished, so `runner_up` is exact among those — including exact
    # ties; the first pruned box distance is strictly greater than
    # `dist` and a valid lower bound for everything else. Using the
    # minimum keeps the ambiguity flag conservative (it may flag,
    # never miss) without ever reporting a phantom zero.
    runner_up = min(runner_up, pruned_lower)
    return Location(element_id=eid, xi=xi, residual=dist,
                    boundary=pinned, runner_up_residual=runner_up)


def closest_point(evaluator, points, element_ids=None, **kwargs):
    """Vector version of :func:`find_location`: list of Locations.

    Each query runs branch-and-bound over cached per-element bounding
    boxes, so cost grows with the handful of elements near each point,
    not with mesh size. Queries are still processed one at a time in
    Python (~1.5 ms each on a 3-D scaffold); restrict ``element_ids``
    where you can.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return [find_location(evaluator, p, element_ids=element_ids, **kwargs)
            for p in points]


def _gauss_newton_multi(evaluator, element_id, point, seeds, tol,
                        max_iterations):
    """Minimise |x(xi) - point|^2 from several starts simultaneously.

    All starts advance together with batched field evaluation — the
    per-call Python overhead is paid once per iteration, not once per
    start per iteration. Returns (distance, xi, boundary_pinned) of the
    best converged start; ``boundary_pinned`` is True when that start
    ended on the element boundary with its unconstrained step pointing
    outside.
    """
    xis = np.array(seeds, dtype=float)          # (k, dim)
    k, dim = xis.shape
    active = np.ones(k, dtype=bool)
    pinned = np.zeros(k, dtype=bool)
    eye = 1e-12 * np.eye(dim)
    for _ in range(max_iterations):
        x, J = evaluator.evaluate_with_derivatives(element_id, xis)
        r = x - point
        g = np.einsum("kdc,kc->kd", J, r)
        H = J @ np.swapaxes(J, 1, 2) + eye
        try:
            steps = np.linalg.solve(H, -g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break  # degenerate geometry: keep current (seed) locations
        raw = xis + steps
        new_xis = np.clip(raw, 0.0, 1.0)
        pinned = np.any(((new_xis <= 0.0) & (raw < -1e-12))
                        | ((new_xis >= 1.0) & (raw > 1.0 + 1e-12)), axis=1)
        moved = np.max(np.abs(new_xis - xis), axis=1)
        xis = np.where(active[:, None], new_xis, xis)
        active = active & (moved >= tol)
        if not active.any():
            break
    x = evaluator.evaluate(element_id, xis)
    dists = np.linalg.norm(x - point, axis=1)
    best = int(np.argmin(dists))
    return float(dists[best]), xis[best], bool(pinned[best])
