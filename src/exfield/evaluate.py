"""Field evaluation and arclength integration.

``Evaluator`` evaluates a field (values and xi-derivatives) at
``(element, xi)``. Assembly follows Zinc exactly: per basis function, the
element dof is a sum of terms, each a node parameter (selected by value
label and version) times a product of scale factors (empty product = 1);
the field value is the dot product of basis function values with element
dofs.

``ArclengthTable`` integrates ``|dx/dxi|`` by Gauss-Legendre quadrature
over a *chain* of 1-D elements. Two hazards it is shaped around (see
README.md, "The API is shaped around silent-wrong-geometry hazards"):

* Over a branching mesh, summing all elements silently sums the branches
  too — the most likely wrong number a user will produce. ``build``
  refuses a mesh whose connectivity is not a simple path unless the
  caller passes ``element_ids``.
* Zinc's ``FieldMeshIntegral`` stops improving above 4 Gauss points; the
  converged answer is slightly larger on strongly curved elements. The
  default here is ``order=16`` (converged); pass ``order="zinc"`` for
  bug-compatible agreement with Zinc 4.2 to ~1e-12.
"""

import warnings

import numpy as np

from .errors import EvaluationError
from .faces import ParentMap, face_to_element_map
from .mesh import Field, Mesh


class Evaluator:
    """Evaluate one field on the elements of one dimension.

    Parameters
    ----------
    field : Field, or a Mesh (in which case ``mesh.coordinates`` is used,
        which raises rather than guessing when several coordinate fields
        exist — real scaffolds carry e.g. 'coordinates', 'straight
        coordinates' and 'vagus coordinates', and evaluating the wrong
        one runs perfectly and returns nonsense).
    dimension : element mesh dimension; defaults to the highest present.
    """

    def __init__(self, field, dimension=None):
        if isinstance(field, Mesh):
            field = field.coordinates
        if not isinstance(field, Field):
            raise TypeError(
                f"Evaluator takes a Field (e.g. mesh.fields['coordinates'])"
                f" or a Mesh, not {type(field).__name__}")
        if field.mesh is None:
            raise ValueError(f"Field {field.name!r} is not part of a mesh")
        self.field = field
        self.model = field.mesh
        if dimension is None:
            dimension = self.model.highest_dimension
            if dimension == 0:
                raise EvaluationError("Mesh has no elements")
        self.dimension = dimension
        self.element_mesh = self.model.element_mesh(dimension)
        self.nodeset = self.model.nodesets[self.element_mesh.nodeset_name]
        self._param_cache = {}
        self._resolve_cache = {}
        self._bounds_cache = {}
        self._parents = ParentMap(self.model)
        self._maybe_warn_normalized()
        self._maybe_warn_coordinate_system()

    def _maybe_warn_coordinate_system(self):
        """Evaluation returns raw components in the field's declared
        coordinate system. For non-cartesian systems that is rarely what
        a caller doing geometry wants — distances and integrals computed
        on raw (r, theta, z) or (lambda, mu, theta) components are
        wrong. Convert explicitly with
        :func:`exfield.to_rectangular_cartesian`."""
        cs = self.field.coordinate_system
        if cs is not None and cs not in ("rectangular cartesian", "fibre"):
            warnings.warn(
                f"Field {self.field.name!r} is declared in "
                f"'{cs}' coordinates; evaluate() returns raw components "
                f"in that system. Use "
                f"exfield.to_rectangular_cartesian(values, {cs!r}"
                f"{', focus=...' if 'spheroidal' in cs else ''}) before "
                f"measuring distances, lengths, areas or volumes.",
                stacklevel=3)

    def _maybe_warn_normalized(self):
        """Warn when the field looks dimensionless (all values in [0,1]).

        Building an Evaluator on a material field (e.g. 'vagus
        coordinates') runs perfectly and computes nonsense lengths."""
        if self.field.value_type != "real":
            return
        lo, hi = np.inf, -np.inf
        for i, node in enumerate(self.nodeset):
            comps = node.fields.get(self.field.name)
            if comps is None:
                continue
            for values in comps:
                if len(values):
                    lo = min(lo, values.min())
                    hi = max(hi, values.max())
            if i >= 200:
                break
        if lo >= -1.0 - 1e-9 and hi <= 1.0 + 1e-9 and hi > lo:
            warnings.warn(
                f"Field {self.field.name!r} values all lie in [-1, 1] — "
                f"this looks like a material/normalised field, not "
                f"geometry. Coordinate-type fields on this mesh: "
                f"{self.model.coordinate_field_names}", stacklevel=3)

    # ------------------------------------------------------------ params

    def element(self, element_id, dimension=None):
        """The :class:`~exfield.mesh.Element` object for an id, on this
        evaluator's dimension unless another is given.

        Raises :class:`~exfield.errors.EvaluationError` if no such
        element exists there. Evaluation itself takes ids directly —
        this is for reaching an element's shape, faces or template.
        """
        dimension = dimension if dimension is not None else self.dimension
        mesh = self.model.element_meshes.get(dimension)
        if mesh is None or element_id not in mesh:
            raise EvaluationError(
                f"No element {element_id} in mesh{dimension}d")
        return mesh[element_id]

    def _has_definition(self, element):
        if element.template is None:
            return False
        if self.field.field_type == "constant":
            return True
        return (self.field.name, 0) in element.template.efts

    def _resolve(self, element_id):
        """Find where the field is actually defined, following face
        inheritance upward. Returns (dimension, element_id, A, b) with
        ``xi_defined = A @ xi + b`` (A None for the identity)."""
        key = element_id
        cached = self._resolve_cache.get(key)
        if cached is not None:
            return cached
        dimension = self.dimension
        eid = element_id
        A = None
        b = None
        while True:
            element = self.element(eid, dimension)
            if self._has_definition(element):
                break
            parent = self._parents.parent_of(dimension, eid)
            if parent is None:
                raise EvaluationError(
                    f"Field {self.field.name!r} is not defined on element "
                    f"{element_id} (mesh{self.dimension}d) nor inherited "
                    f"from any parent element")
            pid, face_index = parent
            A1, b1 = face_to_element_map(dimension + 1, face_index)
            if A is None:
                A, b = A1, b1
            else:
                b = A1 @ b + b1
                A = A1 @ A
            dimension += 1
            eid = pid
        result = (dimension, eid, A, b)
        self._resolve_cache[key] = result
        return result

    def element_parameters(self, element_id, dimension=None):
        """The (number_of_functions, number_of_components) matrix of
        scaled element dofs for this field, cached per element. The
        element must define the field directly (use :meth:`evaluate` for
        inherited evaluation)."""
        dimension = dimension if dimension is not None else self.dimension
        cached = self._param_cache.get((dimension, element_id))
        if cached is not None:
            return cached
        element = self.element(element_id, dimension)
        field = self.field
        if element.template is None:
            raise EvaluationError(
                f"Element {element_id} has no field definitions (it was "
                f"only referenced, never defined)")
        if field.field_type == "constant":
            P = np.asarray(field.values, dtype=float).reshape(1, -1)
            self._param_cache[(dimension, element_id)] = (P, None)
            return self._param_cache[(dimension, element_id)]
        eft0 = element.template.efts.get((field.name, 0))
        if eft0 is None:
            raise EvaluationError(
                f"Field {field.name!r} is not defined on element "
                f"{element_id}")
        n_comp = field.number_of_components
        nodeset = self.model.nodesets[
            self.model.element_meshes[dimension].nodeset_name]
        # exfield requires all components to share one basis (true of all
        # SPARC scaffolds); mixed-basis components would need per-component
        # phi vectors. Check BEFORE filling P: P is sized from component
        # 0's basis, so a later component with more functions would
        # otherwise crash in the fill loop with a bare IndexError instead
        # of this message. Bases are interned in basis._BASIS_CACHE, so
        # identity comparison is exact here.
        basis_by_comp = [element.template.efts[(field.name, c)].basis
                         for c in range(n_comp)]
        for b in basis_by_comp[1:]:
            if b is not basis_by_comp[0]:
                raise EvaluationError(
                    f"Field {field.name!r} components use different bases "
                    f"on element {element_id}; not supported")
        P = np.zeros((basis_by_comp[0].number_of_functions, n_comp))
        for c in range(n_comp):
            eft = element.template.efts[(field.name, c)]
            for fn, terms in enumerate(eft.functions):
                total = 0.0
                for term in terms:
                    node_id = element.nodes[term.local_node - 1]
                    if node_id is None:
                        raise EvaluationError(
                            f"Element {element_id} local node "
                            f"{term.local_node} is unset (-1)")
                    node = nodeset[node_id]
                    value = node.get_parameter(field.name, c, term.label,
                                               term.version)
                    for sf_index in term.scale_factor_indices:
                        value = value * element.scale_factors[sf_index]
                    total += value
                P[fn, c] = total
        result = (P, basis_by_comp[0])
        self._param_cache[(dimension, element_id)] = result
        return result

    def element_bounds(self, element_id):
        """Rigorous axis-aligned bounding box (mins, maxs) of the field
        on an element, from Bezier control points (convex hull property
        — guaranteed to contain the geometry, never sampled). For
        face-inherited elements the defining parent's box is used, which
        is conservative."""
        cached = self._bounds_cache.get(element_id)
        if cached is not None:
            return cached
        from .bounds import element_aabb
        dimension, eid, _A, _b = self._resolve(element_id)
        P, basis = self.element_parameters(eid, dimension)
        if basis is None:
            bounds = (P[0].copy(), P[0].copy())
        else:
            bounds = element_aabb(P, basis)
        self._bounds_cache[element_id] = bounds
        return bounds

    # ---------------------------------------------------------- evaluate

    def _prepare_xi(self, element_id, xi):
        """Resolve inheritance and map xi (single or batch) into the
        defining element's xi space."""
        dimension, eid, A, b = self._resolve(element_id)
        xi = np.asarray(xi, dtype=float)
        single = xi.ndim <= 1
        xis = np.atleast_2d(xi)
        if xis.shape[1] != self.dimension:
            raise ValueError(
                f"xi must have {self.dimension} column(s), got "
                f"{xis.shape[1]}")
        if A is not None:
            xis = xis @ A.T + b
        return dimension, eid, A, xis, single

    # The three evaluation entry points differ only in what they
    # RETURN — values, derivatives, or both fused. All three take one
    # xi or a batch; none of them is "the batch one".

    def evaluate(self, element_id, xi):
        """Field value(s) at (element, xi).

        ``xi`` may be one point ``(dimension,)`` or a batch
        ``(n, dimension)``; returns ``(n_components,)`` or
        ``(n, n_components)``. Elements without their own field
        definition (faces/lines listed in ``Faces:`` sections) are
        evaluated by inheritance from a parent element, as in Zinc.
        """
        dimension, eid, _A, xis, single = self._prepare_xi(element_id, xi)
        P, basis = self.element_parameters(eid, dimension)
        if basis is None:  # constant field
            out = np.broadcast_to(P[0], (xis.shape[0], P.shape[1])).copy()
            return out[0] if single else out
        phi = basis.evaluate(xis)
        out = phi @ P
        return out[0] if single else out

    def evaluate_derivatives(self, element_id, xi):
        """d(value)/d(xi_k) for the element's own xi.

        Single point -> (dimension, n_components); batch ``(n,
        dimension)`` xi -> (n, dimension, n_components). Face-inherited
        elements chain the parent derivatives through the
        face-to-element map."""
        dimension, eid, A, xis, single = self._prepare_xi(element_id, xi)
        P, basis = self.element_parameters(eid, dimension)
        n = xis.shape[0]
        if basis is None:
            out = np.zeros((n, self.dimension,
                            self.field.number_of_components))
            return out[0] if single else out
        dphi = basis.evaluate_derivatives(xis)   # (n, def_dim, n_funcs)
        derivs = dphi @ P                        # (n, def_dim, n_comp)
        if A is not None:
            derivs = np.einsum("kj,nkc->njc", A, derivs)
        return derivs[0] if single else derivs

    def evaluate_values_and_derivatives(self, element_id, xi):
        """Values and their xi-derivatives together, in one pass.

        Same inputs as :meth:`evaluate`; returns exactly what
        :meth:`evaluate` and :meth:`evaluate_derivatives` return
        separately, sharing one element resolve and one monomial
        evaluation — the fast path for Newton-type loops. Single point
        -> ``(values (c,), derivatives (d, c))``; batch ``(n,
        dimension)`` xi -> ``(values (n, c), derivatives (n, d, c))``."""
        dimension, eid, A, xis, single = self._prepare_xi(element_id, xi)
        P, basis = self.element_parameters(eid, dimension)
        n = xis.shape[0]
        if basis is None:
            x = np.broadcast_to(P[0], (n, P.shape[1])).copy()
            J = np.zeros((n, self.dimension, P.shape[1]))
            return (x[0], J[0]) if single else (x, J)
        phi, dphi = basis.evaluate_values_and_derivatives(xis)
        x = phi @ P
        J = dphi @ P
        if A is not None:
            J = np.einsum("kj,nkc->njc", A, J)
        return (x[0], J[0]) if single else (x, J)


# ----------------------------------------------------------- integration


def integrate(evaluator, element_ids=None, order=16, integrand=None):
    """Integrate over elements: length (1-D), area (2-D) or volume (3-D).

    The measure element is the Gram-determinant Jacobian
    ``sqrt(det(J @ J.T))`` with ``J = d(coordinates)/d(xi)``, which
    handles every dimension and codimension uniformly (a 2-D surface in
    3-D space, a 1-D curve in 3-D, a 3-D volume).

    Parameters
    ----------
    evaluator : Evaluator (or Field/Mesh) for the coordinate field.
    element_ids : elements to integrate over; default the whole element
        mesh of the evaluator's dimension. Unlike arclength chains, no
        connectivity is required — this mirrors Zinc's FieldMeshIntegral
        (a sum of independent element integrals).
    order : Gauss-Legendre points per xi direction. The default (16) is
        converged; pass ``order="zinc"`` to reproduce Zinc's 4-point
        FieldMeshIntegral to ~1e-12 (slightly short on strongly curved
        elements).
    integrand : optional callable ``f(values) -> (n,) weights`` applied
        at the quadrature points, where ``values`` is the (n,
        n_components) coordinate field evaluation. Default integrates 1,
        i.e. returns the measure. To integrate a *different* field, pass
        an ``integrand`` that evaluates it (or precompute with your own
        Evaluator).

    Returns
    -------
    float — the integral summed over the given elements.
    """
    if isinstance(evaluator, (Field, Mesh)):
        evaluator = Evaluator(evaluator)
    if order == "zinc":
        order = ArclengthTable.ZINC_ORDER
    dim = evaluator.dimension
    gx, gw = np.polynomial.legendre.leggauss(order)
    gx = 0.5 * (gx + 1.0)
    gw = 0.5 * gw
    grids = np.meshgrid(*([gx] * dim), indexing="ij")
    xis = np.stack([g.reshape(-1) for g in grids], axis=1)
    weights = np.ones(1)
    for _ in range(dim):
        weights = np.outer(weights, gw).reshape(-1)
    if element_ids is None:
        element_ids = sorted(evaluator.element_mesh.elements)
    total = 0.0
    for eid in element_ids:
        J = evaluator.evaluate_derivatives(eid, xis)  # (n, dim, ncomp)
        gram = J @ np.swapaxes(J, 1, 2)               # (n, dim, dim)
        measure = np.sqrt(np.abs(np.linalg.det(gram)))
        if integrand is not None:
            measure = measure * np.asarray(
                integrand(evaluator.evaluate(eid, xis)), dtype=float)
        total += float((weights * measure).sum())
    return total


# ------------------------------------------------------------- arclength


def endpoint_keys(evaluator, element_ids):
    """Endpoint connectivity keys for 1-D elements.

    Uses node identity when every element carries its own nodes — exact,
    and robust to geometrically coincident but topologically distinct
    nodes (seams, touching branches). Falls back to quantised endpoint
    coordinates for face-inherited elements, which carry no nodes of
    their own."""
    mesh = evaluator.element_mesh
    elements = [mesh[eid] for eid in element_ids]
    if all(e.nodes and e.nodes[0] is not None and e.nodes[-1] is not None
           for e in elements):
        return {e.identifier: (("n", e.nodes[0]), ("n", e.nodes[-1]))
                for e in elements}
    ends = np.array([[evaluator.evaluate(eid, [0.0]),
                      evaluator.evaluate(eid, [1.0])]
                     for eid in element_ids])
    scale = float(np.abs(ends).max()) or 1.0
    quantum = scale * 1e-8
    keys = np.round(ends / quantum).astype(np.int64)
    return {eid: (tuple(keys[i, 0]), tuple(keys[i, 1]))
            for i, eid in enumerate(element_ids)}


def build_chain(evaluator, element_ids=None):
    """Order 1-D elements into a chain (simple path).

    If ``element_ids`` is None the whole mesh must form a simple path —
    any endpoint shared by more than two elements means branches, and the
    caller must choose the path by passing ``element_ids`` (in order).
    Connectivity is geometric (shared endpoint coordinates), so it works
    for face-inherited line elements with no nodes of their own.
    Returns the ordered element id list and orientation flags (True if
    the element is traversed xi=1 to xi=0), or raises ValueError.
    """
    element_mesh = evaluator.element_mesh
    if element_ids is not None:
        ids = list(element_ids)
    else:
        ids = sorted(element_mesh.elements)
    node_use = {}
    ends = endpoint_keys(evaluator, ids)
    for eid in ids:
        a, b = ends[eid]
        for n in (a, b):
            node_use.setdefault(n, []).append(eid)
    branch_joints = [es for es in node_use.values() if len(es) > 2]
    if branch_joints and element_ids is None:
        examples = sorted(set(e for es in branch_joints for e in es))[:6]
        raise ValueError(
            f"Mesh connectivity is not a simple path: endpoints shared by "
            f"more than two elements (e.g. elements {examples}) — a "
            f"branching structure. Over a branching mesh a total arclength "
            f"silently sums the branches. Pass element_ids= selecting the "
            f"chain you mean.")
    if branch_joints:
        examples = sorted(set(e for es in branch_joints for e in es))[:6]
        raise ValueError(
            f"element_ids do not form a simple path: endpoints shared by "
            f"more than two of the given elements (e.g. {examples})")
    # find endpoints (nodes used once)
    endpoints = [n for n, es in node_use.items() if len(es) == 1]
    if len(endpoints) != 2:
        raise ValueError(
            f"Elements do not form an open chain: {len(endpoints)} "
            f"endpoint nodes found (expected 2)")
    # walk from one endpoint; prefer starting at the first given element
    a0, b0 = ends[ids[0]]
    if a0 in endpoints:
        start = a0
    elif b0 in endpoints:
        start = b0
    else:
        start = min(endpoints)
    ordered = []
    reversed_flags = []
    node = start
    used = set()
    while True:
        candidates = [e for e in node_use.get(node, []) if e not in used]
        if not candidates:
            break
        eid = candidates[0]
        used.add(eid)
        a, b = ends[eid]
        if a == node:
            ordered.append(eid)
            reversed_flags.append(False)
            node = b
        else:
            ordered.append(eid)
            reversed_flags.append(True)
            node = a
    if len(ordered) != len(ids):
        raise ValueError(
            f"Elements do not form a single connected chain: walked "
            f"{len(ordered)} of {len(ids)}")
    return ordered, reversed_flags


class ArclengthTable:
    """Cumulative arclength over a chain of 1-D elements.

    Build with :meth:`build`. ``total`` is the chain length. Because
    element lengths within one scaffold vary by a factor of two or more,
    ``s * total`` is *not* arclength for any normalised parameter ``s`` —
    use :meth:`arclength_at` / :meth:`arclength_at_parameter`, which
    interpolate the cumulative table.
    """

    #: Gauss order used by Zinc's FieldMeshIntegral (bug-compatibility)
    ZINC_ORDER = 4

    def __init__(self, evaluator, element_ids, reversed_flags, order,
                 samples_per_element=32):
        self.evaluator = evaluator
        self.element_ids = list(element_ids)
        self.reversed_flags = list(reversed_flags)
        self.order = order
        n_sub = samples_per_element
        gx, gw = np.polynomial.legendre.leggauss(order)
        gx = 0.5 * (gx + 1.0)  # map to [0,1]
        gw = 0.5 * gw
        # dense cumulative table: per element, arclength at xi grid
        self._xi_grid = np.linspace(0.0, 1.0, n_sub + 1)
        h = 1.0 / n_sub
        # all quadrature points for one element, in one batch
        all_xi = (self._xi_grid[:-1, None] + h * gx[None, :]).reshape(-1)
        self._cum = []          # per element: (n_sub+1,) cumulative from chain start
        self._element_start = []
        total = 0.0
        for eid, rev in zip(self.element_ids, self.reversed_flags):
            self._element_start.append(total)
            xis = (1.0 - all_xi if rev else all_xi)[:, None]
            d = evaluator.evaluate_derivatives(eid, xis)  # (n, 1, ncomp)
            speeds = np.linalg.norm(d[:, 0, :], axis=1)
            seg = (speeds.reshape(n_sub, order) * gw).sum(axis=1) * h
            cum = np.empty(n_sub + 1)
            cum[0] = total
            cum[1:] = total + np.cumsum(seg)
            total = cum[-1]
            self._cum.append(cum)
        self.total = total
        self._parameter_samples = None

    @classmethod
    def build(cls, evaluator, element_ids=None, order=16,
              samples_per_element=32):
        """Build an arclength table over a chain of 1-D elements.

        Parameters
        ----------
        evaluator : Evaluator over a 1-D element mesh (dimension 1).
        element_ids : explicit chain, required when the mesh branches.
        order : Gauss-Legendre order per sub-interval. The default (16)
            is converged; pass ``order="zinc"`` to reproduce Zinc's
            FieldMeshIntegral (4 Gauss points) to ~1e-12. On strongly
            curved elements Zinc's answer is slightly short.
        """
        if isinstance(evaluator, (Field, Mesh)):
            evaluator = Evaluator(evaluator, dimension=1)
        if evaluator.dimension != 1:
            raise ValueError(
                f"Arclength needs a 1-D element mesh; evaluator is over "
                f"mesh{evaluator.dimension}d")
        if order == "zinc":
            order = cls.ZINC_ORDER
        ordered, reversed_flags = build_chain(evaluator, element_ids)
        return cls(evaluator, ordered, reversed_flags, order,
                   samples_per_element)

    # ----------------------------------------------------------- queries

    def _element_index(self, element_id):
        try:
            return self.element_ids.index(element_id)
        except ValueError:
            raise KeyError(
                f"Element {element_id} is not on this chain") from None

    def arclength_at(self, element_id, xi):
        """Arclength from the chain start to (element, xi)."""
        idx = self._element_index(element_id)
        xi = float(np.atleast_1d(xi)[0])
        if self.reversed_flags[idx]:
            xi = 1.0 - xi
        return float(np.interp(xi, self._xi_grid, self._cum[idx]))

    def location_at(self, arclength):
        """Inverse: (element_id, xi) at a given arclength from start.

        Clamps to the chain ends.
        """
        s = float(arclength)
        if s <= 0.0:
            idx = 0
        elif s >= self.total:
            idx = len(self.element_ids) - 1
        else:
            starts = self._element_start
            idx = int(np.searchsorted(starts, s, side="right") - 1)
            idx = max(0, min(idx, len(self.element_ids) - 1))
            while idx + 1 < len(self.element_ids) and s > self._cum[idx][-1]:
                idx += 1
        cum = self._cum[idx]
        xi = float(np.interp(s, cum, self._xi_grid))
        if self.reversed_flags[idx]:
            xi = 1.0 - xi
        return self.element_ids[idx], xi

    def element_lengths(self):
        """Arclength of each chain element, in chain order."""
        return np.array([c[-1] - c[0] for c in self._cum])

    # ------------------------------------------------ material parameter

    def attach_parameter(self, parameter_field, component=0):
        """Sample a (monotonic) scalar parameter field along the chain so
        :meth:`arclength_at_parameter` can convert parameter values (e.g.
        a material coordinate) to arclength by interpolating the table.
        """
        ev = Evaluator(parameter_field, dimension=1)
        params = []
        arcs = []
        for idx, (eid, rev) in enumerate(
                zip(self.element_ids, self.reversed_flags)):
            xi_vals = 1.0 - self._xi_grid if rev else self._xi_grid
            values = ev.evaluate(eid, xi_vals[:, None])[:, component]
            params.extend(values)
            arcs.extend(self._cum[idx])
        params = np.asarray(params)
        arcs = np.asarray(arcs)
        d = np.diff(params)
        span = abs(params[-1] - params[0])
        tol = 1e-9 * max(1.0, float(np.abs(params).max()))
        increasing = np.all(d >= -tol) and params[-1] > params[0] + tol
        decreasing = np.all(d <= tol) and params[0] > params[-1] + tol
        if not (increasing or decreasing) or span <= tol:
            raise ValueError(
                f"Parameter field {parameter_field.name!r} is not strictly "
                f"monotonic along this chain; cannot convert parameter to "
                f"arclength")
        if params[0] > params[-1]:
            params = params[::-1]
            arcs = arcs[::-1]
        self._parameter_samples = (params, arcs)
        return self

    def arclength_at_parameter(self, s):
        """Arclength at parameter value(s) ``s`` (requires
        :meth:`attach_parameter`). This interpolates the cumulative table
        — it is NOT ``s * total``, which is wrong because element lengths
        vary along the chain."""
        if self._parameter_samples is None:
            raise ValueError(
                "Call attach_parameter(field) before arclength_at_parameter")
        params, arcs = self._parameter_samples
        return np.interp(s, params, arcs)
