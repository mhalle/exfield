"""Finite element basis functions.

Reference: ``finite_element_basis.cpp`` in Zinc. Scope is the bases that
appear in SPARC EX scaffolds: tensor products of linear/quadratic/cubic
Lagrange and cubic Hermite, cubic Hermite serendipity (2-D/3-D), and the
constant basis. Simplex and polygon bases raise
:class:`UnsupportedExFeature`.

Representation
--------------
Like the reference implementation, every basis is stored as a *blending
matrix* over tensor-product monomials: ``phi_f(xi) = sum_m B[f, m] *
mono_m(xi)`` where monomial ``m`` has per-xi powers decoded with the xi1
power varying fastest (Zinc's convention). This makes evaluation at many
points a single matrix product, and derivatives are the same product
against differentiated monomials — there is no separate derivative
matrix.

Function ordering (the #1 source of silently-wrong geometry; verified
against ``cubic_hermite_node_derivative_labels`` and the sort in
``CREATE(FE_basis)``):

* functions are **node-major**: all dofs of local node 1, then node 2…
* local nodes are ordered **xi1-fastest** (node = 1 + i1 + n1*(i2 + n2*i3))
* within a node, dofs follow the derivative bitmask with the xi1 bit
  fastest — for tricubic Hermite:
  value, d/ds1, d/ds2, d2/ds1ds2, d/ds3, d2/ds1ds3, d2/ds2ds3, d3/ds1ds2ds3
"""

from functools import reduce

import numpy as np

from .errors import ExSyntaxError, UnsupportedExFeature

# ------------------------------------------------------ 1D coefficients
# Verbatim from Zinc's blending matrices (finite_element_basis.cpp),
# monomial order [1, x, x^2, ...].

_LINEAR_LAGRANGE_1D = [
    (np.array([1.0, -1.0]), 0),            # node 0 value: 1 - x
    (np.array([0.0, 1.0]), 0),             # node 1 value: x
]
_QUADRATIC_LAGRANGE_1D = [
    (np.array([1.0, -3.0, 2.0]), 0),
    (np.array([0.0, 4.0, -4.0]), 0),
    (np.array([0.0, -1.0, 2.0]), 0),
]
_CUBIC_LAGRANGE_1D = [
    (np.array([1.0, -5.5, 9.0, -4.5]), 0),
    (np.array([0.0, 9.0, -22.5, 13.5]), 0),
    (np.array([0.0, -4.5, 18.0, -13.5]), 0),
    (np.array([0.0, 1.0, -4.5, 4.5]), 0),
]
_CUBIC_HERMITE_1D = [
    (np.array([1.0, 0.0, -3.0, 2.0]), 0),  # node 0 value
    (np.array([0.0, 1.0, -2.0, 1.0]), 1),  # node 0 d/dxi
    (np.array([0.0, 0.0, 3.0, -2.0]), 0),  # node 1 value
    (np.array([0.0, 0.0, -1.0, 1.0]), 1),  # node 1 d/dxi
]
_CONSTANT_1D = [(np.array([1.0]), 0)]


class _Family1D:
    """One tensor factor: per-node lists of (monomial coeffs, deriv order)."""

    def __init__(self, name, functions, nodes):
        self.name = name
        self.nodes = nodes
        self.dofs_per_node = len(functions) // nodes
        self.order = len(functions[0][0]) - 1
        # coeffs[node][dof], deriv[node][dof]
        self.coeffs = [[functions[n * self.dofs_per_node + j][0]
                        for j in range(self.dofs_per_node)]
                       for n in range(nodes)]
        self.deriv = [[functions[n * self.dofs_per_node + j][1]
                       for j in range(self.dofs_per_node)]
                      for n in range(nodes)]


_FAMILY_BY_TOKEN = {
    "l.lagrange": _Family1D("l.Lagrange", _LINEAR_LAGRANGE_1D, 2),
    "q.lagrange": _Family1D("q.Lagrange", _QUADRATIC_LAGRANGE_1D, 3),
    "c.lagrange": _Family1D("c.Lagrange", _CUBIC_LAGRANGE_1D, 4),
    "c.hermite": _Family1D("c.Hermite", _CUBIC_HERMITE_1D, 2),
    "constant": _Family1D("constant", _CONSTANT_1D, 1),
}

_UNSUPPORTED_TOKENS = ("simplex", "polygon",
                       "lagrangehermite", "hermitelagrange")


# --------------------------------------------------------- base machinery


class _MonomialBasis:
    """Shared blending-matrix machinery.

    Subclass constructors must set ``description``, ``dimension``,
    ``orders`` (per-xi polynomial degree), ``B`` (n_functions x
    n_monomials), ``function_node`` and ``function_derivatives``.
    """

    description = None
    dimension = None
    orders = None
    B = None
    function_node = None
    function_derivatives = None

    @property
    def number_of_functions(self):
        return self.B.shape[0]

    @property
    def number_of_nodes(self):
        return self.function_node[-1] + 1 if self.function_node else 0

    # ------------------------------------------------------- monomials

    def _monomials(self, xis, derivative):
        """(n_points, n_monomials) monomial (derivative) values, xi1
        power fastest."""
        n = xis.shape[0]
        M = np.ones((n, 1))
        for k in range(self.dimension):
            order = self.orders[k]
            nd = derivative[k]
            V = np.zeros((n, order + 1))
            x = xis[:, k]
            for p in range(nd, order + 1):
                factor = 1.0
                for q in range(p, p - nd, -1):
                    factor *= q
                V[:, p] = factor * x ** (p - nd)
            # combine: new index = p_k * len(M-part) + previous (p1 fastest)
            M = (V[:, :, None] * M[:, None, :]).reshape(n, -1)
        return M

    def _check_xi(self, xi):
        xi = np.asarray(xi, dtype=float)
        single = xi.ndim == 1
        if single:
            if xi.shape != (self.dimension,):
                raise ValueError(
                    f"xi must have length {self.dimension}, got {xi.shape}")
            xi = xi[None, :]
        elif xi.ndim != 2 or xi.shape[1] != self.dimension:
            raise ValueError(
                f"xi must be ({self.dimension},) or (n, {self.dimension}),"
                f" got {xi.shape}")
        return xi, single

    # ------------------------------------------------------ evaluation

    def evaluate(self, xi, derivative=None):
        """Basis function values at ``xi``.

        ``xi`` may be a single point ``(dimension,)`` or a batch
        ``(n, dimension)``; returns ``(number_of_functions,)`` or
        ``(n, number_of_functions)`` correspondingly. ``derivative`` is
        an optional per-xi derivative-order tuple, e.g. ``(1, 0, 0)``.

        Repeated calls with an identical xi batch (e.g. the same
        quadrature or seed grid over many elements) return a memoized
        array — treat the result as read-only.
        """
        xis, single = self._check_xi(xi)
        ndx = derivative if derivative is not None else (0,) * self.dimension
        phi = self._cached("phi", ndx, xis,
                           lambda: self._monomials(xis, ndx) @ self.B.T)
        return phi[0] if single else phi

    def evaluate_derivatives(self, xi):
        """All first derivatives. Single point -> (dimension,
        n_functions); batch -> (n, dimension, n_functions). Memoized per
        identical xi batch like :meth:`evaluate`."""
        xis, single = self._check_xi(xi)

        def compute():
            out = np.empty((xis.shape[0], self.dimension,
                            self.number_of_functions))
            for k in range(self.dimension):
                d = [0] * self.dimension
                d[k] = 1
                out[:, k, :] = self._monomials(xis, tuple(d)) @ self.B.T
            return out

        out = self._cached("dphi", None, xis, compute)
        return out[0] if single else out

    def values_and_derivatives(self, xi):
        """Values and all first derivatives in one pass, sharing the
        per-dimension power tables (used by Newton loops where xi
        changes every iteration and memoization cannot help).

        Single point -> ``(phi (F,), dphi (dim, F))``; batch ->
        ``(phi (n, F), dphi (n, dim, F))``.
        """
        xis, single = self._check_xi(xi)
        n = xis.shape[0]
        # per-dim value (nd=0) and derivative (nd=1) power tables
        V0 = []
        V1 = []
        for k in range(self.dimension):
            order = self.orders[k]
            x = xis[:, k]
            v0 = np.empty((n, order + 1))
            v1 = np.zeros((n, order + 1))
            v0[:, 0] = 1.0
            for p in range(1, order + 1):
                v0[:, p] = v0[:, p - 1] * x
                v1[:, p] = p * v0[:, p - 1]
            V0.append(v0)
            V1.append(v1)

        def combine(tables):
            M = np.ones((n, 1))
            for t in tables:
                M = (t[:, :, None] * M[:, None, :]).reshape(n, -1)
            return M

        phi = combine(V0) @ self.B.T
        dphi = np.empty((n, self.dimension, self.number_of_functions))
        for k in range(self.dimension):
            tables = [V1[j] if j == k else V0[j]
                      for j in range(self.dimension)]
            dphi[:, k, :] = combine(tables) @ self.B.T
        if single:
            return phi[0], dphi[0]
        return phi, dphi

    def _cached(self, kind, ndx, xis, compute):
        """Memoize the last result per (kind, derivative): quadrature
        grids and seed grids are evaluated repeatedly across elements."""
        cache = getattr(self, "_eval_cache", None)
        if cache is None:
            cache = self._eval_cache = {}
        key = (kind, ndx)
        hit = cache.get(key)
        if hit is not None and hit[0].shape == xis.shape \
                and np.array_equal(hit[0], xis):
            return hit[1]
        result = compute()
        cache[key] = (xis.copy(), result)
        return result

    def __repr__(self):
        return f"{type(self).__name__}({self.description!r})"


def _kron_row(coeff_arrays):
    """Flattened tensor product of per-xi coefficient arrays with the
    xi1 power fastest: reduce kron with the LAST xi outermost."""
    return reduce(np.kron, reversed(list(coeff_arrays)))


# ------------------------------------------------------- tensor product


class TensorProductBasis(_MonomialBasis):
    """Tensor product of 1D basis families, one per xi direction."""

    def __init__(self, description, families):
        self.description = description
        self.families = families
        self.dimension = len(families)
        self.orders = tuple(f.order for f in families)
        node_counts = [f.nodes for f in families]
        dof_counts = [f.dofs_per_node for f in families]
        n_nodes = int(np.prod(node_counts))
        dofs_per_node = int(np.prod(dof_counts))
        n_funcs = n_nodes * dofs_per_node
        n_monomials = int(np.prod([o + 1 for o in self.orders]))
        self.B = np.zeros((n_funcs, n_monomials))
        self.function_node = []
        self.function_derivatives = []
        fn = 0
        for node in range(n_nodes):
            ii = self._unravel(node, node_counts)
            for dof in range(dofs_per_node):
                jj = self._unravel(dof, dof_counts)
                self.B[fn] = _kron_row(
                    families[k].coeffs[ii[k]][jj[k]]
                    for k in range(self.dimension))
                self.function_node.append(node)
                self.function_derivatives.append(tuple(
                    families[k].deriv[ii[k]][jj[k]]
                    for k in range(self.dimension)))
                fn += 1

    @staticmethod
    def _unravel(index, counts):
        """xi1-fastest mixed-radix decode."""
        out = []
        for c in counts:
            out.append(index % c)
            index //= c
        return tuple(out)


# ------------------------------------------------- Hermite serendipity


class SerendipityBasis(_MonomialBasis):
    """Cubic Hermite serendipity (2-D: 12 functions, 3-D: 32).

    Not a tensor product — no cross derivatives, that is the point. From
    Zinc's ``finite_element_basis.cpp``: with Hv/Hd the 1D Hermite
    value/derivative pair for an end and L the linear Lagrange pair, for
    node ends (n1, n2[, n3]):

    2-D, per node [VALUE, D_DS1, D_DS2]::

        VALUE = Hv[n1](x) L[n2](y) + L[n1](x) Hv[n2](y) - L[n1](x) L[n2](y)
        D_DS1 = Hd[n1](x) L[n2](y)
        D_DS2 = L[n1](x) Hd[n2](y)

    3-D, per node [VALUE, D_DS1, D_DS2, D_DS3]::

        VALUE = Hv L L + L Hv L + L L Hv - 2 L L L
        D_DSk = Hd in xi_k, L in the others

    Nodes are in xi1-fastest grid order; the blending matrix spans the
    full bicubic/tricubic monomial set (16/64 columns).
    """

    _HV = (np.array([1.0, 0.0, -3.0, 2.0]), np.array([0.0, 0.0, 3.0, -2.0]))
    _HD = (np.array([0.0, 1.0, -2.0, 1.0]), np.array([0.0, 0.0, -1.0, 1.0]))
    _L = (np.array([1.0, -1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0]))

    def __init__(self, description, dimension):
        if dimension not in (2, 3):
            raise UnsupportedExFeature(
                f"Hermite serendipity of dimension {dimension} is not "
                f"supported")
        self.description = description
        self.dimension = dimension
        self.orders = (3,) * dimension
        n_nodes = 2 ** dimension
        dofs = 1 + dimension
        self.B = np.zeros((n_nodes * dofs, 4 ** dimension))
        self.function_node = []
        self.function_derivatives = []
        HV, HD, L = self._HV, self._HD, self._L
        fn = 0
        for node in range(n_nodes):
            ends = [(node >> k) & 1 for k in range(dimension)]
            row = np.zeros(4 ** dimension)
            for k in range(dimension):
                row += _kron_row(HV[ends[j]] if j == k else L[ends[j]]
                                 for j in range(dimension))
            row -= (dimension - 1.0) * _kron_row(
                L[ends[j]] for j in range(dimension))
            self.B[fn] = row
            self.function_node.append(node)
            self.function_derivatives.append((0,) * dimension)
            fn += 1
            for k in range(dimension):
                self.B[fn] = _kron_row(
                    HD[ends[j]] if j == k else L[ends[j]]
                    for j in range(dimension))
                self.function_node.append(node)
                deriv = [0] * dimension
                deriv[k] = 1
                self.function_derivatives.append(tuple(deriv))
                fn += 1


# -------------------------------------------------------------- parsing


_BASIS_CACHE = {}


def parse_basis(description, line=None):
    """Parse an EX basis description like ``c.Hermite*c.Hermite*l.Lagrange``.

    Mirrors ``FE_basis_string_to_type_array``: factors separated by ``*``,
    each naming a 1D family; ``c.HermiteSerendipity(2[;3])`` marks linked
    serendipity dimensions. Unsupported families (simplex, polygon,
    Lagrange-Hermite mixes) raise UnsupportedExFeature.
    """
    key = " ".join(description.split()).lower()
    cached = _BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    tokens = [t.strip() for t in key.split("*")]
    if any("hermiteserendipity" in t for t in tokens):
        if not all("hermiteserendipity" in t for t in tokens):
            raise UnsupportedExFeature(
                f"Hermite serendipity mixed with other basis families "
                f"('{description}') is not supported", line=line)
        basis = SerendipityBasis(description.strip(), len(tokens))
        _BASIS_CACHE[key] = basis
        return basis
    families = []
    for token in tokens:
        base = token.split("(")[0].strip()
        for bad in _UNSUPPORTED_TOKENS:
            if bad in base:
                raise UnsupportedExFeature(
                    f"Basis family '{token}' in '{description}' is not "
                    f"supported (declined by design)", line=line)
        fam = _FAMILY_BY_TOKEN.get(base)
        if fam is None:
            raise ExSyntaxError(
                f"Unknown basis family '{token}' in '{description}'",
                line=line)
        families.append(fam)
    if not families:
        raise ExSyntaxError(f"Empty basis description '{description}'",
                            line=line)
    basis = TensorProductBasis(description.strip(), families)
    _BASIS_CACHE[key] = basis
    return basis


def constant_basis(dimension):
    """The constant basis of the given dimension (single function 1)."""
    return parse_basis("*".join(["constant"] * dimension))
