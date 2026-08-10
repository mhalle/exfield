"""Embedded point sets: material ``(element, xi)`` addresses that
survive refitting.

A point stored as ``(element, xi)`` keeps its anatomical meaning when
the scaffold geometry is refitted, because the address is in material
space. The projection residual (world distance from the original point
to the mesh) is retained in ``metadata['residual']`` — a landmark
landing 4 mm off the centreline is a different object from one landing
on it — and ``from_world`` enforces a ``max_residual`` by default so the
check is opt-out, not opt-in.

:class:`HostedPath` adds order to the same machinery, for proxy
structures whose world geometry is derived by evaluating a host rather
than measured.

Comparing addresses across meshes is only meaningful when both were
built from the same template with the same discretisation options — see
``exfield.fingerprint``.
"""

from typing import NamedTuple

import numpy as np

from .errors import FingerprintMismatch
from .evaluate import ArclengthTable, Evaluator
from .fingerprint import check_fingerprints
from .inverse import find_location


class ArclengthValues(NamedTuple):
    """Arclengths for an embedded point set. Unpacks as
    ``(values, nan_count)``; off-chain points are NaN and ``nan_count``
    says how many, because a downstream ``nanmean`` drops them
    silently."""
    values: np.ndarray
    nan_count: int


class EmbeddedPoints:
    """A set of named points addressed by ``(element, xi)``."""

    def __init__(self, element_ids, xis, names=None, metadata=None,
                 fingerprint=None):
        self.element_ids = list(element_ids)
        self.xis = [np.atleast_1d(np.asarray(x, dtype=float)) for x in xis]
        n = len(self.element_ids)
        if len(self.xis) != n:
            raise ValueError(
                f"element_ids and xis disagree in length ({n} vs "
                f"{len(self.xis)}); later zips would silently drop points")
        self.names = list(names) if names is not None else [None] * n
        if len(self.names) != n:
            raise ValueError(
                f"names length {len(self.names)} != {n} points")
        self.metadata = metadata if metadata is not None else {}
        self.fingerprint = fingerprint

    def __len__(self):
        return len(self.element_ids)

    def __iter__(self):
        return iter(zip(self.names, self.element_ids, self.xis))

    @classmethod
    def from_world(cls, evaluator, points, element_ids=None, names=None,
                   max_residual=None, **kwargs):
        """Project world points onto the mesh.

        Parameters
        ----------
        max_residual : raise ValueError if any projection lands further
            than this from its source point. Pass ``np.inf`` explicitly
            to opt out — the residual is the whole safety net.
        """
        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(evaluator)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.size == 0:
            raise ValueError(
                f"{cls.__name__}.from_world needs at least one point")
        if max_residual is None:
            raise ValueError(
                "from_world requires max_residual: the projection residual "
                "is the only guard against a landmark landing far off the "
                "mesh. Pass a distance in the mesh's units, or np.inf to "
                "accept any residual.")
        locations = [find_location(evaluator, p, element_ids=element_ids,
                                   **kwargs) for p in points]
        residuals = np.array([loc.residual for loc in locations])
        boundary = np.array([loc.at_boundary for loc in locations])
        bad = residuals > max_residual
        if np.any(bad):
            worst = float(residuals.max())
            raise ValueError(
                f"{int(bad.sum())} of {len(points)} points project with "
                f"residual > {max_residual} (worst {worst:.6g}). These "
                f"landmarks are not on the mesh; raise max_residual only "
                f"if that is expected.")
        obj = cls([loc.element_id for loc in locations],
                  [loc.xi for loc in locations], names=names,
                  fingerprint=evaluator.model.fingerprint)
        obj.metadata["residual"] = residuals
        obj.metadata["at_boundary"] = boundary
        return obj

    def _check_lengths(self):
        """Re-validate collection lengths — they are public lists, and a
        post-construction mutation would make zips silently truncate."""
        n = len(self.element_ids)
        if len(self.xis) != n or len(self.names) != n:
            raise ValueError(
                f"{type(self).__name__} collections were mutated to "
                f"unequal lengths (element_ids {n}, xis {len(self.xis)}, "
                f"names {len(self.names)})")

    def to_world(self, evaluator):
        """Evaluate the addresses on (possibly refitted) geometry."""
        self._check_lengths()
        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(evaluator)
        self._check_fingerprint(evaluator.model.fingerprint)
        return np.array([evaluator.evaluate(eid, xi)
                         for eid, xi in zip(self.element_ids, self.xis)])

    def chain_arclengths(self, table):
        """Arclength along a chain for each point.

        Points whose element is not on the chain correctly return NaN (a
        branch landmark has no position along the trunk). Returns an
        :class:`ArclengthValues` named tuple ``(values, nan_count)`` —
        check the count, because a downstream ``nanmean`` silently drops
        them.
        """
        if not isinstance(table, ArclengthTable):
            raise TypeError("arclength needs an ArclengthTable")
        self._check_lengths()
        values = np.full(len(self), np.nan)
        nan_count = 0
        chain = set(table.element_ids)
        for i, (eid, xi) in enumerate(zip(self.element_ids, self.xis)):
            if eid in chain:
                values[i] = table.arclength_at(eid, xi)
            else:
                values[i] = np.nan
                nan_count += 1
        return ArclengthValues(values, nan_count)

    def _check_fingerprint(self, other):
        try:
            check_fingerprints(self.fingerprint, other)
        except FingerprintMismatch as e:
            raise FingerprintMismatch(
                f"{type(self).__name__} was created on a mesh with a "
                f"different template fingerprint: {e}") from None

    def __repr__(self):
        return f"EmbeddedPoints({len(self)} points)"


class HostedPath(EmbeddedPoints):
    """An *ordered* chain of material addresses in a host scaffold.

    Where :class:`EmbeddedPoints` is an unordered set of landmarks, a
    ``HostedPath`` is a path: the address order is the path order, and
    ``to_world`` returns the vertices in that order. It exists for proxy
    structures — a lymphatic chain running alongside the aorta, say —
    whose geometry is not measured but *derived*: the control addresses
    are anchored in the host's material coordinates, so refitting the
    host to a subject moves the path for free. Nothing about the path
    itself is ever fitted, which is why it stores addresses and not
    coordinates.

    What the path does **not** claim: interpolation between consecutive
    addresses is undefined, because two addresses may sit in different
    host elements and no element's basis spans them. The path is a
    polyline over its addresses and nothing more — its resolution *is*
    the number of addresses, and no cross-element smoothness or
    continuity of tangent is asserted. Callers wanting a smooth curve
    must supply enough addresses; they will not get one for free.

    Parameters
    ----------
    host_group : name of the host group or chain the path is anchored
        to. Carried because an address is only interpretable against the
        structure it was authored on — ``(element 12, xi 0.5)`` means
        nothing without knowing which host numbered that element.
    """

    def __init__(self, element_ids, xis, names=None, metadata=None,
                 fingerprint=None, host_group=None):
        super().__init__(element_ids, xis, names=names, metadata=metadata,
                         fingerprint=fingerprint)
        if len(self.element_ids) == 0:
            raise ValueError(
                "HostedPath requires at least one address: an empty path "
                "has no geometry to derive, and would evaluate to an "
                "empty polyline rather than raising downstream.")
        self.host_group = host_group

    @classmethod
    def from_world(cls, evaluator, points, *args, host_group=None,
                   **kwargs):
        """Project an ordered run of world points onto the host.

        Input order is path order — this does not sort, resample or
        otherwise second-guess the caller's sequence. ``max_residual``
        stays mandatory, per :meth:`EmbeddedPoints.from_world`.
        ``host_group`` is keyword-only: positional arguments pass
        through to the parent unchanged, so ``from_world(ev, pts,
        element_ids)`` restricts the search exactly as it does on
        :class:`EmbeddedPoints` — a signature that silently reassigned
        the third positional to ``host_group`` widened the search to
        the whole mesh with no error.
        """
        obj = super().from_world(evaluator, points, *args, **kwargs)
        obj.host_group = host_group
        return obj

    def polyline_arclengths(self, evaluator):
        """Cumulative polyline arclength at each address, starting 0.0.

        This is the length of the *polyline through the addresses*, not
        the arclength of any host curve between them: it is a lower
        bound on a curved path's true length, and it changes when
        addresses are added. Returns an array of length ``len(self)``;
        a one-address path returns ``[0.0]``. The fingerprint guard
        fires here too, since the geometry comes from ``to_world``.
        """
        xyz = self.to_world(evaluator)
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def __repr__(self):
        return (f"HostedPath({len(self)} addresses, "
                f"host_group={self.host_group!r})")
