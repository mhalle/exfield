"""Embedded point sets: material ``(element, xi)`` addresses that
survive refitting.

A point stored as ``(element, xi)`` keeps its anatomical meaning when
the scaffold geometry is refitted, because the address is in material
space. The projection residual (world distance from the original point
to the mesh) is retained in ``metadata['residual']`` — a landmark
landing 4 mm off the centreline is a different object from one landing
on it — and ``from_world`` enforces a ``max_residual`` by default so the
check is opt-out, not opt-in.

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
        self.names = list(names) if names is not None else [None] * n
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
        if max_residual is None:
            raise ValueError(
                "from_world requires max_residual: the projection residual "
                "is the only guard against a landmark landing far off the "
                "mesh. Pass a distance in the mesh's units, or np.inf to "
                "accept any residual.")
        locations = [find_location(evaluator, p, element_ids=element_ids,
                                   **kwargs) for p in points]
        residuals = np.array([loc.residual for loc in locations])
        boundary = np.array([loc.boundary for loc in locations])
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
        obj.metadata["boundary"] = boundary
        return obj

    def to_world(self, evaluator):
        """Evaluate the addresses on (possibly refitted) geometry."""
        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(evaluator)
        self._check_fingerprint(evaluator.model.fingerprint)
        return np.array([evaluator.evaluate(eid, xi)
                         for eid, xi in zip(self.element_ids, self.xis)])

    def arclength(self, table):
        """Arclength along a chain for each point.

        Points whose element is not on the chain correctly return NaN (a
        branch landmark has no position along the trunk). Returns an
        :class:`ArclengthValues` named tuple ``(values, nan_count)`` —
        check the count, because a downstream ``nanmean`` silently drops
        them.
        """
        if not isinstance(table, ArclengthTable):
            raise TypeError("arclength needs an ArclengthTable")
        values = np.empty(len(self))
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
                f"EmbeddedPoints were created on a mesh with a different "
                f"template fingerprint: {e}") from None

    def __repr__(self):
        return f"EmbeddedPoints({len(self)} points)"
