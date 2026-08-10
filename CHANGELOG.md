# Changelog

exfield is **alpha**: the API and any interchange conventions may
change between 0.x releases. Entries below correspond to the working
rounds recorded in the map-core project log.

## 0.5.0 — 2026-08-10

**Breaking: API naming pass.** Names that had accreted across build
rounds were reworked as one coherent set. Done now, deliberately,
while the library is alpha and every consumer is in-repo — all three
pilots needed **zero** changes (none of them called a renamed API),
and no deprecation aliases are shipped: this is a clean break
documented by the table below.

| old | new | why |
|---|---|---|
| `Evaluator.evaluate_with_derivatives` | `Evaluator.value_and_jacobian` | says what it *returns* (both), instead of differing from `evaluate_derivatives` by the word "with"; follows the `value_and_grad` idiom |
| `Evaluator.evaluate_many` | *deleted* | was an alias for batch `evaluate`; its existence implied `evaluate` was single-point-only, which is false — **all** evaluation methods take one xi or a batch |
| `TensorProductBasis.evaluate_with_derivatives` | `.values_and_derivatives` | basis functions have values and derivatives, not a Jacobian |
| `Mesh.mesh(dimension)` | `Mesh.element_mesh(dimension)` | `mesh.mesh(3)` was opaque; now matches the `mesh.element_meshes` dict it reads from |
| `closest_point(...)` | `find_locations(...)` | it returns a list of `Location`, not points; plural pairs with `find_location` |
| `Location.boundary` | `Location.at_boundary` | it is a bool; reads as an adjective now, like `ambiguous` |
| `EmbeddedPoints.metadata["boundary"]` | `metadata["at_boundary"]` | key follows the attribute |
| `EmbeddedPoints.arclength(table)` | `.chain_arclengths(table)` | plural (returns an array) and says *which* measure: position along a chain |
| `HostedPath.world_arclengths(ev)` | `.polyline_arclengths(ev)` | the method's central caveat — it measures the polyline through the addresses, not a curve — now lives in the name |
| `dump(model, path)`, `dumps(model)` | `dump(mesh, ...)`, `dumps(mesh)` | every other public function calls it `mesh` |

Unchanged and worth stating, since they were the source of the
confusion: `evaluate`, `evaluate_derivatives` and `value_and_jacobian`
differ **only in what they return** — values, derivatives, or both in
one pass. Every one of them accepts a single xi or an `(n, dimension)`
batch.

## 0.4.0 — 2026-08-10

First release packaged as a standalone unit (Apache-2.0 + NOTICE;
previously carried MPL-2.0 metadata matching upstream Zinc).

Added
- `HostedPath` — ordered `(element, xi)` material-address paths in a
  host scaffold, for proxy structures whose world geometry is derived
  by evaluating the host (lymphatic chains). Polyline-only semantics,
  `host_group`, `world_arclengths()`, fingerprint + `max_residual`
  guards.

Fixed (two adversarial review rounds; 186 tests)
- `export_vtu` deduplication uses **per-component quanta**: scaled
  geometry and unscaled attached-field values no longer share one
  quantum (which merged real field seams at large `scale`, overflowed
  the int64 key at small `scale`, and let `dedup=False` merge points).
  `scale=0` is refused.
- `find_location` polishes candidate elements whose bounding box is at
  exactly the best distance instead of pruning them: exact ties at
  shared endpoints are flagged `ambiguous` via a genuinely polished
  runner-up, while merely-touching boxes report their true distance
  instead of a phantom 0.0.
- `Location.ambiguous` uses `<=` so exact ties are ambiguous.
- `HostedPath.from_world` passes positional arguments through to the
  parent (`host_group` is keyword-only) — it previously swallowed
  `element_ids`, silently widening the search.
- Reader rejects field redeclarations whose coordinate system, focus,
  element:xi host mesh, or component names conflict — while treating
  an omitted coordinate system as equal to explicit rectangular
  cartesian (the EX default).
- `EmbeddedPoints` validates collection lengths at construction and
  re-validates before evaluation (post-construction mutation cannot
  return uninitialised memory); `from_world` refuses empty input.

Known gaps (stated, not hidden)
- Fingerprints are **not serialized** into `.exf`: a loaded mesh has
  `fingerprint = None`, and two `None`s pass the comparability check
  without checking. Cross-mesh address safety currently requires
  stamping fingerprints in code.
- The 59-file Zinc corpus regression requires a local Zinc source
  checkout; without it the corpus tests skip (loudly).

## 0.3.0 — 2026-08-08/09

- uv-managed workflow (lockfile, dev group, ruff clean); `uv build`
  produces wheel + sdist.
- VTK bridge: `export_vtu` writes exact Bezier cells (anisotropic
  degrees, face-inherited elements exact via parent-lattice slicing)
  or tessellated linear cells, with attached fields exact under VTK
  probing; `export_markers_vtu` for embedded landmarks. NumPy-only —
  no vtk import at runtime; ordering pinned against VTK 9.6.2 tables.
- Inverse mapping rebuilt as branch-and-bound over rigorous Bernstein
  control-net AABBs with batched multi-start Gauss-Newton.
- `integrate()` (length/area/volume, golden-matched to Zinc),
  `to_rectangular_cartesian()`, template fingerprints, benchmarks.

## 0.2.0 — 2026-08-08

Initial working library: EX version 2/3 reader + writer mirroring
Zinc's productions, tensor-product + Hermite-serendipity bases,
evaluation with face inheritance, `ArclengthTable`, `EmbeddedPoints`,
silent-wrong-geometry guards. Golden-validated against Zinc 4.2.1 on
SPARC dataset 426 (samples < 1e-13; bit-identical round-trip through
Zinc). Out of scope by design: EX v1, simplex/polygon shapes, grid
fields, time sequences, multi-region files (declined loudly, never
half-read).

## 0.1.0

Referenced by the prototype spec; never recovered. This lineage
restarted at 0.2.0.
