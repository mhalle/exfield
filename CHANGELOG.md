# Changelog

exfield is **alpha**: the API and any interchange conventions may
change between 0.x releases. Entries below correspond to the working
rounds recorded in the map-core project log.

## 0.5.3 — 2026-08-13

Fixed
- **Quoted `Group name:` declarations kept their quotes.** `m.groups`
  was keyed `'"Left common carotid artery"'`, so callers reading the
  Auckland whole-body scaffolds needed
  `name if name in m.groups else '"%s"' % name` at every lookup — and
  the same corpus is inconsistent, since `arteries.exf` and `veins.exf`
  quote every name while `nerve_centerlines.exf` quotes none.

  This is a **deliberate divergence from Zinc**, which reads the token
  as rest-of-line and uses it verbatim; exfield was previously
  bug-compatible. The evidence that the quotes are syntax rather than
  content: all 955 group names in `arteries.exf` are quoted, including
  single words like `"systemic"` that need no quoting under any
  escaping convention. Only a *complete* quoted token is stripped,
  using the same escape rules as every other EX string, so an unquoted
  name containing spaces — which Zinc's writer emits verbatim — is
  untouched.

  Group keys therefore change for files that quote them. Defensive
  lookups of the form above keep working (they try the unquoted name
  first); a hardcoded `'"systemic"'` does not.

  The writer still emits names verbatim and is **not** changed to
  re-quote: Zinc would read the quotes back into the name, and Zinc
  reading exfield's output identically is a validated property.
  Round-trip stays stable regardless, since an unquoted rest-of-line
  preserves everything but surrounding whitespace. Writer output for
  the vagus and cube scaffolds is byte-identical to 0.5.2.

  Other name tokens were checked and need no change: node and element
  template names and string field values already go through the
  quote-aware reader, and nodeset, mesh and host-mesh names already go
  through `make_valid_token` on the way out.

## 0.5.2 — 2026-08-12

Bug-fix release. Two behaviour changes worth calling out before the
detail: **`exreader.load` is removed** (never exported, never
documented, imported by nothing — it shadowed `exfield.load` while
dropping gzip support), and two exception contracts tightened —
`export_vtu` now raises `ValueError` for partially-covered
`extra_fields` (was a mid-write `EvaluationError`) and for
>3-component geometry (was a silently corrupt file).

**Three bugs found by review, all in inputs the corpus never reaches.**
Every test fixture is 3-component, and every EFT in the vagus scaffold
references a contiguous prefix of its element's nodes with one basis per
field — so the suite passed 127/127 with all three live. The shape of
the gap was a corpus of one scaffold family standing in for the format's
full range, not weak testing.

* **`export_vtu` silently corrupted non-3-component geometry.** A `.vtu`
  `Points` array is always 3-component; a 2-D scaffold's 2-component
  coordinates were written into it unchanged, so VTK read consecutive
  (x, y) pairs as (x, y, z) triples — a third of the points gone, the
  rest rotated onto wrong axes, no error raised. Missing components are
  now padded with zeros, and more than three is refused.
* **The writer emitted a per-component `#Nodes` its own reader
  rejected.** It wrote the count of *distinct* local nodes an EFT
  references, but `Term.local_node` indexes the element's `Nodes:` list
  directly, so an EFT using a non-prefix subset (local nodes 3 and 4 of
  four) round-tripped to `#Nodes=2` beside labels for node 4 and failed
  re-reading with "Too many nodes referenced". Now the highest index
  referenced. Output for the vagus and cube scaffolds is byte-identical,
  so the Zinc golden validation is unaffected.
* **The mixed-basis guard was unreachable in the failing direction.** It
  ran after the dof matrix was filled, and that matrix is sized from
  component 0's basis — so a later component with *more* functions
  overran it and surfaced as a bare `IndexError` instead of the typed
  message. The check now runs before assembly.

Smaller fixes in the same pass:

* `export_vtu` aborted mid-write when an `extra_fields` entry was
  defined on only some elements — ordinary in EX, where templates vary
  element to element — surfacing the resolver's bare "not defined on
  element N". VTK point data is one array over the whole grid and has
  no representation for partial coverage, so this is now checked up
  front and refused with a message naming the field, the uncovered
  elements, and the way forward (`element_ids=`).
* `exreader.load` is gone. It shadowed `exfield.load` while silently
  dropping gzip support; nothing imported it.
* Dropped a stray `uv sync` code fence sitting above the README title.

Also: seven public docstrings cited `EXFIELD_GOTCHAS.md` and
`EXFIELD_PORTING_SPEC.md`, which stayed behind in map-core when exfield
was extracted — `help(exfield.Mesh)` sent readers to files that do not
exist. Retargeted to README sections, with a test that pins it.

## 0.5.1 — 2026-08-11

Fixed
- `export_vtu`: the point pool copies incoming coordinates instead of
  aliasing them. The tessellation path passes a row view of a whole
  element lattice, so the pool held references into the caller's array
  — exposing pooled points to later mutation and keeping every element
  lattice alive for the length of the export, including rows dedup had
  discarded.

Tests
- A vtk-free check that every lattice slot of a tricubic hex holds the
  control point the ordering specifies. The ordering itself was pinned
  against VTK's tables, and the readback tests pinned the file VTK
  reads — but those skip without the `vtk` extra, so mis-wiring the
  writer's *use* of the ordering left a green default run.
- A lone-hex readback oracle: in the two-hex block every cell has a
  pooled shared face, which can mask an ordering that is
  self-consistent across that face rather than correct per cell.

Repo
- `ruff check` is clean across `src/` **and** `tests/` (an unused
  import had survived since the initial commit because only `src/` was
  being linted).

## 0.5.0 — 2026-08-10

**Breaking: API naming pass.** Names that had accreted across build
rounds were reworked as one coherent set. Done now, deliberately,
while the library is alpha and every consumer is in-repo — all three
pilots needed **zero** changes (none of them called a renamed API),
and no deprecation aliases are shipped: this is a clean break
documented by the table below.

| old | new | why |
|---|---|---|
| `Evaluator.evaluate_with_derivatives` | `Evaluator.evaluate_values_and_derivatives` | says what it *returns* (both) instead of differing from `evaluate_derivatives` by the word "with", and stays parallel with the rest of the family (verb-first, plural, same vocabulary) |
| `Evaluator.evaluate_many` | *deleted* | was an alias for batch `evaluate`; its existence implied `evaluate` was single-point-only, which is false — **all** evaluation methods take one xi or a batch |
| `TensorProductBasis.evaluate_with_derivatives` | `.evaluate_values_and_derivatives` | same name at both layers |
| `Mesh.mesh(dimension)` | `Mesh.element_mesh(dimension)` | `mesh.mesh(3)` was opaque; now matches the `mesh.element_meshes` dict it reads from |
| `closest_point(...)` | `find_locations(...)` | it returns a list of `Location`, not points; plural pairs with `find_location` |
| `Location.boundary` | `Location.at_boundary` | it is a bool; reads as an adjective now, like `ambiguous` |
| `EmbeddedPoints.metadata["boundary"]` | `metadata["at_boundary"]` | key follows the attribute |
| `EmbeddedPoints.arclength(table)` | `.chain_arclengths(table)` | plural (returns an array) and says *which* measure: position along a chain |
| `HostedPath.world_arclengths(ev)` | `.polyline_arclengths(ev)` | the method's central caveat — it measures the polyline through the addresses, not a curve — now lives in the name |
| `dump(model, path)`, `dumps(model)` | `dump(mesh, ...)`, `dumps(mesh)` | every other public function calls it `mesh` |

Unchanged and worth stating, since they were the source of the
confusion: `evaluate`, `evaluate_derivatives` and
`evaluate_values_and_derivatives`
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
