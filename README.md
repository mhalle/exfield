```bash
uv sync
```

# exfield

Read, evaluate and write OpenCMISS-Zinc **EX/EXF finite element
scaffolds** with **NumPy as the only runtime dependency**.

**Status: alpha (0.5.x).** The library is golden-validated against
Zinc 4.2.1 and battle-tested by the map-core pilots, but the API and
any interchange conventions may still change between 0.x releases.
Known gap stated up front: fingerprints are not serialized through
`.exf` files, so cross-mesh address guards protect only meshes whose
fingerprints are stamped in code (see the hazards section).

## Installing

Not published to PyPI (the name is taken, and this experimental
project doesn't claim one). Install from GitHub:

```bash
uv add git+https://github.com/mhalle/exfield
```

(or `pip install git+https://github.com/mhalle/exfield`), from a
checkout (`uv add --editable path/to/exfield`), or build a wheel
(`uv build` → `dist/exfield-*.whl`). Runtime dependency: NumPy only.
Python ≥ 3.10 (suite passes on 3.10, 3.12 and 3.14).

Note: the standalone repo runs 127 of the 186 tests — the 59-file
Zinc corpus regression, the 42-scaffold SPARC-426 sweep and the
live-VTK oracles need sibling checkouts/caches and skip loudly
without them.

Zinc *makes* scaffolds; exfield lets downstream NumPy/VTK code consume
them without inheriting a 63 MB OpenGL-linked binary. Deliberately out
of scope: fitting (use [`scaffoldfitter`](https://pypi.org/project/scaffoldfitter/)),
mesh generation (use [`scaffoldmaker`](https://pypi.org/project/scaffoldmaker/)),
rendering, solvers and time-varying fields.

**Coming from Zinc?** [ZINC.md](ZINC.md) is the translation guide:
concept map, side-by-side snippets for every common task, the traps
that specifically catch Zinc users — and, for Zinc authors, what
mirrors which production, how the golden validation works, and where
exfield deliberately differs.

```python
import exfield

mesh = exfield.load("f001-left_vagus_scaffold.exf")
print(mesh.summary())
assert not mesh.skipped              # anything here is silently missing

ev = mesh.evaluator("coordinates", dimension=1)
x = ev.evaluate(element_id=1155, xi=[0.37])        # world coordinates
xs = ev.evaluate(1155, [[0.1], [0.5], [0.9]])      # batch: (3, 3)
dx = ev.evaluate_derivatives(1155, [0.37])         # d/dxi

trunk = sorted(mesh.groups["orientation anterior"].element_ids(1))
table = exfield.ArclengthTable.build(ev, element_ids=trunk)
print(table.total)                                  # 491.9 mm (in µm)

surface = mesh.evaluator("coordinates", dimension=2)
area = exfield.integrate(                           # length/area/volume
    surface, element_ids=mesh.groups["epineurium"].element_ids(2))

loc = exfield.find_location(ev, x, element_ids=trunk)
points = exfield.EmbeddedPoints.from_world(
    ev, [x], element_ids=trunk, max_residual=50.0)
arcs = points.chain_arclengths(table)               # .values, .nan_count

path = exfield.HostedPath(                          # ordered proxy path
    element_ids=[1155, 1156], xis=[[0.2], [0.8]],   # anchored in host
    host_group="orientation anterior",              # material coords
    fingerprint=mesh.fingerprint)
xyz = path.to_world(ev)                             # derived, never fitted
cum = path.polyline_arclengths(ev)                  # from 0.0

exfield.dump(mesh, "rewritten.exf")                # EX Version 3

exfield.export_vtu(mesh, "vagus.vtu", scale=1e-3,  # µm -> mm
                   extra_fields=["vagus coordinates"])
```

## VTK export

``export_vtu`` writes VTK ``.vtu`` files consumable by ParaView, PyVista,
Slicer and any VTK pipeline — with **no VTK dependency**: the format is
XML + base64, written directly.

* **Default: exact Bezier cells.** Elements become
  ``vtkBezierCurve``/``Quadrilateral``/``Hexahedron`` cells whose
  control points are the exact Bernstein coefficients of the geometry
  (anisotropic degrees supported, e.g. the vagus 4x2x2 boxes). No
  resolution parameter, no cracks; VTK reads the files back and
  evaluates identical geometry (<1e-9, tested). Face-inherited 1-D/2-D
  elements export exactly by slicing the parent's control lattice.
* **Fields ride along exactly.** ``extra_fields`` sharing the
  geometry's basis are attached as point data holding their own
  Bernstein control values — VTK interpolates point data with the
  cell's shape functions, so probing e.g. the material coordinate
  downstream is exact, not sampled (tested against VTK).
* **``tessellate=N``** switches to linear line/quad/hex cells for
  consumers without higher-order support (some Slicer filters, vtk.js);
  shared lattice points make surfaces watertight.
* Annotation groups become 0/1 cell arrays; ``element_id`` cell data
  maps every cell back to its ``(element, xi)`` address; ``scale``
  converts units (µm -> mm for Slicer); ``export_markers_vtu`` writes
  embedded landmarks as vertex cells.

The VTK point-ordering convention (the silently-warped-geometry trap)
is transcribed from ``PointIndexFromIJK`` and pinned against tables
dumped from VTK 9.6.2; the oracle tests run via
``uv run --with vtk pytest tests/test_vtu.py``.

Evaluation follows Zinc's own architecture — a blending matrix over
tensor-product monomials — so batch evaluation is one matrix product,
and basis matrices are memoized for repeated grids (quadrature, seeds).

## Performance vs Zinc

Measured on the f001 vagus scaffold (515 KB; Apple Silicon; Zinc 4.2.1
through its Python bindings; `tests/bench_zinc.py` /
`tests/bench_exfield.py`):

| Operation | Zinc | exfield |
|---|---|---|
| parse | 17 ms | 63 ms |
| 20,000 field evaluations (batched) | 8.5 ms* | 4.2 ms |
| 20,000 field evaluations (one at a time) | 8.5 ms | 370 ms |
| 5,000 x 3 derivative evaluations (batched) | 4.9 ms | 7.4 ms |
| trunk arclength, Gauss-4 (50 elements) | 0.11 ms | 0.65 ms |
| epineurium area (831 elements) | 0.96 ms | 3.6 ms |
| whole-mesh volume (88 elements) | 0.62 ms | 2.3 ms |
| inverse-map 200 points onto mesh3d | 35 ms | 244 ms |
| write | 6.2 ms | 13 ms |

*Zinc has no batch API; a fieldcache loop is its only mode. The moral:
**batch your xi points** — vectorised exfield beats Zinc's per-call
bindings, while per-point Python loops pay ~15 µs/call overhead.

`find_location` uses branch-and-bound over per-element bounding boxes
derived from Bezier control points (``exfield.bounds``) — rigorous by
the convex-hull property, so pruning is exact — with all Newton starts
of an element iterated as one batch. Zinc's equivalent
(``FeMeshFieldRanges`` + shrinking-radius pruning + warm start) remains
~8x faster in compiled C++; the remaining gap is per-query Python
overhead, and restricting ``element_ids`` shrinks it further.

## Validation

Golden-tested against **Zinc 4.2.1** on a real fitted human vagus
scaffold (SPARC dataset 426, sub-f001 left — tricubic Hermite serendipity
surfaces, `c.Hermite*l.Lagrange*l.Lagrange` volumes, scale factors,
face-inherited line elements, 55 annotation groups):

* field values agree to < 1e-15 relative, derivatives < 1e-13, at 1,476
  sampled `(element, xi)` locations over all three mesh dimensions;
* all 157 per-group arclength, surface-area and volume integrals agree
  with Zinc's `FieldMeshIntegral` to < 1e-13 relative;
* a file written by exfield and re-read **by Zinc** evaluates
  bit-identically to the original at all 29,520 golden samples;
* all 42 scaffolds in dataset 426 (21 subjects x L/R plus the sub-M000
  template) load, chain and measure (`tests/sweep_426.py`); the fitted
  trunks span 429-725 mm and sub-M000 is flagged as the unit-scale
  outlier (it ships in mm where the fitted scaffolds use µm).

Regenerate the reference data (the only step that needs Zinc; uv builds
the ephemeral Zinc environment on the fly — no Python 3.13+ wheel exists
for cmlibs.zinc, hence the pinned interpreter):

```bash
uv run --no-project --python 3.12 --with cmlibs.zinc --with numpy python tests/golden_zinc.py scaffold.exf golden.json
```

## Scope

Supported: EX versions 2 and 3; tensor products of
linear/quadratic/cubic Lagrange and cubic Hermite bases; cubic Hermite
serendipity (2-D and 3-D); scale factors including sums of scaled terms;
per-derivative versions (branch nodes); groups; markers/datapoints;
`element_xi` embedded locations; constant fields; face inheritance
(fields evaluated on `Faces:` elements through the parent's
`face_to_element` map, with Zinc's cyclic axis convention).

Declined with `UnsupportedExFeature` **at the point of encounter, so a
file never loads half-read**: EX version 1 (legacy unversioned files),
multiple regions, time sequences, simplex and polygon shapes,
element-based (grid) field values, indexed fields. An
`UnsupportedExFeature` is a scope boundary, not necessarily a bug. If
you extend scope, extend this list rather than silently accepting.

## The API is shaped around silent-wrong-geometry hazards

This is code where everything runs and the geometry is quietly wrong;
basis ordering, wrong coordinate field, wrong element subset and
under-converged quadrature none raise on their own. So:

* **`ArclengthTable.build` refuses a branching mesh** unless you pass
  `element_ids`. Integrating the `left vagus nerve` 1-D group "works"
  and returns thousands of mm because it includes circumferential rings;
  the trunk is the 50-element `orientation anterior` chain at 491.9 mm.
* **`mesh.coordinates` raises when ambiguous.** Real scaffolds carry
  `coordinates`, `straight coordinates` and `vagus coordinates`; all are
  coordinate-type and 3-component, and evaluating the material field
  produces dimensionless nonsense. `Evaluator` also warns when every
  parameter lies in [-1, 1].
* **`find_location` requires `element_ids` on branching 1-D meshes** — a
  branch running alongside the trunk is frequently nearer than the
  trunk. Results carry an explicit `boundary` flag (xi is clipped, so a
  point past the end of a chain projects to the endpoint with a
  plausible-looking address) and a `runner_up_residual` for ambiguity.
* **`EmbeddedPoints.from_world` requires `max_residual`** (pass
  `np.inf` to opt out) and `arclength()` returns the NaN count next to
  the values, because `nanmean` drops off-chain points silently.
* **`HostedPath` claims polyline semantics only.** An ordered chain of
  material addresses in a host scaffold (proxy structures — lymph
  chains — whose world geometry is *derived* by evaluating the host,
  never fitted). Interpolation between consecutive addresses is
  undefined across elements: resolution *is* the number of addresses,
  and `polyline_arclengths` is a lower bound that changes when addresses
  are added. Empty paths are refused. The fingerprint guard runs on
  every evaluation but only *bites* when fingerprints are present —
  EX files carry none, so a loaded mesh has `fingerprint = None` and
  two `None`s pass without checking. Protection requires stamping
  fingerprints in code (as `stitch3.build_mesh` does); serializing
  them through `.exf` is an open gap.
* **`table.arclength_at(...)` / `arclength_at_parameter(...)`
  interpolate the cumulative table.** `s * total` is not arclength:
  element lengths within one scaffold vary by over 2x.
* **Quadrature default is converged (`order=16`); `order="zinc"` is a
  named bug-compatibility mode** reproducing Zinc's 4-point
  `FieldMeshIntegral` to ~1e-12. On strongly curved elements Zinc runs
  slightly short; agreement with Zinc is not the same as correctness.
* **Template fingerprints** (`exfield.make_fingerprint`) record scaffold
  type, version and a hash of the discretisation options. `(element,
  xi)` names the same anatomy only across models built with identical
  options — two option changes on `3D Heart 1` silently destroyed
  correspondence for 64% of elements. Mismatch raises.
* **`mesh.declared_unit`** lets you record what the file cannot: EX
  files state units nowhere (dataset 426 is in µm; its template
  `sub-M000` is in mm — iterating the folder naively yields a 0.6 mm
  nerve).

## Layout

| Module | Mirrors / responsibility |
|---|---|
| `scanner.py` | Zinc `IO_stream` primitives; offset-based so every error names a line |
| `exreader.py` | `EXReader` in `import_finite_element.cpp`, one method per production, names matching, so the two read side by side |
| `exwriter.py` | `EXWriter` in `export_finite_element.cpp`; emits EX Version 3, Zinc number formats |
| `basis.py` | `finite_element_basis.cpp`: tensor products + Hermite serendipity, Zinc's node-major function ordering |
| `faces.py` | `face_to_element` maps from `finite_element_shape.cpp` (cyclic axis convention) |
| `mesh.py` | data model: `Mesh`, fields, nodesets, element meshes, groups |
| `evaluate.py` | `Evaluator`, `ArclengthTable`, `integrate` (length/area/volume via Gram-determinant Jacobian) |
| `coordinates.py` | explicit conversions to rectangular cartesian (formulas from `general/geometry.cpp`); `Evaluator` warns on non-cartesian fields rather than silently converting |
| `inverse.py` | `find_location` / `find_locations` (Gauss-Newton, seeded) |
| `embedded.py` | `EmbeddedPoints`, `HostedPath` (ordered material-address paths for hosted proxies) |
| `bezier.py` | exact Hermite→Bezier change of basis (evaluation/display only) |
| `fingerprint.py` | template identity convention |

Only the parser and writer internals mirror Zinc (that is what makes
divergence reviewable); the public API is plain Python/NumPy — batch
`evaluate`, `mesh.mesh1d/2d/3d`, `mesh.evaluator(...)`, named result
tuples — and does not reproduce Zinc's object model.

`find_locations` has no spatial index — O(elements × samples ×
candidates) with Newton inside. Fine for a centreline; measure before
projecting thousands of points onto 3-D scaffolds.

## Tests

```bash
uv run pytest
```

The 42-scaffold corpus sweep runs when a cache directory is present
(`EXFIELD_426_CACHE=... uv run pytest`, or fetch with
`uv run python tests/sweep_426.py`); it is skipped otherwise.

The suite covers: every EX file in Zinc's own `tests/resources` (parses
or declines with a typed error — never a silent misread), the four
undocumented format semantics (`zero` = no terms; `-1` faces keep their
slot; face count comes from the shape; separator-retaining
tokenisation), derivative versions at branch nodes, the golden Zinc
comparison, write/read roundtrips (byte-stable), and a regression test
for every guard above asserting the guard *fires*.

## License and attribution

Apache-2.0 (see `LICENSE`), with a `NOTICE` file. exfield is a
from-scratch Python reimplementation — no source code was copied —
but its reader/writer deliberately mirror the productions of
OpenCMISS-Zinc's `import_finite_element.cpp` / `export_finite_element.cpp`
(MPL-2.0, Auckland Bioengineering Institute) so the implementations
review side by side, and the VTK higher-order point ordering is
transcribed from VTK (BSD-3-Clause, Kitware). Test data derives from
SPARC dataset 426 (CC-BY-4.0) and the sparc.client test fixtures; the
test corpus optionally exercises a local Zinc source checkout.
