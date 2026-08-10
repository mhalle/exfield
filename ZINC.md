# exfield for Zinc users and authors

Two audiences, one document:

* **Part 1** is for people who know the Zinc Python API and want to do
  the same things with exfield — concept map, side-by-side
  translations, and the traps that specifically catch Zinc users.
* **Part 2** is for people who know Zinc's C++ and want to review,
  trust, or maintain parity with this port — what mirrors what, how
  the golden validation works, and where exfield deliberately differs.

Every Zinc snippet below is taken from code in `tests/` that actually
runs against `cmlibs.zinc` 4.2.1 (`tests/golden_zinc.py`,
`tests/bench_zinc.py`), not written from memory.

---

## Should you use exfield at all?

exfield is not a Zinc replacement. It covers Zinc's **consumer** side —
read, evaluate, measure, write — with NumPy as the only dependency.

| You want to… | Use |
|---|---|
| Generate a scaffold from a template | **Zinc** (`scaffoldmaker`) |
| Fit a scaffold to data | **Zinc** (`scaffoldfitter`) |
| Render, or drive a GUI | **Zinc** (or export to VTK from exfield) |
| Computed-field expressions, solvers, time-varying fields | **Zinc** |
| Read an `.exf` and evaluate fields in NumPy | **exfield** |
| Ship a library/service without a 63 MB OpenGL-linked binary | **exfield** |
| Batch-evaluate tens of thousands of `(element, xi)` | **exfield** (see benchmarks) |
| Run on a Python with no `cmlibs.zinc` wheel (3.13+) | **exfield** |
| Address anatomy by material coordinates across refits | **exfield** (`EmbeddedPoints`, `HostedPath`) |

Mixing is normal and expected: author with scaffoldmaker/scaffoldfitter,
consume with exfield. Files round-trip both ways (see Part 2).

---

# Part 1 — For Zinc users

## The three mental-model shifts

1. **No context/region tree, no field DAG.** Zinc gives you a live
   object graph: `Context` → `Region` → `Fieldmodule` → fields you look
   up by name and *compose* (`createFieldDerivative`,
   `createFieldMeshIntegral`, `createFieldFindMeshLocation`). exfield
   gives you a plain data model — one `Mesh` per file, with dicts of
   fields, nodesets, element meshes and groups — plus free functions
   that take an evaluator. Nothing is lazily recomputed; there is no
   field to construct before you can measure something.

2. **No `Fieldcache`; xi goes in the call.** Zinc's evaluation is
   stateful: set a location on a cache, then evaluate a field against
   it. exfield's is functional and *batched* — pass an array of xi and
   get an array back. This is the single biggest performance lever
   (one matrix product vs a Python-level loop; see the traps below).

3. **Dimension is explicit, not inferred.** In Zinc you pick a mesh
   (`findMeshByDimension`) and then get elements from it. In exfield
   the same field usually lives on 3-D elements *and* their inherited
   faces/lines, so you say which mesh you mean when you build the
   evaluator: `mesh.evaluator("coordinates", dimension=1)`.

## Concept map

| Zinc | exfield |
|---|---|
| `Context` | — (none) |
| `Region` (one, no children) | `Mesh` (the whole file's contents) |
| `region.readFile(path)` | `exfield.load(path)` |
| `region.writeFile(path)` | `exfield.dump(mesh, path)` |
| `Fieldmodule` | attributes of `Mesh` |
| `fm.findFieldByName(n)` | `mesh.fields[n]` |
| `Fieldcache` + `setMeshLocation` | arguments to `evaluate(element_id, xi)` |
| `field.evaluateReal(cache, n)` | `ev.evaluate(element_id, xi)` |
| `fm.createFieldDerivative(f, d)` | `ev.evaluate_derivatives(...)` / `evaluate_with_derivatives(...)` |
| `fm.findMeshByDimension(d)` | `mesh.mesh1d` / `mesh2d` / `mesh3d`, or `mesh.mesh(d)` |
| `mesh.findElementByIdentifier(i)` | element ids are used directly |
| `fm.findNodesetByName("nodes")` | `mesh.nodes`, `mesh.datapoints` (`None` if the file has none) |
| `field.castGroup()` + `getMeshGroup` | `mesh.groups[name].element_ids(dim)` |
| `fm.createFieldMeshIntegral` | `exfield.integrate(ev, element_ids=…)` |
| `fm.createFieldFindMeshLocation` (nearest) | `exfield.find_location(ev, point, …)` |
| stored `element_xi` field (markers) | `node.fields["marker_location"][0]` → `(element_id, xi)` |
| `FieldStoredMeshLocation` on datapoints | `EmbeddedPoints` / `HostedPath` |
| — | `ArclengthTable` (no Zinc equivalent) |
| — | `make_fingerprint` / `check_fingerprints` |
| — | `export_vtu` (exact Bezier cells) |

## Side by side

### Load

```python
# Zinc
context = Context("app")
region = context.getDefaultRegion()
assert region.readFile("scaffold.exf") == RESULT_OK
fm = region.getFieldmodule()

# exfield
mesh = exfield.load("scaffold.exf")      # gzip accepted
assert not mesh.skipped                  # anything here was NOT read
print(mesh.summary())
```

`mesh.skipped` has no Zinc analogue and is worth checking every time:
it lists what the reader could not represent. Files exfield cannot
read faithfully raise `UnsupportedExFeature` at the point of encounter
rather than loading half a model.

### Evaluate a field

```python
# Zinc
cache = fm.createFieldcache()
coordinates = fm.findFieldByName("coordinates")
element = fm.findMeshByDimension(1).findElementByIdentifier(1155)
cache.setMeshLocation(element, [0.37])
result, x = coordinates.evaluateReal(cache, 3)

# exfield
ev = mesh.evaluator("coordinates", dimension=1)
x = ev.evaluate(1155, [0.37])                    # -> (3,) ndarray
xs = ev.evaluate(1155, [[0.1], [0.5], [0.9]])    # -> (3, 3), ONE matmul
```

### Derivatives

```python
# Zinc
d1 = fm.createFieldDerivative(coordinates, 1)
cache.setMeshLocation(element, [0.37])
res, dx = d1.evaluateReal(cache, 3)

# exfield
dx = ev.evaluate_derivatives(1155, [0.37])
x, J = ev.evaluate_with_derivatives(1155, xis)   # fused, batched
```

### Groups

```python
# Zinc
group = fm.findFieldByName("left vagus nerve").castGroup()
mg = group.getMeshGroup(fm.findMeshByDimension(1))
size = mg.getSize()

# exfield
ids = mesh.groups["left vagus nerve"].element_ids(1)
```

### Integrals (length / area / volume)

```python
# Zinc
one = fm.createFieldConstant([1.0])
integral = fm.createFieldMeshIntegral(one, coordinates, mg)
integral.setNumbersOfPoints([4])
res, value = integral.evaluateReal(fm.createFieldcache(), 1)

# exfield
value = exfield.integrate(ev, element_ids=ids)              # order=16
value_like_zinc = exfield.integrate(ev, element_ids=ids, order=4)
```

Dimension decides the measure: 1-D → length, 2-D → area, 3-D → volume,
via the Gram-determinant Jacobian. **The defaults differ deliberately**
— see the traps.

### Inverse mapping (nearest point)

```python
# Zinc
const = fm.createFieldConstant([0.0, 0.0, 0.0])
find = fm.createFieldFindMeshLocation(const, coordinates, mesh3)
find.setSearchMode(find.SEARCH_MODE_NEAREST)
const.assignReal(cache, target)
element, xi = find.evaluateMeshLocation(cache, 3)

# exfield
loc = exfield.find_location(ev, target, element_ids=trunk)
loc.element_id, loc.xi, loc.residual
loc.boundary      # xi pinned at the element edge (projected past the end)
loc.ambiguous     # another element fits nearly/exactly as well
```

exfield's version answers the same question but **refuses ambiguous
framings**: on a branching 1-D mesh it requires `element_ids` (a branch
running alongside the trunk is frequently nearer than the trunk), and
it reports `boundary` and `runner_up_residual` so a plausible-looking
wrong answer is visible. Pass `element_ids="all"` to insist.

### Markers / embedded locations

```python
# exfield: stored element:xi fields, read directly
for node in mesh.nodes:
    if "marker_location" in node.fields:
        element_id, xi = node.fields["marker_location"][0]
        name = node.fields["marker_name"][0]

# and the working form for landmarks that must survive refitting:
pts = exfield.EmbeddedPoints.from_world(ev, xyz, element_ids=trunk,
                                        max_residual=50.0)
xyz_refitted = pts.to_world(refitted_ev)
```

### Write

```python
# Zinc
region.writeFile("out.exf")

# exfield
exfield.dump(mesh, "out.exf")     # EX Version 3
exfield.export_vtu(mesh, "out.vtu", scale=1e-3)   # exact Bezier cells
```

## Traps that specifically catch Zinc users

* **Loop-per-point performance.** Zinc has no batch API, so Zinc habits
  produce `for xi in xis: ev.evaluate(eid, xi)` — which pays ~15 µs of
  Python overhead per call and is ~90× slower than passing the array.
  Batched exfield *beats* Zinc's per-call bindings (4.2 ms vs 8.5 ms
  for 20,000 evaluations); unbatched it loses badly (370 ms).
* **Quadrature default differs.** exfield defaults to `order=16`
  (converged). Zinc's `FieldMeshIntegral` defaults to 4 points, which
  runs slightly short on strongly curved elements. `order="zinc"` is a
  named bug-compatibility mode reproducing Zinc to ~1e-12 — use it to
  compare, not to measure.
* **`mesh.coordinates` raises when ambiguous.** Real scaffolds carry
  `coordinates`, `straight coordinates` and `vagus coordinates` — all
  coordinate-type, 3-component. Zinc will happily evaluate the material
  field and hand you dimensionless nonsense; exfield makes you name it
  (and warns when every parameter lies in [-1, 1]).
* **Arclength is not `s × total`.** Element lengths within one scaffold
  vary by more than 2×. `ArclengthTable` (no Zinc equivalent) does this
  properly, refuses branching element sets, and reports the NaN count
  for off-chain points.
* **Units are your problem, and EX does not record them.** Dataset 426
  is in µm while its own template `sub-M000` is in mm; iterating a
  folder naively yields a 0.6 mm nerve. `mesh.declared_unit` is a
  caller-set annotation, nothing more. exfield refuses to guess.
* **No time.** Zinc time sequences are declined by design
  (`UnsupportedExFeature`), not silently flattened to the first time.
* **Fingerprints are not in the file.** `(element, xi)` names the same
  anatomy only across models built from the same template with the
  same discretisation options. exfield gives you
  `make_fingerprint`/`check_fingerprints` to enforce that — but EX
  files carry no fingerprint, so a freshly loaded mesh has
  `fingerprint = None` and two `None`s pass the check without
  checking. Stamp them in your own code (see README hazards).

---

# Part 2 — For Zinc authors and reviewers

## What this port is, precisely

exfield is an independent Python reimplementation. **No source code was
copied.** The EX/EXF semantics were recovered by reading Zinc's C++ and
written up as a spec, then implemented against NumPy — and the reader
and writer deliberately mirror Zinc's *productions*, method for method
and name for name, so the two implementations can be reviewed side by
side. Everything above that layer (the public API) is plain
Python/NumPy and does not reproduce Zinc's object model.

Licensing follows from that: exfield is Apache-2.0; `NOTICE` records
the derivation from Zinc (MPL-2.0, Auckland Bioengineering Institute)
and from VTK (BSD-3-Clause, Kitware) for the higher-order point
ordering. Corrections to that attribution are welcome.

## Where to look, file by file

| exfield | mirrors / derives from |
|---|---|
| `src/exfield/scanner.py` | Zinc `IO_stream` primitives; offset-based, so every error names a line |
| `src/exfield/exreader.py` | `EXReader` in `import_finite_element.cpp`, one method per production |
| `src/exfield/exwriter.py` | `EXWriter` in `export_finite_element.cpp`; emits EX Version 3 with Zinc's number formats |
| `src/exfield/basis.py` | `finite_element_basis.cpp`: tensor products + Hermite serendipity, Zinc's node-major function ordering |
| `src/exfield/faces.py` | `face_to_element` maps from `finite_element_shape.cpp` (cyclic axis convention) |
| `src/exfield/coordinates.py` | conversion formulas from `general/geometry.cpp` |
| `src/exfield/vtu.py` | VTK `vtkHigherOrder*::PointIndexFromIJK` (transcribed, pinned against VTK 9.6.2 tables) |
| `src/exfield/evaluate.py` | Zinc's blending-matrix-over-monomials evaluation *design*, not its code |

Semantics deliberately reproduced (each has a regression test):
zero local nodes meaning "no terms"; absent faces keeping their slot;
face count from the element shape; per-derivative versions not
collapsing; EX v2+ version-consecutive parameter layout.

## How parity is established

`tests/golden_zinc.py` runs under a real `cmlibs.zinc` and dumps
values, first derivatives, per-group integrals, group sizes and marker
data to JSON; `tests/test_golden_zinc.py` checks exfield against it
with **no Zinc present**. Measured against Zinc 4.2.1 on a fitted human
vagus scaffold (SPARC 426 sub-f001 left — tricubic Hermite serendipity
surfaces, `c.Hermite*l.Lagrange*l.Lagrange` volumes, scale factors,
face-inherited lines, 55 groups):

* field values < 1e-15 relative, derivatives < 1e-13, over 1,476
  sampled `(element, xi)` across all three mesh dimensions;
* all 157 per-group arclength/area/volume integrals < 1e-13 relative
  against `FieldMeshIntegral`;
* **a file written by exfield and re-read by Zinc evaluates
  bit-identically** to the original at all 29,520 golden samples;
* all 42 scaffolds of dataset 426 load, chain and measure
  (`tests/sweep_426.py`);
* every `.ex*` file in Zinc's own `tests/resources` either parses or is
  declined with a typed exception — never a silent misread
  (`TestCorpus`, 59 files; skips loudly if the Zinc source checkout is
  absent).

Regenerate the reference data (the only step needing Zinc; there is no
`cmlibs.zinc` wheel for Python 3.13+, hence the pinned interpreter):

```bash
uv run --no-project --python 3.12 --with cmlibs.zinc --with numpy \
    python tests/golden_zinc.py scaffold.exf golden.json
```

## Where exfield deliberately differs

These are choices, not bugs — flagged here so reviewers don't read them
as porting errors:

* **Quadrature.** Default `order=16` (converged) rather than Zinc's
  4-point default. `order="zinc"` reproduces Zinc to ~1e-12. Agreement
  with Zinc and correctness are not the same claim, and exfield
  documents both.
* **Inverse-mapping bounds.** Per-element AABBs are computed from
  Bernstein control points (convex-hull property ⇒ rigorous), so
  branch-and-bound pruning is *exact*; Zinc's `FeMeshFieldRanges`
  samples and pads by ~1%. Zinc remains ~8× faster (compiled, warm
  start); exfield's version cannot prune away the true nearest element.
* **Ambiguity is surfaced, not resolved.** `find_location` reports
  `boundary`, `runner_up_residual` and `ambiguous`, and *refuses*
  nearest-point queries on branching 1-D meshes without an explicit
  element subset.
* **Output is EX Version 3 only**, though both v2 and v3 are read (v3
  alone would have excluded Zinc's own v2 test corpus).
* **Scope is declined loudly.** EX v1, simplex/polygon shapes,
  grid-based field values, time sequences, multi-region files and
  indexed fields raise `UnsupportedExFeature` at the point of
  encounter. A file never loads half-read.

## Known gaps, and what would help

* **Fingerprints don't round-trip.** exfield has a template-identity
  mechanism but no way to carry it in an `.exf`. If EX ever grows a
  blessed place for model-provenance metadata, that is the hook we'd
  use.
* **Multi-version branch-node data is under-tested.** Version handling
  is covered synthetically; dataset 428 v5 turned out to hold
  Neurolucida XML rather than the branched EX centreline we expected.
  Pointers to real multi-version EX files would be valuable.
* **Grid-based (element-based) field values** are declined rather than
  implemented — a deliberate scope call that could be revisited if
  there's demand.

## If Zinc changes

The pinned semantics live in tests, not prose. After a Zinc release:
regenerate the goldens (command above), re-run `pytest`, and re-run the
corpus test against the new `tests/resources`. A behavioural change in
Zinc shows up as a specific failing assertion naming the production it
came from. Issues and corrections — especially "you mirrored that
production wrong" — are welcome at
<https://github.com/mhalle/exfield>.

This is an experimental project, not an official SPARC or Auckland
product, and claims no affiliation.
