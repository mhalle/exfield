# Working on exfield

## Where this repo lives

Development happens in the `map-core` monorepo, at `map-core/exfield/`.
The public repo <https://github.com/mhalle/exfield> is a **subtree
mirror** — a derived artifact, never edited directly. Pilots in the
monorepo consume the library through editable installs
(`exfield = { path = "../exfield", editable = true }`), so a change to
the source is live for them at the next Python start, with no reinstall.

Publish with one command from the monorepo root:

```bash
./sync-exfield.sh
```

It refuses to run with uncommitted changes under `exfield/`, and pushes
a deterministic subtree split. If it ever reports a non-fast-forward,
the monorepo's history was rewritten — reconcile by hand; **never**
force-push the public repo.

## Checks

```bash
uv run pytest        # tests
uv run ruff check    # lint — bare, so it covers src AND tests
```

Run `ruff check` without a path. Scoping it to `src/` was the habit for
a while, and an unused import survived in `tests/` from the initial
commit until 0.5.1 because of it.

The suite is 187 tests with a few environment-dependent skips, all of
which announce themselves: the 59-file Zinc corpus regression needs a
sibling Zinc source checkout, the 42-scaffold sweep needs
`EXFIELD_426_CACHE`, and the live-VTK oracles need
`uv run --with vtk pytest tests/test_vtu.py`. A skip is not a pass —
if you touched the reader, run the corpus; if you touched `vtu.py`, run
the VTK oracles.

## Release checklist

1. Bump the version in **both** `pyproject.toml` and
   `src/exfield/__init__.py` — they are separate strings and have
   drifted before (0.2.0 vs 0.3.0, caught by review).
2. `uv lock` — the lockfile pins `requires-python` and the version, and
   has lagged a bump twice. Do it in the same commit as step 1.
3. Add a `CHANGELOG.md` entry. Breaking changes get a migration table
   with an old → new → *why* column; this is an alpha library and the
   table is the only migration aid users get, since no deprecation
   aliases are shipped.
4. `uv build`, then install the wheel into a throwaway venv and
   exercise the changed API — not just `import exfield`.
5. Commit, then `./sync-exfield.sh`.

## Conventions worth knowing before you name something

- **The API is shaped around silent-wrong-geometry hazards.** Guards
  that refuse ambiguous questions (branching meshes, mandatory
  `max_residual`, fingerprint mismatch) are the product, not friction.
  If a change makes a guard optional, it needs a stronger argument than
  convenience.
- **Method families stay morphologically parallel** — same part of
  speech, same number, same vocabulary. `evaluate` /
  `evaluate_derivatives` / `evaluate_values_and_derivatives` is
  deliberately verbose at the third name; an earlier
  `value_and_jacobian` was rejected for breaking the family despite
  matching a well-known idiom.
- **A name must not imply an axis that doesn't exist.** `evaluate_many`
  was deleted because its existence implied `evaluate` was
  single-point-only. Every evaluation method takes one xi or a batch;
  they differ only in what they return.
- **Parser and writer internals mirror Zinc's productions** method for
  method, so the two implementations can be read side by side. That
  constraint applies *only* to `exreader.py` / `exwriter.py`; the public
  API is plain Python and should not reproduce Zinc's object model.
  See `ZINC.md`.
- **Docstrings say what the reader can't see** — entry state, what a
  method leaves behind, error contracts. Not a restatement of the name.

## Scope

Consumer-side only: read, evaluate, measure, write. Authoring stays with
`scaffoldmaker` and `scaffoldfitter`. Declined by design and raised
loudly rather than half-read: EX version 1, simplex and polygon shapes,
grid-based field values, time sequences, multiple regions, indexed
fields. Adding one of those is a scope decision, not a bug fix.
