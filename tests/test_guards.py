"""Regression tests for the API guards (acceptance §10.4).

Each asserts the *guard fires*, not merely that the happy path works.
The unifying failure mode these prevent: everything runs and the
geometry is quietly wrong.
"""

import warnings

import numpy as np
import pytest

import exfield


def _line_mesh(points, extra_branch=None, pairs=None):
    """Build a tiny 1-D linear-Lagrange mesh through given points.

    ``pairs`` overrides the default sequential chain with explicit
    (node, node) element connectivity."""
    lines = ["EX Version: 2", "Region: /", "!#nodeset nodes",
             "Shape. Dimension=0", "#Fields=1",
             "1) coordinates, coordinate, rectangular cartesian, real, "
             "#Components=3",
             " x. #Values=1 (value)", " y. #Values=1 (value)",
             " z. #Values=1 (value)"]
    for i, p in enumerate(points, start=1):
        lines.append(f"Node: {i}")
        lines.append(" " + " ".join(f"{v}" for v in p))
    lines += ["!#mesh mesh1d, dimension=1, nodeset=nodes",
              "Shape. Dimension=1, line",
              "#Scale factor sets=0", "#Nodes=2", "#Fields=1",
              "1) coordinates, coordinate, rectangular cartesian, real, "
              "#Components=3"]
    for c in "xyz":
        lines += [f" {c}. l.Lagrange, no modify, standard node based.",
                  "  #Nodes=2",
                  "  1. #Values=1", "   Value labels: value",
                  "  2. #Values=1", "   Value labels: value"]
    if pairs is None:
        pairs = [(i, i + 1) for i in range(1, len(points))]
        if extra_branch:
            pairs.append(extra_branch)
    for e, (a, b) in enumerate(pairs, start=1):
        lines += [f"Element: {e}", " Nodes:", f" {a} {b}"]
    return exfield.loads("\n".join(lines) + "\n")


@pytest.fixture
def chain_mesh():
    # 4 nodes along x with unequal element lengths (1, 3, 1)
    return _line_mesh([(0, 0, 0), (1, 0, 0), (4, 0, 0), (5, 0, 0)])


def make_branching():
    """1-D mesh where node 2 belongs to three elements (1-2, 2-3, 2-4)."""
    pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)]
    return _line_mesh(pts, extra_branch=(2, 4))


class TestArclengthGuards:
    def test_branching_total_refused(self):
        mesh = make_branching()
        ev = exfield.Evaluator(mesh.fields["coordinates"], dimension=1)
        with pytest.raises(ValueError, match="branching|simple path"):
            exfield.ArclengthTable.build(ev)

    def test_explicit_chain_allowed_on_branching_mesh(self):
        mesh = make_branching()
        ev = exfield.Evaluator(mesh.fields["coordinates"], dimension=1)
        table = exfield.ArclengthTable.build(ev, element_ids=[1, 2])
        assert table.total == pytest.approx(2.0)

    def test_disconnected_ids_refused(self, chain_mesh):
        # give elements 1 and 3, skipping 2 — not a connected chain
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        with pytest.raises(ValueError):
            exfield.ArclengthTable.build(ev, element_ids=[1, 3])

    def test_arclength_at_is_not_fraction_times_total(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        table = exfield.ArclengthTable.build(ev)
        # element lengths 1, 3, 1: total 5. Midpoint of element 2 is at
        # arclength 2.5; "element fraction" naive answer would be wrong.
        assert table.total == pytest.approx(5.0)
        assert table.arclength_at(2, 0.5) == pytest.approx(2.5)
        assert table.element_lengths() == pytest.approx([1.0, 3.0, 1.0])
        eid, xi = table.location_at(2.5)
        assert eid == 2 and xi == pytest.approx(0.5)

    def test_nonmonotonic_parameter_refused(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        table = exfield.ArclengthTable.build(ev)
        # y is constant 0 -> not monotonic; must refuse rather than
        # silently produce a garbage parameter mapping
        with pytest.raises(ValueError, match="monotonic"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                table.attach_parameter(chain_mesh.fields["coordinates"],
                                       component=1)

    def test_zinc_order_mode(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        t = exfield.ArclengthTable.build(ev, order="zinc")
        assert t.order == exfield.ArclengthTable.ZINC_ORDER


class TestEvaluatorGuards:
    def test_takes_field_not_arbitrary_object(self):
        with pytest.raises(TypeError, match="Field"):
            exfield.Evaluator(42)

    def test_mesh_resolves_or_raises_on_ambiguity(self, vagus_mesh):
        # vagus mesh has a field literally named 'coordinates' -> resolves
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = exfield.Evaluator(vagus_mesh)
        assert ev.field.name == "coordinates"

    def test_normalized_field_warns(self, vagus_mesh):
        """Building an Evaluator on the material field runs perfectly and
        returns dimensionless nonsense — so it must at least warn."""
        with pytest.warns(UserWarning, match="material|normalis"):
            exfield.Evaluator(vagus_mesh.fields["vagus coordinates"],
                              dimension=1)


class TestInverseGuards:
    def test_branching_requires_element_ids(self):
        mesh = make_branching()
        ev = exfield.Evaluator(mesh.fields["coordinates"], dimension=1)
        with pytest.raises(ValueError, match="branch"):
            exfield.find_location(ev, [1.5, 0.0, 0.0])
        # explicit subset works
        loc = exfield.find_location(ev, [1.5, 0.0, 0.0],
                                    element_ids=[1, 2])
        assert loc.element_id == 2
        assert loc.residual < 1e-9

    def test_boundary_flag_on_projection_past_end(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        loc = exfield.find_location(ev, [7.0, 0.0, 0.0])
        assert loc.element_id == 3
        assert loc.xi[0] == pytest.approx(1.0)
        assert loc.at_boundary          # explicit indicator, not just residual
        assert loc.residual == pytest.approx(2.0)

    def test_exact_tie_at_shared_endpoint_is_ambiguous(self, chain_mesh):
        """A point at a node shared by two elements maps with residual 0
        in both — that address IS ambiguous. A strict < comparison made
        0 < 0 report unambiguous."""
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        loc = exfield.find_location(ev, [1.0, 0.0, 0.0])   # node 2: elem 1|2
        assert loc.residual == pytest.approx(0.0, abs=1e-12)
        assert loc.runner_up_residual == pytest.approx(0.0, abs=1e-12)
        assert loc.ambiguous

    def test_clearly_interior_point_is_not_ambiguous(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        loc = exfield.find_location(ev, [2.5, 0.0, 0.0])   # mid element 2
        assert loc.residual == pytest.approx(0.0, abs=1e-12)
        assert not loc.ambiguous

    def test_touching_box_is_not_ambiguous(self):
        """An element whose AABB merely CONTAINS the query point is not
        a tie: its box distance is 0 but its nearest point is far. The
        box must be polished, not pruned into a phantom runner_up of
        0.0 that flags every exact hit near an overlapping box."""
        mesh = _line_mesh([(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0)],
                          pairs=[(1, 2), (3, 4)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # unit-scale test geometry
            ev = exfield.Evaluator(mesh.fields["coordinates"],
                                   dimension=1)
        loc = exfield.find_location(ev, [0.5, 0.0, 0.0],
                                    element_ids="all")
        assert loc.element_id == 1
        assert loc.residual == pytest.approx(0.0, abs=1e-12)
        # element 2's box contains the point; its true distance is
        # ~0.707 and must be reported, not a box-distance 0
        assert loc.runner_up_residual == pytest.approx(
            0.7071, abs=1e-3)
        assert not loc.ambiguous


class TestEmbeddedGuards:
    def test_max_residual_is_mandatory(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        with pytest.raises(ValueError, match="max_residual"):
            exfield.EmbeddedPoints.from_world(ev, [[1.0, 0.0, 0.0]])

    def test_empty_from_world_refused(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        with pytest.raises(ValueError, match="at least one point"):
            exfield.EmbeddedPoints.from_world(ev, [], max_residual=np.inf)
        with pytest.raises(ValueError, match="at least one point"):
            exfield.HostedPath.from_world(ev, [], max_residual=np.inf)

    def test_mutation_to_unequal_lengths_caught(self, chain_mesh):
        """The collections are public lists; a post-construction append
        must not make arclength return uninitialised memory."""
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        table = exfield.ArclengthTable.build(ev, element_ids=[1, 2])
        p = exfield.EmbeddedPoints(element_ids=[1, 2],
                                   xis=[[0.5], [0.5]])
        p.element_ids.append(3)
        with pytest.raises(ValueError, match="mutated"):
            p.chain_arclengths(table)
        with pytest.raises(ValueError, match="mutated"):
            p.to_world(ev)

    def test_max_residual_fires(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        with pytest.raises(ValueError, match="residual"):
            exfield.EmbeddedPoints.from_world(
                ev, [[1.0, 5.0, 0.0]], max_residual=0.5)

    def test_mismatched_collection_lengths_rejected(self):
        """Unequal element_ids/xis/names used to zip-truncate silently:
        len(obj) said 2 while iteration yielded 1 point."""
        with pytest.raises(ValueError, match="length"):
            exfield.EmbeddedPoints(element_ids=[1, 2], xis=[[0.5]])
        with pytest.raises(ValueError, match="names"):
            exfield.EmbeddedPoints(element_ids=[1, 2],
                                   xis=[[0.5], [0.5]], names=["only-one"])

    def test_nan_count_reported(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        table = exfield.ArclengthTable.build(ev, element_ids=[1, 2])
        emb = exfield.EmbeddedPoints(
            element_ids=[1, 3], xis=[[0.5], [0.5]])
        values, nan_count = emb.chain_arclengths(table)
        assert nan_count == 1              # element 3 is off the chain
        assert np.isnan(values[1])
        assert values[0] == pytest.approx(0.5)


class TestHostedPath:
    """A proxy path is ordered and derived; the guards protect both."""

    def test_to_world_keeps_path_order(self, chain_mesh):
        """Order is the payload: a path that silently sorted its
        addresses would render as a different anatomical route."""
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        path = exfield.HostedPath(element_ids=[3, 1, 2],
                                  xis=[[1.0], [0.0], [1.0]])
        xyz = path.to_world(ev)
        assert xyz[:, 0] == pytest.approx([5.0, 0.0, 4.0])

    def test_polyline_arclengths_match_known_geometry(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        path = exfield.HostedPath(element_ids=[1, 1, 2],
                                  xis=[[0.0], [1.0], [1.0]])
        s = path.polyline_arclengths(ev)
        assert s == pytest.approx([0.0, 1.0, 4.0])
        assert np.all(np.diff(s) >= 0.0)

    def test_single_address_arclength_is_zero(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        path = exfield.HostedPath(element_ids=[2], xis=[[0.5]])
        assert path.polyline_arclengths(ev) == pytest.approx([0.0])

    def test_empty_path_refused(self):
        with pytest.raises(ValueError, match="at least one address"):
            exfield.HostedPath(element_ids=[], xis=[])

    def test_mismatched_collection_lengths_rejected(self):
        with pytest.raises(ValueError, match="length"):
            exfield.HostedPath(element_ids=[1, 2], xis=[[0.5]])

    def test_fingerprint_guard_fires(self, chain_mesh):
        """Same guard as EmbeddedPoints: addresses only mean something
        on the template they were authored against."""
        chain_mesh.fingerprint = exfield.make_fingerprint("t", "1", {"o": 1})
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        path = exfield.HostedPath.from_world(
            ev, [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], max_residual=0.1)
        chain_mesh.fingerprint = exfield.make_fingerprint("t", "1", {"o": 2})
        with pytest.raises(exfield.FingerprintMismatch):
            path.to_world(ev)
        with pytest.raises(exfield.FingerprintMismatch):
            path.polyline_arclengths(ev)

    def test_from_world_returns_hosted_path_with_residuals(self, chain_mesh):
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        path = exfield.HostedPath.from_world(
            ev, [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], max_residual=0.1,
            host_group="aorta")
        assert isinstance(path, exfield.HostedPath)
        assert len(path) == 2
        assert path.metadata["residual"] == pytest.approx([0.0, 0.0],
                                                          abs=1e-9)
        assert path.host_group == "aorta"

    def test_from_world_positional_element_ids_respected(self, chain_mesh):
        """The third positional is element_ids, exactly as on
        EmbeddedPoints — an override that silently reassigned it to
        host_group widened the search to the whole mesh."""
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        pts = [[0.5, 0.0, 0.0], [4.5, 0.0, 0.0]]
        base = exfield.EmbeddedPoints.from_world(
            ev, pts, [1], max_residual=np.inf)
        path = exfield.HostedPath.from_world(
            ev, pts, [1], max_residual=np.inf)
        assert path.element_ids == base.element_ids == [1, 1]
        assert path.host_group is None       # keyword-only now

    def test_host_group_round_trips(self, chain_mesh):
        path = exfield.HostedPath(element_ids=[1], xis=[[0.5]],
                                  host_group="thoracic_aorta")
        assert path.host_group == "thoracic_aorta"
        assert "thoracic_aorta" in repr(path)
        assert exfield.HostedPath(element_ids=[1],
                                  xis=[[0.5]]).host_group is None


class TestFingerprintGuards:
    def test_mismatch_raises(self):
        a = exfield.make_fingerprint("3D Heart 1", "1.0",
                                     {"Number of elements up RV": 4})
        b = exfield.make_fingerprint("3D Heart 1", "1.0",
                                     {"Number of elements up RV": 3})
        with pytest.raises(exfield.FingerprintMismatch):
            exfield.check_fingerprints(a, b)

    def test_one_sided_none_raises(self):
        a = exfield.make_fingerprint("t", "1", {})
        with pytest.raises(exfield.FingerprintMismatch):
            exfield.check_fingerprints(a, None)

    def test_embedded_points_check(self, chain_mesh):
        chain_mesh.fingerprint = exfield.make_fingerprint("t", "1", {"o": 1})
        ev = exfield.Evaluator(chain_mesh.fields["coordinates"], dimension=1)
        emb = exfield.EmbeddedPoints.from_world(
            ev, [[1.0, 0.0, 0.0]], max_residual=0.1)
        chain_mesh.fingerprint = exfield.make_fingerprint("t", "1", {"o": 2})
        with pytest.raises(exfield.FingerprintMismatch):
            emb.to_world(ev)


class TestMixedBasisGuard:
    """Components of one field must share a basis. The check has to run
    BEFORE parameters are assembled: the dof matrix is sized from
    component 0's basis, so a later component with MORE functions used
    to overrun it and surface as a bare IndexError instead of the
    typed message — the failing direction the guard exists for."""

    @staticmethod
    def _mixed_exf(first="l.Lagrange", second="q.Lagrange"):
        def component(name, basis, n):
            return (f" {name}. {basis}, no modify, standard node based.\n"
                    f"  #Nodes={n}\n"
                    + "".join(f"  {i}. #Values=1\n   Value labels: value\n"
                              for i in range(1, n + 1)).rstrip("\n"))

        n1 = 2 if first.startswith("l") else 3
        n2 = 2 if second.startswith("l") else 3
        declaration = ("1) coordinates, coordinate, rectangular cartesian, "
                       "real, #Components=2")
        return f"""EX Version: 3
Region: /
!#nodeset nodes
Define node template: node1
Shape. Dimension=0
#Fields=1
{declaration}
 x.  #Values=1 (value)
 y.  #Values=1 (value)
Node template: node1
Node: 1
 0.0
 0.0
Node: 2
 1.0
 2.0
Node: 3
 2.0
 5.0
!#mesh mesh1d, dimension=1, nodeset=nodes
Define element template: element1
Shape. Dimension=1, line
#Scale factor sets=0
#Nodes=3
#Fields=1
{declaration}
{component("x", first, n1)}
{component("y", second, n2)}
Element template: element1
Element: 1
 Nodes:
 1 2 3
"""

    def test_later_component_with_more_functions(self):
        """x linear, y quadratic — the direction that used to IndexError."""
        mesh = exfield.loads(self._mixed_exf("l.Lagrange", "q.Lagrange"))
        ev = exfield.Evaluator(mesh.fields["coordinates"], dimension=1)
        with pytest.raises(exfield.EvaluationError, match="different bases"):
            ev.evaluate(1, [0.5])

    def test_later_component_with_fewer_functions(self):
        """x quadratic, y linear — the direction that already worked."""
        mesh = exfield.loads(self._mixed_exf("q.Lagrange", "l.Lagrange"))
        ev = exfield.Evaluator(mesh.fields["coordinates"], dimension=1)
        with pytest.raises(exfield.EvaluationError, match="different bases"):
            ev.evaluate(1, [0.5])

    def test_matching_bases_still_evaluate(self):
        mesh = exfield.loads(self._mixed_exf("l.Lagrange", "l.Lagrange"))
        ev = exfield.Evaluator(mesh.fields["coordinates"], dimension=1)
        assert ev.evaluate(1, [0.0]) == pytest.approx([0.0, 0.0])
        assert ev.evaluate(1, [1.0]) == pytest.approx([1.0, 2.0])


class TestDocstringReferences:
    """Docstrings that cite a companion file must cite one that exists.

    Seven public docstrings pointed at EXFIELD_GOTCHAS.md and
    EXFIELD_PORTING_SPEC.md, which stayed behind in map-core when
    exfield was extracted — so help(exfield.Mesh) sent readers to
    nothing. Cheap to re-break, so pin it."""

    def test_no_docstring_cites_a_missing_markdown_file(self):
        import pathlib
        import re
        root = pathlib.Path(exfield.__file__).resolve().parent.parent.parent
        dangling = []
        for path in sorted((root / "src" / "exfield").glob("*.py")):
            for n, line in enumerate(
                    path.read_text().splitlines(), start=1):
                for name in re.findall(r"\b([A-Za-z0-9_./-]+\.md)\b", line):
                    if not (root / name).exists():
                        dangling.append(f"{path.name}:{n} -> {name}")
        assert not dangling, "docstrings cite missing files: " + "; ".join(
            dangling)
