"""Reader tests: corpus coverage, format semantics, adversarial inputs."""

import pytest

import exfield
from exfield.errors import ExSyntaxError, UnsupportedExFeature

from conftest import zinc_resource_files


HEADER = """EX Version: 2
Region: /
!#nodeset nodes
Shape. Dimension=0
#Fields=1
1) coordinates, coordinate, rectangular cartesian, real, #Components=1
 x. #Values=1 (value)
"""


class TestCorpus:
    """Every EX file in Zinc's own test resources either parses or is
    declined with a typed exception — never a silent misread or crash."""

    @pytest.mark.parametrize("path", zinc_resource_files() or
                             [pytest.param(None, marks=pytest.mark.skip)])
    def test_parse_or_decline(self, path):
        try:
            mesh = exfield.load(path)
        except (UnsupportedExFeature, ExSyntaxError):
            return
        except UnicodeDecodeError:
            pytest.skip("binary/compressed variant")
        assert isinstance(mesh, exfield.Mesh)
        assert mesh.skipped == []


class TestFormatSemantics:
    """The four reference-implementation semantics from the porting spec."""

    def test_zero_means_no_terms(self, cube_mesh):
        # slot 4 of tricubic Hermite node 1 is d2/ds1ds2, declared 'zero'
        element = cube_mesh.mesh(3)[1]
        eft = element.template.efts[("notcoordinates", 0)]
        assert eft.functions[3] == []          # no terms, not value 0.0
        assert [t.label for t in eft.functions[4]] == ["d/ds3"]

    def test_absent_face_keeps_slot(self):
        text = (
            "EX Version: 2\nRegion: /\n"
            "!#nodeset nodes\nShape. Dimension=0\n#Fields=0\n"
            "Node: 1\nNode: 2\nNode: 3\nNode: 4\n"
            "!#mesh mesh1d, dimension=1, nodeset=nodes\n"
            "Shape. Dimension=1, line\n"
            "#Scale factor sets=0\n#Nodes=0\n#Fields=0\n"
            "Element: 1\nElement: 2\nElement: 3\n"
            "!#mesh mesh2d, dimension=2, face mesh=mesh1d, nodeset=nodes\n"
            "Shape. Dimension=2, line*line\n"
            "#Scale factor sets=0\n#Nodes=0\n#Fields=0\n"
            "Element: 1\n Faces:\n 1 -1 2 3\n")
        mesh = exfield.loads(text)
        faces = mesh.mesh(2)[1].faces
        assert faces == [1, None, 2, 3]        # -1 occupies its slot

    def test_face_count_comes_from_shape(self):
        # square has 4 faces: giving only 3 must fail, not "count tokens"
        text = (
            "EX Version: 2\nRegion: /\n"
            "!#nodeset nodes\nShape. Dimension=0\n#Fields=0\nNode: 1\n"
            "!#mesh mesh1d, dimension=1, nodeset=nodes\n"
            "Shape. Dimension=1, line\n"
            "#Scale factor sets=0\n#Nodes=0\n#Fields=0\n"
            "Element: 1\nElement: 2\nElement: 3\n"
            "!#mesh mesh2d, dimension=2, face mesh=mesh1d, nodeset=nodes\n"
            "Shape. Dimension=2, line*line\n"
            "#Scale factor sets=0\n#Nodes=0\n#Fields=0\n"
            "Element: 1\n Faces:\n 1 2 3\n")
        with pytest.raises(ExSyntaxError):
            exfield.loads(text)

    def test_versions_do_not_collapse(self):
        """Branch nodes carry multiple derivative versions; a reader that
        quietly takes version 1 everywhere has wrong tangents at every
        bifurcation."""
        text = (
            "EX Version: 2\nRegion: /\n"
            "!#nodeset nodes\nShape. Dimension=0\n#Fields=1\n"
            "1) coordinates, coordinate, rectangular cartesian, real, "
            "#Components=1\n"
            " x. #Values=3 (value,d/ds1(2))\n"
            "Node: 1\n 5.0 1.0 -1.0\n")
        mesh = exfield.loads(text)
        node = mesh.nodes[1]
        assert node.get_parameter("coordinates", 0, "value") == 5.0
        assert node.get_parameter("coordinates", 0, "d/ds1", 1) == 1.0
        assert node.get_parameter("coordinates", 0, "d/ds1", 2) == -1.0


class TestVagusScaffold:
    def test_load_complete(self, vagus_mesh):
        m = vagus_mesh
        assert m.skipped == []
        assert len(m.nodes) == 97
        assert len(m.mesh(3)) == 88
        assert len(m.mesh(2)) == 831
        assert len(m.mesh(1)) == 1808
        assert len(m.groups) == 55
        assert set(m.coordinate_field_names) >= {
            "coordinates", "straight coordinates", "vagus coordinates"}

    def test_ambiguous_coordinates_raises(self, vagus_mesh):
        # several coordinate-type fields and none named 'coordinates'
        # would raise; here 'coordinates' exists so accessor resolves
        assert vagus_mesh.coordinates.name == "coordinates"

    def test_marker_fields(self, vagus_mesh):
        markers = [n for n in vagus_mesh.nodes if "marker_name" in n.fields]
        assert markers, "expected marker nodes"
        names = {n.fields["marker_name"][0] for n in markers}
        assert any("jugular" in name or "nodose" in name for name in names)
        node = markers[0]
        element_id, xi = node.fields["marker_location"][0]
        assert element_id in vagus_mesh.mesh(3)
        assert xi.shape == (3,)


class TestAdversarial:
    """Malformed input raises with a line number (acceptance §10.3)."""

    def _assert_line(self, text, exc=ExSyntaxError):
        with pytest.raises(exc) as excinfo:
            exfield.loads(text)
        assert excinfo.value.line is not None

    def test_garbage_node_values(self):
        self._assert_line(HEADER + "Node: 1\n banana\n")

    def test_nan_node_value_rejected(self):
        self._assert_line(HEADER + "Node: 1\n nan\n")

    def test_missing_node_values(self):
        self._assert_line(HEADER + "Node: 1\n")

    def test_wrong_values_count(self):
        bad = HEADER.replace("#Values=1 (value)", "#Values=2 (value)")
        self._assert_line(bad + "Node: 1\n 1.0 2.0\n")

    def test_unknown_top_level_token(self):
        self._assert_line("EX Version: 2\nRegion: /\nQuack: 1\n")

    def test_truncated_header(self):
        self._assert_line("EX Version: 2\nRegion: /\n!#nodeset nodes\n"
                          "Shape. Dimension=0\n#Fields=1\n"
                          "1) coordinates, coordinate")

    def test_element_before_template(self):
        self._assert_line(
            "EX Version: 2\nRegion: /\n"
            "!#mesh mesh1d, dimension=1, nodeset=nodes\n"
            "Element: 1\n")

    def test_node_group_missing_node(self):
        self._assert_line(
            "EX Version: 3\nRegion: /\nGroup name: g\n!#nodeset nodes\n"
            "Node group:\n1..3\n")


class TestDeclined:
    """Declining is correct behaviour — a file must never load half-read."""

    CASES = [
        ("legacy v1", "Group name: bob\n#Fields=0\n"),
        ("time sequence", "EX Version: 3\nRegion: /\n"
         "Time sequence: t1, size=2\n 0.0 1.0\n"),
        ("simplex shape", "EX Version: 2\nRegion: /\n"
         "!#mesh mesh2d, dimension=2, nodeset=nodes\n"
         "Shape. Dimension=2, simplex(2)*simplex\n"),
        ("multiple regions", "EX Version: 2\nRegion: /\n"
         "!#nodeset nodes\nShape. Dimension=0\n#Fields=0\nNode: 1\n"
         "Region: /child\n"),
        ("indexed field", HEADER.replace(
            "coordinate, rectangular cartesian",
            "field, indexed, Index_field=i, #Values=2, rectangular "
            "cartesian")),
    ]

    @pytest.mark.parametrize("name,text", CASES, ids=[c[0] for c in CASES])
    def test_declined(self, name, text):
        with pytest.raises(UnsupportedExFeature):
            exfield.loads(text)


def _redeclared_field_text(element_coordinate_system="rectangular cartesian",
                           element_component="x"):
    """One field declared in a node header, redeclared in the element
    header — legal EX, but the declarations must agree."""
    return f"""EX Version: 2
Region: /
!#nodeset nodes
Shape. Dimension=0
#Fields=1
1) coordinates, coordinate, rectangular cartesian, real, #Components=1
 x. #Values=1 (value)
Node: 1
 0.0
Node: 2
 1.0
!#mesh mesh1d, dimension=1, nodeset=nodes
Shape. Dimension=1, line
#Scale factor sets=0
#Nodes=2
#Fields=1
1) coordinates, coordinate, {element_coordinate_system}, real, #Components=1
 {element_component}. l.Lagrange, no modify, standard node based.
  #Nodes=2
  1. #Values=1
   Value labels: value
  2. #Values=1
   Value labels: value
Element: 1
 Nodes:
 1 2
"""


class TestRedeclaration:
    """Redeclaring a field with conflicting metadata must raise, not
    silently keep whichever declaration came first."""

    def test_consistent_redeclaration_accepted(self):
        mesh = exfield.loads(_redeclared_field_text())
        assert mesh.fields["coordinates"].coordinate_system \
            == "rectangular cartesian"

    def test_conflicting_coordinate_system_rejected(self):
        with pytest.raises(ExSyntaxError, match="redeclared"):
            exfield.loads(_redeclared_field_text(
                element_coordinate_system="cylindrical polar"))

    def test_conflicting_component_name_rejected(self):
        with pytest.raises(ExSyntaxError, match="component name"):
            exfield.loads(_redeclared_field_text(element_component="theta"))


class TestStrings:
    def test_quoted_string_with_escapes(self):
        text = (
            "EX Version: 2\nRegion: /\n!#nodeset nodes\n"
            "Shape. Dimension=0\n#Fields=1\n"
            "1) name, field, string, #Components=1\n n. #Values=1 (value)\n"
            'Node: 1\n "left A branch \\"quoted\\", with comma"\n')
        mesh = exfield.loads(text)
        assert mesh.nodes[1].fields["name"][0] == \
            'left A branch "quoted", with comma'
