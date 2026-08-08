"""Data model for EX finite element scaffolds.

The top-level object is :class:`Mesh` — the contents of one EX region:
fields, nodesets ("nodes" and "datapoints"), element meshes by dimension,
and named groups. This mirrors what Zinc calls a region; exfield declines
multiple regions so the two notions coincide.

Terminology follows the EX format and Zinc:

* A *field* is declared once and holds parameters at nodes (via node
  field templates) and mappings on elements (via element field templates).
* A *node field template* (``NodeFieldTemplate``) lists, per component,
  the value/derivative labels and version counts stored at a node. EX
  version 2+ stores versions consecutively within each label.
* An *element field template* (``ElementFieldTemplate``) maps basis
  functions to node parameters: per function, a sum of terms, each term a
  (local node, value label, version) triple scaled by a product of scale
  factors.
"""


from .errors import EvaluationError

VALUE_LABELS = (
    "value", "d/ds1", "d/ds2", "d2/ds1ds2",
    "d/ds3", "d2/ds1ds3", "d2/ds2ds3", "d3/ds1ds2ds3",
)


# --------------------------------------------------------------- fields


class Field:
    """A field declaration (shared by node and element headers)."""

    def __init__(self, name, cm_type, value_type, number_of_components,
                 coordinate_system=None, focus=None, field_type="general",
                 host_mesh_name=None, host_mesh_dimension=None):
        self.name = name
        self.cm_type = cm_type                  # "coordinate"|"field"|"anatomical"
        self.value_type = value_type            # "real"|"integer"|"string"|"element_xi"
        self.number_of_components = number_of_components
        self.component_names = [str(i + 1) for i in range(number_of_components)]
        self.coordinate_system = coordinate_system  # e.g. "rectangular cartesian"
        self.focus = focus
        self.field_type = field_type            # "general"|"constant"
        self.host_mesh_name = host_mesh_name
        self.host_mesh_dimension = host_mesh_dimension
        self.values = None                      # constant field values
        self.mesh = None                        # back-reference, set by reader

    @property
    def is_coordinate(self):
        return self.cm_type == "coordinate"

    def matches_declaration(self, other):
        return (self.name == other.name
                and self.cm_type == other.cm_type
                and self.value_type == other.value_type
                and self.number_of_components == other.number_of_components
                and self.field_type == other.field_type)

    def __repr__(self):
        return (f"Field({self.name!r}, {self.cm_type}, {self.value_type}, "
                f"#Components={self.number_of_components})")


class NodeFieldTemplate:
    """Per-component list of (value label, version count) stored at a node.

    The parameter vector layout is label-major with versions consecutive
    within each label (EX version 2+ ordering).
    """

    def __init__(self):
        self.labels = []          # list of (label, versions)
        self._index = {}          # (label, version) -> slot

    def set_value_number_of_versions(self, label, versions):
        for existing, _ in self.labels:
            if existing == label:
                raise ValueError(f"repeated value label {label}")
        base = self.total_values
        self.labels.append((label, versions))
        for v in range(versions):
            self._index[(label, v + 1)] = base + v

    @property
    def total_values(self):
        return sum(v for _, v in self.labels)

    def slot(self, label, version=1):
        """Index of (label, version) in the parameter vector, or None."""
        return self._index.get((label, version))

    def versions(self, label):
        for existing, v in self.labels:
            if existing == label:
                return v
        return 0

    def signature(self):
        return tuple(self.labels)


class Node:
    """One node (or datapoint): identifier plus per-field parameters.

    ``fields[field_name]`` is a list per component; each entry is either a
    numpy array of parameters laid out per the field's NodeFieldTemplate
    (real/integer fields), a string, or an (element_id, xi) tuple for
    element_xi fields.
    """

    __slots__ = ("identifier", "fields", "templates")

    def __init__(self, identifier):
        self.identifier = identifier
        self.fields = {}       # field name -> list per component
        self.templates = {}    # field name -> list of NodeFieldTemplate

    def get_parameter(self, field_name, component, label="value", version=1):
        nfts = self.templates.get(field_name)
        if nfts is None:
            raise EvaluationError(
                f"Field {field_name!r} not defined at node {self.identifier}")
        slot = nfts[component].slot(label, version)
        if slot is None:
            raise EvaluationError(
                f"Node {self.identifier} field {field_name!r} component "
                f"{component + 1} has no {label}({version})")
        return self.fields[field_name][component][slot]


class Nodeset:
    def __init__(self, name):
        self.name = name       # "nodes" or "datapoints"
        self.nodes = {}        # identifier -> Node

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, identifier):
        return self.nodes[identifier]

    def __iter__(self):
        return iter(self.nodes.values())

    def __contains__(self, identifier):
        return identifier in self.nodes

    def get_or_create(self, identifier):
        node = self.nodes.get(identifier)
        if node is None:
            node = Node(identifier)
            self.nodes[identifier] = node
        return node


# --------------------------------------------------------------- shapes


class ElementShape:
    """An element shape. Only tensor products of lines are supported."""

    def __init__(self, dimension, description=None):
        self.dimension = dimension
        if description is None and dimension > 0:
            description = "*".join(["line"] * dimension)
        self.description = description

    @property
    def face_count(self):
        # line: 2 ends; square: 4 edges; cube: 6 faces
        return 2 * self.dimension if self.dimension >= 1 else 0

    def face_shape(self):
        if self.dimension <= 1:
            return None
        return ElementShape(self.dimension - 1)

    def __eq__(self, other):
        return (isinstance(other, ElementShape)
                and self.dimension == other.dimension
                and self.description == other.description)

    def __repr__(self):
        return f"ElementShape({self.dimension}, {self.description!r})"


# ------------------------------------------------- element field templates


class Term:
    """One term of an element field template function: a node parameter
    times a product of scale factors.

    ``local_node`` is the 1-based index into the element template's node
    list (the element's ``Nodes:`` line). ``scale_factor_indices`` are
    0-based indices into the element's concatenated scale factor array;
    empty means unscaled (multiplier exactly 1).
    """

    __slots__ = ("local_node", "label", "version", "scale_factor_indices")

    def __init__(self, local_node, label, version=1, scale_factor_indices=()):
        self.local_node = local_node
        self.label = label
        self.version = version
        self.scale_factor_indices = tuple(scale_factor_indices)

    def __repr__(self):
        s = f"{self.label}"
        if self.version != 1:
            s += f"({self.version})"
        return f"Term(node {self.local_node}, {s})"


class ElementFieldTemplate:
    """Mapping from basis functions to node parameters for one field
    component on one element template."""

    def __init__(self, basis, mapping="node", scale_factor_set=None):
        self.basis = basis
        self.mapping = mapping           # "node" | "field"
        self.scale_factor_set = scale_factor_set
        # functions[fn] = list of Terms (empty list = zero function)
        self.functions = [None] * basis.number_of_functions

    def set_function_terms(self, fn, terms):
        self.functions[fn] = list(terms)

    def validate(self):
        for fn, terms in enumerate(self.functions):
            if terms is None:
                raise ValueError(
                    f"element field template function {fn + 1} has no "
                    f"term mapping")


class ScaleFactorSet:
    def __init__(self, name, count, offset, identifiers=None):
        self.name = name
        self.count = count
        self.offset = offset          # offset into element's concatenated array
        self.identifiers = identifiers  # raw identifiers string or None


class ElementTemplate:
    """Shape plus field definitions shared by a run of elements."""

    def __init__(self, name=""):
        self.name = name
        self.shape = None
        self.scale_factor_sets = []
        self.node_count = 0            # header #Nodes
        self.fields = []               # header order
        self.efts = {}                 # (field name, component) -> EFT
        self.has_element_values = False

    @property
    def total_scale_factors(self):
        return sum(s.count for s in self.scale_factor_sets)

    def find_scale_factor_set(self, name):
        for s in self.scale_factor_sets:
            if s.name == name:
                return s
        return None


class Element:
    """One element: identifier, shape, faces, nodes, scale factors and a
    reference to the template that defines its fields."""

    __slots__ = ("identifier", "shape", "faces", "nodes", "scale_factors",
                 "template")

    def __init__(self, identifier, shape=None, template=None):
        self.identifier = identifier
        self.shape = shape
        # faces: list with one entry per shape face; None = absent (-1 in
        # file). An absent face still occupies its slot — dropping it
        # would shift every later slot.
        self.faces = None
        self.nodes = []               # global node identifiers, template order
        self.scale_factors = None     # concatenated over scale factor sets
        self.template = template


class ElementMesh:
    """All elements of one dimension ("mesh1d", "mesh2d", "mesh3d")."""

    def __init__(self, dimension, name=None):
        self.dimension = dimension
        self.name = name or f"mesh{dimension}d"
        self.face_mesh_name = None
        self.nodeset_name = "nodes"
        self.elements = {}            # identifier -> Element

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, identifier):
        return self.elements[identifier]

    def __iter__(self):
        return iter(self.elements.values())

    def __contains__(self, identifier):
        return identifier in self.elements

    def get_or_create(self, identifier, shape=None):
        element = self.elements.get(identifier)
        if element is None:
            element = Element(identifier, shape=shape)
            self.elements[identifier] = element
        return element


# --------------------------------------------------------------- groups


class Group:
    """A named group of nodes and/or elements."""

    def __init__(self, name):
        self.name = name
        self.nodes = {}       # nodeset name -> set of node identifiers
        self.elements = {}    # dimension -> set of element identifiers

    def add_node(self, nodeset_name, identifier):
        self.nodes.setdefault(nodeset_name, set()).add(identifier)

    def add_element(self, dimension, identifier):
        self.elements.setdefault(dimension, set()).add(identifier)

    def node_ids(self, nodeset_name="nodes"):
        return self.nodes.get(nodeset_name, set())

    def element_ids(self, dimension):
        return self.elements.get(dimension, set())

    def __repr__(self):
        parts = [f"{len(v)} {k}" for k, v in self.nodes.items()]
        parts += [f"{len(v)} {d}-D elements" for d, v in self.elements.items()]
        return f"Group({self.name!r}: {', '.join(parts) or 'empty'})"


# ------------------------------------------------------------------ mesh


class Mesh:
    """The contents of one EX region.

    Attributes
    ----------
    fields : dict of field name -> Field, in declaration order
    nodesets : dict with "nodes" and (if present) "datapoints"
    element_meshes : dict of dimension -> ElementMesh
    groups : dict of group name -> Group
    skipped : list of str — constructs passed over rather than raised on.
        Check this after reading; anything here is silently missing.
    declared_unit : optional unit string set by the caller (EX files state
        units nowhere; see EXFIELD_GOTCHAS.md).
    fingerprint : optional template fingerprint dict (see fingerprint.py).
    """

    def __init__(self, name="/"):
        self.name = name
        self.fields = {}
        self.nodesets = {"nodes": Nodeset("nodes")}
        self.element_meshes = {}
        self.groups = {}
        self.skipped = []
        self.declared_unit = None
        self.fingerprint = None
        self.ex_version = None

    # convenience accessors ------------------------------------------------

    @property
    def nodes(self):
        return self.nodesets["nodes"]

    @property
    def datapoints(self):
        return self.nodesets.get("datapoints")

    def mesh(self, dimension):
        return self.element_meshes[dimension]

    @property
    def mesh1d(self):
        return self.element_meshes[1]

    @property
    def mesh2d(self):
        return self.element_meshes[2]

    @property
    def mesh3d(self):
        return self.element_meshes[3]

    def evaluator(self, field=None, dimension=None):
        """Convenience: an Evaluator for a field (by name, Field object,
        or None for :attr:`coordinates`) on the chosen element mesh
        dimension (default: highest present)."""
        from .evaluate import Evaluator
        if field is None:
            field = self.coordinates
        elif isinstance(field, str):
            field = self.fields[field]
        return Evaluator(field, dimension=dimension)

    @property
    def highest_dimension(self):
        dims = [d for d, m in self.element_meshes.items() if len(m)]
        return max(dims) if dims else 0

    @property
    def coordinate_field_names(self):
        """Names of all coordinate-type fields. Real scaffolds carry
        several (e.g. 'coordinates', 'straight coordinates', 'vagus
        coordinates') and picking the wrong one is silent — see
        EXFIELD_GOTCHAS.md §1."""
        return [f.name for f in self.fields.values() if f.is_coordinate]

    @property
    def coordinates(self):
        """The field named 'coordinates', or the single coordinate-type
        field. Raises if ambiguous rather than guessing."""
        if "coordinates" in self.fields:
            return self.fields["coordinates"]
        names = self.coordinate_field_names
        if len(names) == 1:
            return self.fields[names[0]]
        raise KeyError(
            f"No field named 'coordinates'; coordinate-type fields are "
            f"{names}. Pick one explicitly via mesh.fields[name].")

    def get_or_create_nodeset(self, name):
        ns = self.nodesets.get(name)
        if ns is None:
            ns = Nodeset(name)
            self.nodesets[name] = ns
        return ns

    def get_or_create_element_mesh(self, dimension):
        m = self.element_meshes.get(dimension)
        if m is None:
            m = ElementMesh(dimension)
            self.element_meshes[dimension] = m
        return m

    def get_or_create_group(self, name):
        g = self.groups.get(name)
        if g is None:
            g = Group(name)
            self.groups[name] = g
        return g

    def summary(self):
        lines = [f"Mesh {self.name!r} (EX version {self.ex_version})"]
        lines.append(f"  fields: {', '.join(self.fields)}")
        for name, ns in self.nodesets.items():
            lines.append(f"  {name}: {len(ns)}")
        for d in sorted(self.element_meshes):
            lines.append(f"  mesh{d}d: {len(self.element_meshes[d])} elements")
        lines.append(f"  groups: {len(self.groups)}")
        if self.skipped:
            lines.append(f"  SKIPPED: {self.skipped}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"Mesh({self.name!r}, {len(self.fields)} fields, "
                f"{len(self.nodes)} nodes)")
