"""EX/EXF file reader.

Structured to mirror Zinc's ``EXReader`` in
``src/finite_element/import_finite_element.cpp``: one method per
production, names matching the reference, so the two read side by side.
This is the single most useful convention in the project — it makes
divergence reviewable.

Supported: EX versions 2 and 3 (version 3 is what current Zinc writes;
version 2 differs only in template handling and covers Zinc's own test
corpus). Declined with :class:`UnsupportedExFeature` at the point of
encounter, so a file never loads half-read: EX version 1 (legacy,
unversioned), multiple regions, time sequences, simplex and polygon
shapes, element-based (grid) field values, indexed fields.

Format semantics that are not in the documentation (from the reference
implementation; summarised in README.md's Tests section):

1. ``zero`` in a value-label list means the function has *no terms*, not
   a term whose value is zero.
2. A face identifier of ``-1`` denotes an absent face that still
   occupies its slot.
3. The number of face identifiers to read comes from the element shape,
   not from counting tokens on the line.
4. Tokenising retains the separator that terminated each token: in
   ``d/ds1(2)+d/ds2`` the ``(`` announces a version and the ``+``
   announces another term.

The format is a stream, not a line grammar: values wrap lines freely and
several tokens may share a line.
"""

import numpy as np

from .basis import parse_basis, constant_basis
from .errors import ExSyntaxError, UnsupportedExFeature
from .mesh import (ElementFieldTemplate, ElementShape,
                   ElementTemplate, Field, Mesh, NodeFieldTemplate,
                   ScaleFactorSet, Term, VALUE_LABELS)
from .scanner import LineTokenizer, Scanner

_CM_FIELD_TYPES = ("field", "coordinate", "anatomical")
_VALUE_TYPES = ("real", "integer", "string", "element_xi")
_COORDINATE_SYSTEMS = (
    "rectangular cartesian", "cylindrical polar", "spherical polar",
    "prolate spheroidal", "oblate spheroidal", "fibre",
)
_UNSUPPORTED_VALUE_TYPES = (
    "double", "double_array", "real_array", "float", "float_array",
    "integer_array", "short", "short_array", "unsigned", "unsigned_array",
    "url",
)


class _NodeTemplate:
    """Reader-side node template: field list plus per-field NFTs."""

    def __init__(self, name=""):
        self.name = name
        self.header_fields = []            # Field objects in header order
        self.nfts = {}                     # field name -> [NodeFieldTemplate]


class EXReader:
    """Reads one EX region into a :class:`~exfield.mesh.Mesh`."""

    def __init__(self, text):
        self.s = Scanner(text)
        self.exVersion = 1
        self.model = None                  # Mesh, created by Region:
        self.useData = False
        self.nodeset = None                # current Nodeset
        self.mesh = None                   # current ElementMesh
        self.fieldGroup = None             # current Group
        self.nodeTemplates = []
        self.nodeTemplate = None           # active _NodeTemplate
        self.elementTemplates = []
        self.elementTemplate = None        # active ElementTemplate

    # ------------------------------------------------------------ helpers

    def error(self, message):
        """Build (do not raise) an :class:`ExSyntaxError` at the current
        scanner line. Callers write ``raise self.error(...)``."""
        return self.s.error(message)

    def unsupported(self, message):
        """Build (do not raise) an :class:`UnsupportedExFeature` at the
        current scanner line. Callers write ``raise self.unsupported(...)``."""
        return UnsupportedExFeature(message, line=self.s.line)

    def clearTemplates(self):
        """Drop all node and element templates and deactivate both.

        Templates are scoped to the current domain, so this is called on
        every nodeset/mesh switch: a template defined under one !#nodeset
        or !#mesh is not nameable from another.
        """
        self.nodeTemplates = []
        self.nodeTemplate = None
        self.elementTemplates = []
        self.elementTemplate = None

    def requireModel(self):
        """Return the Mesh, raising if no ``Region:`` has been read yet."""
        if self.model is None:
            raise self.error("Region: must be set before this construct")
        return self.model

    def setNodeset(self, nodeset):
        """Make ``nodeset`` current and clear the mesh, so element
        constructs error until a !#mesh directive arrives.

        Templates are cleared. Below EX version 3 there is no ``Define
        node template``, so a blank template is created *and activated*
        here for subsequent ``#Fields=`` headers to fill in.
        """
        self.mesh = None
        self.nodeset = nodeset
        self.clearTemplates()
        if self.exVersion < 3:
            # older versions required default blank node template
            self.nodeTemplate = _NodeTemplate()
            self.nodeTemplates.append(self.nodeTemplate)

    def setMesh(self, element_mesh):
        """Make ``element_mesh`` current, and *also* switch the current
        nodeset to the one the mesh declares, creating it if needed —
        elements name nodes from that nodeset. Templates are cleared.
        """
        self.mesh = element_mesh
        self.nodeset = self.requireModel().get_or_create_nodeset(
            element_mesh.nodeset_name)
        self.clearTemplates()

    # ------------------------------------------------------- productions

    def readEXVersion(self):
        """'E' has been consumed; 'X' is next."""
        if self.exVersion > 1:
            raise self.error("EX Version has already been specified")
        if self.model is not None:
            raise self.error("EX Version number must be first token")
        if not self.s.match_literal("X Version :"):
            raise self.error("Error reading EX Version: number")
        version = self.s.read_int("Error reading EX Version: number")
        if version < 2:
            raise self.unsupported(
                "EX version 1 (legacy) files are not supported")
        if version > 3:
            raise self.unsupported(
                f"Cannot read EX Version {version}, only versions 2-3")
        self.exVersion = version

    def readCommentOrDirective(self):
        """'!' consumed. Directives: !#nodeset NAME / !#mesh NAME, ..."""
        c = self.s.getc()
        if c == "":
            return
        nodesetDirective = meshDirective = False
        if c == "#":
            if self.s.match_literal("nodeset "):
                nodesetDirective = True
            elif self.s.match_literal("mesh "):
                meshDirective = True
        if not (nodesetDirective or meshDirective):
            if c not in ("\n", "\r"):
                self.s.read_rest_of_line()
            return
        model = self.requireModel()
        name = self.s.read_ex_string()
        keyValueMap = self.s.read_key_value_map(",")
        if nodesetDirective:
            if name == "nodes":
                domain = "nodes"
            elif name == "datapoints":
                domain = "datapoints"
            else:
                domain = "datapoints" if self.useData else "nodes"
            self.setNodeset(model.get_or_create_nodeset(domain))
        else:
            dimensionString = keyValueMap.get("dimension")
            dimension = int(dimensionString) if dimensionString else 0
            if not 1 <= dimension <= 3:
                raise self.error(
                    "Missing or invalid dimension in mesh directive")
            element_mesh = model.get_or_create_element_mesh(dimension)
            element_mesh.face_mesh_name = keyValueMap.get(
                "face mesh", element_mesh.face_mesh_name)
            element_mesh.nodeset_name = keyValueMap.get(
                "nodeset", element_mesh.nodeset_name)
            self.setMesh(element_mesh)

    def readTimeSequence(self):
        """'T' consumed: always raises — time sequences are declined by
        design, so this production never returns."""
        raise self.unsupported(
            "Time sequences are not supported (declined by design)")

    def readField(self):
        """Read a field declaration (shared by node/element headers).

        Also returns nothing extra: time sequences are declined, so the
        ``timeSequence`` out-parameter of the reference has no analogue.
        """
        s = self.s
        s.skip_whitespace()
        s.read_int("Missing field number in header")
        if not s.match_literal(") "):
            raise self.error("Missing ')' after field number")
        field_name = s.read_charset("^,").rstrip()
        s.match_literal(", ")
        if not field_name:
            raise self.error("No field name")

        next_block = s.read_charset("^,").strip()
        s.match_literal(", ")
        if next_block not in _CM_FIELD_TYPES:
            raise self.error(
                f"Field {field_name} has unknown CM field type '{next_block}'")
        cm_type = next_block

        next_block = s.read_charset("^,").strip()
        s.match_literal(", ")
        field_type = "general"
        if next_block == "constant":
            field_type = "constant"
            next_block = s.read_charset("^,").strip()
            s.match_literal(", ")
        elif next_block == "indexed":
            raise self.unsupported(
                f"Indexed field {field_name} is not supported "
                f"(declined by design)")

        coordinate_system = None
        focus = None
        value_type = None
        lowered = next_block.lower()
        if lowered in _COORDINATE_SYSTEMS:
            coordinate_system = lowered
            if lowered in ("prolate spheroidal", "oblate spheroidal"):
                if s.match_literal(" focus="):
                    focus = s.read_real("Missing focus value")
                else:
                    focus = 1.0
                s.match_literal(" ,")
            if lowered == "fibre":
                value_type = "real"
            next_block = s.read_charset("^,\n\r").strip()
            s.match_literal(", ")

        lowered = next_block.lower()
        if lowered in _VALUE_TYPES:
            value_type = lowered
            next_block = s.read_charset("^,\n\r")
        elif lowered in _UNSUPPORTED_VALUE_TYPES:
            raise self.unsupported(
                f"Field {field_name} value type '{next_block}' is not "
                f"supported")
        elif value_type is None:
            raise self.error(
                f"Field {field_name} has unknown value type '{next_block}'")
        else:
            # fibre with no explicit value type — next_block is #Components
            pass

        stripped = next_block.strip()
        if not stripped.startswith("#Components="):
            raise self.error(f"Field {field_name} missing #Components")
        try:
            number_of_components = int(stripped[len("#Components="):])
        except ValueError:
            number_of_components = 0
        if number_of_components < 1:
            raise self.error(f"Field {field_name} invalid #Components")

        keyValueMap = s.read_key_value_map(",")
        host_mesh_name = None
        host_mesh_dimension = None
        if value_type == "element_xi":
            dimensionString = keyValueMap.pop("host mesh dimension", None)
            if dimensionString is None:
                raise self.error(
                    f"Missing 'host mesh dimension=N' for element:xi valued "
                    f"field {field_name}")
            host_mesh_dimension = int(dimensionString)
            host_mesh_name = keyValueMap.pop("host mesh", None)
            if host_mesh_name is None:
                raise self.error(
                    f"Missing 'host mesh=~' for element:xi valued field "
                    f"{field_name}")
        if "time sequence" in keyValueMap:
            raise self.unsupported(
                f"Field {field_name} uses a time sequence (declined by "
                f"design)")

        field = Field(field_name, cm_type, value_type, number_of_components,
                      coordinate_system=coordinate_system, focus=focus,
                      field_type=field_type, host_mesh_name=host_mesh_name,
                      host_mesh_dimension=host_mesh_dimension)
        return self._mergeField(field)

    def _mergeField(self, field):
        """Merge declaration into the model, returning the shared Field."""
        model = self.requireModel()
        existing = model.fields.get(field.name)
        if existing is not None:
            if not existing.matches_declaration(field):
                raise self.error(
                    f"Field {field.name} redeclared with a conflicting "
                    f"declaration (type, components, coordinate system, "
                    f"focus, or element:xi host mesh differs)")
            return existing
        field.mesh = model
        model.fields[field.name] = field
        return field

    def readFieldValues(self):
        """'V' consumed: "alues:" then constants for header fields."""
        if not (self.s.match_literal("alues ") and self.s.match_one_of(":")):
            raise self.error("Truncated read of Values: token")
        template = self.nodeTemplate or self.elementTemplate
        if template is None:
            raise self.error(
                "Must have a current node or element template to read "
                "field values")
        header_fields = (template.header_fields
                         if isinstance(template, _NodeTemplate)
                         else template.fields)
        for field in header_fields:
            if field.field_type != "constant":
                continue
            n = field.number_of_components
            if field.value_type == "real":
                values = [self._readFiniteReal("field value")
                          for _ in range(n)]
            elif field.value_type == "integer":
                values = [self.s.read_int("Error reading integer field value")
                          for _ in range(n)]
            elif field.value_type == "string":
                values = [self.s.read_ex_string() for _ in range(n)]
            else:
                raise self.error(
                    f"Unsupported constant field value type "
                    f"{field.value_type}")
            field.values = values

    def _readFiniteReal(self, what):
        value = self.s.read_real(f"Error reading {what}")
        if not np.isfinite(value):
            raise self.error(f"Infinity or NAN {what} read")
        return value

    # ------------------------------------------------------ node headers

    def readNodeValueLabelsVersions(self, nft):
        """Parse e.g. ``(value,d/ds1(2),d/ds2,d2/ds1ds2)``."""
        s = self.s
        if s.next_non_space_char() != "(":
            raise self.error("Missing '(' before value/derivative labels")
        while True:
            label_name = s.read_charset("^,()\n\r").strip()
            if (not nft.labels and not label_name
                    and s.check_consume_next_char(")")):
                break  # empty list ()
            if label_name not in VALUE_LABELS:
                if not label_name and s.next_non_space_char() == ")":
                    break
                raise self.error(
                    f"Unrecognised value/derivative label name {label_name!r}")
            versionCount = 1
            if s.peekc() == "(":
                s.getc()
                versionCount = s.read_int("Invalid version count")
                if not s.match_literal(")"):
                    raise self.error("Missing ')' after version count")
                if versionCount < 2:
                    raise self.error(
                        "Value/derivative version count must be > 1 if "
                        "specified")
            try:
                nft.set_value_number_of_versions(label_name, versionCount)
            except ValueError:
                raise self.error(
                    f"Invalid derivative, number of versions or repeated "
                    f"derivative {label_name}")
            nc = s.next_non_space_char()
            if nc == ")":
                break
            if nc != ",":
                raise self.error(
                    "Invalid character in derivative/value versions list")

    def readNodeHeaderField(self):
        """One field of a node header: declaration then one line per
        component, ``name. #Values=N (labels)``.

        Appends the merged :class:`Field` and its per-component
        :class:`NodeFieldTemplate` list to ``self.nodeTemplate``, which
        must already be active. Constant fields carry no ``#Values`` —
        the component name alone is the whole entry, values arrive later
        in ``Values:``. The declared ``#Values`` count is checked against
        the value/version list. Component names are recorded on first
        declaration and must match on any redeclaration. Returns True to
        mirror the reference's bool.
        """
        field = self.readField()
        nfts = []
        for c in range(field.number_of_components):
            self.s.skip_whitespace()
            componentName = self.s.read_charset("^.").strip()
            if not componentName:
                raise self.error(
                    f"Error getting component name for field {field.name}")
            if (field.component_names_declared
                    and field.component_names[c] != componentName):
                raise self.error(
                    f"Field {field.name} redeclared with different component "
                    f"name {componentName!r} (was "
                    f"{field.component_names[c]!r})")
            field.component_names[c] = componentName
            if self.s.getc() != ".":
                raise self.error("Missing required '.' after component name")
            nft = NodeFieldTemplate()
            if field.field_type == "general":
                if not self.s.match_literal(" #Values="):
                    raise self.error(
                        f"Failed to read node field {field.name} component "
                        f"{componentName} #Values")
                valuesCount = self.s.read_int("Invalid #Values")
                self.readNodeValueLabelsVersions(nft)
                if nft.total_values != valuesCount:
                    raise self.error(
                        f"Count of value/derivative versions did not match "
                        f"number specified for field {field.name} component "
                        f"{componentName}")
                self.s.read_key_value_map()
            else:
                self.s.read_key_value_map(",")
            nfts.append(nft)
        field.component_names_declared = True
        self.nodeTemplate.header_fields.append(field)
        self.nodeTemplate.nfts[field.name] = nfts
        return True

    def readNodeHeader(self):
        """'#' consumed: ``Fields=N`` then N node field declarations.

        Requires a nodeset and no current mesh. Below EX version 3 a
        bare header at top level starts a fresh template (all earlier
        templates are discarded); in version 3+ it fills in the template
        opened by ``Define node template``, and errors if none is active.
        """
        if (self.mesh is not None) or (self.nodeset is None):
            raise self.error("Nodeset not set")
        if self.exVersion < 3:
            self.clearTemplates()
            self.nodeTemplate = _NodeTemplate()
            self.nodeTemplates.append(self.nodeTemplate)
        elif self.nodeTemplate is None:
            raise self.error(
                "Define node template needed before this in EX version 3+")
        if not self.s.match_literal("Fields="):
            raise self.error("Error reading number of fields")
        fieldCount = self.s.read_int("Error reading number of fields")
        for _ in range(fieldCount):
            self.readNodeHeaderField()

    def findNodeTemplateByName(self, name):
        """Node template of that name in the current nodeset's scope, or
        None. Templates from other domains have been cleared away."""
        for nt in self.nodeTemplates:
            if nt.name == name:
                return nt
        return None

    def readDefineNodeTemplate(self):
        """'D' consumed."""
        if not (self.s.match_literal("efine node template ")
                and self.s.match_one_of(":")):
            raise self.error("Truncated Define node template: token")
        if self.fieldGroup is not None:
            raise self.error(
                "Define node template not allowed in group definition")
        name = self.s.read_ex_string(",\n\r\t")
        self.s.read_key_value_map(",")
        if self.findNodeTemplateByName(name):
            raise self.error(f"Node template {name} already exists")
        self.nodeTemplate = _NodeTemplate(name)
        self.nodeTemplates.append(self.nodeTemplate)
        self.s.skip_whitespace()
        if self.s.getc() != "S":
            raise self.error("Missing Shape")
        self.readElementShape()
        self.s.skip_whitespace()
        if self.s.getc() != "#":
            raise self.error("Missing #Fields")
        self.readNodeHeader()
        self.s.skip_whitespace()
        if self.s.peekc() == "V":
            self.s.getc()
            self.readFieldValues()
        self.nodeTemplate = None  # must be separately activated

    def readNodeTemplate(self):
        """``Node template:`` consumed: name (plus ignored key=value
        trailer) of an already-defined template, which becomes active.

        Activation is the only job — the template was populated by an
        earlier ``Define node template``, which deliberately left it
        deactivated. Not permitted inside a group definition.
        """
        if self.fieldGroup is not None:
            raise self.error("Node template not allowed in group definition")
        name = self.s.read_ex_string(",\n\r\t")
        self.s.read_key_value_map(",")
        template = self.findNodeTemplateByName(name)
        if template is None:
            raise self.error(
                f"Node template {name} not found in scope of current nodeset")
        self.nodeTemplate = template

    # ------------------------------------------------------- node values

    def readElementXiValue(self, field):
        """Read ``ELEMENT_NUMBER xi1 .. xiD`` (EX version 2+ format)."""
        model = self.requireModel()
        dimension = field.host_mesh_dimension
        host_mesh = model.get_or_create_element_mesh(dimension)
        elementIdentifier = self.s.read_int(
            "Missing element number in element:xi value")
        xi = np.empty(dimension)
        for d in range(dimension):
            xi[d] = self._readFiniteReal("xi value")
        if elementIdentifier < 0:
            return (None, xi)
        host_mesh.get_or_create(elementIdentifier,
                                shape=ElementShape(dimension))
        return (elementIdentifier, xi)

    def readNode(self):
        """Read a node, adding or merging into the current nodeset."""
        if (self.fieldGroup is not None) and (self.exVersion >= 3):
            raise self.error(
                "Node not allowed in group definition; expecting Node group")
        if self.nodeset is None:
            raise self.error("Can't read node as no nodeset set")
        nodeIdentifier = self.s.read_int(
            "Error reading Node token or node number")
        existing = nodeIdentifier in self.nodeset
        if not existing and self.nodeTemplate is None:
            raise self.error(
                "Node not found, must be first defined with a template "
                "shape and field header")
        node = self.nodeset.get_or_create(nodeIdentifier)
        if self.nodeTemplate is None:
            return node  # listing nodes in a group without headers
        for field in self.nodeTemplate.header_fields:
            if field.field_type != "general":
                continue
            nfts = self.nodeTemplate.nfts[field.name]
            componentCount = field.number_of_components
            components = []
            if field.value_type == "element_xi":
                for _c in range(componentCount):
                    components.append(self.readElementXiValue(field))
            elif field.value_type == "real":
                for c in range(componentCount):
                    count = nfts[c].total_values
                    values = np.empty(count)
                    for k in range(count):
                        values[k] = self._readFiniteReal(
                            f"real value for field {field.name} at node "
                            f"{nodeIdentifier}")
                    components.append(values)
            elif field.value_type == "integer":
                for c in range(componentCount):
                    count = nfts[c].total_values
                    values = np.empty(count, dtype=int)
                    for k in range(count):
                        values[k] = self.s.read_int(
                            f"Error reading int value for field "
                            f"{field.name} at node {nodeIdentifier}")
                    components.append(values)
            elif field.value_type == "string":
                for _c in range(componentCount):
                    components.append(self.s.read_ex_string())
            else:
                raise self.error(
                    f"Unsupported value type {field.value_type} for node "
                    f"{nodeIdentifier}")
            node.fields[field.name] = components
            node.templates[field.name] = nfts
        return node

    def _readIdentifierRanges(self, add, what):
        """Read compact ranges ``1,3..7,22..150`` calling add(identifier)."""
        while True:
            start = self.s.read_int(f"Missing {what} identifier")
            self.s.skip_whitespace()
            stop = start
            if self.s.peekc() == ".":
                if not self.s.match_literal(".."):
                    raise self.error("Malformed start..stop range")
                stop = self.s.read_int("Malformed start..stop range")
                self.s.skip_whitespace()
                if start > stop:
                    raise self.error("Decreasing start..stop range")
            for identifier in range(start, stop + 1):
                add(identifier)
            if self.s.peekc() == ",":
                self.s.getc()
            else:
                break

    def readNodeGroup(self):
        """``Node group:`` consumed: compact identifier ranges added to
        the current group for the current nodeset.

        Membership only — every identifier must already exist in the
        nodeset, otherwise it is an error; a group never creates nodes.
        """
        if self.fieldGroup is None:
            raise self.error(
                "Node group: may only be used within a Group definition")
        if self.nodeset is None:
            raise self.error(
                "Region/Group/nodeset must be set before reading node group")
        group = self.fieldGroup
        nodeset = self.nodeset

        def add(identifier):
            if identifier not in nodeset:
                raise self.error(f"Node {identifier} not found")
            group.add_node(nodeset.name, identifier)

        self._readIdentifierRanges(add, "node")

    def readNodeOrTemplate(self):
        """'N' consumed: Node, Node group or Node template."""
        token = self.s.read_charset("^:").rstrip()
        isNode = token == "ode"
        isNodeGroup = token == "ode group"
        isNodeTemplate = token == "ode template"
        if ((self.exVersion < 3) and not isNode) or not (
                isNode or isNodeGroup or isNodeTemplate):
            raise self.error(f"Unrecognised token N{token}")
        self.s.skip_whitespace()
        if not self.s.match_one_of(":"):
            raise self.error("Missing : separator")
        if isNode:
            node = self.readNode()
            if self.fieldGroup is not None:
                self.fieldGroup.add_node(self.nodeset.name, node.identifier)
            return
        if isNodeGroup:
            self.readNodeGroup()
            return
        self.readNodeTemplate()

    # --------------------------------------------------- element headers

    def readElementShape(self):
        """'S' consumed. ``Shape. Dimension=D[, description]``"""
        if not self.s.match_literal("hape. Dimension="):
            raise self.error("Error reading element shape dimension")
        dimension = self.s.read_int("Error reading element shape dimension")
        if not 0 <= dimension <= 3:
            raise self.error(f"Invalid shape dimension {dimension}")
        self.requireModel()
        if self.exVersion < 2:
            raise self.unsupported("EX version 1 shapes are not supported")
        if dimension == 0:
            if (self.mesh is not None) or (self.nodeset is None):
                raise self.error("!#nodeset not specified before 0-D Shape")
        else:
            if self.mesh is None:
                raise self.error("!#mesh not specified before Shape")
            if self.mesh.dimension != dimension:
                raise self.error("Shape dimension does not match current !#mesh")
        if self.exVersion < 3:
            # v2: Shape at top level starts a new blank template
            self.clearTemplates()
            if self.mesh is not None:
                self.elementTemplate = ElementTemplate()
                self.elementTemplates.append(self.elementTemplate)
            else:
                self.nodeTemplate = _NodeTemplate()
                self.nodeTemplates.append(self.nodeTemplate)
        if dimension == 0:
            return  # node templates have no shape object
        self.s.match_literal(",")
        description = self.s.read_rest_of_line().strip()
        if description:
            factors = [f.strip() for f in description.split("*")]
            if len(factors) != dimension:
                raise self.error(
                    f"Shape description '{description}' does not match "
                    f"dimension {dimension}")
            for factor in factors:
                base = factor.split("(")[0].strip()
                if base in ("simplex", "polygon"):
                    raise self.unsupported(
                        f"Element shape '{description}' contains "
                        f"'{base}' which is not supported (declined by "
                        f"design)")
                if base != "line":
                    raise self.error(
                        f"Invalid shape description '{description}'")
        shape = ElementShape(dimension)
        self.elementTemplate.shape = shape

    def readBasis(self):
        """Read a basis description up to (but not consuming) the next
        ',' and return the parsed Basis; the caller eats the separator."""
        description = self.s.read_charset("^,").strip()
        if not description:
            raise self.error("Error reading basis description")
        return parse_basis(description, line=self.s.line)

    def readElementHeaderField(self):
        """One field of an element header: declaration then, per
        component, a basis line and its parameter map.

        Builds an :class:`ElementFieldTemplate` per component into
        ``self.elementTemplate.efts[(field name, component)]`` and
        appends the merged Field to ``template.fields``. Requires mesh,
        nodeset and element template to be set.

        Non-general (constant) fields and ``field based`` components get
        a synthetic constant-basis EFT with one empty function rather
        than a parsed map. For ``standard node based`` components the
        map is read function by function: a local node index expression
        (``0.`` meaning no terms, ``1.``, ``1+2.``), ``#Values=`` giving
        how many consecutive basis functions the following labels cover,
        a ``Value labels:`` line, and — only when the component names a
        scale factor set — a ``Scale factor indices:`` line. Duplicate
        fields in one header, out-of-range local node indexes and node
        counts exceeding ``#Nodes`` are errors; each EFT is validated
        before the next component. ``element based``/``grid based``
        mappings and legacy ``Value indices`` are declined.
        """
        if not (self.mesh is not None and self.nodeset is not None
                and self.elementTemplate is not None):
            raise self.error("Mesh/nodeset/element template not set")
        s = self.s
        template = self.elementTemplate
        field = self.readField()
        for existing in template.fields:
            if existing is field:
                raise self.error(
                    f"Field {field.name} appears more than once in header")
        dimension = self.mesh.dimension
        for c in range(field.number_of_components):
            s.skip_whitespace()
            componentName = s.read_charset("^.").strip()
            if not componentName:
                raise self.error(
                    f"Error reading component name for field {field.name}")
            if (field.component_names_declared
                    and field.component_names[c] != componentName):
                raise self.error(
                    f"Field {field.name} redeclared with different component "
                    f"name {componentName!r} (was "
                    f"{field.component_names[c]!r})")
            field.component_names[c] = componentName
            if s.getc() != ".":
                raise self.error("Missing required '.' after component name")
            if field.field_type != "general":
                # component name is sufficient; constant basis, field mapping
                eft = ElementFieldTemplate(constant_basis(dimension),
                                           mapping="field")
                eft.functions = [[]]
                template.efts[(field.name, c)] = eft
                continue
            s.match_literal(" ")
            basis = self.readBasis()
            s.match_literal(", ")
            modifyName = s.read_charset("^,").strip()
            if modifyName != "no modify":
                raise self.unsupported(
                    f"Basis modify function '{modifyName}' is not supported")
            s.match_literal(", ")
            mapTypeName = s.read_charset("^.").strip()
            s.match_literal(".")
            if mapTypeName == "standard node based":
                pass
            elif mapTypeName in ("element based", "grid based"):
                raise self.unsupported(
                    f"Element-based (grid) field values for field "
                    f"{field.name} are not supported (declined by design)")
            elif mapTypeName == "field based":
                eft = ElementFieldTemplate(constant_basis(dimension),
                                           mapping="field")
                eft.functions = [[]]
                template.efts[(field.name, c)] = eft
                continue
            else:
                raise self.error(
                    f"Invalid element parameter mapping mode '{mapTypeName}'")
            keyValueMapBase = s.read_key_value_map()
            scaleFactorSetName = keyValueMapBase.get("scale factor set")
            sfSet = None
            if scaleFactorSetName is not None:
                sfSet = template.find_scale_factor_set(scaleFactorSetName)
                if sfSet is None:
                    raise self.error(
                        f"Could not find scale factor set "
                        f"{scaleFactorSetName}")
            eft = ElementFieldTemplate(basis, mapping="node",
                                       scale_factor_set=sfSet)
            template.efts[(field.name, c)] = eft
            if not s.match_literal(" #Nodes="):
                raise self.error(
                    f"Error reading field {field.name} component "
                    f"{componentName} number of nodes")
            nodeCount = s.read_int("Invalid #Nodes")
            s.read_key_value_map(",")
            scaleFactorCount = sfSet.count if sfSet else 0
            scaleFactorOffset = sfSet.offset if sfSet else 0

            functionCount = basis.number_of_functions
            fn = 1
            while fn <= functionCount:
                # one or more local node indexes summing terms:
                # "0." (zero terms), "1.", "1+2.", "3+1+3."
                termNodes = []
                while True:
                    s.skip_whitespace()
                    nodeIndexString = s.read_charset("^.+").strip()
                    next_char = s.next_non_space_char()
                    try:
                        nodeIndex = int(nodeIndexString)
                    except ValueError:
                        nodeIndex = -1
                    if (nodeIndex < 0 or nodeIndex > template.node_count
                            or (nodeIndex == 0 and termNodes)):
                        raise self.error(
                            f"Node index '{nodeIndexString}' invalid or "
                            f"out of range")
                    if nodeIndex == 0:
                        if next_char != ".":
                            raise self.error(
                                "Node index 0 (zero terms) must be on its "
                                "own ('0.')")
                        break
                    termNodes.append(nodeIndex)
                    if next_char == ".":
                        break
                    if next_char != "+":
                        raise self.error(
                            "Invalid character in local node index "
                            "expression")
                termCount = len(termNodes)
                if not s.match_literal(" #Values="):
                    raise self.error("Invalid #Values")
                valueCount = s.read_int("Invalid #Values")
                if valueCount < 1:
                    raise self.error("Invalid #Values")
                if fn + valueCount - 1 > functionCount:
                    raise self.error(
                        "#Values would exceed number of basis functions")
                dofMappingType = s.read_charset("^:")
                if "Value labels" not in dofMappingType:
                    if "Value indices" in dofMappingType:
                        raise self.unsupported(
                            "Legacy 'Value indices' mapping (EX version 1) "
                            "is not supported")
                    raise self.error(
                        'Missing "Value labels:" token')
                s.match_literal(": ")
                rest = s.read_rest_of_line()
                terms_per_function = self._parseValueLabelExpressions(
                    rest, valueCount, termCount, termNodes)
                # scale factor indices, only if scaling in EX Version 2+
                if scaleFactorCount > 0:
                    if not (s.match_literal(" Scale factor indices ")
                            and s.match_one_of(":")):
                        raise self.error(
                            'Missing "Scale factor indices:" token')
                    s.match_literal(" ")
                    rest = s.read_rest_of_line()
                    self._parseScaleFactorExpressions(
                        rest, valueCount, terms_per_function,
                        scaleFactorOffset, scaleFactorCount)
                for v in range(valueCount):
                    eft.set_function_terms(fn + v - 1, terms_per_function[v])
                fn += valueCount
            if nodeCount and max(
                    (t.local_node for terms in eft.functions if terms
                     for t in terms), default=0) > nodeCount:
                raise self.error("Too many nodes referenced")
            eft.validate()
        field.component_names_declared = True
        template.fields.append(field)

    def _parseValueLabelExpressions(self, rest, valueCount, termCount,
                                    termNodes):
        """Parse e.g. ``value d/ds1(2) zero d/ds1(2)+d/ds2(3)``.

        Returns list per function of list of Terms (without scaling).
        """
        tk = LineTokenizer(rest)
        terms_per_function = []
        termLimit = termCount if termCount > 0 else 1
        for _v in range(valueCount):
            terms = []
            for t in range(1, termLimit + 1):
                token, nextchar = tk.next_token("(+")
                if termCount <= 1 and token == "zero":
                    if nextchar not in ("", " ") and not nextchar.isspace():
                        raise self.error(
                            f"Invalid character '{nextchar}' after value "
                            f"label 'zero'")
                    terms = []
                elif termCount == 0:
                    raise self.error(
                        "Require value label 'zero' when there are no terms")
                else:
                    if token not in VALUE_LABELS:
                        raise self.error(
                            f"Invalid node value/derivative label '{token}'")
                    version = 1
                    if nextchar == "(":
                        versionToken, nextchar = tk.next_token(")")
                        try:
                            version = int(versionToken)
                        except ValueError:
                            version = 0
                        if nextchar != ")" or version < 1:
                            raise self.error(
                                "Invalid version number specification")
                        tk.skip_spaces()
                        nextchar = tk.peek()
                        if nextchar == "+":
                            tk.pos += 1
                    terms.append(Term(termNodes[t - 1], token, version))
                if t < termCount:
                    if nextchar != "+":
                        raise self.error(
                            "Require '+' followed by additional term(s)")
                elif nextchar == "+":
                    raise self.error(
                        f"Too many terms, unexpected character '{nextchar}'")
            terms_per_function.append(terms)
        token, _ = tk.next_token("")
        if token:
            raise self.error(f"Unexpected text '{token}' after labels")
        return terms_per_function

    def _parseScaleFactorExpressions(self, rest, valueCount,
                                     terms_per_function, offset, count):
        """Parse e.g. ``0`` / ``60`` / ``1*2`` / ``3*4+1*2*3`` per term.

        Indices in the file are 1-based across the whole element template;
        stored 0-based into the element's concatenated array.
        """
        tk = LineTokenizer(rest)
        for v in range(valueCount):
            terms = terms_per_function[v]
            termLimit = len(terms) if terms else 1
            for t in range(1, termLimit + 1):
                indexes = []
                while True:
                    token, nextchar = tk.next_token("*+")
                    try:
                        scaleFactorIndex = int(token)
                    except ValueError:
                        scaleFactorIndex = -1
                    valid = ((not indexes and scaleFactorIndex == 0)
                             or (offset < scaleFactorIndex <= offset + count))
                    if scaleFactorIndex < 0 or not valid:
                        raise self.error(
                            f"Invalid scale factor index '{token}' for "
                            f"scale factor set")
                    if scaleFactorIndex == 0:
                        if nextchar == "*":
                            raise self.error(
                                "Scale factor index 0 (no scaling) cannot "
                                "be used in product with other scale factors")
                        break
                    indexes.append(scaleFactorIndex - 1)  # store 0-based global
                    if nextchar != "*":
                        break
                if t <= len(terms):
                    terms[t - 1].scale_factor_indices = tuple(indexes)
                if t < len(terms):
                    if nextchar != "+":
                        raise self.error(
                            "Require '+' followed by additional term(s)")
                elif nextchar not in ("", "+") and not nextchar.isspace():
                    raise self.error(f"Unexpected character '{nextchar}'")

    def readElementHeader(self):
        """'#' consumed: #Scale factor sets / #Nodes / #Fields + fields."""
        if self.exVersion < 3:
            if (self.elementTemplate is None
                    or self.elementTemplate.shape is None):
                raise self.error("No element shape set")
            # v2: start a new blank template with the last element shape
            lastShape = self.elementTemplate.shape
            self.clearTemplates()
            self.elementTemplate = ElementTemplate()
            self.elementTemplate.shape = lastShape
            self.elementTemplates.append(self.elementTemplate)
        elif self.elementTemplate is None:
            raise self.error(
                "Define element template needed before this in EX version 3+")
        template = self.elementTemplate
        if not self.s.match_literal("Scale factor sets="):
            raise self.error("Error reading #Scale factor sets")
        scaleFactorSetCount = self.s.read_int(
            "Error reading #Scale factor sets")
        scaleFactorOffset = 0
        for _ in range(scaleFactorSetCount):
            name = self.s.read_ex_string(",\n\r\t")
            keyValueMap = self.s.read_key_value_map(",")
            countString = keyValueMap.get("#Scale factors")
            if countString is None:
                raise self.error("Missing #Scale factors")
            count = int(countString)
            if count <= 0:
                raise self.error("Must have positive #Scale factors")
            if template.find_scale_factor_set(name):
                raise self.error(f"Scale factor set {name} already defined")
            template.scale_factor_sets.append(ScaleFactorSet(
                name, count, scaleFactorOffset,
                identifiers=keyValueMap.get("identifiers")))
            scaleFactorOffset += count
        if not self.s.match_literal(" #Nodes="):
            raise self.error("Error reading #Nodes")
        template.node_count = self.s.read_int("Error reading #Nodes")
        if not self.s.match_literal(" #Fields="):
            raise self.error("Error reading number of fields")
        fieldCount = self.s.read_int("Error reading number of fields")
        for _ in range(fieldCount):
            self.readElementHeaderField()

    def findElementTemplateByName(self, name):
        """Element template of that name in the current mesh's scope, or
        None. Templates from other domains have been cleared away."""
        for et in self.elementTemplates:
            if et.name == name:
                return et
        return None

    def readDefineElementTemplate(self):
        """'D' consumed: the whole ``Define element template:`` block —
        name, ``Shape.``, ``#Scale factor sets``/``#Nodes``/``#Fields``
        with its field entries, and an optional trailing ``Values:``.

        The template is created and registered but left *deactivated* on
        exit (``self.elementTemplate`` is None); a later ``Element
        template:`` line must name it. Redefining a name is an error,
        and the block is not allowed inside a group definition.
        """
        if not (self.s.match_literal("efine element template ")
                and self.s.match_one_of(":")):
            raise self.error("Truncated Define element template: token")
        if self.fieldGroup is not None:
            raise self.error(
                "Define element template not allowed in group definition")
        name = self.s.read_ex_string(",\n\r\t")
        self.s.read_key_value_map(",")
        if self.findElementTemplateByName(name):
            raise self.error(f"Element template {name} already exists")
        self.elementTemplate = ElementTemplate(name)
        self.elementTemplates.append(self.elementTemplate)
        self.s.skip_whitespace()
        if self.s.getc() != "S":
            raise self.error("Missing Shape")
        self.readElementShape()
        self.s.skip_whitespace()
        if self.s.getc() != "#":
            raise self.error("Missing #Scale factor sets")
        self.readElementHeader()
        self.s.skip_whitespace()
        if self.s.peekc() == "V":
            self.s.getc()
            self.readFieldValues()
        self.elementTemplate = None  # must be separately activated

    def readElementTemplate(self):
        """``Element template:`` consumed: name (plus ignored key=value
        trailer) of an already-defined template, which becomes active.
        Not permitted inside a group definition."""
        if self.fieldGroup is not None:
            raise self.error(
                "Element template not allowed in group definition")
        name = self.s.read_ex_string(",\n\r\t")
        self.s.read_key_value_map(",")
        template = self.findElementTemplateByName(name)
        if template is None:
            raise self.error(
                f"Element template {name} not found in scope of current mesh")
        self.elementTemplate = template

    # ------------------------------------------------------ element data

    def readElementIdentifier(self):
        """One element number. EX version 2+ writes a plain integer, so
        this is a bare int read; it is also used for face identifiers,
        where -1 is legal and means an absent face."""
        return self.s.read_int("Error reading element identifier")

    def readElement(self):
        """``Element:`` consumed: identifier then the optional
        ``Faces:``, ``Nodes:`` and ``Scale factors:`` sections.

        Creates or merges the element in the current mesh and returns
        it. With no active template it returns immediately after
        get-or-create — that is how a group lists existing elements
        without repeating headers.

        The counts are taken from the shape and template, never from
        counting tokens: ``shape.face_count`` faces, ``template.
        node_count`` nodes, ``template.total_scale_factors`` reals. A
        face identifier of -1 is an absent face that still occupies its
        slot (stored as None); referenced faces and nodes are
        get-or-created in the (d-1) mesh and the current nodeset
        respectively, so a file may mention them before defining them.
        Redefining an element with a different shape dimension is an
        error, and a template carrying element-based values is declined.
        """
        if (self.fieldGroup is not None) and (self.exVersion >= 3):
            raise self.error(
                "Element not allowed in group definition; expecting "
                "Element group")
        if self.mesh is None:
            raise self.error("Can't read element as no mesh set")
        elementIdentifier = self.readElementIdentifier()
        if elementIdentifier < 0:
            raise self.error("Negative element identifier is not permitted")
        template = self.elementTemplate
        existing = elementIdentifier in self.mesh
        if not existing and template is None:
            raise self.error(
                "Element not found, must be first defined with a shape and "
                "field header")
        element = self.mesh.get_or_create(elementIdentifier)
        if template is None:
            return element  # listing elements in a group without headers
        if element.shape is not None and element.template is not None \
                and element.shape.dimension != template.shape.dimension:
            raise self.error(
                f"Element {elementIdentifier} redefined with different shape")
        element.shape = template.shape
        element.template = template

        # Faces: (optional). The count comes from the element shape.
        if self.s.match_literal(" Faces") and self.s.match_one_of(":"):
            faceCount = element.shape.face_count
            model = self.requireModel()
            if self.mesh.dimension < 2 and faceCount > 0:
                raise self.error("Failed to find face mesh")
            faceMesh = model.get_or_create_element_mesh(
                self.mesh.dimension - 1)
            faces = []
            faceShape = element.shape.face_shape()
            for _ in range(faceCount):
                faceIdentifier = self.readElementIdentifier()
                if faceIdentifier == -1:
                    # absent face still occupies its slot
                    faces.append(None)
                    continue
                if faceIdentifier < 0:
                    raise self.error(
                        "Negative face identifier is not permitted")
                faceMesh.get_or_create(faceIdentifier, shape=faceShape)
                faces.append(faceIdentifier)
            element.faces = faces

        # Values: only for element-based fields, which are declined, so a
        # Values section here means an unsupported construct slipped by.
        if template.has_element_values:
            raise self.unsupported(
                "Element field values are not supported (declined by design)")

        # Nodes: if any in element header
        if template.node_count > 0:
            if not (self.s.match_literal(" Nodes")
                    and self.s.match_one_of(":")):
                raise self.error(
                    'Truncated read of required " Nodes:" token in element')
            nodes = []
            for _ in range(template.node_count):
                nodeIdentifier = self.s.read_int(
                    "Error reading node identifier")
                if nodeIdentifier < -1:
                    raise self.error(
                        f"Invalid node identifier {nodeIdentifier}")
                if nodeIdentifier >= 0:
                    self.nodeset.get_or_create(nodeIdentifier)
                    nodes.append(nodeIdentifier)
                else:
                    nodes.append(None)
            element.nodes = nodes

        # Scale factors: if any scale factor sets in element header
        if template.scale_factor_sets:
            if not (self.s.match_literal(" Scale factors")
                    and self.s.match_one_of(":")):
                raise self.error(
                    'Truncated read of required " Scale factors:" token in '
                    'element')
            total = template.total_scale_factors
            scaleFactors = np.empty(total)
            for i in range(total):
                scaleFactors[i] = self._readFiniteReal("scale factor")
            element.scale_factors = scaleFactors
        return element

    def readElementGroup(self):
        """``Element group:`` consumed: compact identifier ranges added
        to the current group at the current mesh's dimension.

        Membership only — every identifier must already exist in the
        mesh, otherwise it is an error; a group never creates elements.
        """
        if self.fieldGroup is None:
            raise self.error(
                "Element group: may only be used within a Group definition")
        if self.mesh is None:
            raise self.error(
                "Region/Group/mesh must be set before reading element group")
        group = self.fieldGroup
        mesh = self.mesh

        def add(identifier):
            if identifier not in mesh:
                raise self.error(f"Element {identifier} not found")
            group.add_element(mesh.dimension, identifier)

        self._readIdentifierRanges(add, "element")

    def readElementOrTemplate(self):
        """'E' consumed and next is 'l'."""
        token = self.s.read_charset("^:").rstrip()
        isElement = token == "lement"
        isElementGroup = token == "lement group"
        isElementTemplate = token == "lement template"
        if ((self.exVersion < 3) and not isElement) or not (
                isElement or isElementGroup or isElementTemplate):
            raise self.error(f"Unrecognised token E{token}")
        if not self.s.match_one_of(":"):
            raise self.error("Missing : separator")
        if isElement:
            element = self.readElement()
            if self.fieldGroup is not None:
                self.fieldGroup.add_element(self.mesh.dimension,
                                            element.identifier)
            return
        if isElementGroup:
            self.readElementGroup()
            return
        self.readElementTemplate()

    # -------------------------------------------------------- region/group

    def readRegionOrGroupName(self, first_char):
        """'R' or 'G' consumed (passed in as ``first_char``): the rest of
        ``Region: /`` or ``Group name: NAME``.

        ``Region:`` creates the Mesh — it is what makes ``requireModel``
        succeed — clears any current group and selects the ``nodes``
        nodeset. Only the root path ``/`` is accepted; a sub-region, or a
        second ``Region:`` block, is declined as unsupported.

        ``Group name:`` gets or creates the group and makes it current,
        which switches later Node/Element constructs into membership
        mode. Below EX version 3 a group may legally precede any
        ``Region:``, so the model is created implicitly here.
        """
        s = self.s
        if first_char == "R":
            if not (s.match_literal("egion ") and s.match_one_of(":")):
                raise self.error("Truncated 'Region :' token")
        else:
            if not (s.match_literal("roup name ") and s.match_one_of(":")):
                raise self.error("Truncated 'Group name :' token")
        name = s.read_rest_of_line().strip(" \t")
        if first_char == "R":
            if not name.startswith("/"):
                raise self.error(
                    f"Missing '/' at start of region path '{name}'")
            if self.model is not None:
                raise self.unsupported(
                    "Multiple Region: blocks are not supported "
                    "(declined by design)")
            if name != "/":
                raise self.unsupported(
                    f"Sub-region '{name}' is not supported; only the root "
                    f"region '/' (declined by design)")
            self.model = Mesh(name)
            self.model.ex_version = self.exVersion
            self.fieldGroup = None
            self.setNodeset(self.model.nodesets["nodes"])
        else:
            if self.model is None:
                if self.exVersion >= 3:
                    raise self.error(
                        "Region must be set before Group name in EX "
                        "version 3+")
                self.model = Mesh("/")
                self.model.ex_version = self.exVersion
                self.setNodeset(self.model.nodesets["nodes"])
            self.fieldGroup = self.model.get_or_create_group(name)

    # -------------------------------------------------------------- read

    def read(self):
        """Top-level dispatch on the first character of each token."""
        s = self.s
        while True:
            s.skip_whitespace()
            first = s.getc()
            if first == "":
                break  # end of file
            if self.exVersion == 1 and first not in ("E", "!"):
                raise self.unsupported(
                    "File does not start with 'EX Version:' — EX version 1 "
                    "(legacy) files are not supported")
            if first in ("R", "G"):
                self.readRegionOrGroupName(first)
            elif first == "S":
                if self.exVersion < 3:
                    self.readElementShape()
                else:
                    raise self.error(
                        "Shape must be within Define template in EX "
                        "version 3+")
            elif first == "!":
                self.readCommentOrDirective()
            elif first == "#":
                if self.exVersion < 3:
                    if self.mesh is not None:
                        self.readElementHeader()
                    elif self.nodeset is not None:
                        self.readNodeHeader()
                    else:
                        raise self.error(
                            "Region/Group not set before field header")
                else:
                    raise self.error(
                        "#tokens must be within Define template in EX "
                        "version 3+")
            elif first == "D":
                if self.exVersion < 3:
                    raise self.error(f"Invalid token 'D' in EX version "
                                     f"{self.exVersion} file")
                if self.mesh is not None:
                    self.readDefineElementTemplate()
                elif self.nodeset is not None:
                    self.readDefineNodeTemplate()
                else:
                    raise self.error("Nodeset/Mesh not set before Define")
            elif first == "N":
                self.readNodeOrTemplate()
            elif first == "E":
                if s.peekc() == "l":
                    self.readElementOrTemplate()
                elif s.peekc() == "X":
                    self.readEXVersion()
                else:
                    rest = s.read_rest_of_line()
                    raise self.error(f"Invalid token 'E{rest}'")
            elif first == "T":
                if self.exVersion < 3:
                    rest = s.read_rest_of_line()
                    raise self.error(f"Invalid token 'T{rest}'")
                self.readTimeSequence()
            elif first == "V":
                if self.exVersion < 3:
                    self.readFieldValues()
                else:
                    raise self.error(
                        "Field Values must be at end of Define template in "
                        "EX version 3+")
            else:
                rest = s.read_rest_of_line()
                raise self.error(
                    f"Invalid token '{first}{rest}' in EX version "
                    f"{self.exVersion} file")
        if self.model is None:
            raise ExSyntaxError("Empty or truncated EX file: no Region/Group")
        return self.model


def loads(text):
    """Read EX content from a string, returning a Mesh."""
    return EXReader(text).read()


def load(path):
    """Read an EX/EXF file, returning a Mesh."""
    with open(path, encoding="utf8") as fh:
        return loads(fh.read())
