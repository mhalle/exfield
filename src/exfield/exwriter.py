"""EX/EXF file writer.

Mirrors Zinc's ``EXWriter`` in
``src/finite_element/export_finite_element.cpp`` and emits EX Version 3
only (what current Zinc writes). Number formatting matches Zinc:
``%22.15e`` reals with a single space separator, five scale factors per
line, group identifier ranges wrapped at ten columns.

Scope matches the reader: no time sequences, no simplex/polygon shapes,
no grid fields, single region.
"""

import warnings

import numpy as np

from .mesh import Mesh

_NEEDS_QUOTING = set(" \t\n\r\f\v,;=#\"'\\$")


def make_valid_token(name):
    """Mirror of Zinc's ``make_valid_token``: quote the string if it is
    empty, is "/", or contains whitespace or ,;=#"'\\$; escape
    " ' \\ $ with a backslash inside quotes."""
    if name and name != "/" and not any(c in _NEEDS_QUOTING for c in name):
        return name
    out = ['"']
    for c in name:
        if c in "\"'\\$":
            out.append("\\")
        out.append(c)
    out.append('"')
    return "".join(out)


def _real(value):
    """One real in Zinc's output format: space + %22.15e (23 chars)."""
    return " %22.15e" % value


class _IdentifierRanges:
    """Compact ``1,3..7,22..150`` range writer (10-column wrapping)."""

    COLUMN_LIMIT = 10

    def __init__(self, identifiers):
        self.ranges = []
        for identifier in sorted(identifiers):
            if self.ranges and identifier == self.ranges[-1][1] + 1:
                self.ranges[-1][1] = identifier
            else:
                self.ranges.append([identifier, identifier])

    def write(self, out):
        columns = 0
        first = True
        for start, stop in self.ranges:
            if not first:
                if columns >= self.COLUMN_LIMIT:
                    out.append(",\n")
                    columns = 0
                else:
                    out.append(",")
            first = False
            if start == stop:
                out.append(str(start))
                columns += 1
            else:
                out.append(f"{start}..{stop}")
                columns += 2
        if self.ranges:
            out.append("\n")


class EXWriter:
    """Writes a :class:`~exfield.mesh.Mesh` as EX Version 3 text."""

    def __init__(self, model):
        if not isinstance(model, Mesh):
            raise TypeError("EXWriter takes an exfield Mesh")
        self.model = model
        self.out = []
        self.nodeTemplateNumber = 0
        self.elementTemplateNumber = 0
        # per-nodeset template bookkeeping (cleared on domain switch,
        # numbering file-global, mirroring Zinc)
        self.nodeTemplates = {}       # signature -> name
        self.nodeTemplate = None      # active signature
        self.elementTemplates = {}    # id(template) -> (name, sf names)
        self.elementTemplate = None   # active template object

    # ------------------------------------------------------------- write

    def write(self):
        """Serialise the whole model and return it as one string.

        Resets the output buffer, so calling it twice is safe. Order is
        fixed: ``nodes``, then the element meshes by ascending dimension,
        then ``datapoints``, then the groups. Datapoints come after the
        meshes because their element:xi fields name host elements, which
        should already be defined at that point.
        """
        self.out = []
        self.out.append("EX Version: 3\n")
        self.out.append("Region: /\n")
        self.writeNodeset("nodes")
        for dimension in sorted(self.model.element_meshes):
            self.writeMesh(dimension)
        self.writeNodeset("datapoints")
        for group in self.model.groups.values():
            self.writeGroup(group)
        return "".join(self.out)

    # ------------------------------------------------------ field header

    def writeFieldHeader(self, fieldIndex, field):
        """One ``N) name, cm type, ..., #Components=C`` header line,
        shared by node and element headers.

        ``fieldIndex`` is the 1-based position within this header, not
        anything stored on the field. Prolate/oblate coordinate systems
        always emit ``focus=``, defaulting to 1.0 when unset; element:xi
        fields always emit host mesh and host mesh dimension, falling
        back to a ``meshNd`` name. The fibre branch is a mirror of the
        reference's special case and emits the same ``, real`` either way.
        """
        o = self.out
        o.append(f"{fieldIndex}) {field.name}, {field.cm_type}")
        if field.field_type == "constant":
            o.append(", constant")
        if field.coordinate_system:
            o.append(f", {field.coordinate_system}")
            if field.coordinate_system in ("prolate spheroidal",
                                           "oblate spheroidal"):
                o.append(", focus=%22.15e" % (field.focus
                                              if field.focus is not None
                                              else 1.0))
        if not (field.coordinate_system == "fibre"
                and field.value_type == "real"):
            o.append(f", {field.value_type}")
        else:
            o.append(", real")
        o.append(f", #Components={field.number_of_components}")
        if field.value_type == "element_xi":
            host_name = field.host_mesh_name or \
                f"mesh{field.host_mesh_dimension}d"
            o.append(f", host mesh={make_valid_token(host_name)}"
                     f", host mesh dimension={field.host_mesh_dimension}")
        o.append("\n")

    def writeFieldValues(self, fields):
        """Trailing ``Values:`` block for the constant fields in
        ``fields``, one line each, in header order.

        Writes nothing at all — not even the token — when no field in
        the header is constant with values set.
        """
        constants = [f for f in fields
                     if f.field_type == "constant" and f.values is not None]
        if not constants:
            return
        self.out.append("Values:\n")
        for field in constants:
            for value in field.values:
                if field.value_type == "real":
                    self.out.append(_real(value))
                elif field.value_type == "integer":
                    self.out.append(f" {int(value)}")
                else:
                    self.out.append(f" {make_valid_token(str(value))}")
            self.out.append("\n")

    # ------------------------------------------------------------- nodes

    def _nodeSignature(self, node):
        sig = []
        for name, field in self.model.fields.items():
            if name in node.templates:
                sig.append((name, tuple(nft.signature()
                                        for nft in node.templates[name])))
        return tuple(sig)

    def writeNodeset(self, nodeset_name):
        """``!#nodeset NAME`` directive followed by its nodes, each
        preceded by its template line when the template changes.

        Emits nothing when the nodeset is missing or empty, with one
        exception: the model's constant fields are homed on ``nodes``,
        so if any exist that directive is still written and carries a
        node template that is defined but never activated — it exists
        only to hold the ``Values:`` block.

        Per-nodeset template bookkeeping is reset here (templates are
        not nameable across domains) but the ``nodeN`` numbering stays
        file-global, mirroring Zinc.
        """
        nodeset = self.model.nodesets.get(nodeset_name)
        constants = [f for f in self.model.fields.values()
                     if f.field_type == "constant" and f.values is not None]
        has_constants = nodeset_name == "nodes" and constants
        if (nodeset is None or len(nodeset) == 0) and not has_constants:
            return
        self.out.append(f"!#nodeset {make_valid_token(nodeset_name)}\n")
        self.nodeTemplates = {}
        self.nodeTemplate = None
        if has_constants:
            # constants are carried by a template that is never activated
            self.nodeTemplateNumber += 1
            name = f"node{self.nodeTemplateNumber}"
            self.out.append(f"Define node template: {name}\n")
            self.out.append("Shape. Dimension=0\n")
            self.out.append(f"#Fields={len(constants)}\n")
            for i, field in enumerate(constants):
                self.writeFieldHeader(i + 1, field)
                for c in range(field.number_of_components):
                    self.out.append(f" {field.component_names[c]}.\n")
            self.writeFieldValues(constants)
        if nodeset is None:
            return
        for identifier in sorted(nodeset.nodes):
            node = nodeset.nodes[identifier]
            self.writeNodeTemplate(node)
            self.writeNode(node)

    def writeNodeTemplate(self, node):
        """Ensure the template ``node`` needs is defined and active,
        emitting nothing if it is already the active one.

        Templates are pooled by value signature rather than by identity,
        so nodes that happen to agree on fields and value labels share
        one ``Define node template`` and later occurrences cost only a
        ``Node template:`` line. Header field order follows the model's
        field order, not the node's.

        Warns for a general component with no value labels: exfield
        reads such files back, but Zinc 4.2 crashes on them.
        """
        signature = self._nodeSignature(node)
        if signature == self.nodeTemplate:
            return
        name = self.nodeTemplates.get(signature)
        if name is None:
            self.nodeTemplateNumber += 1
            name = f"node{self.nodeTemplateNumber}"
            self.nodeTemplates[signature] = name
            o = self.out
            o.append(f"Define node template: {name}\n")
            o.append("Shape. Dimension=0\n")
            o.append(f"#Fields={len(signature)}\n")
            fieldIndex = 0
            for field_name, _nft_sigs in signature:
                field = self.model.fields[field_name]
                fieldIndex += 1
                self.writeFieldHeader(fieldIndex, field)
                nfts = node.templates[field_name]
                for c in range(field.number_of_components):
                    o.append(f" {field.component_names[c]}.")
                    if field.field_type == "general":
                        nft = nfts[c]
                        if nft.total_values == 0:
                            # Zinc 4.2 segfaults reading '#Values=0 ()'
                            # general node fields (found compiling marker
                            # fields with empty templates); exfield reads
                            # them, but warn so files stay interoperable.
                            warnings.warn(
                                f"Node field {field.name!r} component "
                                f"{field.component_names[c]!r} has no value "
                                f"labels; Zinc crashes reading such files. "
                                f"Give the template a 'value' label.",
                                stacklevel=4)
                        o.append(f" #Values={nft.total_values} (")
                        o.append(",".join(
                            label + (f"({versions})" if versions > 1 else "")
                            for label, versions in nft.labels))
                        o.append(")")
                    o.append("\n")
        self.out.append(f"Node template: {name}\n")
        self.nodeTemplate = signature

    def writeNode(self, node):
        """``Node: id`` and its values, one line per component.

        Assumes the matching template is already active. Only general
        fields contribute values — constant fields were emitted once in
        the template's ``Values:`` block. An element:xi component with no
        host writes ``-1`` followed by integer zeros for the xi slots;
        a real or integer component with no values writes no line at all.
        """
        o = self.out
        o.append(f"Node: {node.identifier}\n")
        for field_name, _sig in self._nodeSignature(node):
            field = self.model.fields[field_name]
            if field.field_type != "general":
                continue
            components = node.fields[field_name]
            if field.value_type == "element_xi":
                for element_id, xi in components:
                    if element_id is None:
                        o.append(" -1")
                        o.append(" 0" * len(xi))
                    else:
                        o.append(f" {element_id}")
                        for x in xi:
                            o.append(_real(x))
                    o.append("\n")
            elif field.value_type == "real":
                for values in components:
                    for value in values:
                        o.append(_real(value))
                    if len(values):
                        o.append("\n")
            elif field.value_type == "integer":
                for values in components:
                    for value in values:
                        o.append(f" {int(value)}")
                    if len(values):
                        o.append("\n")
            elif field.value_type == "string":
                for value in components:
                    o.append(f" {make_valid_token(value)}\n")

    # ---------------------------------------------------------- elements

    def writeMesh(self, dimension):
        """``!#mesh`` directive for one dimension followed by its
        elements, each preceded by its template line when it changes.

        Emits nothing for an empty mesh. ``face mesh=`` is only written
        above dimension 1. Elements with no template are skipped: those
        are placeholders created by a face reference or an element:xi
        host, and belong to whichever mesh actually defines them.
        Per-mesh template bookkeeping is reset here; the ``elementN``
        numbering stays file-global.
        """
        element_mesh = self.model.element_meshes[dimension]
        if len(element_mesh) == 0:
            return
        o = self.out
        o.append(f"!#mesh {make_valid_token(element_mesh.name)}"
                 f", dimension={dimension}")
        if dimension > 1:
            face_name = element_mesh.face_mesh_name or f"mesh{dimension - 1}d"
            o.append(f", face mesh={make_valid_token(face_name)}")
        o.append(f", nodeset={make_valid_token(element_mesh.nodeset_name)}")
        o.append("\n")
        self.elementTemplates = {}
        self.elementTemplate = None
        for identifier in sorted(element_mesh.elements):
            element = element_mesh.elements[identifier]
            if element.template is None:
                # placeholder element (face or element_xi host) — written
                # by its owner mesh only if it has its own definition
                continue
            self.writeElementTemplate(element.template)
            self.writeElement(element)

    def _eftDistinctNodeCount(self, eft):
        seen = set()
        for terms in eft.functions:
            for term in terms:
                seen.add(term.local_node)
        return len(seen)

    def writeElementTemplate(self, template):
        """Ensure ``template`` is defined and active, emitting nothing if
        it is already the active one.

        Unlike node templates these are pooled by object identity, not
        by value: two equal-but-distinct templates are written out
        twice. First sight also fixes the ``scaling1..N`` names for the
        template's scale factor sets, which the field components refer
        to by name.
        """
        if template is self.elementTemplate:
            return
        entry = self.elementTemplates.get(id(template))
        if entry is None:
            self.elementTemplateNumber += 1
            name = f"element{self.elementTemplateNumber}"
            sf_names = {id(s): f"scaling{i + 1}"
                        for i, s in enumerate(template.scale_factor_sets)}
            entry = (name, sf_names)
            self.elementTemplates[id(template)] = entry
            self._defineElementTemplate(template, name, sf_names)
        self.out.append(f"Element template: {entry[0]}\n")
        self.elementTemplate = template

    def _defineElementTemplate(self, template, name, sf_names):
        o = self.out
        o.append(f"Define element template: {name}\n")
        shape = template.shape
        o.append(f"Shape. Dimension={shape.dimension}, "
                 f"{shape.description}\n")
        o.append(f"#Scale factor sets={len(template.scale_factor_sets)}\n")
        for sfSet in template.scale_factor_sets:
            identifiers = sfSet.identifiers
            if identifiers is None:
                identifiers = ("element_patch("
                               + ",".join(["0"] * sfSet.count) + ")")
            o.append(f"  {sf_names[id(sfSet)]}, "
                     f"#Scale factors={sfSet.count}, "
                     f"identifiers=\"{identifiers}\"\n")
        o.append(f"#Nodes={template.node_count}\n")
        o.append(f"#Fields={len(template.fields)}\n")
        for fieldIndex, field in enumerate(template.fields, start=1):
            self.writeFieldHeader(fieldIndex, field)
            for c in range(field.number_of_components):
                self.writeElementHeaderFieldComponent(
                    template, field, c, sf_names)
        self.writeFieldValues(template.fields)

    def writeElementHeaderFieldComponent(self, template, field, c, sf_names):
        """One component entry of an element field header: basis line,
        ``#Nodes``, and the function-by-function parameter map.

        Non-general fields and field-mapped EFTs collapse to a single
        line with no map. Otherwise ``#Nodes`` is the count of *distinct*
        local nodes the EFT actually references, which can be below the
        template's node count.

        Consecutive functions are merged into one ``#Values=`` run when
        they share both an identical term local-node list and the same
        basis node — the run never spans basis nodes, mirroring Zinc's
        grouping. A function with no terms writes local node ``0`` and
        the label ``zero``. The ``Scale factor indices:`` line is emitted
        only when the EFT names a scale factor set, with indices
        converted back to the file's 1-based numbering (0 = no scaling).
        """
        o = self.out
        componentName = field.component_names[c]
        if field.field_type != "general":
            o.append(f" {componentName}.\n")
            return
        eft = template.efts[(field.name, c)]
        if eft.mapping == "field":
            o.append(f" {componentName}. constant, no modify, field based.\n")
            return
        o.append(f" {componentName}. {eft.basis.description}, no modify, "
                 f"standard node based.")
        if eft.scale_factor_set is not None:
            o.append(f" scale factor set="
                     f"{sf_names[id(eft.scale_factor_set)]}")
        o.append("\n")
        o.append(f"  #Nodes={self._eftDistinctNodeCount(eft)}\n")
        scaled = eft.scale_factor_set is not None
        # group consecutive functions with identical term node lists,
        # never spanning basis nodes (mirrors Zinc's grouping)
        functions = eft.functions
        basis = eft.basis
        fn = 0
        while fn < len(functions):
            nodes_key = tuple(t.local_node for t in functions[fn])
            basis_node = basis.function_node[fn]
            run = 1
            while (fn + run < len(functions)
                   and basis.function_node[fn + run] == basis_node
                   and tuple(t.local_node
                             for t in functions[fn + run]) == nodes_key):
                run += 1
            if nodes_key:
                o.append("  " + "+".join(str(n) for n in nodes_key))
            else:
                o.append("  0")
            o.append(f". #Values={run}\n")
            o.append("   Value labels:")
            for f2 in range(fn, fn + run):
                terms = functions[f2]
                if not terms:
                    o.append(" zero")
                else:
                    o.append(" " + "+".join(
                        t.label + (f"({t.version})" if t.version > 1 else "")
                        for t in terms))
            o.append("\n")
            if scaled:
                o.append("   Scale factor indices:")
                for f2 in range(fn, fn + run):
                    terms = functions[f2]
                    if not terms:
                        o.append(" 0")
                    else:
                        o.append(" " + "+".join(
                            "*".join(str(i + 1)
                                     for i in t.scale_factor_indices)
                            if t.scale_factor_indices else "0"
                            for t in terms))
                o.append("\n")
            fn += run

    def writeElement(self, element):
        """``Element: id`` and its ``Faces:``, ``Nodes:`` and ``Scale
        factors:`` sections.

        Assumes the element's template is already active. Each section
        is written only if the template or shape calls for it, since the
        reader takes its counts from there rather than from the line.
        Absent faces and nodes are written as -1. An element whose
        template has scale factor sets but no stored values gets a full
        run of zeros rather than a missing section; five reals per line.
        """
        o = self.out
        o.append(f"Element: {element.identifier}\n")
        template = element.template
        if element.faces is not None and element.shape.face_count > 0:
            o.append(" Faces:\n")
            for face in element.faces:
                o.append(f" {face if face is not None else -1}")
            o.append("\n")
        if template.node_count > 0:
            o.append(" Nodes:\n")
            for node_id in element.nodes:
                o.append(f" {node_id if node_id is not None else -1}")
            o.append("\n")
        if template.scale_factor_sets:
            o.append(" Scale factors:\n")
            count = 0
            values = element.scale_factors
            if values is None:
                values = np.zeros(template.total_scale_factors)
            for value in values:
                o.append(_real(value))
                count += 1
                if count % 5 == 0:
                    o.append("\n")
            if count % 5 != 0:
                o.append("\n")

    # ------------------------------------------------------------ groups

    def writeGroup(self, group):
        """``Group name:`` and the group's members as compact identifier
        ranges, nodes first then elements by ascending dimension.

        Each block re-emits the ``!#nodeset``/``!#mesh`` directive it
        applies to, because the group section comes after all the domain
        content and the reader's current domain has moved on. Empty
        per-domain sets are skipped. A dimension with no element mesh in
        the model still writes a synthesised ``meshNd`` name.
        """
        o = self.out
        o.append(f"Group name: {group.name}\n")
        for nodeset_name in ("nodes", "datapoints"):
            ids = group.nodes.get(nodeset_name)
            if not ids:
                continue
            o.append(f"!#nodeset {make_valid_token(nodeset_name)}\n")
            o.append("Node group:\n")
            _IdentifierRanges(ids).write(o)
        for dimension in sorted(group.elements):
            ids = group.elements[dimension]
            if not ids:
                continue
            element_mesh = self.model.element_meshes.get(dimension)
            mesh_name = (element_mesh.name if element_mesh
                         else f"mesh{dimension}d")
            o.append(f"!#mesh {make_valid_token(mesh_name)}"
                     f", dimension={dimension}")
            if dimension > 1:
                face_name = (element_mesh.face_mesh_name
                             if element_mesh and element_mesh.face_mesh_name
                             else f"mesh{dimension - 1}d")
                o.append(f", face mesh={make_valid_token(face_name)}")
            nodeset_name = (element_mesh.nodeset_name if element_mesh
                            else "nodes")
            o.append(f", nodeset={make_valid_token(nodeset_name)}\n")
            o.append("Element group:\n")
            _IdentifierRanges(ids).write(o)


def dumps(mesh):
    """Serialise a Mesh to EX Version 3 text."""
    return EXWriter(mesh).write()


def dump(mesh, path):
    """Write a Mesh to an EX/EXF file."""
    with open(path, "w", encoding="utf8") as fh:
        fh.write(dumps(mesh))
