"""exfield — read, evaluate and write OpenCMISS-Zinc EX/EXF scaffolds
with NumPy as the only dependency.

Zinc *makes* scaffolds; exfield lets downstream NumPy/VTK code consume
them without inheriting a 63 MB OpenGL-linked binary. Fitting stays in
``scaffoldfitter``, mesh generation in ``scaffoldmaker``; exfield covers
the scientific core: the EX reader and writer, field evaluation at
``(element, xi)``, arclength by quadrature, inverse mapping, and
embedded point sets that survive refitting.

Quick start::

    import exfield
    mesh = exfield.load("scaffold.exf")
    print(mesh.summary())          # check mesh.skipped is empty!
    ev = exfield.Evaluator(mesh.fields["coordinates"], dimension=1)
    x = ev.evaluate(element_id=5, xi=[0.5])
    table = exfield.ArclengthTable.build(ev, element_ids=[...])
    print(table.total)

An :class:`~exfield.errors.UnsupportedExFeature` exception is a scope
boundary, not necessarily a bug: EX version 1, multiple regions, time
sequences, simplex/polygon shapes, grid field values and indexed fields
are declined at the point of encounter so a file never loads half-read.
"""

import gzip as _gzip

from .basis import TensorProductBasis, parse_basis
from .bezier import hermite_to_bezier_1d, hermite_to_bezier_matrix
from .embedded import ArclengthValues, EmbeddedPoints
from .errors import (EvaluationError, ExError, ExSyntaxError,
                     FingerprintMismatch, UnsupportedExFeature)
from .coordinates import to_rectangular_cartesian
from .evaluate import ArclengthTable, Evaluator, build_chain, integrate
from .exreader import EXReader, loads
from .exwriter import EXWriter, dump, dumps
from .vtu import export_markers_vtu, export_vtu
from .fingerprint import check_fingerprints, eft_signature, make_fingerprint
from .inverse import Location, closest_point, find_location
from .mesh import (Element, ElementFieldTemplate, ElementShape,
                   ElementTemplate, Field, Group, Mesh, Node,
                   NodeFieldTemplate, Nodeset, ScaleFactorSet, Term,
                   VALUE_LABELS)

__version__ = "0.2.0"

__all__ = [
    "load", "loads", "dump", "dumps",
    "export_vtu", "export_markers_vtu",
    "Mesh", "Field", "Node", "Nodeset", "Element", "ElementShape",
    "ElementTemplate", "ElementFieldTemplate", "ScaleFactorSet", "Term",
    "NodeFieldTemplate", "Group", "VALUE_LABELS",
    "EXReader", "EXWriter",
    "Evaluator", "ArclengthTable", "build_chain", "integrate",
    "to_rectangular_cartesian",
    "Location", "find_location", "closest_point",
    "EmbeddedPoints", "ArclengthValues",
    "TensorProductBasis", "parse_basis",
    "hermite_to_bezier_1d", "hermite_to_bezier_matrix",
    "make_fingerprint", "check_fingerprints", "eft_signature",
    "ExError", "ExSyntaxError", "UnsupportedExFeature", "EvaluationError",
    "FingerprintMismatch",
]


def load(path):
    """Read an EX/EXF file (gzip-compressed accepted), returning a Mesh."""
    with open(path, "rb") as fh:
        head = fh.read(2)
        fh.seek(0)
        data = fh.read()
    if head == b"\x1f\x8b":
        data = _gzip.decompress(data)
    return loads(data.decode("utf8"))
