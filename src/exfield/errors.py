"""Exceptions for exfield.

Two kinds of failure are distinguished deliberately:

* ``ExSyntaxError`` — the file is malformed. Always carries a line number.
* ``UnsupportedExFeature`` — the file is (probably) valid EX but uses a
  format feature outside exfield's scope. Raised at the point of encounter
  so a file never loads half-read. An exception of this type is a scope
  boundary, not necessarily a bug.
"""


class ExError(Exception):
    """Base class for all exfield errors."""


class ExSyntaxError(ExError):
    """Malformed EX input. Carries the 1-based line number where found."""

    def __init__(self, message, line=None):
        self.line = line
        if line is not None:
            message = f"{message} (line {line})"
        super().__init__(message)


class UnsupportedExFeature(ExError):
    """Valid EX format feature that exfield deliberately does not support.

    The declined list (see the Scope section of README.md): EX version 1
    (legacy, unversioned), multiple regions, time sequences, simplex and
    polygon element shapes, element-based (grid) field values, indexed
    fields.
    If you extend scope, extend this list rather than silently accepting.
    """

    def __init__(self, message, line=None):
        self.line = line
        if line is not None:
            message = f"{message} (line {line})"
        super().__init__(message)


class EvaluationError(ExError):
    """Raised when a field cannot be evaluated at the requested location."""


class FingerprintMismatch(ExError):
    """Raised when comparing addresses across meshes whose template
    fingerprints differ. ``(element, xi)`` addresses are only comparable
    across models built from the same scaffold template with the same
    discretisation options."""
