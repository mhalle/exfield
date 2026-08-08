"""Template fingerprinting.

``(element, xi)`` names the same anatomy only across models built from
the same scaffold type AND the same discretisation options. Measured on
'3D Heart 1' (EXFIELD_PORTING_SPEC.md §8): geometry parameters moved a
fixed (element, xi) up to 0.17 in world space but preserved anatomical
labels on all 392 elements; changing just two discretisation options
silently destroyed correspondence for 64% of elements.

Nothing in the EX format records which options generated a mesh, so this
is a convention, not an algorithm: write a fingerprint into every mesh
you produce, and check it before any cross-model comparison. Mismatch
raises, not warns.
"""

import hashlib
import json

from .errors import FingerprintMismatch


def make_fingerprint(scaffold_type, version, options):
    """Build a fingerprint dict.

    Parameters
    ----------
    scaffold_type : e.g. "3D Heart 1"
    version : the scaffold/scaffoldmaker version string
    options : the full discretisation option dict used to generate the
        mesh. Hashed canonically (sorted keys); the raw options are also
        kept for inspection.
    """
    canonical = json.dumps(options, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf8")).hexdigest()[:16]
    return {
        "scaffold_type": scaffold_type,
        "version": version,
        "options_hash": digest,
        "options": options,
    }


def check_fingerprints(a, b):
    """Raise FingerprintMismatch unless a and b are comparable.

    Two None fingerprints pass with no check (nothing recorded — the
    caller takes responsibility). One-sided None fails: if one mesh
    records provenance and the other doesn't, comparability is unknown.
    """
    if a is None and b is None:
        return
    if a is None or b is None:
        raise FingerprintMismatch(
            "one mesh has a template fingerprint and the other does not; "
            "cross-model (element, xi) comparison is unverifiable")
    keys = ("scaffold_type", "version", "options_hash")
    for k in keys:
        if a.get(k) != b.get(k):
            raise FingerprintMismatch(
                f"template fingerprints differ in {k}: {a.get(k)!r} != "
                f"{b.get(k)!r}. (element, xi) addresses are not comparable "
                f"across models built with different templates or "
                f"discretisation options.")


def eft_signature(eft, scale_factors=None):
    """Canonical serializable form of an element field template.

    Structural identity of an EFT: basis, per-function term lists
    (local node, value label, version, scale-factor index tuple) and —
    when given — the element's scale-factor values. Two elements whose
    signatures compare equal are instances of the same generation
    pattern. Promoted from the vagus pilot's extractor; organ-agnostic.
    """
    return {
        "basis": eft.basis.description,
        "functions": [
            [[t.local_node, t.label, t.version,
              list(t.scale_factor_indices)] for t in terms]
            for terms in eft.functions
        ],
        "scale_factors": [float(v) for v in scale_factors]
        if scale_factors is not None else None,
    }
