"""Face inheritance: evaluating fields on elements that have no direct
field definition by mapping into a parent element.

In real scaffolds the 1-D and 2-D meshes are largely *faces* of the 3-D
elements: they appear in ``Faces:`` lists but carry no field header of
their own. Zinc evaluates fields on them by transforming the face xi
into the parent element's xi space (``face_to_element`` matrices in
``finite_element_shape.cpp``); exfield mirrors that.

Face numbering for tensor-product line shapes (verified against the
reference source): for each xi axis in ascending order, two faces — the
xi=0 face then the xi=1 face. So a cube's faces are
(xi1=0, xi1=1, xi2=0, xi2=1, xi3=0, xi3=1).

Face xi -> element xi mapping is **cyclic**, not lexicographic: the
face's xi columns are assigned to the element's other axes starting
after the face axis and wrapping. For a cube:

* xi1 faces: (u, v) -> (xi2, xi3) = (u, v)
* xi2 faces: (u, v) -> (xi3, xi1) = (u, v)   i.e. xi1 = v, xi3 = u
* xi3 faces: (u, v) -> (xi1, xi2) = (u, v)

Getting the xi2 face wrong is silent: geometry still looks plausible.
This module has a regression test against Zinc on real data.
"""

import numpy as np


def face_axis_value(face_index):
    """(axis, boundary value) for a face of a line-tensor shape."""
    return face_index // 2, float(face_index % 2)


def face_to_element_map(parent_dimension, face_index):
    """Affine map from face xi to parent xi: ``xi_parent = A @ u + b``.

    ``A`` is (parent_dimension, parent_dimension - 1); ``b`` has the face
    axis pinned at 0 or 1. Column assignment is cyclic per the reference
    implementation (see module docstring).
    """
    D = parent_dimension
    axis, value = face_axis_value(face_index)
    A = np.zeros((D, D - 1))
    b = np.zeros(D)
    b[axis] = value
    k = axis + 1
    if k >= D:
        k = 1
    for j in range(D):
        if j == axis:
            continue
        A[j, k - 1] = 1.0
        k += 1
        if k >= D:
            k = 1
    return A, b


class ParentMap:
    """Face element -> (parent element, face index) lookup per dimension.

    When a face is shared by several parents the first parent in
    ascending element order wins, mirroring Zinc's behaviour; for
    continuous fields either parent evaluates identically on the face.
    """

    def __init__(self, model):
        self.model = model
        self._maps = {}   # child dimension -> {face_id: (parent_id, face#)}

    def parent_of(self, dimension, element_id):
        """(parent_id, face_index) in dimension+1, or None."""
        m = self._maps.get(dimension)
        if m is None:
            m = {}
            parent_mesh = self.model.element_meshes.get(dimension + 1)
            if parent_mesh is not None:
                for pid in sorted(parent_mesh.elements):
                    parent = parent_mesh.elements[pid]
                    if parent.faces is None:
                        continue
                    for i, fid in enumerate(parent.faces):
                        if fid is not None and fid not in m:
                            m[fid] = (pid, i)
            self._maps[dimension] = m
        return m.get(element_id)
