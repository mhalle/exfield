"""Generate golden reference data from Zinc for comparison with exfield.

Run with a python that has cmlibs.zinc installed (this is the only place
Zinc is needed; exfield itself depends only on NumPy):

    uv run --no-project --python 3.12 --with cmlibs.zinc --with numpy \\
        python tests/golden_zinc.py <file.exf> <out.json>

For each element mesh dimension, samples field values and first
xi-derivatives at a fixed grid of (element, xi) for every real-valued
field, integrates arclength/area/volume per dimension with 4-point Gauss
(Zinc's converged order), and records group membership sizes and marker
data.
"""

import json
import sys

from cmlibs.zinc.context import Context
from cmlibs.zinc.field import Field
from cmlibs.zinc.result import RESULT_OK

XI_1D = [[0.0], [0.25], [0.5], [0.75], [1.0]]
XI_2D = [[a, b] for a in (0.0, 0.3, 1.0) for b in (0.0, 0.7, 1.0)]
XI_3D = [[a, b, c] for a in (0.0, 0.4, 1.0) for b in (0.0, 0.6, 1.0)
         for c in (0.0, 0.5, 1.0)]
XIS = {1: XI_1D, 2: XI_2D, 3: XI_3D}


def main(path, out_path):
    context = Context("golden")
    region = context.getDefaultRegion()
    assert region.readFile(path) == RESULT_OK, f"Zinc could not read {path}"
    fm = region.getFieldmodule()
    cache = fm.createFieldcache()

    result = {"file": path, "fields": {}, "meshes": {}, "groups": {},
              "integrals": {}}

    # real-valued fields
    field_names = []
    it = fm.createFielditerator()
    f = it.next()
    while f.isValid():
        if f.getValueType() == Field.VALUE_TYPE_REAL and f.isManaged():
            field_names.append(f.getName())
        f = it.next()
    result["fields"] = {n: fm.findFieldByName(n).getNumberOfComponents()
                        for n in field_names}

    for dim in (1, 2, 3):
        mesh = fm.findMeshByDimension(dim)
        n = mesh.getSize()
        if n == 0:
            continue
        mesh_data = {"size": n, "samples": []}
        # sample a deterministic subset of elements
        el_iter = mesh.createElementiterator()
        element = el_iter.next()
        ids = []
        while element.isValid():
            ids.append(element.getIdentifier())
            element = el_iter.next()
        ids.sort()
        step = max(1, len(ids) // 12)
        sample_ids = ids[::step][:12]
        for eid in sample_ids:
            element = mesh.findElementByIdentifier(eid)
            for name in field_names:
                field = fm.findFieldByName(name)
                ncomp = field.getNumberOfComponents()
                for xi in XIS[dim]:
                    cache.setMeshLocation(element, xi)
                    res, values = field.evaluateReal(cache, ncomp)
                    if res != RESULT_OK:
                        continue
                    entry = {"element": eid, "field": name, "xi": xi,
                             "values": values if isinstance(values, list)
                             else [values]}
                    derivs = []
                    ok = True
                    for d in range(1, dim + 1):
                        dfield = fm.createFieldDerivative(field, d)
                        cache.setMeshLocation(element, xi)
                        res, dv = dfield.evaluateReal(cache, ncomp)
                        if res != RESULT_OK:
                            ok = False
                            break
                        derivs.append(dv if isinstance(dv, list) else [dv])
                    if ok:
                        entry["derivatives"] = derivs
                    mesh_data["samples"].append(entry)
        result["meshes"][str(dim)] = mesh_data

    # group sizes and per-group 1-D arclength integral of coordinates
    coordinates = fm.findFieldByName("coordinates")
    it = fm.createFielditerator()
    f = it.next()
    while f.isValid():
        group = f.castGroup()
        if group.isValid():
            gname = f.getName()
            ginfo = {}
            measure_key = {1: "arclength_gauss4", 2: "area_gauss4",
                           3: "volume_gauss4"}
            for dim in (1, 2, 3):
                mesh = fm.findMeshByDimension(dim)
                mg = group.getMeshGroup(mesh)
                if mg.isValid() and mg.getSize() > 0:
                    ginfo[f"mesh{dim}d"] = mg.getSize()
                    if coordinates.isValid():
                        one = fm.createFieldConstant([1.0])
                        integral = fm.createFieldMeshIntegral(
                            one, coordinates, mg)
                        integral.setNumbersOfPoints([4])
                        cache2 = fm.createFieldcache()
                        res, value = integral.evaluateReal(cache2, 1)
                        if res == RESULT_OK:
                            ginfo[measure_key[dim]] = value
            nodes = fm.findNodesetByName("nodes")
            ng = group.getNodesetGroup(nodes)
            if ng.isValid() and ng.getSize() > 0:
                ginfo["nodes"] = ng.getSize()
            if ginfo:
                result["groups"][gname] = ginfo
        f = it.next()

    with open(out_path, "w") as fh:
        json.dump(result, fh)
    print(f"wrote {out_path}: "
          f"{sum(len(m['samples']) for m in result['meshes'].values())} "
          f"samples, {len(result['groups'])} groups")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
