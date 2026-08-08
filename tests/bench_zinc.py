"""Benchmark Zinc (cmlibs.zinc) on the f001 vagus scaffold.

Run: uv run --no-project --python 3.12 --with cmlibs.zinc,numpy \\
        python tests/bench_zinc.py
Prints JSON timings to stdout (last line).
"""

import json
import os
import tempfile
import time

import numpy as np
from cmlibs.zinc.context import Context
from cmlibs.zinc.result import RESULT_OK

PATH = "tests/data/sub-f001_L_vagus_scaffold.exf"
N_EVAL = 20000
rng = np.random.default_rng(0)


def timeit(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


timings = {}

# ---- parse
context = Context("bench")


def load():
    region = context.getDefaultRegion().createChild(
        f"r{time.monotonic_ns()}")
    assert region.readFile(PATH) == RESULT_OK
    return region


timings["parse"], region = timeit(load)

fm = region.getFieldmodule()
coordinates = fm.findFieldByName("coordinates")
mesh3 = fm.findMeshByDimension(3)
mesh2 = fm.findMeshByDimension(2)
mesh1 = fm.findMeshByDimension(1)

# collect 3-D element handles
it = mesh3.createElementiterator()
elements = []
e = it.next()
while e.isValid():
    elements.append(e)
    e = it.next()

xis = rng.random((N_EVAL, 3)).tolist()
which = rng.integers(0, len(elements), N_EVAL).tolist()


# ---- bulk evaluation (fieldcache loop is Zinc's API for this)
def eval_values():
    cache = fm.createFieldcache()
    total = 0.0
    for i in range(N_EVAL):
        cache.setMeshLocation(elements[which[i]], xis[i])
        _, v = coordinates.evaluateReal(cache, 3)
        total += v[0]
    return total


timings["eval_20k"], _ = timeit(eval_values)


# ---- derivatives (3 directions)
dfields = [fm.createFieldDerivative(coordinates, d) for d in (1, 2, 3)]


def eval_derivs():
    cache = fm.createFieldcache()
    total = 0.0
    for i in range(0, N_EVAL, 4):   # 5k points x 3 derivatives
        cache.setMeshLocation(elements[which[i]], xis[i])
        for df in dfields:
            _, v = df.evaluateReal(cache, 3)
            total += v[0]
    return total


timings["derivs_5k"], _ = timeit(eval_derivs)


# ---- integrals: trunk length, epineurium area, whole volume, gauss 4
def group_mesh(name, mesh):
    field = fm.findFieldByName(name)
    return field.castGroup().getMeshGroup(mesh)


def integral(mesh_group):
    one = fm.createFieldConstant([1.0])
    f = fm.createFieldMeshIntegral(one, coordinates, mesh_group)
    f.setNumbersOfPoints([4])
    cache = fm.createFieldcache()
    res, v = f.evaluateReal(cache, 1)
    assert res == RESULT_OK
    return v


trunk = group_mesh("orientation anterior", mesh1)
epi = group_mesh("epineurium", mesh2)
timings["arclength_trunk"], _ = timeit(lambda: integral(trunk))
timings["area_epineurium"], _ = timeit(lambda: integral(epi))
timings["volume_mesh3d"], _ = timeit(lambda: integral(mesh3))

# ---- inverse mapping: 200 points onto the 3-D mesh
targets = []
cache = fm.createFieldcache()
for i in range(200):
    cache.setMeshLocation(elements[which[i]], xis[i])
    _, v = coordinates.evaluateReal(cache, 3)
    targets.append(v)


def find_locations():
    find = fm.createFieldFindMeshLocation(
        fm.createFieldConstant([0.0, 0.0, 0.0]), coordinates, mesh3)
    find.setSearchMode(find.SEARCH_MODE_NEAREST)
    n = 0
    cache2 = fm.createFieldcache()
    const = fm.createFieldConstant([0.0, 0.0, 0.0])
    find2 = fm.createFieldFindMeshLocation(const, coordinates, mesh3)
    find2.setSearchMode(find2.SEARCH_MODE_NEAREST)
    for t in targets:
        const.assignReal(cache2, t)
        el, xi = find2.evaluateMeshLocation(cache2, 3)
        if el.isValid():
            n += 1
    return n


timings["find_location_200"], found = timeit(find_locations, repeat=1)
timings["find_location_found"] = found

# ---- write
tmp = tempfile.mktemp(suffix=".exf")


def write():
    assert region.writeFile(tmp) == RESULT_OK
    return os.path.getsize(tmp)


timings["write"], _ = timeit(write)
os.unlink(tmp)

print(json.dumps(timings))
