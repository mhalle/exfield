"""Benchmark exfield on the f001 vagus scaffold (mirror of bench_zinc.py).

Run: uv run python tests/bench_exfield.py
Prints JSON timings to stdout (last line).
"""

import json
import os
import tempfile
import time
import warnings

import numpy as np

import exfield

PATH = "tests/data/sub-f001_L_vagus_scaffold.exf"
N_EVAL = 20000
rng = np.random.default_rng(0)
warnings.simplefilter("ignore")


def timeit(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


timings = {}

timings["parse"], mesh = timeit(lambda: exfield.load(PATH))

ev3 = mesh.evaluator("coordinates", dimension=3)
ev2 = mesh.evaluator("coordinates", dimension=2)
ev1 = mesh.evaluator("coordinates", dimension=1)

element_ids = sorted(mesh.mesh3d.elements)
xis = rng.random((N_EVAL, 3))
which = rng.integers(0, len(element_ids), N_EVAL)

# group xi rows by element so batching is used (the natural exfield way)
by_element = {}
for i in range(N_EVAL):
    by_element.setdefault(element_ids[which[i]], []).append(i)
grouped = {eid: xis[idx] for eid, idx in by_element.items()}


def eval_values_batch():
    total = 0.0
    for eid, x in grouped.items():
        total += ev3.evaluate(eid, x)[:, 0].sum()
    return total


def eval_values_loop():
    total = 0.0
    for i in range(N_EVAL):
        total += ev3.evaluate(element_ids[which[i]], xis[i])[0]
    return total


timings["eval_20k_batch"], _ = timeit(eval_values_batch)
timings["eval_20k_loop"], _ = timeit(eval_values_loop, repeat=1)

grouped_5k = {}
for i in range(0, N_EVAL, 4):
    grouped_5k.setdefault(element_ids[which[i]], []).append(i)
grouped_5k = {e: xis[idx] for e, idx in grouped_5k.items()}


def eval_derivs_batch():
    total = 0.0
    for eid, x in grouped_5k.items():
        total += ev3.evaluate_derivatives(eid, x)[:, :, 0].sum()
    return total


timings["derivs_5k_batch"], _ = timeit(eval_derivs_batch)

trunk = sorted(mesh.groups["orientation anterior"].element_ids(1))
epi = sorted(mesh.groups["epineurium"].element_ids(2))

timings["arclength_trunk"], _ = timeit(
    lambda: exfield.integrate(ev1, element_ids=trunk, order="zinc"))
timings["area_epineurium"], _ = timeit(
    lambda: exfield.integrate(ev2, element_ids=epi, order="zinc"))
timings["volume_mesh3d"], _ = timeit(
    lambda: exfield.integrate(ev3, order="zinc"))

# inverse mapping: 200 points onto the 3-D mesh
targets = [ev3.evaluate(element_ids[which[i]], xis[i]) for i in range(200)]


def find_locations():
    n = 0
    for t in targets:
        loc = exfield.find_location(ev3, t, element_ids="all",
                                    n_sample=4, n_candidates=4)
        if loc.residual < 1e-6:
            n += 1
    return n


timings["find_location_200"], found = timeit(find_locations, repeat=1)
timings["find_location_found"] = found

tmp = tempfile.mktemp(suffix=".exf")


def write():
    exfield.dump(mesh, tmp)
    return os.path.getsize(tmp)


timings["write"], _ = timeit(write)
os.unlink(tmp)

print(json.dumps(timings))
