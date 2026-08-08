"""Sweep every fitted scaffold in SPARC dataset 426 (acceptance §10.1).

Fetches all 42 scaffolds (21 subjects x L/R, plus the sub-M000 template)
into a cache directory if not present, then for each: parse, locate the
trunk chain, build an arclength table, and report trunk length.

sub-M000 is the generic template and ships in millimetres where the 40
fitted scaffolds use microns; the sweep flags it as a unit-scale outlier
rather than silently averaging a 0.6 mm nerve into the population.

Run:  python tests/sweep_426.py [cache_dir]
"""

import json
import os
import sys
import urllib.parse
import urllib.request
import warnings

import numpy as np

import exfield

API = "https://api.pennsieve.io/discover"
DS = 426


def fetch_all(cache):
    os.makedirs(cache, exist_ok=True)
    ver = json.load(urllib.request.urlopen(
        f"{API}/datasets/{DS}", timeout=60))["version"]

    def browse(path):
        u = (f"{API}/datasets/{DS}/versions/{ver}/files/browse"
             f"?limit=300&path={urllib.parse.quote(path)}")
        return json.load(urllib.request.urlopen(u, timeout=90)).get(
            "files", [])

    subs = [f["name"] for f in browse("files/derivative")
            if f["type"] == "Directory"]
    for sub in subs:
        for side in ("L", "R"):
            dest = os.path.join(cache, f"{sub}_{side}.exf")
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
            try:
                fl = browse(f"files/derivative/{sub}/{side}/030-scaffold")
            except Exception:
                continue
            exf = next((f for f in fl if f["name"].endswith(".exf")), None)
            if exf is None:
                continue
            s = exf["uri"]
            bucket = s.split("/")[2]
            key = "/".join(s.split("/")[3:])
            data = urllib.request.urlopen(
                f"https://{bucket}.s3.amazonaws.com/{key}",
                timeout=180).read()
            with open(dest, "wb") as fh:
                fh.write(data)
            print(f"fetched {sub}_{side}.exf")


def trunk_group(mesh):
    """The trunk centreline group. In the 426 scaffolds this is
    'orientation anterior' — the 1-D group named after the whole nerve
    includes circumferential rings and is NOT the centreline."""
    if "orientation anterior" in mesh.groups:
        return mesh.groups["orientation anterior"]
    return None


def sweep_one(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = exfield.load(path)
        assert mesh.skipped == [], f"{path}: skipped constructs"
        ev = mesh.evaluator("coordinates", dimension=1)
        group = trunk_group(mesh)
        if group is None:
            return dict(nodes=len(mesh.nodes), trunk_mm=None)
        trunk = sorted(group.element_ids(1))
        table = exfield.ArclengthTable.build(ev, element_ids=trunk)
    return dict(nodes=len(mesh.nodes), trunk_elements=len(trunk),
                trunk_mm=table.total / 1000.0)


def main(cache):
    fetch_all(cache)
    files = sorted(f for f in os.listdir(cache) if f.endswith(".exf"))
    results = {}
    for name in files:
        try:
            results[name] = sweep_one(os.path.join(cache, name))
        except Exception as e:
            results[name] = dict(error=f"{type(e).__name__}: {e}")
    lengths = {n: r["trunk_mm"] for n, r in results.items()
               if r.get("trunk_mm")}
    median = float(np.median(list(lengths.values()))) if lengths else 0.0
    print(f"{'file':28s} {'nodes':>5s} {'trunk':>6s} {'mm':>9s}")
    outliers = []
    for name, r in results.items():
        if "error" in r:
            print(f"{name:28s} ERROR {r['error']}")
            continue
        mm = r.get("trunk_mm")
        flag = ""
        if mm and median and not (0.2 < mm / median < 5.0):
            flag = "  << unit-scale outlier (template ships in mm, "\
                   "fitted scaffolds in um)"
            outliers.append(name)
        print(f"{name:28s} {r['nodes']:5d} {r.get('trunk_elements', 0):6d} "
              f"{mm:9.1f}{flag}" if mm else f"{name:28s} {r['nodes']:5d}"
              f"   (no trunk group)")
    ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\n{ok}/{len(results)} scaffolds loaded and measured; "
          f"median trunk {median:.1f} mm; unit outliers: {outliers}")
    return results, outliers


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "data", "scaffolds426")
    main(cache)
