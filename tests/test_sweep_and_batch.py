"""Corpus sweep (when the 426 cache is present) and batch-API tests."""

import os
import warnings

import numpy as np
import pytest

import exfield

CACHE_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "data", "scaffolds426"),
    os.environ.get("EXFIELD_426_CACHE", ""),
]


def _cache_dir():
    for c in CACHE_CANDIDATES:
        if c and os.path.isdir(c) and any(
                f.endswith(".exf") for f in os.listdir(c)):
            return c
    return None


@pytest.mark.skipif(_cache_dir() is None,
                    reason="426 corpus cache not present; run "
                           "tests/sweep_426.py to fetch")
def test_all_426_scaffolds_load_and_measure():
    """Acceptance §10.1: all fitted scaffolds load without error;
    sub-M000 loads and is flagged as a different unit scale."""
    import sweep_426
    results, outliers = sweep_426.main(_cache_dir())
    errors = {n: r for n, r in results.items() if "error" in r}
    assert not errors, errors
    assert len(results) >= 40
    fitted = {n: r["trunk_mm"] for n, r in results.items()
              if r.get("trunk_mm") and not n.startswith("sub-M000")}
    assert all(300 < mm < 900 for mm in fitted.values())
    assert sorted(outliers) == ["sub-M000_L.exf", "sub-M000_R.exf"]


class TestBatchAPI:
    def test_batch_matches_single(self, vagus_mesh):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=3)
        eid = sorted(vagus_mesh.mesh3d.elements)[0]
        xis = np.random.default_rng(1).random((50, 3))
        batch = ev.evaluate(eid, xis)
        assert batch.shape == (50, 3)
        for i in (0, 17, 49):
            assert np.allclose(batch[i], ev.evaluate(eid, xis[i]),
                               atol=1e-12)
        dbatch = ev.evaluate_derivatives(eid, xis)
        assert dbatch.shape == (50, 3, 3)
        assert np.allclose(dbatch[17],
                           ev.evaluate_derivatives(eid, xis[17]),
                           atol=1e-12)

    def test_batch_on_inherited_element(self, vagus_mesh):
        """Face-inherited elements support batch evaluation with the
        affine xi map applied per row."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=1)
        trunk = sorted(
            vagus_mesh.groups["orientation anterior"].element_ids(1))
        xis = np.linspace(0, 1, 9)[:, None]
        batch = ev.evaluate(trunk[0], xis)
        assert batch.shape == (9, 3)
        assert np.allclose(batch[4], ev.evaluate(trunk[0], [0.5]),
                           atol=1e-9)

    def test_mesh_conveniences(self, vagus_mesh):
        assert vagus_mesh.mesh1d is vagus_mesh.mesh(1)
        assert vagus_mesh.mesh3d is vagus_mesh.mesh(3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator()          # default: coordinates
        assert ev.field.name == "coordinates"

    def test_arclength_values_named_tuple(self, vagus_mesh):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = vagus_mesh.evaluator("coordinates", dimension=1)
        trunk = sorted(
            vagus_mesh.groups["orientation anterior"].element_ids(1))
        table = exfield.ArclengthTable.build(ev, element_ids=trunk)
        emb = exfield.EmbeddedPoints(element_ids=[trunk[0]], xis=[[0.5]])
        result = emb.arclength(table)
        values, nan_count = result           # tuple unpacking still works
        assert result.nan_count == 0         # and attribute access
        assert result.values[0] > 0
