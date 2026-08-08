import glob
import os
import warnings

import pytest

import exfield

DATA = os.path.join(os.path.dirname(__file__), "data")
VAGUS_EXF = os.path.join(DATA, "sub-f001_L_vagus_scaffold.exf")
GOLDEN_JSON = os.path.join(DATA, "golden_f001_L.json")

ZINC_RESOURCES = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "zinc", "tests", "resources"))
# vendored copy of sparc.client's tricubic-Hermite box fixture
SPARC_CLIENT_CUBE = os.path.join(DATA, "cube.exf")


def zinc_resource_files():
    if not os.path.isdir(ZINC_RESOURCES):
        return []
    return sorted(glob.glob(os.path.join(ZINC_RESOURCES, "**", "*.ex*"),
                            recursive=True))


@pytest.fixture(scope="session")
def vagus_mesh():
    if not os.path.exists(VAGUS_EXF):
        pytest.skip("vagus scaffold test file not present "
                    "(fetch from SPARC dataset 426)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return exfield.load(VAGUS_EXF)


@pytest.fixture(scope="session")
def vagus_coordinates_1d(vagus_mesh):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return exfield.Evaluator(vagus_mesh.fields["coordinates"],
                                 dimension=1)


@pytest.fixture(scope="session")
def cube_mesh():
    if not os.path.exists(SPARC_CLIENT_CUBE):
        pytest.skip("cube.exf not present")
    return exfield.load(SPARC_CLIENT_CUBE)
