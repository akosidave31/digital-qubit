import numpy as np
import pytest
from digital_qubit import StateArrow, TwoQubit


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def rand1(rng):
    return lambda: StateArrow(*(rng.normal(size=2) + 1j * rng.normal(size=2)))


@pytest.fixture
def rand2(rng):
    return lambda: TwoQubit(rng.normal(size=4) + 1j * rng.normal(size=4))


@pytest.fixture
def rand_unitary(rng):
    def f():
        q, r = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        return q * (np.diag(r) / np.abs(np.diag(r)))
    return f


@pytest.fixture
def unit_dirs(rng):
    def f(n):
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return f
