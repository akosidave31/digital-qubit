import numpy as np
from digital_qubit import QubitDevice, QubitDataset

dev = QubitDevice()
ds = QubitDataset(dev)


def test_device_matches_formulas_at_t0(rng):
    for a, b in rng.uniform(0, [np.pi, 2 * np.pi], (200, 2)):
        assert np.isclose(dev.true_p1(a, b, 0, 2), np.sin(a / 2) ** 2)
        assert np.isclose(dev.true_p1(a, b, 0, 0), (1 - np.sin(a) * np.cos(b)) / 2)
        assert np.isclose(dev.true_p1(a, b, 0, 1), (1 - np.sin(a) * np.sin(b)) / 2)


def test_long_wait_relaxes(rng):
    for a, b in rng.uniform(0, [np.pi, 2 * np.pi], (50, 2)):
        assert dev.true_p1(a, b, 1000, 2) < 1e-6
        assert abs(dev.true_p1(a, b, 1000, 0) - 0.5) < 1e-6


def test_dataset_shapes_and_ranges(rng):
    d = ds.generate(3000, 200, rng)
    assert d["X"].shape == (3000, 8)
    assert np.all(np.isfinite(d["X"])) and np.all((d["y"] >= 0) & (d["y"] <= 1))
    assert np.all(np.abs(d["X"][:, :4]) <= 1) and np.all((d["X"][:, 4] >= 0) & (d["X"][:, 4] <= 1))
    assert np.allclose(d["X"][:, 5:].sum(1), 1)


def test_label_noise_is_binomial(rng):
    d = ds.generate(5000, 200, rng)
    resid = d["y"] - d["y_true"]
    assert abs(resid.mean()) < 0.003
    ratio = np.var(resid) / np.mean(d["y_true"] * (1 - d["y_true"]) / 200)
    assert 0.85 < ratio < 1.15


def test_reproducible():
    a = ds.generate(50, 200, np.random.default_rng(7))
    b = ds.generate(50, 200, np.random.default_rng(7))
    assert all(np.array_equal(a[k], b[k]) for k in a)
