import numpy as np
import pytest
from digital_qubit import (MLP, train_mlp, convergence, best_at, Ensemble, QubitDevice, QubitDataset,
                           build_ensemble, read_T1, T2_with_uncertainty)


def bce(net, X, y):
    p = np.clip(net.predict(X), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def test_mlp_gradient_check():
    r = np.random.default_rng(0)
    net = MLP([8, 6, 5, 1], r)
    X, y = r.normal(size=(40, 8)), r.uniform(0, 1, 40)
    _, acts = net.forward(X)
    grads = net.backward(acts, y)
    params = net.W + net.b
    eps = 1e-6
    for p, g in zip(params, grads):
        for _ in range(5):
            idx = tuple(r.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps
            up = bce(net, X, y)
            p[idx] = old - eps
            dn = bce(net, X, y)
            p[idx] = old
            assert abs((up - dn) / (2 * eps) - g[idx]) < 1e-6


def test_training_learns_something(rng):
    ds = QubitDataset(QubitDevice())
    tr, va = ds.generate(3000, 200, rng), ds.generate(500, 200, rng)
    net = train_mlp([8, 32, 32, 1], tr["X"], tr["y"], va["X"], va["y"], seed=1, epochs=20, verbose=False)
    mse = np.mean((net.predict(va["X"]) - va["y"]) ** 2)
    const = np.mean((va["y"] - tr["y"].mean()) ** 2)
    assert mse < const / 4


def test_training_records_history(rng):
    ds = QubitDataset(QubitDevice())
    tr, va = ds.generate(800, 200, rng), ds.generate(200, 200, rng)
    net = train_mlp([8, 8, 1], tr["X"], tr["y"], va["X"], va["y"], seed=1, epochs=7, verbose=False)
    assert len(net.history) == 7
    assert np.isclose(np.mean((net.predict(va["X"]) - va["y"]) ** 2), min(net.history))


def test_convergence_detector():
    flat = [1.0] * 30 + [0.5] * 50
    falling = list(np.linspace(1.0, 0.5, 80))
    assert convergence(flat)[1] is False
    gain, improving = convergence(falling)
    assert improving and gain > 0.1
    assert best_at(falling, 40) > best_at(falling, 80)


def test_ensemble_save_load(tmp_path):
    ds = QubitDataset(QubitDevice())
    r = np.random.default_rng(3)
    ens = Ensemble([MLP([8, 4, 1], r), MLP([8, 6, 1], r)], ds)
    X = r.normal(size=(20, 8))
    ens.save(tmp_path / "ens.npz")
    back = Ensemble.load(tmp_path / "ens.npz", ds)
    assert np.allclose(back.predict(X), ens.predict(X))


@pytest.mark.slow
def test_full_ensemble_discovers_physics():
    dev = QubitDevice()
    hidden = dev.hidden_params()
    ens = build_ensemble(dev, verbose=False)
    ds = ens.ds
    te = ds.generate(4000, 200, np.random.default_rng(99))
    floor = np.mean(te["y_true"] * (1 - te["y_true"]) / 200)
    assert np.mean((ens.predict(te["X"]) - te["y"]) ** 2) < 1.5 * floor
    aa, bb = np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 60)
    assert np.max(np.abs(ens.p1(aa, 0.0, 0.0, 2) - np.sin(aa / 2) ** 2)) < 0.05
    assert np.max(np.abs(ens.p1(np.pi / 2, bb, 0.0, 0) - (1 - np.cos(bb)) / 2)) < 0.05
    assert abs(read_T1(ens) - hidden["T1"]) / hidden["T1"] < 0.15
    u = T2_with_uncertainty(ens)
    assert abs(u["T2"] - hidden["T2"]) / hidden["T2"] < 0.05
    assert abs(u["T2"] - hidden["T2"]) < 3 * u["se"]
