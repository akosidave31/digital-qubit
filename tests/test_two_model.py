"""Two-qubit device, tomography, softmax model, and the two-qubit checker."""
import numpy as np
import pytest
from digital_qubit import (TwoQubitDevice, TwoQubitDataset, QubitDevice, MLP, Ensemble2,
                           TwoChecker, tomography, death_time, PairNoise, build_ensemble2)

dev2 = TwoQubitDevice()
dev1 = QubitDevice()
BELL = (np.pi / 2, 0.0, 0.0, 0.0, 1)


def test_no_cnot_means_independent_qubits(rng):
    for _ in range(100):
        a0, a1 = rng.uniform(0, np.pi, 2)
        b0, b1 = rng.uniform(0, 2 * np.pi, 2)
        t = rng.uniform(0, 150)
        kA, kB = rng.integers(0, 3, 2)
        pA = dev1.true_p1(a0, b0, t, kA)
        pB = dev1.true_p1(a1, b1, t, kB)
        expect = np.kron([1 - pA, pA], [1 - pB, pB])
        assert np.allclose(dev2.true_probs(a0, b0, a1, b1, 0, t, kA, kB), expect)


def test_bell_pair_correlations():
    for k in (0, 2):
        assert np.allclose(dev2.true_probs(*BELL, 0.0, k, k), [0.5, 0, 0, 0.5])


def test_probabilities_valid(rng):
    ds = TwoQubitDataset(dev2)
    d = ds.generate(300, 1000, rng)
    assert d["X"].shape == (300, 16) and d["y"].shape == (300, 4)
    assert np.allclose(d["y"].sum(1), 1) and np.allclose(d["y_true"].sum(1), 1)
    assert np.all(d["y"] >= 0) and np.all(d["y_true"] >= 0)


def test_phase_features_are_opt_in_and_correct(rng):
    plain, rich = TwoQubitDataset(dev2), TwoQubitDataset(dev2, phase_features=True)
    assert plain.n_features == 16 and rich.n_features == 20
    b0, b1 = rng.uniform(0, 6.28, 50), rng.uniform(0, 6.28, 50)
    args = (rng.uniform(0, 3, 50), b0, rng.uniform(0, 3, 50), b1, np.ones(50), rng.uniform(0, 150, 50),
            np.zeros(50, int), np.ones(50, int))
    Xp, Xr = plain.encode(*args), rich.encode(*args)
    assert Xp.shape == (50, 16) and Xr.shape == (50, 20)
    assert np.allclose(Xr[:, :16], Xp)
    assert np.allclose(Xr[:, 16:], np.column_stack([np.cos(b0 + b1), np.sin(b0 + b1),
                                                    np.cos(b0 - b1), np.sin(b0 - b1)]))
    assert rich.generate_early_cnot(20, 100, rng)["X"].shape == (20, 20)


def test_input_size_mismatch_is_caught():
    rich_model = Ensemble2([MLP([20, 8, 4], np.random.default_rng(0))], TwoQubitDataset(dev2))
    with pytest.raises(ValueError, match="phase_features=True"):
        rich_model.probs(*BELL, 5.0, 0, 0)
    ok = Ensemble2([MLP([20, 8, 4], np.random.default_rng(0))], TwoQubitDataset(dev2, phase_features=True))
    assert ok.probs(*BELL, 5.0, 0, 0).shape == (1, 4)


def test_false_entanglement_metric():
    assert TwoChecker(Stub(dev2.true_probs), dev2).false_entanglement(n=5) < 1e-6  # sqrt(eigenvalue) amplifies ~1e-18 round-off to ~1e-9
    fake = Stub(lambda a0, b0, a1, b1, e, t, kA, kB: dev2.true_probs(*BELL, t, kA, kB))
    assert TwoChecker(fake, dev2).false_entanglement(n=3) > 0.5


def test_early_cnot_data(rng):
    d = TwoQubitDataset(dev2).generate_early_cnot(400, 1000, rng)
    raw = d["raw"]
    assert d["X"].shape == (400, 16) and np.allclose(d["y"].sum(1), 1)
    assert np.all(raw[:, 4] == 1)
    assert np.all((raw[:, 5] >= 0) & (raw[:, 5] <= 10)) and np.sum(raw[:, 5] == 0) == 200


def test_tomography_recovers_the_exact_state(rng):
    for _ in range(20):
        c = (*rng.uniform(0, np.pi, 1), *rng.uniform(0, 6.28, 1), *rng.uniform(0, np.pi, 1),
             *rng.uniform(0, 6.28, 1), int(rng.integers(0, 2)), rng.uniform(0, 100))
        rho = tomography(lambda i, j: dev2.true_probs(*c, i, j))
        assert np.allclose(rho, dev2.state(*c), atol=1e-12)


def test_sudden_death_seen_through_tomography():
    T1, Tp = dev2.hidden_params()["T1"], dev2.hidden_params()["T_phi"]
    ts = np.arange(15, 30, 0.05)
    c = [PairNoise().concurrence(tomography(lambda i, j, t=t: dev2.true_probs(*BELL, t, i, j))) for t in ts]
    f = lambda t: np.exp(-2 * t / Tp) - (1 - np.exp(-t / T1))
    lo, hi = 0.0, 500.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    assert abs(death_time(ts, c, 1e-9) - (lo + hi) / 2) < 0.1


def test_softmax_gradient_check():
    r = np.random.default_rng(0)
    net = MLP([6, 5, 4], r)
    X = r.normal(size=(30, 6))
    Y = r.dirichlet(np.ones(4), 30)
    _, acts = net.forward(X)
    grads = net.backward(acts, Y)
    loss = lambda: -np.mean(np.sum(Y * np.log(net.predict(X)), axis=1))
    eps = 1e-6
    for p, g in zip(net.W + net.b, grads):
        for _ in range(5):
            idx = tuple(r.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps; up = loss()
            p[idx] = old - eps; dn = loss()
            p[idx] = old
            assert abs((up - dn) / (2 * eps) - g[idx]) < 1e-6


class Stub:
    def __init__(self, f):
        self.f = f

    def probs(self, a0, b0, a1, b1, e, t, kA, kB):
        arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, dtype=float)) for v in (a0, b0, a1, b1, e, t)])
        out = [self.f(*z, kA, kB) for z in zip(*[x.ravel() for x in arrs])]
        return np.array(out).reshape(arrs[0].shape + (4,))


def test_two_checker_passes_perfect_and_catches_bad(rng):
    perfect = Stub(dev2.true_probs)
    uniform = Stub(lambda *a: np.full(4, 0.25))
    no_cnot = Stub(lambda a0, b0, a1, b1, e, t, kA, kB: dev2.true_probs(a0, b0, a1, b1, 0, t, kA, kB))
    no_noise = Stub(lambda a0, b0, a1, b1, e, t, kA, kB: dev2.true_probs(a0, b0, a1, b1, e, 0.0, kA, kB))
    assert TwoChecker(perfect, dev2).batch(200, rng=rng)["summary"]["verdict"] == "PASS"
    for bad in (uniform, no_cnot, no_noise):
        assert TwoChecker(bad, dev2).batch(200, rng=rng)["summary"]["verdict"] == "FAIL"


def test_learned_entanglement_check_on_perfect_model():
    ck = TwoChecker(Stub(dev2.true_probs), dev2)
    c = ck.concurrence_curve(*BELL, np.linspace(0, 40, 41))
    assert np.allclose(c["model"], c["true"], atol=1e-6)


def test_ensemble2_guards():
    ds = TwoQubitDataset(dev2)
    ens = Ensemble2([MLP([16, 8, 4], np.random.default_rng(0))], ds)
    p = ens.probs(*BELL, 10.0, 0, 2)
    assert p.shape == (1, 4) and np.isclose(p.sum(), 1)
    with pytest.raises(ValueError):
        ens.probs(*BELL, -1.0, 0, 2)
    with pytest.raises(ValueError):
        ens.probs(*BELL, 200.0, 0, 2)
    with pytest.raises(ValueError):
        ens.probs(*BELL, 10.0, 3, 0)


@pytest.mark.slow
def test_network_learns_entanglement():
    """Spec for the v0.6.0 default model (see README 'Known limitations')."""
    ens = build_ensemble2(dev2, verbose=False)
    ck = TwoChecker(ens, dev2)
    S = ck.batch(500, rng=np.random.default_rng(3))["summary"]
    assert S["verdict"] == "PASS"
    c = ck.concurrence_curve(*BELL, np.linspace(0, 40, 81))
    assert np.max(np.abs(c["model"] - c["true"])) < 0.2
    tm, tt = death_time(c["t"], c["model"]), death_time(c["t"], c["true"])
    assert tm is not None and abs(tm - tt) / tt < 0.15
    assert ck.false_entanglement(n=20) < 0.1
