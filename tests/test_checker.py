"""The checker must say MATCH for a correct model and catch wrong ones."""
import numpy as np
import pytest
from digital_qubit import Checker, QubitDevice, build_ensemble

dev = QubitDevice()


class Stub:
    """A fake model defined by a function f(a, b, t, basis) -> P(1)."""

    def __init__(self, f):
        self.f = f
        self.members = [self]

    def p1(self, a, b, t, basis, model=None, allow_extrapolation=False):
        a, b, t = np.broadcast_arrays(np.atleast_1d(a), np.atleast_1d(b), np.atleast_1d(t))
        v = [self.f(x, y, z, basis) for x, y, z in zip(a.ravel(), b.ravel(), t.ravel())]
        return np.array(v).reshape(a.shape)


perfect = Stub(dev.true_p1)
constant = Stub(lambda a, b, t, k: 0.5)
biased = Stub(lambda a, b, t, k: min(1.0, dev.true_p1(a, b, t, k) + 0.05))
no_noise = Stub(lambda a, b, t, k: dev.true_p1(a, b, 0.0, k))


def test_perfect_model_passes(rng):
    S = Checker(perfect, dev).batch(300, rng=rng)["summary"]
    assert S["verdict"] == "PASS" and S["agree_fraction"] == 1.0 and S["max_abs_error"] < 1e-12


def test_constant_model_is_caught(rng):
    S = Checker(constant, dev).batch(300, rng=rng)["summary"]
    assert S["verdict"] == "FAIL" and S["agree_fraction"] < 0.5


def test_small_bias_is_caught(rng):
    S = Checker(biased, dev).batch(300, rng=rng)["summary"]
    assert S["verdict"] == "FAIL"


def test_model_that_ignores_noise_is_caught(rng):
    S = Checker(no_noise, dev).batch(300, rng=rng)["summary"]
    assert S["verdict"] == "FAIL"


def test_single_run_and_measurement(rng):
    ck = Checker(perfect, dev)
    r = ck.run(1.57, 0.0, 20.0, 0, shots=1000, rng=rng)
    assert r["agree"] and abs(r["error"]) < 1e-12
    assert abs(r["measured"] - r["truth"]) < 5 * r["shot_sd"] + 1e-3
    assert not Checker(constant, dev).run(0.3, 0.0, 5.0, 2, rng=rng)["agree"]


def test_sweep_shapes(rng):
    s = Checker(perfect, dev).sweep(1.0, 0.5, 1, np.linspace(0, 150, 31), rng=rng)
    assert all(len(s[k]) == 31 for k in ("t", "model", "sd", "truth", "measured"))
    assert np.allclose(s["model"], s["truth"])


@pytest.mark.slow
def test_real_ensemble_reproduces_physics():
    ens = build_ensemble(dev, verbose=False)
    S = Checker(ens, dev).batch(500, rng=np.random.default_rng(7))["summary"]
    print("\n", S)
    assert S["verdict"] == "PASS"
    assert S["rmse"] < S["measured_rmse"]
