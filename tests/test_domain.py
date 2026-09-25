"""Input domain of the learned model (v0.1.1).
First demonstrate the defect (raw out-of-range tilts), then confirm the fix."""
import warnings
import numpy as np
import pytest
from digital_qubit import QubitDevice, QubitDataset, Ensemble, train_mlp

dev = QubitDevice()
ds = QubitDataset(dev)


@pytest.fixture(scope="module")
def small_ens():
    r = np.random.default_rng(5)
    tr, va = ds.generate(4000, 1000, r), ds.generate(500, 1000, r)
    m = train_mlp([8, 32, 32, 1], tr["X"], tr["y"], va["X"], va["y"], seed=2, epochs=25, verbose=False)
    return Ensemble([m], ds)


def p1_vec(ens, a, b, t, k):
    out = np.empty(len(a))
    for kk in range(3):
        m = k == kk
        out[m] = ens.p1(a[m], b[m], t[m], kk)
    return out


def test_canonical_is_the_same_physical_state(rng):
    for a, b in rng.uniform(-10, 10, (500, 2)):
        ac, bc = ds.canonical(a, b)
        assert 0 <= ac <= np.pi and 0 <= bc < 2 * np.pi
        for k in range(3):
            assert np.isclose(dev.true_p1(a, b, 7.0, k), dev.true_p1(ac, bc, 7.0, k))


def test_out_of_range_tilt_defect_and_fix(small_ens):
    r = np.random.default_rng(6)
    n = 1000
    a_in = r.uniform(0.2, np.pi - 0.2, n)
    a_out = 2 * np.pi - a_in
    b, t, k = r.uniform(0, 2 * np.pi, n), r.uniform(0, 100, n), r.integers(0, 3, n)
    truth_in = np.array([dev.true_p1(*z) for z in zip(a_in, b, t, k)])
    truth_out = np.array([dev.true_p1(*z) for z in zip(a_out, b, t, k)])
    err_in = np.mean(np.abs(p1_vec(small_ens, a_in, b, t, k) - truth_in))
    raw = small_ens.members[0].predict(ds.encode(a_out, b, t, k))
    err_raw = np.mean(np.abs(raw - truth_out))
    err_fixed = np.mean(np.abs(p1_vec(small_ens, a_out, b, t, k) - truth_out))
    print(f"\nmean error  in range: {err_in:.4f}   out of range raw: {err_raw:.4f}   "
          f"out of range fixed: {err_fixed:.4f}")
    assert err_raw > 2 * err_in, "defect not reproduced: raw out-of-range input was not worse"
    assert err_fixed < 1.5 * err_in + 0.005


def test_negative_time_rejected(small_ens):
    with pytest.raises(ValueError):
        small_ens.p1(1.0, 0.0, -1.0, 2)


def test_time_beyond_range_guarded(small_ens):
    small_ens.p1(1.0, 0.0, ds.T_MAX, 2)
    with pytest.raises(ValueError):
        small_ens.p1(1.0, 0.0, ds.T_MAX + 1, 2)
    with pytest.warns(UserWarning):
        small_ens.p1(1.0, 0.0, ds.T_MAX + 1, 2, allow_extrapolation=True)


def test_bad_basis_rejected(small_ens):
    with pytest.raises(ValueError):
        small_ens.p1(1.0, 0.0, 5.0, 3)
