import numpy as np
from digital_qubit import Damping, Sphere, StateArrow, ZAxis, XAxis, Projection, I2

D, S = Damping(), Sphere()


def test_kraus_complete(rng):
    for g in rng.uniform(0, 1, 50):
        K0, K1 = D.kraus(g)
        assert np.allclose(K0.conj().T @ K0 + K1.conj().T @ K1, I2)


def test_output_physical_and_bloch_map(rand1, rng):
    for _ in range(300):
        s, g = rand1(), rng.uniform(0, 1)
        out = D.apply(s.density(), g)
        assert S.is_physical(out)
        r0, r1 = s.vector(), S.to_vector(out)
        assert np.allclose(r1[:2], np.sqrt(1 - g) * r0[:2])
        assert np.isclose(r1[2], g + (1 - g) * r0[2])


def test_ground_state_stable_and_full_decay(rand1):
    g0 = StateArrow(1, 0).density()
    assert np.allclose(D.apply(g0, 0.7), g0)
    for _ in range(50):
        assert np.allclose(S.to_vector(D.apply(rand1().density(), 1.0)), [0, 0, 1])


def test_t1_decay():
    T1 = 50.0
    ts = np.linspace(0, 200, 41)
    p1 = [ZAxis().probs_density(D.evolve(StateArrow(0, 1).density(), t, T1))[1] for t in ts]
    assert np.allclose(p1, np.exp(-ts / T1))


def test_t2_laws(rng):
    T1, Tp = 50.0, 40.0
    ts = np.linspace(0, 200, 41)
    plus = StateArrow(*XAxis.PLUS).density()
    coh = [Projection().coherence(D.evolve(plus, t, T1)) for t in ts]
    assert np.allclose(coh, np.exp(-ts / (2 * T1)))
    T2 = Damping.T2(T1, Tp)
    coh = [Projection().coherence(D.evolve(plus, t, T1, Tp)) for t in ts]
    assert np.allclose(coh, np.exp(-ts / T2))
    assert all(Damping.T2(a, b) <= 2 * a + 1e-12 for a, b in rng.uniform(1, 100, (200, 2)))


def test_learn_T1_from_shots(rng):
    T1, shots = 50.0, 20000
    tg = np.linspace(0.1, 2.0, 15) * T1
    p1 = np.array([rng.binomial(shots, ZAxis().probs_density(D.evolve(StateArrow(0, 1).density(), t, T1))[1]) / shots
                   for t in tg])
    T1_hat = -1 / np.polyfit(tg, np.log(p1), 1)[0]
    assert abs(T1_hat - T1) / T1 < 0.05
