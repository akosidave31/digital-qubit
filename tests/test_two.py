import numpy as np
from digital_qubit import (TwoQubit, Entangle, PairNoise, Damping, StateArrow, Theta, Projection,
                           ZAxis, Sphere, CNOT, CZ, H, I2, ptrace_B)

En, PN = Entangle(), PairNoise()


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def test_product_states_reduce(rand1):
    for _ in range(200):
        a, b = rand1(), rand1()
        s = TwoQubit.product(a, b)
        assert np.allclose(s.vector_A(), a.vector()) and np.allclose(s.vector_B(), b.vector())
        assert np.isclose(purity(s.reduced_A()), 1)


def test_local_gates_are_local(rand1, rand_unitary):
    for _ in range(100):
        a, b, U = rand1(), rand1(), rand_unitary()
        s = TwoQubit.product(a, b).apply_A(U)
        assert np.allclose(s.vector_A(), a.apply(U).vector()) and np.allclose(s.vector_B(), b.vector())


def test_product_measurement(rand1, rng):
    Zx = ZAxis()
    for _ in range(100):
        a, b = rand1(), rand1()
        s = TwoQubit.product(a, b)
        pa, pb = Zx.probs(a), Zx.probs(b)
        assert np.allclose(s.probs(), [pa[0] * pb[0], pa[0] * pb[1], pa[1] * pb[0], pa[1] * pb[1]])
        out, after = s.measure_A(rng)
        assert np.allclose(after.vector_A(), [0, 0, 1 if out == 0 else -1])
        assert np.allclose(after.vector_B(), b.vector())


def test_cnot():
    assert np.allclose(CNOT @ np.eye(4), np.eye(4)[:, [0, 1, 3, 2]])
    assert np.allclose(CNOT @ CNOT, np.eye(4))
    assert np.allclose(np.kron(I2, H) @ CNOT @ np.kron(I2, H), CZ)


def test_bell_states():
    B = np.array([En.bell(k).psi for k in range(4)])
    assert np.allclose(B.conj() @ B.T, np.eye(4))
    assert np.allclose(En.bell(0).psi, np.array([1, 0, 0, 1]) / np.sqrt(2))
    for k in range(4):
        s = En.bell(k)
        assert np.isclose(En.concurrence(s), 1)
        assert np.allclose(s.vector_A(), 0) and np.allclose(s.vector_B(), 0)
        assert np.isclose(En.entropy(s), 1)


def test_bell_correlations(rng, rand_unitary):
    z = En.bell(0).sample(20000, rng)
    assert set(np.unique(z)) <= {0, 3}
    assert abs(np.mean(z == 0) - 0.5) < 0.02
    singlet = En.bell(3)
    for _ in range(30):
        U = rand_unitary()
        assert set(np.unique(singlet.apply_A(U).apply_B(U).sample(1000, rng))) <= {1, 2}


def test_no_signaling_and_local_invariance(rand2, rand_unitary):
    for k in range(4):
        s = En.bell(k)
        assert np.allclose(s.apply_B(rand_unitary()).reduced_A(), s.reduced_A())
    for _ in range(100):
        s = rand2()
        assert np.isclose(En.concurrence(s.apply_A(rand_unitary()).apply_B(rand_unitary())), En.concurrence(s))


def test_concurrence_dial(rng):
    for t in rng.uniform(0, np.pi, 100):
        a = Theta().make(t, 0.0)
        s = TwoQubit.product(a, StateArrow(1, 0)).apply(CNOT)
        assert np.isclose(En.concurrence(s), np.sin(t))
        assert np.isclose(En.concurrence(s), Projection().horizontal(a.vector()))


def test_chsh(rand1, rng):
    assert np.isclose(En.chsh(En.bell(0)), 2 * np.sqrt(2))
    assert np.isclose(PN.chsh_max(En.bell(0).density()), 2 * np.sqrt(2))
    for _ in range(500):
        s = TwoQubit.product(rand1(), rand1())
        assert abs(En.chsh(s, *rng.uniform(0, 2 * np.pi, 4))) <= 2 + 1e-9


def test_pair_noise_is_local_noise(rand2, rng):
    for _ in range(100):
        s, t = rand2(), rng.uniform(0, 150)
        out = PN.evolve(s.density(), t, 50.0, 40.0)
        assert np.all(np.linalg.eigvalsh(out) > -1e-9) and np.isclose(np.trace(out), 1)
        assert np.allclose(ptrace_B(out), Damping().evolve(s.reduced_A(), t, 50.0, 40.0))


def test_wootters_matches_pure(rand2):
    for _ in range(200):
        s = rand2()
        assert np.isclose(PN.concurrence(s.density()), En.concurrence(s))


def test_sudden_death():
    T1, Tp = 50.0, 40.0
    a, b = np.sqrt(1 / 3), np.sqrt(2 / 3)
    rho = TwoQubit([a, 0, 0, b]).density()
    t_death = -T1 * np.log(1 - a / b)
    assert PN.concurrence(PN.evolve(rho, 0.99 * t_death, T1)) > 1e-4
    assert PN.concurrence(PN.evolve(rho, 1.01 * t_death, T1)) < 1e-9
    phi = En.bell(0).density()
    f = lambda t: np.exp(-2 * t / Tp) - (1 - np.exp(-t / T1))
    lo, hi = 0.0, 500.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    t_th = (lo + hi) / 2
    assert PN.concurrence(PN.evolve(phi, 0.98 * t_th, T1, Tp)) > 0
    assert PN.concurrence(PN.evolve(phi, 1.02 * t_th, T1, Tp)) < 1e-9
