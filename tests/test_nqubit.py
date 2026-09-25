import numpy as np
import pytest
from digital_qubit import (NQubit, NDensity, gate_matrix, SWAP, StateArrow, TwoQubit, Damping,
                           PairNoise, Sphere, CNOT, CZ)

NAMES = ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "p"]
T1, TP = 50.0, 40.0


def rand_ops(rng, n, depth, two=0.3):
    ops = []
    for _ in range(depth):
        if n > 1 and rng.random() < two:
            a, b = rng.choice(n, 2, replace=False)
            ops.append(("cx" if rng.random() < 0.7 else "cz", int(a), int(b)))
        else:
            ops.append(("1", NAMES[rng.integers(10)], int(rng.integers(n)), rng.uniform(0, 2 * np.pi)))
    return ops


def run(reg, ops):
    for op in ops:
        if op[0] == "1":
            reg = reg.gate(op[1], op[2], op[3])
        elif op[0] == "cx":
            reg = reg.cx(op[1], op[2])
        else:
            reg = reg.cz(op[1], op[2])
    return reg


def rand_nq(rng, n):
    return NQubit(n, rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n))


def test_one_qubit_matches_statearrow(rng):
    for _ in range(100):
        ops = rand_ops(rng, 1, 12)
        s = StateArrow(1, 0)
        for op in ops:
            s = s.apply(gate_matrix(op[1], op[3]))
        r = run(NQubit(1), ops)
        assert abs(np.vdot(r.psi, s.psi)) ** 2 > 1 - 1e-12
        assert np.allclose(r.bloch(0), s.vector())


def test_two_qubits_match_twoqubit(rng):
    CX10 = SWAP @ CNOT @ SWAP
    for _ in range(100):
        ops = rand_ops(rng, 2, 14)
        s = TwoQubit([1, 0, 0, 0])
        for op in ops:
            if op[0] == "1":
                U = gate_matrix(op[1], op[3])
                s = s.apply_A(U) if op[2] == 0 else s.apply_B(U)
            elif op[0] == "cx":
                s = s.apply(CNOT if op[1] == 0 else CX10)
            else:
                s = s.apply(CZ)
        r = run(NQubit(2), ops)
        assert abs(np.vdot(r.psi, s.psi)) ** 2 > 1 - 1e-12
        assert np.allclose(r.reduced([0]), s.reduced_A()) and np.allclose(r.reduced([1]), s.reduced_B())


def test_cx_any_pair_is_the_right_permutation(rng):
    n = 4
    s = rand_nq(rng, n)
    for c in range(n):
        for t in range(n):
            if c == t:
                continue
            out = np.zeros_like(s.psi)
            for i in range(2 ** n):
                bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                if bits[c]:
                    bits[t] ^= 1
                j = int("".join(map(str, bits)), 2)
                out[j] = s.psi[i]
            assert np.allclose(s.cx(c, t).psi, out)


def test_normalized_and_local_gates_commute(rng):
    r = run(NQubit(8), rand_ops(rng, 8, 120))
    assert np.isclose(np.linalg.norm(r.psi), 1)
    s = rand_nq(rng, 5)
    U, V = gate_matrix("rx", 0.7), gate_matrix("ry", 1.9)
    assert np.allclose(s.apply1(U, 1).apply1(V, 3).psi, s.apply1(V, 3).apply1(U, 1).psi)


def test_product_state_reduces_to_parts(rng):
    parts = [StateArrow(*(rng.normal(size=2) + 1j * rng.normal(size=2))) for _ in range(4)]
    r = NQubit.product(parts)
    for q, p in enumerate(parts):
        assert np.allclose(r.reduced([q]), p.density())
        assert np.isclose(r.marginal(q)[0], abs(p.psi[0]) ** 2)


def test_ghz(rng):
    n = 5
    g = NQubit(n).gate("h", 0)
    for q in range(n - 1):
        g = g.cx(q, q + 1)
    p = g.probs()
    assert np.isclose(p[0], 0.5) and np.isclose(p[-1], 0.5)
    assert all(np.allclose(g.bloch(q), 0) for q in range(n))
    assert set(np.unique(g.sample(5000, rng))) <= {0, 2 ** n - 1}
    for _ in range(20):
        out, after = g.measure(2, rng)
        assert all(np.isclose(after.marginal(q)[out], 1) for q in range(n))


def test_reduced_states_are_physical(rng):
    s = rand_nq(rng, 5)
    for keep in ([0], [1, 3], [0, 2, 4]):
        rho = s.reduced(keep)
        assert np.isclose(np.trace(rho), 1) and np.all(np.linalg.eigvalsh(rho) > -1e-10)
    assert np.isclose(s.marginal(2)[0], np.real(s.reduced([2])[0, 0]))


def test_density_engine_matches_state_engine(rng):
    for _ in range(30):
        ops = rand_ops(rng, 4, 20)
        assert np.allclose(run(NDensity(4), ops).rho, run(NQubit(4), ops).density())


def test_idle_noise_matches_one_and_two_qubit_code(rng):
    for _ in range(50):
        t = rng.uniform(0, 150)
        s1 = NQubit(1, rng.normal(size=2) + 1j * rng.normal(size=2))
        assert np.allclose(NDensity.from_state(s1).idle(t, T1, TP).rho, Damping().evolve(s1.density(), t, T1, TP))
        s2 = rand_nq(rng, 2)
        assert np.allclose(NDensity.from_state(s2).idle(t, T1, TP).rho, PairNoise().evolve(s2.density(), t, T1, TP))


def test_noise_is_local(rng):
    s = rand_nq(rng, 4)
    t = 17.0
    noisy = NDensity.from_state(s).idle(t, T1, TP)
    for q in range(4):
        assert np.allclose(noisy.reduced([q]), Damping().evolve(s.reduced([q]), t, T1, TP))
    assert np.isclose(np.trace(noisy.rho), 1) and np.all(np.linalg.eigvalsh(noisy.rho) > -1e-10)


def test_bigger_ghz_loses_coherence_n_times_faster():
    T2 = Damping.T2(T1, TP)
    for n in range(1, 6):
        g = NQubit(n).gate("h", 0)
        for q in range(n - 1):
            g = g.cx(q, q + 1)
        for t in (5.0, 15.0, 30.0):
            coh = abs(NDensity.from_state(g).idle(t, T1, TP).rho[0, -1])
            assert np.isclose(coh, 0.5 * np.exp(-n * t / T2))


def test_twelve_qubits(rng):
    r = run(NQubit(12), rand_ops(rng, 12, 60))
    assert np.isclose(np.linalg.norm(r.psi), 1)
    assert np.isclose(sum(r.marginal(5)), 1)


def test_bad_indices_rejected():
    with pytest.raises(ValueError):
        NQubit(3).gate("h", 3)
    with pytest.raises(ValueError):
        NQubit(3).cx(1, 1)
    with pytest.raises(ValueError):
        NQubit(3).gate("rx", 0)
