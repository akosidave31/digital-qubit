"""Independent cross-check against Qiskit. Skipped if qiskit is not installed.
Qubit mapping: our A = Qiskit qubit 1, our B = Qiskit qubit 0."""
import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, concurrence
from qiskit_aer.noise import thermal_relaxation_error
from digital_qubit import (StateArrow, TwoQubit, Damping, PairNoise, XAxis, YAxis, Equator,
                           Phi, H, Z, CNOT, CZ)

A_Q, B_Q = 1, 0
GATES = ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "p"]


def ours1(state, name, p):
    Xx, Yx, Eq, Ph = XAxis(), YAxis(), Equator(), Phi()
    return {"h": lambda: state.apply(H), "x": lambda: Xx.flip(state), "y": lambda: Yx.flip(state),
            "z": lambda: state.apply(Z), "s": lambda: state.apply(Ph.gate(np.pi / 2)),
            "t": lambda: state.apply(Ph.gate(np.pi / 4)), "rx": lambda: Xx.rotate(state, p),
            "ry": lambda: Yx.rotate(state, p), "rz": lambda: Eq.rz(state, p),
            "p": lambda: state.apply(Ph.gate(p))}[name]()


def ours_matrix(name, p):
    return np.column_stack([ours1(StateArrow(1, 0), name, p).psi, ours1(StateArrow(0, 1), name, p).psi])


def qk(qc, name, p, q):
    if name in ("rx", "ry", "rz", "p"):
        getattr(qc, name)(p, q)
    else:
        getattr(qc, name)(q)


def fid(a, b):
    return abs(np.vdot(a, b)) ** 2


def test_single_qubit_circuits(rng):
    for _ in range(100):
        s, qc = StateArrow(1, 0), QuantumCircuit(1)
        for _ in range(12):
            name, p = GATES[rng.integers(10)], rng.uniform(0, 2 * np.pi)
            s = ours1(s, name, p)
            qk(qc, name, p, 0)
        assert fid(s.psi, Statevector(qc).data) > 1 - 1e-10


def test_two_qubit_circuits(rng):
    for _ in range(100):
        s, qc = TwoQubit([1, 0, 0, 0]), QuantumCircuit(2)
        for _ in range(14):
            r = rng.random()
            if r < 0.2:
                s = s.apply(CNOT); qc.cx(A_Q, B_Q)
            elif r < 0.3:
                s = s.apply(CZ); qc.cz(A_Q, B_Q)
            else:
                name, p = GATES[rng.integers(10)], rng.uniform(0, 2 * np.pi)
                if rng.random() < 0.5:
                    s = s.apply_A(ours_matrix(name, p)); qk(qc, name, p, A_Q)
                else:
                    s = s.apply_B(ours_matrix(name, p)); qk(qc, name, p, B_Q)
        assert fid(s.psi, Statevector(qc).data) > 1 - 1e-10


def test_thermal_relaxation(rand1, rng):
    for _ in range(100):
        s = rand1()
        t = rng.uniform(0, 200)
        T1, Tp = rng.uniform(10, 100, 2)
        ch = thermal_relaxation_error(T1, Damping.T2(T1, Tp), t).to_quantumchannel()
        ref = DensityMatrix(s.density()).evolve(ch).data
        assert np.max(np.abs(Damping().evolve(s.density(), t, T1, Tp) - ref)) < 1e-10


def test_noisy_pair_concurrence(rand2, rng):
    PN = PairNoise()
    T1, Tp = 50.0, 40.0
    for _ in range(50):
        rho = rand2().density()
        t = rng.uniform(0, 150)
        ch = thermal_relaxation_error(T1, Damping.T2(T1, Tp), t).to_quantumchannel()
        ref = DensityMatrix(rho).evolve(ch, qargs=[A_Q]).evolve(ch, qargs=[B_Q])
        mine = PN.evolve(rho, t, T1, Tp)
        assert np.max(np.abs(mine - ref.data)) < 1e-10
        assert abs(PN.concurrence(mine) - concurrence(ref)) < 1e-8


# ---------------- N qubits (v0.2.0) ------------------------
# Our qubit q = Qiskit qubit n-1-q (both then use the same amplitude order).
from digital_qubit import NQubit, NDensity


def _random_nq_circuit(rng, n, depth):
    ours, qc = NQubit(n), QuantumCircuit(n)
    for _ in range(depth):
        r = rng.random()
        if r < 0.3:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            if r < 0.2:
                ours = ours.cx(a, b); qc.cx(n - 1 - a, n - 1 - b)
            else:
                ours = ours.cz(a, b); qc.cz(n - 1 - a, n - 1 - b)
        else:
            name, p, q = GATES[rng.integers(10)], rng.uniform(0, 2 * np.pi), int(rng.integers(n))
            ours = ours.gate(name, q, p)
            qk(qc, name, p, n - 1 - q)
    return ours, qc


def test_n_qubit_circuits(rng):
    for n in range(3, 7):
        for _ in range(20):
            ours, qc = _random_nq_circuit(rng, n, 30)
            assert fid(ours.psi, Statevector(qc).data) > 1 - 1e-10


def test_n_qubit_noise(rng):
    n, T1, Tp = 3, 50.0, 40.0
    for _ in range(20):
        ours, qc = _random_nq_circuit(rng, n, 20)
        t = rng.uniform(0, 100)
        ch = thermal_relaxation_error(T1, Damping.T2(T1, Tp), t).to_quantumchannel()
        ref = DensityMatrix(qc)
        for q in range(n):
            ref = ref.evolve(ch, qargs=[q])
        mine = NDensity.from_state(ours).idle(t, T1, Tp)
        assert np.max(np.abs(mine.rho - ref.data)) < 1e-10
