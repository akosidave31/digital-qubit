"""N qubits.
NQubit   : statevector engine (pure states), practical up to ~20+ qubits.
NDensity : density-matrix engine with per-qubit T1/T2 noise, practical up to ~8-10 qubits.
Qubit 0 is written first (leftmost / most significant bit), matching TwoQubit (A = 0, B = 1).
All operations return a new object; the original is unchanged."""
import numpy as np
from .paulis import X, Y, Z, H, S, T, CNOT, CZ, rx, ry, rz, phase
from .single import Sphere
from .two import PairNoise

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_FIXED = {"h": H, "x": X, "y": Y, "z": Z, "s": S, "t": T}
_PARAM = {"rx": rx, "ry": ry, "rz": rz, "p": phase}


def gate_matrix(name, param=None):
    """2x2 matrix for a named single-qubit gate: h x y z s t (fixed) or rx ry rz p (need param)."""
    if name in _FIXED:
        return _FIXED[name]
    if name in _PARAM:
        if param is None:
            raise ValueError(f"gate '{name}' needs a parameter")
        return _PARAM[name](param)
    raise ValueError(f"unknown gate '{name}'")


def _on_axes(tensor, U, axes):
    """Apply a 2^k x 2^k matrix U to the listed axes of a (2,2,...,2) tensor."""
    k = len(axes)
    Ut = U.reshape((2,) * (2 * k))
    out = np.tensordot(Ut, tensor, axes=(list(range(k, 2 * k)), list(axes)))
    return np.moveaxis(out, list(range(k)), list(axes))


class _Base:
    def _check(self, *qs):
        for q in qs:
            if not (isinstance(q, (int, np.integer)) and 0 <= q < self.n):
                raise ValueError(f"qubit index {q} out of range for {self.n} qubits")
        if len(set(qs)) != len(qs):
            raise ValueError("two-qubit gates need two different qubits")

    def gate(self, name, q, param=None):
        return self.apply1(gate_matrix(name, param), q)

    def cx(self, control, target):
        return self.apply2(CNOT, control, target)

    def cz(self, a, b):
        return self.apply2(CZ, a, b)

    def swap(self, a, b):
        return self.apply2(SWAP, a, b)

    def marginal(self, q):
        """(P(qubit q = 0), P(qubit q = 1))."""
        self._check(q)
        p = self.probs().reshape((2,) * self.n)
        m = p.sum(axis=tuple(i for i in range(self.n) if i != q))
        return float(m[0]), float(m[1])

    def bloch(self, q):
        return Sphere().to_vector(self.reduced([q]))

    def sample(self, shots, rng):
        """Joint outcomes as integers; bits(i) gives the bit string (qubit 0 first)."""
        return rng.choice(2 ** self.n, size=shots, p=self.probs())

    def bits(self, index):
        return format(int(index), f"0{self.n}b")


class NQubit(_Base):

    def __init__(self, n, amps=None):
        self.n = int(n)
        if amps is None:
            v = np.zeros(2 ** self.n, dtype=complex)
            v[0] = 1
        else:
            v = np.asarray(amps, dtype=complex).reshape(2 ** self.n)
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                raise ValueError("zero vector is not a state")
            v = v / nv
        self.psi = v

    @classmethod
    def product(cls, states):
        """Independent single qubits (StateArrow list) -> one register."""
        v = np.array([1], dtype=complex)
        for s in states:
            v = np.kron(v, s.psi)
        return cls(len(states), v)

    def _t(self):
        return self.psi.reshape((2,) * self.n)

    def apply1(self, U, q):
        self._check(q)
        return NQubit(self.n, _on_axes(self._t(), U, [q]).reshape(-1))

    def apply2(self, U4, q1, q2):
        """4x4 gate acting on (q1, q2) in that order (q1 = first/left)."""
        self._check(q1, q2)
        return NQubit(self.n, _on_axes(self._t(), U4, [q1, q2]).reshape(-1))

    def probs(self):
        p = np.abs(self.psi) ** 2
        return p / p.sum()

    def measure(self, q, rng):
        p0, _ = self.marginal(q)
        out = 0 if rng.random() < p0 else 1
        t = self._t().copy()
        idx = [slice(None)] * self.n
        idx[q] = 1 - out
        t[tuple(idx)] = 0
        return out, NQubit(self.n, t.reshape(-1))

    def reduced(self, qubits):
        """Density matrix of a subset of qubits (returned in increasing qubit order)."""
        keep = sorted(qubits)
        self._check(*keep)
        others = [i for i in range(self.n) if i not in keep]
        t = self._t()
        rho = np.tensordot(t, t.conj(), axes=(others, others))
        d = 2 ** len(keep)
        return rho.reshape(d, d)

    def density(self):
        return np.outer(self.psi, self.psi.conj())

    def fidelity(self, other):
        return float(abs(np.vdot(self.psi, other.psi)) ** 2)


class NDensity(_Base):

    def __init__(self, n, rho=None):
        self.n = int(n)
        D = 2 ** self.n
        if rho is None:
            rho = np.zeros((D, D), dtype=complex)
            rho[0, 0] = 1
        self.rho = np.asarray(rho, dtype=complex).reshape(D, D)

    @classmethod
    def from_state(cls, s):
        return cls(s.n, s.density())

    def _t(self):
        return self.rho.reshape((2,) * (2 * self.n))

    def _sandwich(self, t, K, qs):
        t = _on_axes(t, K, qs)
        return _on_axes(t, K.conj(), [self.n + q for q in qs])

    def _wrap(self, t):
        D = 2 ** self.n
        return NDensity(self.n, t.reshape(D, D))

    def apply1(self, U, q):
        self._check(q)
        return self._wrap(self._sandwich(self._t(), U, [q]))

    def apply2(self, U4, q1, q2):
        self._check(q1, q2)
        return self._wrap(self._sandwich(self._t(), U4, [q1, q2]))

    def channel1(self, kraus, q):
        """Any single-qubit channel given by Kraus operators."""
        self._check(q)
        t0 = self._t()
        return self._wrap(sum(self._sandwich(t0, K, [q]) for K in kraus))

    def idle(self, t, T1, T_phi=None, qubits=None):
        """Every listed qubit (default: all) waits time t under its own T1/T_phi noise."""
        K = PairNoise().kraus(t, T1, T_phi)
        out = self
        for q in (range(self.n) if qubits is None else qubits):
            out = out.channel1(K, q)
        return out

    def probs(self):
        p = np.clip(np.real(np.diag(self.rho)), 0, None)
        return p / p.sum()

    def reduced(self, qubits):
        keep = sorted(qubits)
        self._check(*keep)
        t, cur = self._t(), self.n
        for q in sorted((i for i in range(self.n) if i not in keep), reverse=True):
            t = np.trace(t, axis1=q, axis2=cur + q)
            cur -= 1
        d = 2 ** len(keep)
        return t.reshape(d, d)

    def purity(self):
        return float(np.real(np.trace(self.rho @ self.rho)))
