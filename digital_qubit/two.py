"""Two qubits: joint states, CNOT, entanglement, and noise on pairs.
Amplitude order: |00>, |01>, |10>, |11> with qubit A written first."""
import numpy as np
from .paulis import I2, X, Y, Z, H, CNOT, YY
from .single import Sphere
from .noise import Damping


class TwoQubit:

    def __init__(self, amps):
        v = np.asarray(amps, dtype=complex).reshape(4)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("zero vector is not a state")
        self.psi = v / n

    @classmethod
    def product(cls, a, b):
        return cls(np.kron(a.psi, b.psi))

    def density(self):
        return np.outer(self.psi, self.psi.conj())

    def apply(self, U4):
        return TwoQubit(U4 @ self.psi)

    def apply_A(self, U):
        return self.apply(np.kron(U, I2))

    def apply_B(self, U):
        return self.apply(np.kron(I2, U))

    def probs(self):
        p = np.abs(self.psi) ** 2
        return p / p.sum()

    def reduced_A(self):
        return ptrace_B(self.density())

    def reduced_B(self):
        return ptrace_A(self.density())

    def vector_A(self):
        return Sphere().to_vector(self.reduced_A())

    def vector_B(self):
        return Sphere().to_vector(self.reduced_B())

    def measure_A(self, rng):
        p = self.probs()
        out = 0 if rng.random() < p[0] + p[1] else 1
        keep = np.zeros(4)
        keep[2 * out:2 * out + 2] = 1
        return out, TwoQubit(self.psi * keep)

    def measure_B(self, rng):
        p = self.probs()
        out = 0 if rng.random() < p[0] + p[2] else 1
        keep = np.array([1, 0, 1, 0]) if out == 0 else np.array([0, 1, 0, 1])
        return out, TwoQubit(self.psi * keep)

    def sample(self, shots, rng):
        return rng.choice(4, size=shots, p=self.probs())


def ptrace_A(rho4):
    """Trace out qubit A -> state of B."""
    return np.einsum("ijil->jl", rho4.reshape(2, 2, 2, 2))


def ptrace_B(rho4):
    """Trace out qubit B -> state of A."""
    return np.einsum("ijkj->ik", rho4.reshape(2, 2, 2, 2))


class Entangle:

    def bell(self, k):
        """k = 0..3 -> Phi+, Psi+, Phi-, Psi-."""
        return TwoQubit(np.eye(4)[k]).apply_A(H).apply(CNOT)

    def concurrence(self, s):
        return float(abs(s.psi @ YY @ s.psi))

    def entropy(self, s):
        lam = np.clip(np.linalg.eigvalsh(s.reduced_A()), 1e-15, 1)
        return float(-np.sum(lam * np.log2(lam)))

    def correlation(self, s, alpha, beta):
        MA = np.cos(alpha) * Z + np.sin(alpha) * X
        MB = np.cos(beta) * Z + np.sin(beta) * X
        return float(np.real(s.psi.conj() @ np.kron(MA, MB) @ s.psi))

    def chsh(self, s, a=0.0, a2=np.pi / 2, b=np.pi / 4, b2=-np.pi / 4):
        E = self.correlation
        return E(s, a, b) + E(s, a, b2) + E(s, a2, b) - E(s, a2, b2)


class PairNoise:
    """The same T1/T_phi noise acting independently on each qubit of a pair."""

    def kraus(self, t, T1, T_phi=None):
        K = list(Damping().kraus(1 - np.exp(-t / T1)))
        if T_phi is None:
            return K
        p = (1 - np.exp(-t / T_phi)) / 2
        D = [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]
        return [d @ k for d in D for k in K]

    def evolve(self, rho4, t, T1, T_phi=None):
        K = self.kraus(t, T1, T_phi)
        out = np.zeros((4, 4), dtype=complex)
        for ka in K:
            for kb in K:
                M = np.kron(ka, kb)
                out += M @ rho4 @ M.conj().T
        return out

    def concurrence(self, rho4):
        """Wootters concurrence (pure or mixed)."""
        rt = YY @ rho4.conj() @ YY
        lam = np.sqrt(np.clip(np.real(np.linalg.eigvals(rho4 @ rt)), 0, None))
        lam = np.sort(lam)[::-1]
        return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))

    def chsh_max(self, rho4):
        P = [X, Y, Z]
        T = np.array([[np.real(np.trace(rho4 @ np.kron(a, b))) for b in P] for a in P])
        u = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
        return float(2 * np.sqrt(u[0] + u[1]))
