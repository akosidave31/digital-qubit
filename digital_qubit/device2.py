"""A black-box noisy PAIR of qubits and its experiments.
One experiment:
  1. prepare qubit A at (a0, b0) and qubit B at (a1, b1)
  2. if e = 1, apply CNOT (A control, B target) -> can create entanglement
  3. both qubits wait time t (hidden T1, T_phi on each)
  4. measure A in basis kA and B in basis kB (0 = X, 1 = Y, 2 = Z)
Output: 4 joint probabilities P(00), P(01), P(10), P(11); outcome 0 = |+>, |+i> or |0>."""
import numpy as np
from .paulis import I2, H, S_DAG, CNOT
from .single import Theta
from .two import TwoQubit, PairNoise
from .noise import Damping

ROT = [H, H @ S_DAG, I2]


class TwoQubitDevice:

    def __init__(self, T1=50.0, T_phi=40.0):
        self._T1, self._T_phi = T1, T_phi

    def hidden_params(self):
        return {"T1": self._T1, "T_phi": self._T_phi, "T2": Damping.T2(self._T1, self._T_phi)}

    def state(self, a0, b0, a1, b1, e, t):
        s = TwoQubit.product(Theta().make(a0, b0), Theta().make(a1, b1))
        if int(e):
            s = s.apply(CNOT)
        return PairNoise().evolve(s.density(), t, self._T1, self._T_phi)

    def true_probs(self, a0, b0, a1, b1, e, t, kA, kB):
        U = np.kron(ROT[int(kA)], ROT[int(kB)])
        p = np.clip(np.real(np.diag(U @ self.state(a0, b0, a1, b1, e, t) @ U.conj().T)), 0, None)
        return p / p.sum()

    def run(self, a0, b0, a1, b1, e, t, kA, kB, shots, rng):
        return rng.multinomial(shots, self.true_probs(a0, b0, a1, b1, e, t, kA, kB)) / shots


class TwoQubitDataset:
    T_MAX = 150.0

    def __init__(self, device, phase_features=False):
        """phase_features=True adds cos/sin of (b0 + b1) and (b0 - b1): 16 -> 20 inputs.
        After CNOT the correlations depend on these phase combinations directly."""
        self.dev = device
        self.phase_features = bool(phase_features)

    @property
    def n_features(self):
        return 20 if self.phase_features else 16

    def sample_controls(self, n, rng, early_frac=0.2):
        a0, a1 = rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n)
        b0, b1 = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        e = rng.integers(0, 2, n)
        t = rng.uniform(0, self.T_MAX, n)
        early = rng.random(n) < early_frac
        t_early = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, 10, n))
        t = np.where(early, t_early, t)
        kA, kB = rng.integers(0, 3, n), rng.integers(0, 3, n)
        return a0, b0, a1, b1, e, t, kA, kB

    def encode(self, a0, b0, a1, b1, e, t, kA, kB):
        b0, b1 = np.asarray(b0, dtype=float), np.asarray(b1, dtype=float)
        cols = [np.cos(a0), np.sin(a0), np.cos(b0), np.sin(b0),
                np.cos(a1), np.sin(a1), np.cos(b1), np.sin(b1),
                np.asarray(e, dtype=float), np.asarray(t) / self.T_MAX,
                np.eye(3)[np.asarray(kA, dtype=int)], np.eye(3)[np.asarray(kB, dtype=int)]]
        if self.phase_features:
            cols += [np.cos(b0 + b1), np.sin(b0 + b1), np.cos(b0 - b1), np.sin(b0 - b1)]
        return np.column_stack(cols).astype(np.float64)

    def generate_early_cnot(self, n, shots, rng):
        """Targeted data for the hardest region: CNOT on, early times (half at t = 0, half in [0, 10])."""
        a0, a1 = rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n)
        b0, b1 = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        e = np.ones(n, dtype=int)
        t = np.concatenate([np.zeros(n // 2), rng.uniform(0, 10, n - n // 2)])
        kA, kB = rng.integers(0, 3, n), rng.integers(0, 3, n)
        c = (a0, b0, a1, b1, e, t, kA, kB)
        y_true = np.array([self.dev.true_probs(*z) for z in zip(*c)])
        y = np.array([rng.multinomial(shots, p) / shots for p in y_true])
        return {"X": self.encode(*c), "y": y, "y_true": y_true, "raw": np.column_stack(c)}

    def generate(self, n, shots, rng, early_frac=0.2):
        c = self.sample_controls(n, rng, early_frac)
        y_true = np.array([self.dev.true_probs(*z) for z in zip(*c)])
        y = np.array([rng.multinomial(shots, p) / shots for p in y_true])
        return {"X": self.encode(*c), "y": y, "y_true": y_true, "raw": np.column_stack(c)}
