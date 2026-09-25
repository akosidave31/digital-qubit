"""A black-box noisy qubit and the dataset built from its experiments.
One experiment: prepare (theta=a, phi=b) -> wait t -> measure in X, Y or Z."""
import numpy as np
from .single import Theta, XAxis, YAxis
from .noise import Damping


class QubitDevice:

    def __init__(self, T1=50.0, T_phi=40.0):
        self._T1, self._T_phi = T1, T_phi
        self._basis = [XAxis.PLUS, YAxis.PLUS_I, np.array([1, 0], dtype=complex)]

    def hidden_params(self):
        """For testing only: the values a model is supposed to discover."""
        return {"T1": self._T1, "T_phi": self._T_phi, "T2": Damping.T2(self._T1, self._T_phi)}

    def true_p1(self, a, b, t, basis):
        rho = Damping().evolve(Theta().make(a, b).density(), t, self._T1, self._T_phi)
        v = self._basis[int(basis)]
        return float(1 - np.real(np.conj(v) @ rho @ v))

    def run(self, a, b, t, basis, shots, rng):
        return rng.binomial(shots, self.true_p1(a, b, t, basis)) / shots


class QubitDataset:
    T_MAX = 150.0
    N_FEATURES = 8

    def __init__(self, device):
        self.dev = device

    def sample_controls(self, n, rng):
        a = rng.uniform(0, np.pi, n)
        b = rng.uniform(0, 2 * np.pi, n)
        t = rng.uniform(0, self.T_MAX, n)
        basis = rng.integers(0, 3, n)
        return a, b, t, basis

    @staticmethod
    def canonical(a, b):
        """Map any (a, b) to the same physical state with a in [0, pi], b in [0, 2pi).
        (a, b) and (2pi - a, b + pi) differ only by a global phase."""
        a = np.mod(np.asarray(a, dtype=float), 2 * np.pi)
        flip = a > np.pi
        a = np.where(flip, 2 * np.pi - a, a)
        b = np.where(flip, np.asarray(b, dtype=float) + np.pi, np.asarray(b, dtype=float))
        return a, np.mod(b, 2 * np.pi)

    def encode(self, a, b, t, basis):
        onehot = np.eye(3)[np.asarray(basis, dtype=int)]
        return np.column_stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b),
                                np.asarray(t) / self.T_MAX, onehot]).astype(np.float64)

    def generate(self, n, shots, rng):
        a, b, t, basis = self.sample_controls(n, rng)
        y_true = np.array([self.dev.true_p1(*z) for z in zip(a, b, t, basis)])
        y = rng.binomial(shots, y_true) / shots
        return {"X": self.encode(a, b, t, basis), "y": y.astype(np.float64),
                "y_true": y_true, "raw": np.column_stack([a, b, t, basis])}
