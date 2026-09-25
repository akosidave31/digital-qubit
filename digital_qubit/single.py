"""Single-qubit components: the Bloch sphere and everything on it."""
import numpy as np
from .paulis import I2, X, Y, Z, H as _H, S_DAG as _S_DAG, rx, ry, rz, phase


class Sphere:
    """Bloch vectors r = (rx, ry, rz) <-> 2x2 density matrices."""

    def to_density(self, r):
        rx_, ry_, rz_ = r
        return 0.5 * (I2 + rx_ * X + ry_ * Y + rz_ * Z)

    def to_vector(self, rho):
        return np.real([np.trace(rho @ X), np.trace(rho @ Y), np.trace(rho @ Z)])

    def radius(self, r):
        return float(np.linalg.norm(r))

    def classify(self, r, tol=1e-9):
        n = self.radius(r)
        if n > 1 + tol:
            return "invalid"
        if abs(n - 1) <= tol:
            return "pure"
        return "mixed"

    def purity(self, r):
        rho = self.to_density(r)
        return float(np.real(np.trace(rho @ rho)))

    def is_physical(self, rho, tol=1e-9):
        eig = np.linalg.eigvalsh(rho)
        return bool(np.all(eig >= -tol) and abs(np.trace(rho) - 1) < tol)


class StateArrow:
    """|psi> = alpha|0> + beta|1>, always normalized."""

    def __init__(self, alpha, beta):
        v = np.array([alpha, beta], dtype=complex)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("zero vector is not a qubit state")
        self.psi = v / n

    @classmethod
    def from_vector(cls, r):
        r = np.asarray(r, dtype=float)
        r = r / np.linalg.norm(r)
        theta = np.arccos(np.clip(r[2], -1, 1))
        phi = np.arctan2(r[1], r[0])
        return cls(np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2))

    @classmethod
    def from_angles(cls, theta, phi=0.0):
        return cls(np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2))

    def density(self):
        return np.outer(self.psi, self.psi.conj())

    def vector(self):
        a, b = self.psi
        return np.array([2 * np.real(np.conj(a) * b),
                         2 * np.imag(np.conj(a) * b),
                         abs(a) ** 2 - abs(b) ** 2])

    def apply(self, U):
        return StateArrow(*(U @ self.psi))

    def fidelity(self, other):
        return float(abs(np.vdot(self.psi, other.psi)) ** 2)

    def __repr__(self):
        a, b = self.psi
        return f"StateArrow({a:.4f}|0> + {b:.4f}|1>)"


class ZAxis:
    """Measurement in the computational (Z) basis."""
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    def probs(self, state):
        p0 = float(abs(state.psi[0]) ** 2)
        return p0, 1 - p0

    def probs_density(self, rho):
        p0 = float(np.real(np.trace(self.P0 @ rho)))
        return p0, 1 - p0

    def expectation(self, state):
        p0, p1 = self.probs(state)
        return p0 - p1

    def measure(self, state, rng):
        p0, _ = self.probs(state)
        if rng.random() < p0:
            return 0, StateArrow(1, 0)
        return 1, StateArrow(0, 1)

    def sample(self, state, shots, rng):
        p0, _ = self.probs(state)
        return (rng.random(shots) >= p0).astype(int)


class XAxis:
    """X gate, Rx rotations, and measurement in the |+>/|-> basis."""
    PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
    MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
    H = _H

    def flip(self, state):
        return state.apply(X)

    def rotate(self, state, angle):
        return state.apply(rx(angle))

    def probs(self, state):
        pp = float(abs(np.vdot(self.PLUS, state.psi)) ** 2)
        return pp, 1 - pp

    def expectation(self, state):
        pp, pm = self.probs(state)
        return pp - pm

    def measure(self, state, rng):
        pp, _ = self.probs(state)
        if rng.random() < pp:
            return 0, StateArrow(*self.PLUS)
        return 1, StateArrow(*self.MINUS)

    def sample(self, state, shots, rng):
        pp, _ = self.probs(state)
        return (rng.random(shots) >= pp).astype(int)


class YAxis:
    """Y gate, Ry rotations, and measurement in the |+i>/|-i> basis."""
    PLUS_I = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    MINUS_I = np.array([1, -1j], dtype=complex) / np.sqrt(2)
    S_DAG = _S_DAG

    def flip(self, state):
        return state.apply(Y)

    def rotate(self, state, angle):
        return state.apply(ry(angle))

    def probs(self, state):
        pp = float(abs(np.vdot(self.PLUS_I, state.psi)) ** 2)
        return pp, 1 - pp

    def expectation(self, state):
        pp, pm = self.probs(state)
        return pp - pm

    def measure(self, state, rng):
        pp, _ = self.probs(state)
        if rng.random() < pp:
            return 0, StateArrow(*self.PLUS_I)
        return 1, StateArrow(*self.MINUS_I)

    def sample(self, state, shots, rng):
        pp, _ = self.probs(state)
        return (rng.random(shots) >= pp).astype(int)


class Equator:
    """The 50/50 superpositions (|0> + e^{i phi}|1>)/sqrt(2)."""

    def make(self, phi):
        return StateArrow(1, np.exp(1j * phi))

    def on_equator(self, state, tol=1e-9):
        return abs(state.vector()[2]) < tol

    def phase(self, state):
        rx_, ry_, _ = state.vector()
        return float(np.arctan2(ry_, rx_) % (2 * np.pi))

    def project(self, state):
        rx_, ry_, _ = state.vector()
        if np.hypot(rx_, ry_) < 1e-12:
            raise ValueError("poles have no defined phase")
        return self.make(self.phase(state))

    def rz(self, state, angle):
        return state.apply(rz(angle))


class Theta:
    """Polar angle theta: the 0-vs-1 mix. P(0) = cos^2(theta/2)."""

    def make(self, theta, phi=0.0):
        return StateArrow.from_angles(theta, phi)

    def get(self, state):
        return float(np.arccos(np.clip(state.vector()[2], -1, 1)))

    def set(self, state, theta):
        rx_, ry_, _ = state.vector()
        phi = np.arctan2(ry_, rx_) if np.hypot(rx_, ry_) > 1e-12 else 0.0
        return self.make(theta, phi)

    def p0(self, theta):
        return np.cos(theta / 2) ** 2

    def estimate(self, n_ones, shots):
        p0_hat = 1 - n_ones / shots
        return float(2 * np.arccos(np.sqrt(np.clip(p0_hat, 0, 1))))


class Phi:
    """Relative phase phi, phase gates, and interference."""

    def get(self, state):
        rx_, ry_, _ = state.vector()
        if np.hypot(rx_, ry_) < 1e-12:
            raise ValueError("poles have no defined phase")
        return float(np.arctan2(ry_, rx_) % (2 * np.pi))

    def set(self, state, phi):
        return Theta().make(Theta().get(state), phi)

    def gate(self, lam):
        return phase(lam)

    def interferometer(self, lam):
        s = StateArrow(1, 0).apply(_H).apply(phase(lam)).apply(_H)
        return ZAxis().probs(s)[0]

    def estimate(self, ones_x, ones_y, shots):
        ex = 1 - 2 * ones_x / shots
        ey = 1 - 2 * ones_y / shots
        return float(np.arctan2(ey, ex) % (2 * np.pi))


class Projection:
    """Projection lines: vertical = populations, horizontal = coherence."""

    def foot(self, r):
        return np.array([r[0], r[1], 0.0])

    def vertical(self, r):
        return float(abs(r[2]))

    def horizontal(self, r):
        return float(np.hypot(r[0], r[1]))

    def foot_angle(self, r):
        return float(np.arctan2(r[1], r[0]) % (2 * np.pi))

    def coherence(self, rho):
        return float(2 * abs(rho[0, 1]))

    def dephase(self, rho, p):
        return (1 - p) * rho + p * (Z @ rho @ Z)
