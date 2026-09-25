"""Pauli matrices and standard gates."""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
S = np.diag([1, 1j]).astype(complex)
S_DAG = S.conj().T
T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
YY = np.kron(Y, Y)


def rx(a):
    """Turn the arrow by angle a around X."""
    return np.cos(a / 2) * I2 - 1j * np.sin(a / 2) * X


def ry(a):
    """Turn the arrow by angle a around Y."""
    return np.cos(a / 2) * I2 - 1j * np.sin(a / 2) * Y


def rz(a):
    """Turn the arrow by angle a around Z."""
    return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)]).astype(complex)


def phase(lam):
    """Phase gate P(lam) = diag(1, e^{i lam}): adds lam to phi."""
    return np.diag([1, np.exp(1j * lam)]).astype(complex)
