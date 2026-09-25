"""T1 (amplitude damping) and T2 (dephasing) noise for one qubit."""
import numpy as np
from .single import Projection


class Damping:

    def kraus(self, gamma):
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        return K0, K1

    def apply(self, rho, gamma):
        K0, K1 = self.kraus(gamma)
        return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

    def evolve(self, rho, t, T1, T_phi=None):
        """Wait time t: energy loss (T1) plus optional pure dephasing (T_phi)."""
        rho = self.apply(rho, 1 - np.exp(-t / T1))
        if T_phi is not None:
            p = (1 - np.exp(-t / T_phi)) / 2
            rho = Projection().dephase(rho, p)
        return rho

    @staticmethod
    def T2(T1, T_phi=None):
        return 2 * T1 if T_phi is None else 1 / (1 / (2 * T1) + 1 / T_phi)
