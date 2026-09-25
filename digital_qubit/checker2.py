"""Check a two-qubit model against the true physics, including ENTANGLEMENT.
Tomography: 9 basis pairs x 4 probabilities -> the full 4x4 state -> concurrence."""
import numpy as np
from .paulis import I2, X, Y, Z
from .two import PairNoise

PAULI = [X, Y, Z]


def tomography(prob_fn):
    """prob_fn(kA, kB) -> 4 joint probabilities. Returns the reconstructed 4x4 density matrix."""
    P = {(i, j): np.asarray(prob_fn(i, j), dtype=float) for i in range(3) for j in range(3)}
    rho = np.kron(I2, I2).astype(complex)
    for i in range(3):
        a = np.mean([P[i, j][0] + P[i, j][1] - P[i, j][2] - P[i, j][3] for j in range(3)])
        b = np.mean([P[j, i][0] - P[j, i][1] + P[j, i][2] - P[j, i][3] for j in range(3)])
        rho += a * np.kron(PAULI[i], I2) + b * np.kron(I2, PAULI[i])
        for j in range(3):
            c = P[i, j][0] - P[i, j][1] - P[i, j][2] + P[i, j][3]
            rho += c * np.kron(PAULI[i], PAULI[j])
    return rho / 4


class TwoChecker:

    def __init__(self, model, device, tol=0.03):
        self.model, self.device, self.tol = model, device, tol
        ds = getattr(model, "ds", None)
        self.t_max = ds.T_MAX if ds is not None else 150.0

    def batch(self, n=500, shots=1000, rng=None):
        """Random experiments -> total-variation distance between model and true outcome odds."""
        rng = rng or np.random.default_rng()
        a0, a1 = rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n)
        b0, b1 = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        e, t = rng.integers(0, 2, n), rng.uniform(0, self.t_max, n)
        kA, kB = rng.integers(0, 3, n), rng.integers(0, 3, n)
        model = np.empty((n, 4))
        for i in range(3):
            for j in range(3):
                m = (kA == i) & (kB == j)
                if m.any():
                    model[m] = self.model.probs(a0[m], b0[m], a1[m], b1[m], e[m], t[m], i, j)
        truth = np.array([self.device.true_probs(*z) for z in zip(a0, b0, a1, b1, e, t, kA, kB)])
        measured = np.array([rng.multinomial(shots, p) / shots for p in truth])
        tvd = 0.5 * np.abs(model - truth).sum(1)
        tvd_meas = 0.5 * np.abs(measured - truth).sum(1)
        agree = float(np.mean(tvd < self.tol))
        ent = e == 1
        summary = {"n": n, "mean_tvd": float(tvd.mean()), "max_tvd": float(tvd.max()),
                   "agree_fraction": agree,
                   "agree_entangling": float(np.mean(tvd[ent] < self.tol)) if ent.any() else None,
                   "agree_product": float(np.mean(tvd[~ent] < self.tol)) if (~ent).any() else None,
                   "measured_mean_tvd": float(tvd_meas.mean()),
                   "verdict": "PASS" if agree >= 0.95 else "FAIL"}
        return {"summary": summary, "tvd": tvd, "e": e, "t": t, "model": model, "truth": truth}

    def concurrence_curve(self, a0, b0, a1, b1, e, t_grid):
        """Entanglement over time: learned (via tomography of model outputs) vs true."""
        t_grid = np.asarray(t_grid, dtype=float)
        P = {(i, j): self.model.probs(a0, b0, a1, b1, e, t_grid, i, j) for i in range(3) for j in range(3)}
        pn = PairNoise()
        c_model = np.array([pn.concurrence(tomography(lambda i, j, k=k: P[i, j][k]))
                            for k in range(len(t_grid))])
        c_true = np.array([pn.concurrence(self.device.state(a0, b0, a1, b1, e, t)) for t in t_grid])
        return {"t": t_grid, "model": c_model, "true": c_true}

    def false_entanglement(self, n=20, t_grid=None, rng=None):
        """Largest concurrence the model reports for pairs that are NEVER entangled
        (random preparations, no CNOT). True value is exactly 0."""
        rng = rng or np.random.default_rng(0)
        t_grid = np.linspace(0, 5, 11) if t_grid is None else t_grid
        worst = 0.0
        for _ in range(n):
            a0, a1 = rng.uniform(0, np.pi, 2)
            b0, b1 = rng.uniform(0, 2 * np.pi, 2)
            worst = max(worst, float(np.max(self.concurrence_curve(a0, b0, a1, b1, 0, t_grid)["model"])))
        return worst


def death_time(t, c, threshold=0.02):
    """First time the concurrence falls below the threshold (None if it never does)."""
    below = np.where(np.asarray(c) < threshold)[0]
    return float(t[below[0]]) if len(below) else None
