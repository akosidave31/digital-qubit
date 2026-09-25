"""Feed experiments to the neural model AND the true physics, and check the model's output.
One experiment: prepare (tilt a, phase b) -> wait t -> measure in basis 0=X, 1=Y, 2=Z.
Works with any model that has p1(a, b, t, basis); an Ensemble also gives +/- from member spread."""
import numpy as np

BASIS_NAME = {0: "X", 1: "Y", 2: "Z"}


class Checker:

    def __init__(self, model, device, tol=0.03):
        self.model, self.device, self.tol = model, device, tol
        ds = getattr(model, "ds", None)
        self.t_max = ds.T_MAX if ds is not None else 150.0

    def predict(self, a, b, t, basis):
        """Model output: mean P(1) and member spread (0 for a single model)."""
        mean = np.asarray(self.model.p1(a, b, t, basis), dtype=float)
        members = getattr(self.model, "members", [])
        if len(members) > 1:
            sd = np.std([self.model.p1(a, b, t, basis, model=m) for m in members], axis=0, ddof=1)
        else:
            sd = np.zeros_like(mean)
        return mean, sd

    def run(self, a, b, t, basis, shots=1000, rng=None):
        """One experiment through the model, the true physics, and a simulated measurement."""
        rng = rng or np.random.default_rng()
        m, sd = self.predict(a, b, t, basis)
        m, sd = float(np.ravel(m)[0]), float(np.ravel(sd)[0])
        truth = self.device.true_p1(a, b, t, basis)
        measured = self.device.run(a, b, t, basis, shots, rng)
        err = m - truth
        return {"model": m, "sd": sd, "truth": truth, "measured": measured, "error": err,
                "shot_sd": float(np.sqrt(truth * (1 - truth) / shots)), "agree": abs(err) < self.tol}

    def batch(self, n=500, shots=1000, rng=None):
        """Many random experiments -> summary of how well the model reproduces the physics."""
        rng = rng or np.random.default_rng()
        a = rng.uniform(0, np.pi, n)
        b = rng.uniform(0, 2 * np.pi, n)
        t = rng.uniform(0, self.t_max, n)
        k = rng.integers(0, 3, n)
        model, sd = np.empty(n), np.empty(n)
        for kk in range(3):
            m = k == kk
            if m.any():
                model[m], sd[m] = self.predict(a[m], b[m], t[m], kk)
        truth = np.array([self.device.true_p1(*z) for z in zip(a, b, t, k)])
        measured = rng.binomial(shots, truth) / shots
        err = model - truth
        agree = float(np.mean(np.abs(err) < self.tol))
        summary = {"n": n,
                   "mean_abs_error": float(np.mean(np.abs(err))),
                   "max_abs_error": float(np.max(np.abs(err))),
                   "rmse": float(np.sqrt(np.mean(err ** 2))),
                   "agree_fraction": agree,
                   "coverage_2sd": float(np.mean(np.abs(err) <= 2 * sd)) if np.any(sd > 0) else None,
                   "measured_rmse": float(np.sqrt(np.mean((measured - truth) ** 2))),
                   "verdict": "PASS" if agree >= 0.95 else "FAIL"}
        return {"summary": summary, "a": a, "b": b, "t": t, "basis": k, "model": model, "sd": sd,
                "truth": truth, "measured": measured, "error": err}

    def sweep(self, a, b, basis, t_grid, shots=1000, rng=None):
        """Model vs physics along a whole decay curve."""
        rng = rng or np.random.default_rng()
        t_grid = np.asarray(t_grid, dtype=float)
        mean, sd = self.predict(np.full(t_grid.shape, a), np.full(t_grid.shape, b), t_grid, basis)
        truth = np.array([self.device.true_p1(a, b, t, basis) for t in t_grid])
        return {"t": t_grid, "model": mean, "sd": sd, "truth": truth,
                "measured": rng.binomial(shots, truth) / shots}
