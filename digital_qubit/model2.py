"""Learned model for the qubit PAIR: ensemble of softmax MLPs (16 inputs -> 4 joint probabilities)."""
import numpy as np
from .model import train_mlp, Ensemble
from .device import QubitDataset
from .device2 import TwoQubitDataset


class Ensemble2(Ensemble):

    def probs(self, a0, b0, a1, b1, e, t, kA, kB, model=None):
        """Predicted joint probabilities (..., 4). Tilts are canonicalized; t must be in [0, T_MAX]."""
        if kA not in (0, 1, 2) or kB not in (0, 1, 2):
            raise ValueError("bases must be 0 (X), 1 (Y) or 2 (Z)")
        arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, dtype=float)) for v in (a0, b0, a1, b1, e, t)])
        a0, b0, a1, b1, e, t = [x.ravel() for x in arrs]
        if np.any(t < 0) or np.any(t > self.ds.T_MAX):
            raise ValueError(f"wait time t must be in [0, {self.ds.T_MAX}]")
        a0, b0 = QubitDataset.canonical(a0, b0)
        a1, b1 = QubitDataset.canonical(a1, b1)
        X = self.ds.encode(a0, b0, a1, b1, e, t, np.full(t.size, kA), np.full(t.size, kB))
        n_in = self.members[0].sizes[0]
        if X.shape[1] != n_in:
            raise ValueError(f"model expects {n_in} inputs but the dataset encodes {X.shape[1]}: "
                             f"load it with TwoQubitDataset(device, phase_features={n_in == 20})")
        return (model or self).predict(X).reshape(arrs[0].shape + (4,))


def build_ensemble2(device, n_members=3, n_train=40_000, shots=1000, hidden=128,
                    data_seed=5200, init_seed=600, val_seed=51, epochs=160, n_early_cnot=20_000,
                    early_seed=5300, phase_features=False, verbose=True):
    """Defaults = v0.6.0 release settings (EXPERIMENTS.md, v0.5.2): 160 epochs + 20k early-CNOT
    experiments per member. phase_features is experimental (v0.5.3: no improvement), off by default."""
    ds = TwoQubitDataset(device, phase_features=phase_features)
    val = ds.generate(4000, shots, np.random.default_rng(val_seed))
    members = []
    for i in range(n_members):
        if verbose:
            print(f"member {i + 1}/{n_members}: generating {n_train} experiments")
        tr = ds.generate(n_train, shots, np.random.default_rng(data_seed + i))
        Xtr, ytr = tr["X"], tr["y"]
        if n_early_cnot:
            ex = ds.generate_early_cnot(n_early_cnot, shots, np.random.default_rng(early_seed + i))
            Xtr, ytr = np.vstack([Xtr, ex["X"]]), np.vstack([ytr, ex["y"]])
        members.append(train_mlp([ds.n_features, hidden, hidden, 4], Xtr, ytr, val["X"], val["y"],
                                 seed=init_seed + i, epochs=epochs, verbose=verbose))
    return Ensemble2(members, ds)
