"""The learned model: a pure-NumPy MLP, and a bagged ensemble of them.
Recipe (from the notebook experiments, Cells 12-22):
  64-64 tanh MLP, Adam, 80 epochs, early-time calibration data,
  1000-shot labels, 5 members each on its own fresh data sample."""
import time
import warnings
import numpy as np
from .device import QubitDataset


class MLP:

    def __init__(self, sizes, rng):
        self.sizes = list(sizes)
        self.W = [rng.normal(0, np.sqrt(1 / m), (m, n)) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.step = 0

    def forward(self, X):
        """tanh hidden layers; output = sigmoid (1 output) or softmax (k outputs)."""
        acts, h = [X], X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < last:
                h = np.tanh(z)
            elif self.sizes[-1] == 1:
                h = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            else:
                z = z - z.max(axis=1, keepdims=True)
                e = np.exp(z)
                h = e / e.sum(axis=1, keepdims=True)
            acts.append(h)
        return (h[:, 0] if self.sizes[-1] == 1 else h), acts

    def predict(self, X):
        return self.forward(np.asarray(X, dtype=np.float64))[0]

    def backward(self, acts, y):
        """Gradients of mean cross-entropy (sigmoid or softmax output, soft labels allowed)."""
        P = acts[-1]
        d = (P - np.asarray(y, dtype=np.float64).reshape(P.shape)) / len(P)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i], gb[i] = acts[i].T @ d, d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (1 - acts[i] ** 2)
        return gW + gb

    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.step += 1
        for k, (p, g) in enumerate(zip(self.W + self.b, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.step)
            vh = self.v[k] / (1 - b2 ** self.step)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def get(self):
        return [p.copy() for p in self.W + self.b]

    def set(self, params):
        n = len(self.W)
        self.W = [p.copy() for p in params[:n]]
        self.b = [p.copy() for p in params[n:]]


def train_mlp(sizes, X, y, X_val, y_val, seed=12, epochs=80, batch=128, lr0=3e-3, verbose=True):
    """Adam, lr halves every 25 epochs, keeps the best-validation weights.
    The per-epoch validation MSE is stored on the returned model as .history."""
    r = np.random.default_rng(seed)
    m = MLP(sizes, r)
    best, best_p = np.inf, m.get()
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        lr = lr0 * (0.5 ** (ep // 25))
        idx = r.permutation(len(y))
        for s in range(0, len(y), batch):
            j = idx[s:s + batch]
            _, acts = m.forward(X[j])
            m.adam(m.backward(acts, y[j]), lr)
        v = float(np.mean((m.predict(X_val) - y_val) ** 2))
        history.append(v)
        if v < best:
            best, best_p = v, m.get()
        if verbose and ep % 20 == 0:
            print(f"  epoch {ep:3d}  val MSE {v:.5f}  (best {best:.5f})  {time.time() - t0:.0f}s")
    m.set(best_p)
    m.history = history
    return m


def convergence(history, window=20, threshold=0.01):
    """Did the best validation error still improve by more than `threshold` (relative)
    during the last `window` epochs? Returns (relative_gain, still_improving)."""
    h = np.asarray(history, dtype=float)
    if len(h) <= window:
        return float("nan"), True
    before, after = h[:-window].min(), h.min()
    gain = float((before - after) / before)
    return gain, gain > threshold


def best_at(history, epoch):
    """Best validation MSE reached by a given epoch (1-based)."""
    return float(np.min(history[:epoch]))


def make_training_set(device, seed, n_base=20_000, n_early=5_000, shots=1000):
    """Uniform experiments plus early-time calibration experiments."""
    ds = QubitDataset(device)
    r = np.random.default_rng(seed)
    base = ds.generate(n_base, shots, r)
    a = r.uniform(0, np.pi, n_early)
    b = r.uniform(0, 2 * np.pi, n_early)
    t = np.concatenate([np.zeros(n_early // 2), r.uniform(0, 10, n_early - n_early // 2)])
    k = r.integers(0, 3, n_early)
    yt = np.array([device.true_p1(*z) for z in zip(a, b, t, k)])
    X = np.vstack([base["X"], ds.encode(a, b, t, k)])
    y = np.concatenate([base["y"], r.binomial(shots, yt) / shots])
    return X, y


class Ensemble:

    def __init__(self, members, dataset):
        self.members = list(members)
        self.ds = dataset

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)

    def p1(self, a, b, t, basis, model=None, allow_extrapolation=False):
        """Predicted probability of outcome 1 for given controls (broadcasts).
        Tilts outside [0, pi] are mapped to the equivalent in-range state.
        Wait times must be in [0, T_MAX] unless allow_extrapolation=True."""
        if basis not in (0, 1, 2):
            raise ValueError("basis must be 0 (X), 1 (Y) or 2 (Z)")
        a, b, t = np.broadcast_arrays(np.atleast_1d(np.asarray(a, dtype=float)),
                                      np.atleast_1d(np.asarray(b, dtype=float)),
                                      np.atleast_1d(np.asarray(t, dtype=float)))
        if np.any(t < 0):
            raise ValueError("wait time t must be >= 0")
        if np.any(t > self.ds.T_MAX):
            if not allow_extrapolation:
                raise ValueError(f"t > {self.ds.T_MAX} is outside the training range; "
                                 "pass allow_extrapolation=True to predict anyway (unreliable)")
            warnings.warn(f"extrapolating beyond t = {self.ds.T_MAX}: predictions are unreliable")
        ac, bc = self.ds.canonical(a.ravel(), b.ravel())
        X = self.ds.encode(ac, bc, t.ravel(), np.full(a.size, basis))
        return (model or self).predict(X).reshape(a.shape)

    def save(self, path):
        data = {"n": np.array(len(self.members))}
        for i, m in enumerate(self.members):
            data[f"sizes{i}"] = np.array(m.sizes)
            for j, p in enumerate(m.get()):
                data[f"m{i}_{j}"] = p
        np.savez(path, **data)

    @classmethod
    def load(cls, path, dataset):
        f = np.load(path)
        members = []
        for i in range(int(f["n"])):
            sizes = [int(s) for s in f[f"sizes{i}"]]
            m = MLP(sizes, np.random.default_rng(0))
            m.set([f[f"m{i}_{j}"] for j in range(2 * (len(sizes) - 1))])
            members.append(m)
        return cls(members, dataset)


def build_ensemble(device, n_members=5, data_seed=2200, init_seed=300, val_seed=11,
                   epochs=80, verbose=True):
    ds = QubitDataset(device)
    val = ds.generate(4000, 200, np.random.default_rng(val_seed))
    members = []
    for i in range(n_members):
        if verbose:
            print(f"member {i + 1}/{n_members}")
        X, y = make_training_set(device, data_seed + i)
        members.append(train_mlp([8, 64, 64, 1], X, y, val["X"], val["y"],
                                 seed=init_seed + i, epochs=epochs, verbose=verbose))
    return Ensemble(members, ds)


def log_fit(ts, sig):
    """Decay time from an exponential fit on the log scale."""
    return float(-1 / np.polyfit(ts, np.log(np.clip(sig, 1e-4, None)), 1)[0])


def read_T1(ens, t_grid=None, model=None):
    """Probe: prepare |1>, wait, measure Z. P(1) = e^{-t/T1}."""
    t_grid = np.linspace(5, 100, 30) if t_grid is None else t_grid
    return log_fit(t_grid, ens.p1(np.pi, 0.0, t_grid, 2, model=model))


def read_T2(ens, t_grid=None, model=None):
    """Probe at 4 points on the equator (X and Y bases); returns the average."""
    t_grid = np.linspace(3, 57, 30) if t_grid is None else t_grid
    reads = []
    for b, k, sign in [(0.0, 0, 1), (np.pi, 0, -1), (np.pi / 2, 1, 1), (3 * np.pi / 2, 1, -1)]:
        sig = sign * (1 - 2 * ens.p1(np.pi / 2, b, t_grid, k, model=model))
        reads.append(log_fit(t_grid, sig))
    return float(np.mean(reads))


def T2_with_uncertainty(ens, t_grid=None):
    """T2 from each member -> mean, spread of one model, standard error of the mean."""
    reads = np.array([read_T2(ens, t_grid, model=m) for m in ens.members])
    return {"T2": float(reads.mean()), "sd": float(reads.std(ddof=1)),
            "se": float(reads.std(ddof=1) / np.sqrt(len(reads))), "members": reads}
