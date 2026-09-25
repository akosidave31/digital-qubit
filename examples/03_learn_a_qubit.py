"""Train a neural model on a hidden noisy qubit from shot data only, then check it.
QUICK = True takes about a minute; set False for the full release recipe (~3 min)."""
import numpy as np
import digital_qubit as dq

QUICK = True
dev = dq.QubitDevice(T1=50, T_phi=40)
ens = dq.build_ensemble(dev, n_members=2 if QUICK else 5, epochs=30 if QUICK else 80)

summary = dq.Checker(ens, dev).batch(300, rng=np.random.default_rng(0))["summary"]
print("checker:", summary["verdict"], f"({summary['agree_fraction'] * 100:.1f}% agree)")
u = dq.T2_with_uncertainty(ens)
print(f"learned T2 = {u['T2']:.2f} +/- {u['se']:.2f}   hidden truth = {dev.hidden_params()['T2']:.2f}")
