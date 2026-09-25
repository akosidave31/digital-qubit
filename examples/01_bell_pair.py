"""Make a Bell pair, measure it, and show it beats any classical system (CHSH > 2)."""
import numpy as np
import digital_qubit as dq

c = dq.Circuit(2).add("h", 0).add2("cx", 0, 1)
print(c.to_text())

state = c.state()
print("probabilities 00, 01, 10, 11:", np.round(state.probs(), 3))
print("qubit 0 Bloch vector:", np.round(state.bloch(0), 3), "(at the center = entangled)")

rng = np.random.default_rng(0)
outcomes, counts = np.unique(state.sample(1000, rng), return_counts=True)
print("1000 shots:", {state.bits(k): int(v) for k, v in zip(outcomes, counts)})

pair = dq.Entangle().bell(0)
print("CHSH value:", round(dq.Entangle().chsh(pair), 4), "(any classical system: at most 2)")
