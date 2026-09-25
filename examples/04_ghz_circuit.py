"""Build a 5-qubit GHZ state and see bigger entangled states lose coherence faster."""
import digital_qubit as dq

n = 5
c = dq.Circuit(n).add("h", 0)
for q in range(n - 1):
    c.add2("cx", q, q + 1)
print(c.to_text())

p = c.state().probs()
print(f"P(00000) = {p[0]:.3f}   P(11111) = {p[-1]:.3f}")
for t in [0, 5, 10]:
    rho = c.noisy(t, T1=50, T_phi=40)
    print(f"t = {t:>2}: coherence |<00000|rho|11111>| = {abs(rho.rho[0, -1]):.4f}")
