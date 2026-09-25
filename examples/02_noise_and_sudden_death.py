"""Watch noise destroy a Bell pair's entanglement - it dies at a finite time."""
import digital_qubit as dq

T1, T_phi = 50.0, 40.0
pn = dq.PairNoise()
rho0 = dq.Entangle().bell(0).density()
print("single-qubit T2 =", round(dq.Damping.T2(T1, T_phi), 2))
for t in [0, 5, 10, 15, 20, 21, 22, 30]:
    rho = pn.evolve(rho0, t, T1, T_phi)
    print(f"t = {t:>3}: concurrence {pn.concurrence(rho):.3f}   best CHSH {pn.chsh_max(rho):.3f}")
