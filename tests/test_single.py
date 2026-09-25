import numpy as np
from digital_qubit import (Sphere, StateArrow, ZAxis, XAxis, YAxis, Equator, Theta, Phi,
                           Projection, I2, X, Y, Z, H, S_DAG)

S = Sphere()


def test_sphere_poles():
    assert np.allclose(S.to_density([0, 0, 1]), [[1, 0], [0, 0]])
    assert np.allclose(S.to_density([0, 0, -1]), [[0, 0], [0, 1]])


def test_sphere_regions(rng, unit_dirs):
    surf = unit_dirs(300)
    inner = unit_dirs(300) * rng.uniform(0, 0.99, (300, 1))
    outer = unit_dirs(300) * rng.uniform(1.01, 2, (300, 1))
    assert all(S.classify(r) == "pure" and np.isclose(S.purity(r), 1)
               and S.is_physical(S.to_density(r)) for r in surf)
    assert all(S.classify(r) == "mixed" and S.purity(r) < 1
               and S.is_physical(S.to_density(r)) for r in inner)
    assert all(S.classify(r) == "invalid" and not S.is_physical(S.to_density(r)) for r in outer)
    assert np.isclose(S.purity([0, 0, 0]), 0.5)


def test_sphere_roundtrip(unit_dirs):
    for r in unit_dirs(300):
        assert np.allclose(S.to_vector(S.to_density(r)), r, atol=1e-12)


def test_state_basis_arrows():
    assert np.allclose(StateArrow(1, 0).vector(), [0, 0, 1])
    assert np.allclose(StateArrow(0, 1).vector(), [0, 0, -1])
    assert np.allclose(StateArrow(1, 1).vector(), [1, 0, 0])
    assert np.allclose(StateArrow(1, 1j).vector(), [0, 1, 0])


def test_state_arrow_consistent(rand1):
    for _ in range(300):
        s = rand1()
        assert np.isclose(np.linalg.norm(s.psi), 1)
        assert S.classify(s.vector()) == "pure"
        assert np.allclose(S.to_vector(s.density()), s.vector(), atol=1e-12)
        assert s.fidelity(StateArrow.from_vector(s.vector())) > 1 - 1e-12


def test_global_phase_invisible(rand1):
    s = rand1()
    g = StateArrow(*(np.exp(1j * 1.234) * s.psi))
    assert np.allclose(s.vector(), g.vector())


def test_z_probs_and_collapse(rand1, rng):
    Zx = ZAxis()
    for _ in range(200):
        s = rand1()
        assert np.isclose(Zx.probs(s)[0], (1 + s.vector()[2]) / 2)
        out, after = Zx.measure(s, rng)
        assert np.allclose(after.vector(), [0, 0, 1 if out == 0 else -1])
        assert all(Zx.measure(after, rng)[0] == out for _ in range(3))


def test_z_blind_to_phase():
    base = StateArrow.from_angles(1.0, 0.0)
    for phi in np.linspace(0, 2 * np.pi, 13):
        assert np.allclose(ZAxis().probs(StateArrow.from_angles(1.0, phi)), ZAxis().probs(base))


def test_z_sampling(rand1, rng):
    Zx = ZAxis()
    for _ in range(10):
        s = rand1()
        p1 = Zx.probs(s)[1]
        freq = Zx.sample(s, 20000, rng).mean()
        assert abs(freq - p1) < 4 * np.sqrt(p1 * (1 - p1) / 20000) + 1e-3


def test_x_axis(rand1, rng):
    Xx = XAxis()
    assert np.allclose(Xx.flip(StateArrow(1, 0)).vector(), [0, 0, -1])
    assert np.allclose(Xx.flip(StateArrow(*Xx.PLUS)).vector(), [1, 0, 0])
    for _ in range(200):
        s = rand1()
        assert np.isclose(Xx.probs(s)[0], (1 + s.vector()[0]) / 2)
        assert np.isclose(Xx.probs(s)[0], ZAxis().probs(s.apply(H))[0])
        a = rng.uniform(0, 2 * np.pi)
        x0, y0, z0 = s.vector()
        x1, y1, z1 = Xx.rotate(s, a).vector()
        assert np.allclose([x1, y1, z1], [x0, y0 * np.cos(a) - z0 * np.sin(a), y0 * np.sin(a) + z0 * np.cos(a)])


def test_y_axis(rand1, rng):
    Yx = YAxis()
    assert np.allclose(StateArrow(*Yx.PLUS_I).vector(), [0, 1, 0])
    for _ in range(200):
        s = rand1()
        assert np.isclose(Yx.probs(s)[0], (1 + s.vector()[1]) / 2)
        assert np.isclose(Yx.probs(s)[0], ZAxis().probs(s.apply(S_DAG).apply(H))[0])
        a = rng.uniform(0, 2 * np.pi)
        x0, y0, z0 = s.vector()
        x1, y1, z1 = Yx.rotate(s, a).vector()
        assert np.allclose([x1, y1, z1], [x0 * np.cos(a) + z0 * np.sin(a), y0, -x0 * np.sin(a) + z0 * np.cos(a)])


def test_pauli_algebra():
    assert np.allclose(X @ Y, 1j * Z) and np.allclose(Y @ Z, 1j * X) and np.allclose(Z @ X, 1j * Y)
    assert all(np.allclose(P @ P, I2) for P in (X, Y, Z, H))


def test_tomography(rand1, rng):
    Xx, Yx, Zx = XAxis(), YAxis(), ZAxis()
    for _ in range(10):
        s = rand1()
        est = np.array([1 - 2 * ax.sample(s, 20000, rng).mean() for ax in (Xx, Yx, Zx)])
        assert np.linalg.norm(est - s.vector()) < 0.05


def test_equator(rng):
    Eq = Equator()
    for p in rng.uniform(0, 2 * np.pi, 200):
        e = Eq.make(p)
        assert np.allclose(e.vector(), [np.cos(p), np.sin(p), 0])
        assert np.allclose(ZAxis().probs(e), [0.5, 0.5])
        assert np.isclose(ZAxis().probs(e.apply(H))[0], np.cos(p / 2) ** 2)
        a = rng.uniform(-np.pi, np.pi)
        moved = Eq.rz(e, a)
        assert Eq.on_equator(moved)
        d = (Eq.phase(moved) - (p + a) + np.pi) % (2 * np.pi) - np.pi
        assert abs(d) < 1e-9


def test_theta(rng):
    Th = Theta()
    for t in rng.uniform(0, np.pi, 200):
        s = Th.make(t, rng.uniform(0, 2 * np.pi))
        assert abs(Th.get(s) - t) < 1e-7
        assert np.isclose(ZAxis().probs(s)[0], np.cos(t / 2) ** 2)
    true_t = rng.uniform(0.3, np.pi - 0.3, 50)
    est = [Th.estimate(rng.binomial(10000, 1 - Th.p0(t)), 10000) for t in true_t]
    assert np.mean(np.abs(np.array(est) - true_t)) < 0.02


def test_phi_interference_and_learning(rng):
    Ph, Th, Xx, Yx = Phi(), Theta(), XAxis(), YAxis()
    assert np.isclose(Ph.interferometer(0), 1) and np.isclose(Ph.interferometer(np.pi), 0)
    lams = np.linspace(0, 2 * np.pi, 91)
    assert np.allclose([Ph.interferometer(l) for l in lams], np.cos(lams / 2) ** 2)
    errs = []
    for t, p in zip(rng.uniform(0.5, np.pi - 0.5, 30), rng.uniform(0, 2 * np.pi, 30)):
        s = Th.make(t, p)
        est = Ph.estimate(rng.binomial(10000, Xx.probs(s)[1]), rng.binomial(10000, Yx.probs(s)[1]), 10000)
        errs.append(abs((est - p + np.pi) % (2 * np.pi) - np.pi))
    assert np.mean(errs) < 0.03


def test_projection_and_dephasing(rand1, rng):
    Pr = Projection()
    for _ in range(200):
        s = rand1()
        r = s.vector()
        assert np.isclose(Pr.horizontal(r) ** 2 + Pr.vertical(r) ** 2, 1)
        assert np.isclose(Pr.horizontal(r), Pr.coherence(s.density()))
        p = rng.uniform(0, 0.5)
        r1 = S.to_vector(Pr.dephase(s.density(), p))
        assert np.isclose(Pr.horizontal(r1), (1 - 2 * p) * Pr.horizontal(r))
        assert np.isclose(Pr.vertical(r1), Pr.vertical(r))
