import sys
import numpy as np
import pytest
import digital_qubit as dq
from digital_qubit import Circuit, NQubit, NDensity


def test_ghz_circuit_matches_engine():
    c = Circuit(3).add("h", 0).add2("cx", 0, 1).add2("cx", 1, 2)
    direct = NQubit(3).gate("h", 0).cx(0, 1).cx(1, 2)
    assert np.allclose(c.state().psi, direct.psi)
    p = c.state().probs()
    assert np.isclose(p[0], 0.5) and np.isclose(p[7], 0.5)


def test_all_gate_types_run(rng):
    c = Circuit(3)
    for g in ["h", "x", "y", "z", "s", "t"]:
        c.add(g, int(rng.integers(3)))
    for g in ["rx", "ry", "rz", "p"]:
        c.add(g, int(rng.integers(3)), rng.uniform(0, 6))
    c.add2("cx", 2, 0).add2("cz", 0, 1).add2("swap", 1, 2)
    assert np.isclose(np.linalg.norm(c.state().psi), 1)
    assert len(c.ops) == 13


def test_undo_and_clear():
    c = Circuit(2).add("h", 0).add2("cx", 0, 1)
    c.undo()
    assert np.allclose(c.state().psi, NQubit(2).gate("h", 0).psi)
    c.clear()
    assert c.ops == [] and np.allclose(c.state().psi, NQubit(2).psi)
    c.undo()
    assert c.ops == []


def test_invalid_input_rejected():
    c = Circuit(2)
    for bad in (lambda: c.add("h", 2), lambda: c.add("rx", 0), lambda: c.add("foo", 0),
                lambda: c.add("cx", 0), lambda: c.add2("cx", 1, 1), lambda: c.add2("h", 0, 1),
                lambda: Circuit(0), lambda: Circuit(27)):
        with pytest.raises(ValueError):
            bad()
    assert c.ops == []


def test_noisy_matches_engine():
    c = Circuit(3).add("h", 0).add2("cx", 0, 1).add("ry", 2, 1.1)
    ref = NDensity.from_state(c.state()).idle(12.0, 50.0, 40.0)
    assert np.allclose(c.noisy(12.0, 50.0, 40.0).rho, ref.rho)


def test_text_diagram():
    c = Circuit(3).add("h", 0).add2("cx", 0, 2).add("rx", 1, 1.5)
    lines = c.to_text().split("\n")
    assert len(lines) == 3
    assert "H" in lines[0] and "●" in lines[0]
    assert "┼" in lines[1] and "Rx(1.50)" in lines[1]
    assert "⊕" in lines[2]
    assert len({len(l) for l in lines}) == 1


def test_core_library_does_not_need_ui_packages():
    assert "digital_qubit.app" not in sys.modules or hasattr(dq, "Circuit")
    assert not hasattr(dq, "launch")
