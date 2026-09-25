"""The examples must keep working: run them as part of the test suite."""
import pathlib
import runpy
import pytest

EX = pathlib.Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("name", ["01_bell_pair.py", "02_noise_and_sudden_death.py", "04_ghz_circuit.py"])
def test_example_runs(name, capsys):
    runpy.run_path(str(EX / name), run_name="__main__")
    assert capsys.readouterr().out.strip()


@pytest.mark.slow
def test_learning_example_runs():
    runpy.run_path(str(EX / "03_learn_a_qubit.py"), run_name="__main__")
