"""digital_qubit: a software qubit in pure NumPy.

Physics engine (1 and 2 qubits, T1/T2 noise, entanglement) verified
against Qiskit, plus a learned neural model (bagged MLP ensemble).
"""
__version__ = "0.6.0"

from .paulis import I2, X, Y, Z, H, S, S_DAG, T, CNOT, CZ, YY, rx, ry, rz, phase
from .single import (Sphere, StateArrow, ZAxis, XAxis, YAxis, Equator,
                     Theta, Phi, Projection)
from .noise import Damping
from .two import TwoQubit, Entangle, PairNoise, ptrace_A, ptrace_B
from .nqubit import NQubit, NDensity, gate_matrix, SWAP
from .circuit import Circuit
from .checker import Checker
from .device2 import TwoQubitDevice, TwoQubitDataset
from .model2 import Ensemble2, build_ensemble2
from .checker2 import TwoChecker, tomography, death_time
from .device import QubitDevice, QubitDataset
from .model import (MLP, train_mlp, convergence, best_at, make_training_set, Ensemble, build_ensemble,
                    log_fit, read_T1, read_T2, T2_with_uncertainty)
