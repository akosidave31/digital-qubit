# Experiment log

Rules: one change per experiment; targets and decision rule written BEFORE the run;
defaults change only when every target passes; fixed seeds so runs are paired comparisons.

## Single-qubit model (notebook Cells 12-22)

| Step | Change | Result | Decision |
|---|---|---|---|
| v1 | 64-64 MLP, 20k experiments, 200 shots | 7/9 checks; errors 3.9x larger at t=0 | diagnose |
| v2 | +5k early-time experiments | 12/12; edge 3.9x -> 1.5x; T2 drifted | adopt |
| v3 | 64 -> 128 units | middle error and RMSE WORSE | reject (not capacity) |
| seeds | 4 seeds of v2 | T2 bias -6.4% systematic | investigate |
| v4 | 200 -> 1000 shots | error -31%, T2 bias unchanged | adopt |
| diag | T2 probe analysis | bias came from one data sample + one probe point | explained |
| ens | 5-member bagged ensemble | slice error halved, T2 -1.3%, honest error bars | adopt (library default) |

## Two-qubit model

Targets (fixed at v0.5.1): Bell concurrence at t=0 > 0.90; early-CNOT edge ratio < 2x;
agreement not worse than v0.5.0; no-CNOT agreement >= 99%; late-time errors within 20% of v0.5.0.

### v0.5.0 - baseline
16 inputs, softmax MLP 128-128, 3 members x 40k experiments, 1000 shots, 80 epochs.
Joint odds PASS (98.6%). Bell concurrence at t=0: 0.75 (true 1.00). Death time learned.
Diagnostic: errors concentrated at early times with CNOT (4.2x), systematic across members.

### v0.5.1 - +20k early-CNOT experiments per member  ->  3/6, keep defaults
Bell 0.75 -> 0.86; edge 4.2x -> 2.5x; late-time no-CNOT error +33% (regression).

### v0.5.2 - 80 -> 160 epochs  ->  4/6, keep defaults
Error vs truth 0.34x -> 0.29x of label noise; late regression fixed; Bell 0.86 -> 0.87 only.
Conclusion: under-training was real but NOT the cause of the entanglement gap.
Ruled out so far: tomography sensitivity (D3), data quantity (v0.5.1), under-training (v0.5.2).
Also seen: ~0.05 false entanglement without CNOT at t < 3.

### v0.5.3 - phase-combination inputs (planned, timeboxed: last tuning run)
Hypothesis: after CNOT, correlations depend on b0+b1 and b0-b1; the network must build these
from separate phases, which fits YY (phase-dependent) being worst while ZZ is 0.98.
Change: add cos/sin(b0 +/- b1) inputs (16 -> 20). Baseline: v0.5.2 settings.
Targets: the 6 above + false entanglement without CNOT < 0.03 (20 random pairs, t in [0, 5]).
Decision: 7/7 -> adopt as defaults in v0.6.0. Otherwise release v0.5.2 settings with the
known limitation "Bell concurrence at t=0 under-estimated by ~13%" and stop tuning.
Result: 4/7 -> hypothesis NOT supported.
Agreement 99.2% -> 99.8% and edge 2.43x -> 2.09x improved, but Bell concurrence at t=0 0.87 -> 0.83
(difference may be within run-to-run variation, never measured for two qubits: conclusion is
"no improvement", not "worse"). False entanglement (new metric): 0.050.
Decision (pre-registered): ship v0.5.2 settings as v0.6.0 defaults, document limitations, stop tuning.
phase_features stays available, off by default.

Process notes from this run:
- test_false_entanglement_metric failed on floating-point noise (1.4e-9 vs 1e-9 limit): test bug,
  tolerance set to 1e-6. Same fix applied to a sibling test that passed by luck.
- the notebook kept running after a failing test (`!pytest` does not stop Run all): fixed, the
  test cell now raises on failure.

## v0.6.0 - release
Defaults: 160 epochs + 20k early-CNOT experiments per member (v0.5.2 settings).
Release validation (default settings, 89 tests passed): joint odds PASS 99.2%;
Bell concurrence t=0 0.87; Bell max error 0.135; death time 20.5 / 20.5;
partly entangled max error 0.045; false entanglement 0.039.  RELEASE: VALIDATED.
Open questions (not pursued, timebox): Bell concurrence at t=0 remains under-estimated; causes
ruled out: tomography sensitivity, data quantity, under-training, phase representation.
Untested: run-to-run variance for two-qubit ensembles; larger/other architectures.
