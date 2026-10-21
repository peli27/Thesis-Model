# One-mode nonlinear diagnostic

This folder is independent of training and evaluation. No networks are trained or loaded. All output goes to `results/`.

Run from the project root:

```sh
diagnostics/one_mode/.venv/bin/python diagnostics/one_mode/run_diagnostic.py
```

To recreate the isolated environment, use `python3 -m venv diagnostics/one_mode/.venv` and install `diagnostics/one_mode/requirements.txt` with its Python's `-m pip install -r` command.

The script extracts only `calculate_physics_estimates` from Models 4 and 5 using Python's AST, verifies their estimates agree exactly, and never imports their training modules. The nonlinear approximation calls the original `simulate_beam(N=1)`. The reference is the saved `N=7` dataset response. Both time-domain models use the original DOP853 solver, rotary inertia, modal damping, pressure participation, von Karman stretching, zero initial conditions, 10 s duration, 0.002 s saved timestep, `rtol=1e-8`, and `atol=1e-10`. Peak absolute displacement and sampled RMS are measured at x/L=0.65 over the second half (5–10 s inclusive).

For N=1, the existing modal equation reduces to `M1*q'' + C1*q' + K1*q + (E*A*L*k1**4/8)*q**3 = (2*b*p*L/pi)*sin(2*pi*f*t)`. No new approximation of the nonlinearity or integration method is introduced.

The original analytical estimate is a sinusoidal steady-state linear amplitude with RMS equal to peak/sqrt(2). This differs intentionally from the finite-window time-domain definition because it is the physics estimate actually used by the models. A supplementary linear N=1 integration on 30 cases helps expose this distinction; it is not part of the three-method timing comparison.

Sampling uses seed 20261019 and 200 unique input conditions. It deliberately balances operating regimes and oversamples strong responses; pooled metrics are not estimates of average error over the original 5000-case population. The 50 strong cases are selected using the existing seven-mode stiffness indicator eta >= 0.1. Half come from its upper quartile. No approximation errors are used in selection. The selected source row is zero-based and is retained for traceability.

All 200 reference peak/RMS pairs are reused. Thirty cases (five from each sampling group) are rerun with N=7 for timing and reference checks. A tolerance of `rtol=1e-5, atol=1e-10 m` allows platform/integration roundoff; actual differences are recorded. The simulator and main dataset hashes must match the existing difficult-response metadata. Existing project artifact hashes are compared before and after execution.

Timing excludes imports, file I/O and plotting, and includes solver setup and response reconstruction. Solver warmups are excluded. Paired solver calls alternate order. These are single timing observations per case, not rigorous microbenchmarks. Linear timings use 1000 vectorised batches of 200 conditions, reported per case; this amortises Python/pandas overhead and should not be read as single-request latency. All machine and package details are saved in metadata.

Outputs:

- `selected_cases.csv`, `case_results.csv`: inputs, source identifiers, reference and estimated outputs, individual nonlinear runtimes.
- `accuracy_metrics.csv`: overall and sampling-group RMSE, MAE, 95th percentile and maximum absolute errors in mm, relative errors and counts above 5%/10%.
- `largest_errors.csv`: ten largest absolute and relative errors per approximation and output, with full case context.
- `paired_solver_timing.csv`, `timing_summary.csv`: measured runtimes and reference reproducibility differences.
- `parity.png`, `errors_by_frequency.png`, `timing.png`, `response_histories.png`: comparison figures.
- `metadata.json`: settings, hashes, versions and artifact-preservation check.
- `FINDINGS.md`: interpretation of this completed run.

This diagnostic assesses physics-estimate quality only. It cannot establish the effect on trained Model 4 or Model 5 accuracy, choose a new regularisation weight, or establish convergence beyond the existing seven-mode reference.
