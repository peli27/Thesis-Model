# Findings — 19 October 2026

The nonlinear one-mode approximation is an excellent estimate of the existing seven-mode peak and RMS displacement across these 200 conditions. It is worth investigating as a new physics target for Model 5. For Model 4, its cost and already tiny error raise a more fundamental question: would a residual network provide enough improvement over simply running one nonlinear mode?

## Accuracy

All errors below are against the stored seven-mode nonlinear response, in millimetres.

| Estimate | Response | RMSE | MAE | 95th-percentile absolute error | Largest absolute error |
|---|---|---:|---:|---:|---:|
| Existing linear analytical | Peak | 2.166992 | 0.823251 | 3.668196 | 20.036645 |
| Nonlinear one mode | Peak | 0.000242 | 0.000175 | 0.000488 | 0.000778 |
| Existing linear analytical | RMS | 1.523930 | 0.536403 | 1.922977 | 14.616119 |
| Nonlinear one mode | RMS | 0.000171 | 0.000123 | 0.000345 | 0.000542 |

The largest nonlinear one-mode relative errors were 0.1127% for peak and 0.1145% for RMS. No case exceeded even 1%. The 95th-percentile relative errors were 0.0814% and 0.0972%. These findings held across the sampling groups: peak RMSE was 0.000020 mm for low forcing, 0.000203 mm for the general near-resonance group, and 0.000412 mm for the strongest nonlinear group.

By comparison, the linear estimate exceeded 5% error in 142/200 peak predictions and 90/200 RMS predictions. Its worst case was diagnostic case 185 (difficult-response source row 371), at frequency ratio 1.004335 and pressure 20.21 Pa: peak error 20.04 mm (656%) and RMS error 14.62 mm (854%). The largest absolute one-mode errors occurred in case 91 (main source row 4777), while its largest relative errors occurred in case 73 (main source row 1712). `largest_errors.csv` retains both rankings with full inputs.

## What explains the difference?

The linear formula cannot capture amplitude-dependent stretching stiffness. The nonlinear one-mode integration retains exactly that term, along with the original forcing, damping, initial conditions and time window. Its close agreement with seven modes indicates that the first mode dominates these displacement summaries over the tested loading range. This does not establish accuracy for other response locations, higher-frequency excitation, stresses, different loading shapes or arbitrary initial conditions.

There is also a response-definition difference: the existing analytical estimate assumes a fully settled sinusoid, whereas the dataset measures 5–10 seconds after starting from rest. Low forcing alone therefore does not guarantee a negligible peak error from the analytical formula. The supplementary linear time-domain calculation uses the same finite window as the nonlinear models. It still has substantial errors on the 30 checked cases (peak RMSE 2.194 mm; RMS RMSE 1.218 mm), so changing the window alone does not eliminate the need for nonlinearity. Those 30-case metrics must not be compared directly with the 200-case metrics as if they were the same sample.

## Computation cost

| Approach | Median time per condition | Measurement |
|---|---:|---|
| Linear analytical | 0.000000636 s | Amortised over vectorised batches of 200 |
| Nonlinear one mode | 0.234 s | 30 matched timing conditions |
| Nonlinear seven modes | 1.332 s | Same 30 conditions |

The median paired speedup was 5.46 times. Mean times were 0.252 s and 1.314 s. Individual one-mode times ranged from 0.112 to 0.500 s; seven-mode times ranged from 0.792 to 2.086 s. The median one-mode time over all 200 initial runs was 0.219 s. These timings include ordinary solver and reconstruction work but exclude startup and file I/O. Analytical batch timing is not single-request latency.

One mode is cheaper than seven, but remains an ODE solve. At the measured mean, 5000 one-mode estimates would take roughly 21 minutes serially, versus approximately 109 minutes for seven modes on these timing conditions. These are rough extrapolations, not full-dataset measurements. Reused reference labels cost no additional simulation time, so the seven-mode figures come from actual reruns rather than CSV loading.

## Implications for Models 4 and 5

**Model 5: justified for a controlled future experiment.** Precomputing and caching one-mode estimates would provide a much more faithful peak/RMS regularisation target at a practical offline cost. In the current implementation these targets enter the training loss; no ODE solve is required at inference. A new regularisation weight would need evaluation because the target bias changes substantially. This diagnostic does not show that stronger physics regularisation will improve a trained network.

**Model 4: accurate enough, but only conditionally attractive.** The residual becomes very small, which could simplify the learning task, but its scale and numerical sensitivity change. The existing trained residual network cannot be used with a replaced baseline. Moreover, the existing Model 4 prediction pipeline adds the physics estimate at inference, so a direct replacement would introduce roughly 0.23 seconds of physics computation per new condition. That weakens the case for a fast neural surrogate. Before retraining, compare a proposed hybrid against the standalone nonlinear one-mode solver: the latter already gives below 0.115% relative error here. The diagnostic supports a physics-accuracy investigation, not an unconditional deployment recommendation.

## Scope and checks

The experiment used 40 low-forcing cases (pressure <=3 Pa), 30 below resonance, 30 above resonance, 50 general near-resonance cases, and 50 confirmed nonlinear cases. Twenty-five of the nonlinear cases came from the upper quartile of the existing strong subset. Sampling is intentionally balanced and uses stored response indicators; pooled errors are not population-weighted performance estimates.

All 200 seven-mode reference pairs were reused. Thirty reruns reproduced stored peaks to within 4.49e-12 m and RMS values to within 9.87e-12 m. This is far smaller than the one-mode discrepancies. Simulator and source metadata hashes were checked. Hash comparison confirmed that all 587 monitored existing project files were unchanged. No training scripts, network weights or pre-existing results were changed, and no harmonic balance or retraining was performed.

The diagnostic uses the existing seven-mode reference as requested; it is not a new modal-convergence or solver-convergence study. The extremely small one-mode errors are evidence about this particular simulator, response definition and parameter range, rather than a general claim about nonlinear beams.

See `parity.png`, `errors_by_frequency.png`, `timing.png` and `response_histories.png`; all numeric details are in the adjacent CSV files. Settings and environment versions are recorded in `metadata.json`, and rerun instructions are in `../README.md`.
