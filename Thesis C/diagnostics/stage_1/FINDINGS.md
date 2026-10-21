# Stage 1 — implementation and saved-artifact audit

Completed 20 October 2026. Scope: inspect original Models 4 and 5, their inference path, baseline, dataset definitions, saved artifacts and previous lambda sensitivity experiment. No training, model changes, new approximation, or ODE simulations were performed.

## Assessment

**No implementation error was found in residual signs, units, inverse scaling, displacement reconstruction, output order, physics-loss standardisation or train/test separation in the inspected original training paths.** Saved artifacts are consistent with the current dataset, fixed split, training histories and existing Test 1 provenance. This supports implementation consistency; it does not prove that every historical run used the current source.

There are confirmed differences in objective weighting and confirmed nonphysical predictions. These must be distinguished from possible explanations of the accuracy gap. Stage 1 identifies mechanisms and constraints for later experiments; it does not establish which change would improve either model.

## 1. Source and training setup

Primary training scripts:

- `models/model_4_residual/train_residual_nn.py`
- `models/model_5_physics_regularised/train_physics_regularised_nn.py`
- Baseline: `models/model_1_baseline/train_baseline_nn.py`

The main inference implementation is `evaluation/evaluate_basic_interpolation.py`, especially `prepare_inputs`, `validate_checkpoint`, `validate_scaler` and `predict`. The unseen-geometry, uncertainty and difficult-response evaluations reuse this shared implementation. The reduced-data evaluator constructs the same residual targets and physics-regularised loss, but trains separate artifacts. Its source was reviewed; its many individual checkpoints were not independently replayed here.

All three original networks use seven inputs, hidden layers 64–64–32, ReLU, six linear outputs, Adam at 0.001, batch size 64, seed 42, maximum 500 epochs and patience 30. There is no explicit positivity constraint. Each feature/target is standardised independently using training rows only. Architecture and optimiser settings are broadly matched, but the optimisation objectives are different.

The actual fixed split contains **3,499 training, 750 validation and 751 test rows**, totalling 5,000. The baseline's fractional split construction yields this small departure from 3,500/750/750. It is not a dataset-size mismatch. The split partitions all rows exactly once, with no overlapping indices.

## 2. Model 4 checklist

| Check | Finding |
|---|---|
| Units and definitions | Inputs use metres, Pa, kg/m³ and Hz; angular excitation frequency is converted using 2π. Both analytical outputs and displacement labels are metres at x/L = 0.65. |
| Residual sign | Targets are `true − physics`; inference adds physics back. Exact algebraic reconstruction passes numerical checks. |
| Scaling | The target scaler fits the four frequency labels and two displacement residuals on training rows only. Saved means, variances and sample counts match recomputed training statistics. |
| Reconstruction | Inference first inverse-transforms all six network outputs, then adds analytical displacement to columns 4:6 only. No addition to standardised quantities. |
| Frequencies | f1–f4 are direct Hz predictions, not residuals. Labels agree with the simulator's modal-frequency formula including rotary inertia. Their scaler statistics equal the baseline's. |
| Physics parity | The original Model 4 and Model 5 estimators agree exactly over all 5,000 cases. Shared inference and sensitivity-script estimates agree to tight numerical tolerances. |
| Saved artifacts | Architecture, target order, residual definition, width 0.04 m, location 0.65, scalers and recorded best validation objective agree. The saved residual-target CSV also matches recomputation. |
| Validation | Correctly measures standardised residual-target MSE across all six outputs. Best epoch 124; training stops after epoch 154. This is not the same weighting of final physical errors as the baseline objective. |

The analytical modal mass, stiffness, damping and force follow the first-mode reduction of the simulator:

`M = rho*(A + I*k²)*L/2`, `K = E*I*k⁴*L/2`, `C = 2*zeta*M*sqrt(K/M)`, `Q = 2*pressure*b*L/pi`.

The displacement amplitude is `abs(Q*sin(pi*0.65)/sqrt((K − M*omega²)² + (C*omega)²))`; RMS is amplitude/√2. Pressure-to-line-load conversion, modal participation and rotary inertia are consistent. No sign or factor-of-width error was found.

### Confirmed objective-weighting difference

For a displacement output, final physical error equals residual-prediction error: `(physics + predicted_residual) − truth = predicted_residual − residual_target`. However, training divides that error by the **residual** training standard deviation, whereas the baseline divides by the **true-response** training standard deviation.

| Training statistic | Peak | RMS |
|---|---:|---:|
| Direct-response standard deviation, mm | 1.05591 | 0.670912 |
| Residual standard deviation, mm | 2.26546 | 1.61426 |
| Residual/direct standard-deviation ratio | 2.1455 | 2.4061 |
| Dimensional squared-error coefficient relative to baseline | 0.21724 | 0.17274 |

Consequently, relative to the unchanged frequency-error weights, Model 4 places only about 22% and 17% of the baseline's weight on a given dimensional peak and RMS squared error. Standardising residuals is internally correct, but it does not hold the evaluation objective constant across the two strategies. It also affects validation selection. Comparing their reported standardised losses directly would be misleading.

The largest 1% of training residuals by magnitude account for 70.4% of peak and 71.1% of RMS residual squared magnitude. The largest training analytical peak is 48.90 mm, versus a maximum true peak of 5.424 mm; these maxima need not refer to the same case. Peak residuals span −45.85 to +4.143 mm. This confirms a heavy-tailed correction task involving substantial cancellation. It does not independently establish an optimisation failure.

All seven varying physical/loading parameters are supplied, and the analytical estimate is a deterministic function of them. There is no demonstrated missing independent input. Providing the estimate or frequency ratio explicitly could change how easily the MLP learns the correction; that is a representation hypothesis for a later controlled experiment.

## 3. Model 5 checklist and exact loss interpretation

| Check | Finding |
|---|---|
| Compatible scales | Data and physics targets use the same training-label mean and standard deviation for each displacement. No mixture of dimensional and standardised quantities was found. |
| Intended outputs | Physics MSE uses only columns 4:6, peak and RMS. Frequencies are not directly penalised, although shared hidden layers can transmit optimisation effects. |
| Saved weight | Checkpoint records λ = 0.01, matching source. |
| Genuine constraint | A differentiable, nonzero-gradient penalty pulls predicted peak/RMS toward the linear estimates. It is a soft approximate-response target, not a nonlinear equation residual or a hard physical constraint. |
| Model selection | Both training and validation use data MSE + λ times physics MSE. Checkpoint minimises validation total. Best epoch 123; stopped after 153. |
| Leakage | Scalers fit training rows only. Validation labels guide stopping; test labels are not used for gradient updates or checkpoint selection in the inspected training source. Computing deterministic physics for every row is not target leakage. |

For batch size N, the implemented objective is

`L = (1/(6N))*sum_all_outputs((z−t)²) + lambda*(1/(2N))*sum_displacements((z−p)²)`.

Thus each displacement's physics-to-data coefficient ratio is **3λ**, not λ. At λ = 0.01 it is 0.03. This is correct for the written two-MSE objective, but must be stated when comparing weights or implementations. The audit verifies the gradient against this formula using the extracted original loss function.

For an independently adjustable displacement prediction, completing the square gives an effective target `(t + 3*lambda*p)/(1 + 3*lambda)` and displacement coefficient `(1 + 3*lambda)` relative to each frequency term. At λ = 0.01, physics contributes **2.913%** to this target blend. This is an exact objective identity, not a claim that the finite shared network reaches that optimum. Regularisation here can be understood as approximate-label blending plus a change in output weighting.

At the saved validation checkpoint:

- Data loss: 0.007438515.
- Physics loss: 4.011004.
- Weighted physics term: 0.04011004; total: 0.04754856.
- Weighted physics term / data loss: 5.392.

The large scalar physics term does **not** establish gradient dominance: completing the square reveals an irreducible prediction-independent term due to disagreement between physics and labels. Gradient magnitudes and directions would need separate measurement. Large reported total losses cannot be compared to baseline data losses as though they were the same objective.

Combined-loss stopping could select a less accurate checkpoint in principle. **In this saved run, the smallest logged validation data loss also occurs at epoch 123.** There is no evidence of a checkpoint-selection penalty within this recorded trajectory. A different stopping rule could still alter the length of a future run; later unobserved epochs cannot be assessed from this history.

## 4. Confirmed limitations and possible causes

### Confirmed observations / limitations

1. **Different error weighting:** Model 4's residual standardisation changes the physical displacement/frequency trade-off; Model 5 changes both displacement target and weighting. Matching architecture does not make these identical optimisation comparisons.
2. **Physical response definitions differ:** analytical physics assumes a steady single sinusoid, but simulator labels are sampled peak absolute displacement and sampled RMS over 5–10 seconds of a zero-initial-condition, seven-mode nonlinear simulation (10 s total, 0.002 s output interval). The location and units match; the underlying response approximation/window do not. Deviations can include transient/window, higher-mode and nonlinear effects. Calling all disagreement “nonlinearity error” would be unsupported.
3. **Nonphysical outputs occur:** on the 751-row Test 1 set, negative peak/RMS counts are 72/84 for Model 4, 45/36 for Model 5, and 11/4 for baseline. Negative physical magnitudes are invalid. Negative residual targets themselves are valid. Model 5's soft positive-target penalty does not ensure positive final predictions. Predictions were not clipped during this audit.
4. **Minor training-history reporting issue:** scripts average batch losses equally rather than weighting by batch size. The last of 55 training batches contains 43 samples, so the logged training average slightly overweights it. Backpropagation still uses the intended per-batch mean; validation uses the full set and is unaffected. This is not evidence for the large Model 4 gap.
5. **Historical provenance is incomplete:** original checkpoints lack embedded dataset/source/scaler hashes and environment/optimiser state. Existing Test 1 metadata links the current artifacts by hashes, and recomputed losses support consistency. They cannot establish the exact source revision or hardware used during training. This is a reproducibility limitation, not a confirmed artifact mismatch.

### Hypotheses requiring Stage 2 experiments

Sharp near-resonance physics, heavy residual tails, cancellation, output weighting, feature representation, shared-network optimisation and approximation bias are plausible contributors. No single one has been isolated causally. The weaker training fit as well as test fit of Model 4 merits investigation, but does not by itself diagnose a learning-rate or optimiser bug. No nonlinear one-mode replacement has been made or assumed necessary.

## 5. Reproduced performance and previous sensitivity experiment

The following Test 1 RMSEs are in **mm**, from the existing saved networks on the fixed test rows:

| Model | Peak RMSE | RMS RMSE |
|---|---:|---:|
| Baseline | 0.176883 | 0.110542 |
| Model 4 | 1.273936 | 0.898310 |
| Model 5, λ = 0.01 | 0.188349 | 0.117272 |

The previous lambda sensitivity script uses the **same main dataset and fixed train/validation split**, train-only scalers, architecture, seed, optimiser settings and combined-loss stopping. It runs on CPU and reports **validation**, not held-out test, errors:

| λ | Best epoch | Validation peak RMSE, mm | Validation RMS RMSE, mm |
|---|---:|---:|---:|
| 0.01 | 123 | 0.150771 | 0.0995055 |
| 0.1 | 284 | 0.462021 | 0.326902 |
| 1.0 | 125 | 1.565930 | 1.125693 |

The λ = 0.01 row agrees with the current saved Model 5 validation checkpoint. The larger-lambda rows are existing summary results, not independently retrained here; their checkpoints/histories are not saved by that script. There is no λ = 0 control or multi-seed replication in that experiment. It supports a validation-set trend under this implementation, not a general causal conclusion or a fair direct numerical comparison with the five-test tables. Historical human decisions about hyperparameters cannot be audited from training code alone.

## 6. Reproduction and evidence

Run from the project root using the existing environment:

```sh
/Users/112708_admin/opt/anaconda3/envs/thesis_py310/bin/python diagnostics/stage_1/run_audit.py
```

The script extracts pure estimator/loss functions with AST and never imports a training module. It loads existing artifacts, runs inference and writes only its own results. Python 3.10.18, NumPy 1.26.4, pandas 2.3.1, PyTorch 2.2.2 and scikit-learn 1.7.1 were used, matching existing Test 1 metadata. CPU inference uses one PyTorch thread.

Evidence files in `results/`:

- `audit.json`: checkpoint metadata, recomputed objectives, checks, package versions and SHA-256 hashes; **587 original files checked unchanged** before/after execution.
- `metrics.csv`: train/validation/test RMSE, MAE and negative counts for all six outputs and three saved models.
- `scalers.csv`: saved means, standard deviations and variances.
- `residual_statistics.csv`: correction ranges, scale changes and tail concentration by split.

Predictions reproduce the existing Test 1 outputs within floating-point tolerance; small CPU arithmetic differences are recorded separately for frequencies and displacements. No new test data were generated or used to tune a model. The next stage can select controlled alternatives using training and validation data, with the previously inspected test set explicitly treated as already observed.
