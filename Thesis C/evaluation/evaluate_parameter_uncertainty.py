"""
TEST 3 - PARAMETER UNCERTAINTY AND HIDDEN ERROR

Evaluate models at +/-2.5%, +/-5% and +/-10% property-error bands.
Test 3A supplies actual perturbed E, rho and h to each model. 
Test 3B supplies nominal properties while comparing with the same perturbed simulator responses.

"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import evaluate_basic_interpolation as shared


# ----- TEST SETTINGS -----

TEST_NAME = "test_3_parameter_uncertainty"
LEVELS = (2.5, 5.0, 10.0)
N_NOMINAL_CASES = 500
PERTURBED_COLUMNS = ("E", "rho", "h")
TRAINING_RANGES = {
    "L": (1.6, 2.4),
    "h": (0.008, 0.012),
    "E": (185e9, 225e9),
    "rho": (7450.0, 8250.0),
    "zeta": (0.005, 0.020),
    "pressure_amplitude": (0.1, 30.0),
    "frequency_ratio": (0.50, 1.50),
}


# ----- DATASET AND PAIRED-EXPERIMENT CHECKS -----

def input_view(cases, known):
    """Use actual inputs for 3A; nominal inputs for 3B and the reference."""
    if known:
        return cases[shared.INPUT_COLUMNS].copy()
    inputs = cases[[f"nominal_{c}" for c in shared.INPUT_COLUMNS]].copy()
    inputs.columns = shared.INPUT_COLUMNS
    return inputs


def validate_dataset(data, original, split):
    required = shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS + [
        "sample_id", "nominal_case_id", "nominal_row_index", "uncertainty_level_percent", "frequency_ratio",
        "E_error_percent", "rho_error_percent", "h_error_percent",
    ] + [f"nominal_{c}" for c in shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS + ["frequency_ratio"]]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Uncertainty dataset is missing columns: {missing}")
    if not np.all(np.isfinite(data[required].to_numpy(float))):
        raise ValueError("Uncertainty dataset contains non-finite values.")
    if not data.sample_id.is_unique or len(data) != N_NOMINAL_CASES * len(LEVELS):
        raise ValueError("Expected 1,500 cases with unique sample IDs.")
    if set(data.uncertainty_level_percent.unique()) != set(LEVELS):
        raise ValueError("Expected uncertainty bands of 2.5%, 5% and 10%.")

    reference_ids = None
    reference_factors = None
    for level in LEVELS:
        group = data[data.uncertainty_level_percent == level].sort_values("nominal_case_id")
        if len(group) != N_NOMINAL_CASES or not group.nominal_case_id.is_unique or not group.nominal_row_index.is_unique:
            raise ValueError(f"Expected 500 unique nominal cases at band {level}%.")
        row_values = group.nominal_row_index.to_numpy(float)
        if not np.all(row_values == np.floor(row_values)):
            raise ValueError("Nominal row indices must be integers.")
        indices = row_values.astype(int)
        if not np.all(np.isin(indices, split["test_indices"])):
            raise ValueError("Nominal cases must belong to the original held-out test split.")

        columns = shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS + ["frequency_ratio"]
        nominal_values = group[[f"nominal_{c}" for c in columns]].to_numpy(float)
        if not np.allclose(nominal_values, original.iloc[indices][columns].to_numpy(float), rtol=1e-10, atol=1e-12):
            raise ValueError("Stored nominal inputs or outputs disagree with the original dataset.")
        ids = group[["nominal_case_id", "nominal_row_index"]].to_numpy()
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(ids, reference_ids):
            raise ValueError("Uncertainty bands do not use the same paired nominal beams.")

        factors = []
        for column in shared.INPUT_COLUMNS:
            actual = group[column].to_numpy(float)
            nominal = group[f"nominal_{column}"].to_numpy(float)
            if column in PERTURBED_COLUMNS:
                errors = group[f"{column}_error_percent"].to_numpy(float)
                if np.any(np.abs(errors) > level + 1e-10):
                    raise ValueError(f"{column} errors exceed the prescribed {level}% band.")
                if not np.allclose(actual, nominal * (1.0 + errors / 100.0), rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Incorrect perturbation calculation for {column}.")
                factors.append(errors / level)
            elif not np.allclose(actual, nominal, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{column} should remain fixed in this experiment.")

        factors = np.column_stack(factors)
        if reference_factors is None:
            reference_factors = factors
        elif not np.allclose(factors, reference_factors, rtol=1e-10, atol=1e-12):
            raise ValueError("Error factors must be paired consistently across uncertainty bands.")

        f1 = shared.calculate_physics_features(group)[:, 0]
        if not np.allclose(group.f1, f1, rtol=1e-9, atol=1e-12):
            raise ValueError("Actual f1 does not match the actual beam properties.")
        if not np.allclose(group.excitation_frequency / f1, group.frequency_ratio, rtol=1e-9, atol=1e-12):
            raise ValueError("Actual frequency ratios are inconsistent.")
        if (group[shared.OUTPUT_COLUMNS] < 0).any().any():
            raise ValueError("True simulator outputs cannot be negative.")


def collect_metrics(true, predicted, ratios, name, mode, level):
    rows = shared.metric_rows(true, predicted, name)
    regimes = np.where(ratios < 0.85, "below_resonance", np.where(ratios <= 1.20, "near_resonance", "above_resonance"))
    for regime in ("below_resonance", "near_resonance", "above_resonance"):
        mask = regimes == regime
        if mask.any():
            rows.extend(shared.metric_rows(true[mask], predicted[mask], name, regime))
    for row in rows:
        row["test_case"] = mode
        row["uncertainty_level_percent"] = level
    return rows


def prediction_records(cases, inputs, predicted, true, mode, level):
    records = cases.copy()
    records["test_case"] = mode
    records["uncertainty_level_percent"] = level
    for column in shared.INPUT_COLUMNS:
        records[f"model_input_{column}"] = inputs[column].to_numpy(float)
    for i, output in enumerate(shared.OUTPUT_COLUMNS):
        records[output] = true[:, i]
        records.rename(columns={output: f"true_{output}"}, inplace=True)
        records[f"predicted_{output}"] = predicted[:, i]
        records[f"error_{output}"] = predicted[:, i] - true[:, i]
        records[f"absolute_error_{output}"] = np.abs(predicted[:, i] - true[:, i])
    return records


def timed_prediction(inputs, prefix, checkpoint, model, x_scaler, y_scaler, repeats):
    prediction = shared.predict(inputs, prefix, checkpoint, model, x_scaler, y_scaler)
    for _ in range(3):
        shared.predict(inputs, prefix, checkpoint, model, x_scaler, y_scaler)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        shared.predict(inputs, prefix, checkpoint, model, x_scaler, y_scaler)
        elapsed.append(time.perf_counter() - start)
    return prediction, float(np.median(elapsed))


# ----- LOAD AND RUN THE MODELS -----

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timing-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be at least 1")
    root = args.project_root.resolve()
    dataset_file = root / "data" / "test_datasets" / "parameter_uncertainty_dataset.csv"
    metadata_file = dataset_file.with_name("parameter_uncertainty_generation_metadata.json")
    main_file = root / "data" / "main" / "nonlinear_beam_dataset.csv"
    split_file = root / "data" / "splits" / "main_split_indices.npz"

    # ----- 1. VALIDATE DATASET IDENTITY AND PAIRED CASES -----

    metadata = json.loads(metadata_file.read_text())
    if metadata["main_dataset_sha256"] != shared.file_hash(main_file) or metadata["split_sha256"] != shared.file_hash(split_file):
        raise ValueError("The original dataset or split has changed since uncertainty generation.")
    data = pd.read_csv(dataset_file)
    original = pd.read_csv(main_file)
    required = shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS
    if any(column not in original for column in required) or not np.all(np.isfinite(original[required].to_numpy(float))):
        raise ValueError("Main dataset contains missing columns or non-finite values.")
    split = shared.load_split(split_file, len(original))
    validate_dataset(data, original, split)
    train = original.iloc[split["train_indices"]]
    groups = {level: data[data.uncertainty_level_percent == level].sort_values("nominal_case_id").reset_index(drop=True) for level in LEVELS}
    first_group = groups[LEVELS[0]]
    nominal_inputs = input_view(first_group, known=False)
    nominal_true = first_group[[f"nominal_{c}" for c in shared.OUTPUT_COLUMNS]].to_numpy(float)

    # Zero-uncertainty records describe the nominal structure, not a perturbed case.
    reference_cases = first_group.copy()
    reference_cases["sample_id"] = -1  # No new simulation ID for reused nominal responses.
    for column in shared.INPUT_COLUMNS:
        reference_cases[column] = nominal_inputs[column]
    for column in PERTURBED_COLUMNS:
        reference_cases[f"{column}_error_percent"] = 0.0
    reference_cases["frequency_ratio"] = reference_cases["nominal_frequency_ratio"]

    print("TEST 3 - PARAMETER UNCERTAINTY AND HIDDEN ERROR")
    print("500 paired beams per band; 3A uses actual inputs and 3B uses nominal inputs.")
    torch.set_num_threads(1)
    completed = []

    # ----- 2. RESTORE EACH MODEL AND EVALUATE THE NOMINAL REFERENCE -----

    for folder, prefix, name, input_count in shared.MODEL_SPECS:
        saved = root / "models" / folder / "saved_model"
        model_file = saved / f"{prefix}_nn_model.pt"
        x_file = shared.find_scaler(saved, prefix, "X")
        y_file = shared.find_scaler(saved, prefix, "Y")
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        shared.validate_checkpoint(checkpoint, prefix, input_count)
        x_scaler, y_scaler = joblib.load(x_file), joblib.load(y_file)
        train_inputs, physics = shared.prepare_inputs(train, prefix, checkpoint)
        train_targets = train[shared.OUTPUT_COLUMNS].to_numpy(float).copy()
        if prefix == "residual":
            train_targets[:, 4:6] -= physics
        shared.validate_scaler(x_scaler, train_inputs, f"{prefix} input scaler")
        shared.validate_scaler(y_scaler, train_targets, f"{prefix} output scaler")
        model = shared.SurrogateNN(input_count)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        nominal_prediction, nominal_seconds = timed_prediction(nominal_inputs, prefix, checkpoint, model, x_scaler, y_scaler, args.timing_repeats)
        nominal_ratios = first_group.nominal_frequency_ratio.to_numpy(float)
        metrics = collect_metrics(nominal_true, nominal_prediction, nominal_ratios, name, "Nominal reference", 0.0)
        records = [prediction_records(reference_cases, nominal_inputs, nominal_prediction, nominal_true, "Nominal reference", 0.0)]
        timings = [{"model": name, "test_case": "Nominal reference", "uncertainty_level_percent": 0.0, "median_batch_seconds": nominal_seconds}]

        # ----- 3. TEST 3A AND 3B AGAINST IDENTICAL PERTURBED TARGETS -----

        for level, cases in groups.items():
            true = cases[shared.OUTPUT_COLUMNS].to_numpy(float)
            actual_inputs = input_view(cases, known=True)
            actual_prediction, actual_seconds = timed_prediction(actual_inputs, prefix, checkpoint, model, x_scaler, y_scaler, args.timing_repeats)

            # Same nominal beams/order at every band: reuse exactly the same hidden-input prediction.
            assumed_inputs = input_view(cases, known=False)
            if not np.array_equal(assumed_inputs.to_numpy(), nominal_inputs.to_numpy()):
                raise ValueError("Test 3B nominal inputs changed across bands.")

            for mode, inputs, prediction, seconds in (
                ("3A Known perturbation", actual_inputs, actual_prediction, actual_seconds),
                ("3B Unknown perturbation", assumed_inputs, nominal_prediction, nominal_seconds),
            ):
                metrics.extend(collect_metrics(true, prediction, cases.frequency_ratio.to_numpy(float), name, mode, level))
                records.append(prediction_records(cases, inputs, prediction, true, mode, level))
                timings.append({"model": name, "test_case": mode, "uncertainty_level_percent": level, "median_batch_seconds": seconds})

            known_rmse = np.sqrt(np.mean((actual_prediction[:, 4:6] - true[:, 4:6])**2, axis=0))
            hidden_rmse = np.sqrt(np.mean((nominal_prediction[:, 4:6] - true[:, 4:6])**2, axis=0))
            print(f"{name} | +/-{level:g}% | peak RMSE: 3A={known_rmse[0]:.6e}, 3B={hidden_rmse[0]:.6e} m")

        timing_frame = pd.DataFrame(timings)
        timing_frame["device"] = "cpu"
        timing_frame["torch_threads"] = 1
        timing_frame["n_samples"] = len(first_group)
        timing_frame["timing_repeats"] = args.timing_repeats
        timing_frame["amortised_seconds_per_sample"] = timing_frame.median_batch_seconds / len(first_group)
        timing_frame["timing_note"] = np.where(
            timing_frame.test_case == "3B Unknown perturbation",
            "Reused timing of identical nominal-reference inputs",
            "Measured independently on this input batch",
        )
        provenance = {
            "model": name,
            "checkpoint_sha256": shared.file_hash(model_file),
            "input_scaler_sha256": shared.file_hash(x_file),
            "output_scaler_sha256": shared.file_hash(y_file),
        }
        completed.append((folder, pd.concat(records, ignore_index=True), pd.DataFrame(metrics), timing_frame, provenance))

    # ----- 4. SAVE INDIVIDUAL RESULTS -----

    collected = []
    for folder, predictions, metrics, timing, provenance in completed:
        results = root / "models" / folder / "results" / TEST_NAME
        results.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(results / "predictions.csv", index=False)
        metrics.to_csv(results / "metrics.csv", index=False)
        timing.to_csv(results / "prediction_timing.csv", index=False)
        collected.append(pd.read_csv(results / "metrics.csv"))

    comparison = root / "comparison" / "results"
    comparison.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(collected, ignore_index=True)
    combined.to_csv(comparison / f"{TEST_NAME}_metrics.csv", index=False)
    overall = combined[combined.regime == "all"]
    summary_rows = []
    for (name, mode, level), group in overall.groupby(["model", "test_case", "uncertainty_level_percent"], sort=False):
        row = {"model": name, "test_case": mode, "uncertainty_level_percent": level, "n_test_samples": int(group.n_samples.iloc[0])}
        for _, metric in group.iterrows():
            for key in ("RMSE", "MAE", "R2", "P95_absolute_error", "maximum_absolute_error"):
                row[f"{metric['output']}_{key}"] = metric[key]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    for (mode, level), group in summary.groupby(["test_case", "uncertainty_level_percent"]):
        baseline = group[group.model == "Baseline MLP"].iloc[0]
        for output in shared.OUTPUT_COLUMNS[4:]:
            key = f"{output}_RMSE"
            denominator = float(baseline[key])
            improvement = 100.0 * (denominator - group[key]) / denominator if denominator > 0 else np.nan
            summary.loc[group.index, f"{output}_RMSE_improvement_vs_baseline_percent"] = improvement
    for name, group in summary.groupby("model"):
        reference = group[group.test_case == "Nominal reference"].iloc[0]
        for output in shared.OUTPUT_COLUMNS[4:]:
            key = f"{output}_RMSE"
            denominator = float(reference[key])
            summary.loc[group.index, f"{output}_RMSE_ratio_to_nominal_reference"] = group[key] / denominator if denominator > 0 else np.nan

    comparison_file = comparison / f"{TEST_NAME}_comparison.csv"
    summary.to_csv(comparison_file, index=False)
    pd.concat([item[3] for item in completed], ignore_index=True).to_csv(comparison / f"{TEST_NAME}_timing.csv", index=False)

    # ----- 5. DIAGNOSTICS (RESPONSE CHANGES AND DOMAIN DEPARTURES) -----

    response_changes = []
    audit_rows = []
    for level, cases in groups.items():
        true = cases[shared.OUTPUT_COLUMNS].to_numpy(float)
        changes = shared.metric_rows(true, nominal_true, "Nominal simulator response (diagnostic)")
        for row in changes:
            row["uncertainty_level_percent"] = level
        response_changes.extend(changes)
        for column, (low, high) in TRAINING_RANGES.items():
            values = cases[column].to_numpy(float)
            audit_rows.append({
                "uncertainty_level_percent": level,
                "parameter": column,
                "actual_min": float(values.min()),
                "actual_max": float(values.max()),
                "original_sampling_min": low,
                "original_sampling_max": high,
                "count_outside_original_sampling_range": int(np.sum((values < low) | (values > high))),
            })
    pd.DataFrame(response_changes).to_csv(comparison / f"{TEST_NAME}_simulator_response_change.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(comparison / f"{TEST_NAME}_domain_audit.csv", index=False)
    run_metadata = {
        "test": TEST_NAME,
        "dataset_sha256": shared.file_hash(dataset_file),
        "generation_metadata": metadata,
        "shared_preprocessing_sha256": shared.file_hash(Path(shared.__file__)),
        "known_inputs": "actual perturbed properties; physics features recalculated from actual inputs",
        "unknown_inputs": "nominal properties only; same predictions reused across bands",
        "regime_labels": "actual frequency ratios, used only for evaluation diagnostics",
        "reference": "same 500 held-out nominal beams, original simulator outputs; not all 751 Test 1 samples",
        "prediction_clipping": False,
        "models": [item[4] for item in completed],
    }
    (comparison / f"{TEST_NAME}_run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print("\nComparison saved:", comparison_file)


if __name__ == "__main__":
    main()