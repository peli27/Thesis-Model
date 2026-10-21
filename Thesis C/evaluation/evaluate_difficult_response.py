"""
TEST 5 - DIFFICULT NONLINEAR RESPONSE

Test 5A evaluates all 500 near-resonance cases. Test 5B evaluates the subset
whose nonlinear first-mode restoring-stiffness contribution reaches at least
10% of linear stiffness during the simulator's second-half response window.

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


# ----- PREDEFINED TEST CRITERIA -----

TEST_NAME = "test_5_difficult_response"
STRONG_THRESHOLD = 0.10
NEAR_RANGES = {
    "L": (1.6, 2.4),
    "h": (0.008, 0.012),
    "E": (185e9, 225e9),
    "rho": (7450.0, 8250.0),
    "zeta": (0.005, 0.020),
    "pressure_amplitude": (0.1, 30.0),
    "frequency_ratio": (0.85, 1.20),
}
GROUP_NAMES = ("5A Near resonance", "5B Strong nonlinear response near resonance")


# ----- CHECK DATASET AND DEFINE THE STRONG SUBSET -----

def validate_dataset(data, original):
    required = shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS + [
        "sample_id", "frequency_ratio", "nonlinear_stiffness_ratio_peak",
        "nonlinear_stiffness_ratio_mean", "first_modal_peak", "peak_displacement_to_thickness",
    ]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Difficult-response dataset is missing columns: {missing}")
    if len(data) != 500 or not data.sample_id.is_unique:
        raise ValueError("Expected 500 cases with unique sample IDs.")
    if not np.all(np.isfinite(data[required].to_numpy(float))):
        raise ValueError("Dataset contains missing or non-finite values.")
    for column, (low, high) in NEAR_RANGES.items():
        if ((data[column] < low) | (data[column] > high)).any():
            raise ValueError(f"{column} falls outside the predefined range {low}-{high}.")
    if (data[shared.OUTPUT_COLUMNS] < 0).any().any():
        raise ValueError("True simulator outputs cannot be negative.")
    f1 = shared.calculate_physics_features(data)[:, 0]
    if not np.allclose(data.f1, f1, rtol=1e-9, atol=1e-12):
        raise ValueError("Stored first frequencies disagree with the beam properties.")
    if not np.allclose(data.excitation_frequency / f1, data.frequency_ratio, rtol=1e-9, atol=1e-12):
        raise ValueError("Stored frequency ratios are inconsistent.")

    eta_peak = data.nonlinear_stiffness_ratio_peak.to_numpy(float)
    eta_mean = data.nonlinear_stiffness_ratio_mean.to_numpy(float)
    if np.any(eta_peak < 0) or np.any(eta_mean < 0) or np.any(eta_mean > eta_peak + 1e-12):
        raise ValueError("Invalid nonlinear stiffness indicators.")
    if (data.first_modal_peak < 0).any():
        raise ValueError("First-modal peak must be nonnegative.")
    lower_bound = 3.0 * (data.first_modal_peak / data.h)**2
    if np.any(eta_peak + 1e-10 < lower_bound):
        raise ValueError("Nonlinear indicator is inconsistent with the first-modal peak.")
    if not np.allclose(data.peak_displacement / data.h, data.peak_displacement_to_thickness, rtol=1e-10, atol=1e-12):
        raise ValueError("Displacement-to-thickness indicator is inconsistent.")
    if len(data.merge(original[shared.INPUT_COLUMNS], on=shared.INPUT_COLUMNS, how="inner")):
        raise ValueError("Test inputs duplicate original dataset inputs.")

    strong = eta_peak >= STRONG_THRESHOLD
    if not strong.any():
        raise ValueError("No cases satisfy the predefined strong-nonlinearity criterion; Test 5B cannot be evaluated on this dataset.")
    return strong


def metrics_for_groups(true, predicted, strong, name):
    rows = []
    for group_name, mask in ((GROUP_NAMES[0], np.ones(len(true), dtype=bool)), (GROUP_NAMES[1], strong)):
        group_rows = shared.metric_rows(true[mask], predicted[mask], name)
        for row in group_rows:
            row["test_group"] = group_name
        rows.extend(group_rows)
    return pd.DataFrame(rows)


# ----- EVALUATE THE FIVE SAVED MODELS -----

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timing-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be at least 1")
    root = args.project_root.resolve()
    dataset_file = root / "data" / "test_datasets" / "difficult_response_dataset.csv"
    metadata_file = dataset_file.with_name("difficult_response_generation_metadata.json")
    main_file = root / "data" / "main" / "nonlinear_beam_dataset.csv"
    split_file = root / "data" / "splits" / "main_split_indices.npz"

    # ----- SIMULATOR SETTINGS AND CASE DEFINITIONS -----

    metadata = json.loads(metadata_file.read_text())
    if metadata["strong_threshold"] != STRONG_THRESHOLD:
        raise ValueError("Generation and evaluation strong-response thresholds differ.")
    expected_simulation = {"b": 0.04, "N": 7, "t_end": 10.0, "dt": 0.002, "use_nonlinearity": True}
    if metadata["simulation"] != expected_simulation or metadata["n_samples"] != 500:
        raise ValueError("Generation settings do not match the original simulator experiment.")
    if metadata["main_dataset_sha256"] != shared.file_hash(main_file):
        raise ValueError("The main dataset has changed since generating Test 5.")
    data = pd.read_csv(dataset_file)
    original = pd.read_csv(main_file)
    required = shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS
    if any(column not in original for column in required) or not np.all(np.isfinite(original[required].to_numpy(float))):
        raise ValueError("Original dataset has missing columns or non-finite values.")
    strong = validate_dataset(data, original)
    split = shared.load_split(split_file, len(original))
    train = original.iloc[split["train_indices"]]
    true = data[shared.OUTPUT_COLUMNS].to_numpy(float)
    print("TEST 5 - DIFFICULT NONLINEAR RESPONSE")
    print(f"5A: {len(data)} near-resonance cases | 5B: {int(strong.sum())} strong cases")
    if strong.sum() < 30:
        print("The strong subset has fewer than 30 cases; interpret 5B as preliminary.")
    torch.set_num_threads(1)
    completed = []

    # ----- PREDICT AND SCORE -----

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
        predicted = shared.predict(data, prefix, checkpoint, model, x_scaler, y_scaler)

        for _ in range(3):
            shared.predict(data, prefix, checkpoint, model, x_scaler, y_scaler)
        elapsed = []
        for _ in range(args.timing_repeats):
            start = time.perf_counter()
            shared.predict(data, prefix, checkpoint, model, x_scaler, y_scaler)
            elapsed.append(time.perf_counter() - start)
        seconds = float(np.median(elapsed))
        timing = {
            "model": name,
            "device": "cpu",
            "torch_threads": 1,
            "n_samples": len(data),
            "timing_repeats": args.timing_repeats,
            "median_batch_seconds": seconds,
            "amortised_seconds_per_sample": seconds / len(data),
            "timing_scope": "All 500 cases; the strong subset reuses these predictions",
        }

        metrics = metrics_for_groups(true, predicted, strong, name)
        predictions = data.copy()
        predictions["included_in_test_5A"] = True
        predictions["included_in_test_5B"] = strong
        for i, output in enumerate(shared.OUTPUT_COLUMNS):
            predictions.rename(columns={output: f"true_{output}"}, inplace=True)
            predictions[f"predicted_{output}"] = predicted[:, i]
            predictions[f"error_{output}"] = predicted[:, i] - true[:, i]
            predictions[f"absolute_error_{output}"] = np.abs(predicted[:, i] - true[:, i])
        provenance = {
            "model": name,
            "checkpoint_sha256": shared.file_hash(model_file),
            "input_scaler_sha256": shared.file_hash(x_file),
            "output_scaler_sha256": shared.file_hash(y_file),
        }
        completed.append((folder, predictions, metrics, timing, provenance))
        for group_name in GROUP_NAMES:
            peak = metrics[(metrics.test_group == group_name) & (metrics.output == "peak_displacement")].iloc[0]
            print(f"{name} | {group_name} | peak RMSE = {peak.RMSE:.6e} m | R2 = {peak.R2:.4f}")

    # ----- INDIVIDUAL RESULTS AND COMPARISON -----

    collected = []
    for folder, predictions, metrics, timing, provenance in completed:
        results = root / "models" / folder / "results" / TEST_NAME
        results.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(results / "predictions.csv", index=False)
        metrics.to_csv(results / "metrics.csv", index=False)
        pd.DataFrame([timing]).to_csv(results / "prediction_timing.csv", index=False)
        collected.append(pd.read_csv(results / "metrics.csv"))

    comparison = root / "comparison" / "results"
    comparison.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(collected, ignore_index=True)
    combined.to_csv(comparison / f"{TEST_NAME}_metrics.csv", index=False)
    rows = []
    for group_name in GROUP_NAMES:
        for _, _, name, _ in shared.MODEL_SPECS:
            selected = combined[(combined.model == name) & (combined.test_group == group_name)]
            row = {"model": name, "test_group": group_name, "n_test_samples": int(selected.n_samples.iloc[0])}
            for _, metric in selected.iterrows():
                for key in ("RMSE", "MAE", "R2", "P95_absolute_error", "maximum_absolute_error"):
                    row[f"{metric['output']}_{key}"] = metric[key]
            rows.append(row)
    summary = pd.DataFrame(rows)
    for group_name, group in summary.groupby("test_group"):
        baseline = group[group.model == "Baseline MLP"].iloc[0]
        for output in shared.OUTPUT_COLUMNS[4:]:
            column = f"{output}_RMSE"
            denominator = float(baseline[column])
            improvement = 100.0 * (denominator - group[column]) / denominator if denominator > 0 else np.nan
            summary.loc[group.index, f"{output}_RMSE_improvement_vs_baseline_percent"] = improvement
    comparison_file = comparison / f"{TEST_NAME}_comparison.csv"
    summary.to_csv(comparison_file, index=False)
    pd.DataFrame([item[3] for item in completed]).to_csv(comparison / f"{TEST_NAME}_timing.csv", index=False)

    # ----- 4. SAVE CASE-MIX SUMMARIES AND THE PREDEFINED CRITERION -----

    dataset_summaries = []
    for group_name, mask in ((GROUP_NAMES[0], np.ones(len(data), dtype=bool)), (GROUP_NAMES[1], strong)):
        subset = data.loc[mask]
        row = {"test_group": group_name, "n_samples": len(subset), "strong_threshold": STRONG_THRESHOLD}
        for column in ("frequency_ratio", "pressure_amplitude", "nonlinear_stiffness_ratio_peak", "peak_displacement", "rms_displacement"):
            for key, value in (("min", subset[column].min()), ("median", subset[column].median()), ("max", subset[column].max())):
                row[f"{column}_{key}"] = float(value)
        dataset_summaries.append(row)
    pd.DataFrame(dataset_summaries).to_csv(comparison / f"{TEST_NAME}_dataset_summary.csv", index=False)
    run_metadata = {
        "test": TEST_NAME,
        "dataset_sha256": shared.file_hash(dataset_file),
        "split_sha256": shared.file_hash(split_file),
        "shared_preprocessing_sha256": shared.file_hash(Path(shared.__file__)),
        "generation_metadata": metadata,
        "strong_subset_count": int(strong.sum()),
        "subset_relationship": "Test 5B is contained within Test 5A",
        "indicator_use": "Simulator-based evaluation labels only; never used in model inputs",
        "prediction_clipping": False,
        "models": [item[4] for item in completed],
    }
    (comparison / f"{TEST_NAME}_run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print("\nComparison saved:", comparison_file)
    print("Compare models within each shared group; 5A and 5B have different response distributions.")


if __name__ == "__main__":
    main()
