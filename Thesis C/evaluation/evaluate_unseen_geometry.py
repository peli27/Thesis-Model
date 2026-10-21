"""
TEST 2 - UNSEEN GEOMETRY

Compare all five trained models on 250 unseen-length and 250 unseen-thickness cases. 
Each case changes only one geometry input outside the training range.

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


# ----- ORIGINAL SAMPLING RANGES FROM generate_dataset.py -----

TEST_NAME = "test_2_unseen_geometry"
TRAINING_RANGES = {
    "L": (1.6, 2.4),
    "h": (0.008, 0.012),
    "E": (185e9, 225e9),
    "rho": (7450.0, 8250.0),
    "zeta": (0.005, 0.020),
    "pressure_amplitude": (0.1, 30.0),
    "frequency_ratio": (0.50, 1.50),
}


# ----- VALIDATE THE GEOMETRY EXPERIMENT -----

def outside_range(values, bounds):
    low, high = bounds
    return (values < low) | (values > high)


def validate_dataset(data):
    required = shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS + [
        "sample_id", "test_case", "extrapolation_side", "frequency_ratio"
    ]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Test 2 dataset is missing columns: {missing}")
    if len(data) != 500 or data["sample_id"].isna().any() or not data["sample_id"].is_unique:
        raise ValueError("Test 2 requires 500 samples with unique, non-missing sample IDs.")
    numeric = shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS + ["frequency_ratio"]
    if not np.all(np.isfinite(data[numeric].to_numpy(float))):
        raise ValueError("Test 2 contains missing or non-finite numerical values.")
    if (data[["L", "h", "E", "rho", "excitation_frequency"]] <= 0).any().any():
        raise ValueError("Geometry, material properties and excitation frequency must be positive.")
    if (data[shared.OUTPUT_COLUMNS] < 0).any().any():
        raise ValueError("True frequency/displacement outputs cannot be negative.")

    counts = data["test_case"].value_counts().to_dict()
    if counts != {"Unseen length": 250, "Unseen thickness": 250}:
        raise ValueError(f"Expected 250 cases per geometry group; found {counts}.")

    errors = []
    length_mask = data["test_case"] == "Unseen length"
    thickness_mask = data["test_case"] == "Unseen thickness"
    length_outside = outside_range(data["L"], TRAINING_RANGES["L"])
    thickness_outside = outside_range(data["h"], TRAINING_RANGES["h"])

    invalid_length = length_mask & (~length_outside | thickness_outside)
    invalid_thickness = thickness_mask & (~thickness_outside | length_outside)
    if invalid_length.any():
        errors.append(f"{invalid_length.sum()} unseen-length cases do not extrapolate only length outside 1.6-2.4 m")
    if invalid_thickness.any():
        errors.append(f"{invalid_thickness.sum()} unseen-thickness cases do not extrapolate only thickness outside 0.008-0.012 m")

    for column in ("E", "rho", "zeta", "pressure_amplitude", "frequency_ratio"):
        invalid = outside_range(data[column], TRAINING_RANGES[column])
        if invalid.any():
            errors.append(f"{invalid.sum()} cases have {column} outside {TRAINING_RANGES[column]}")

    for case_name, variable in (("Unseen length", "L"), ("Unseen thickness", "h")):
        group = data[data["test_case"] == case_name]
        expected_side = np.where(group[variable] < TRAINING_RANGES[variable][0], "Below training range", "Above training range")
        if not np.array_equal(group["extrapolation_side"].to_numpy(), expected_side):
            errors.append(f"Incorrect extrapolation-side labels for {case_name}")
        if group["extrapolation_side"].value_counts().to_dict() != {"Below training range": 125, "Above training range": 125}:
            errors.append(f"Expected 125 cases per extrapolation side for {case_name}")

    # Calculate f1 from raw inputs rather than using the stored output as a feature.
    physics_f1 = shared.calculate_physics_features(data)[:, 0]
    if not np.allclose(data["excitation_frequency"] / physics_f1, data["frequency_ratio"], rtol=1e-9, atol=1e-12):
        errors.append("Frequency ratios do not match the raw beam inputs and excitation frequency")
    if not np.allclose(data["f1"], physics_f1, rtol=1e-9, atol=1e-12):
        errors.append("Stored f1 does not match the beam physics used during training")

    if errors:
        raise ValueError("Invalid Test 2 dataset:\n- " + "\n- ".join(errors) + "\nRun generate_unseen_dataset.py before evaluation.")


def subset_metrics(true, predicted, data, name):
    rows = []
    ratios = data["frequency_ratio"].to_numpy(float)
    regimes = np.where(ratios < 0.85, "below_resonance", np.where(ratios <= 1.20, "near_resonance", "above_resonance"))

    for group_name in ("all", "Unseen length", "Unseen thickness"):
        group_mask = np.ones(len(data), dtype=bool) if group_name == "all" else (data["test_case"] == group_name).to_numpy()
        subsets = [("all", "all", group_mask)]
        if group_name != "all":
            for side in ("Below training range", "Above training range"):
                subsets.append((side, "all", group_mask & (data["extrapolation_side"] == side).to_numpy()))
        for regime in ("below_resonance", "near_resonance", "above_resonance"):
            subsets.append(("all", regime, group_mask & (regimes == regime)))

        for side, regime, mask in subsets:
            if not mask.any():
                continue
            metrics = shared.metric_rows(true[mask], predicted[mask], name, regime)
            for row in metrics:
                row["test_group"] = group_name
                row["extrapolation_side"] = side
            rows.extend(metrics)

    return pd.DataFrame(rows)


# ----- RUN THE FIVE MODELS -----

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, help="Optional dataset path, relative to the project root or absolute.")
    parser.add_argument("--timing-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be at least 1")

    root = args.project_root.resolve()
    test_file = args.dataset or Path("data/test_datasets/unseen_geometry_dataset.csv")
    if not test_file.is_absolute():
        test_file = root / test_file
    main_file = root / "data" / "main" / "nonlinear_beam_dataset.csv"
    split_file = root / "data" / "splits" / "main_split_indices.npz"

    # ----- 1. VALIDATE THE SHARED TEST DATASET -----

    test = pd.read_csv(test_file)
    validate_dataset(test)
    true = test[shared.OUTPUT_COLUMNS].to_numpy(float)
    print("Dataset checks passed: 250 unseen-length and 250 unseen-thickness cases.")

    # ----- 2. CHECK THE ORIGINAL TRAINING DATA AND SPLIT -----

    original = pd.read_csv(main_file)
    required = shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS + ["frequency_ratio"]
    if any(column not in original for column in required) or not np.all(np.isfinite(original[required].to_numpy(float))):
        raise ValueError("Original training dataset has missing columns or non-finite values.")
    for column, bounds in TRAINING_RANGES.items():
        if outside_range(original[column], bounds).any():
            raise ValueError(f"Main dataset's {column} values disagree with generate_dataset.py. Confirm the training ranges before running Test 2.")
    split = shared.load_split(split_file, len(original))
    train = original.iloc[split["train_indices"]]
    completed = []
    torch.set_num_threads(1)

    # ----- 3. LOAD MODELS, PREDICT AND CALCULATE METRICS -----

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
        predicted = shared.predict(test, prefix, checkpoint, model, x_scaler, y_scaler)

        # Measure the same warmed, end-to-end batch prediction operation as Test 1.
        for _ in range(3):
            shared.predict(test, prefix, checkpoint, model, x_scaler, y_scaler)
        elapsed = []
        for _ in range(args.timing_repeats):
            start = time.perf_counter()
            shared.predict(test, prefix, checkpoint, model, x_scaler, y_scaler)
            elapsed.append(time.perf_counter() - start)
        batch_seconds = float(np.median(elapsed))
        timing = {
            "model": name,
            "device": "cpu",
            "torch_threads": 1,
            "n_samples": len(test),
            "timing_repeats": args.timing_repeats,
            "median_batch_seconds": batch_seconds,
            "amortised_seconds_per_sample": batch_seconds / len(test),
        }

        metrics = subset_metrics(true, predicted, test, name)
        predictions = test.copy()
        predictions.insert(0, "dataset_row_index", np.arange(len(test)))
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
        overall = metrics[(metrics.test_group == "all") & (metrics.regime == "all")]
        peak_rmse = overall.loc[overall.output == "peak_displacement", "RMSE"].iloc[0]
        rms_rmse = overall.loc[overall.output == "rms_displacement", "RMSE"].iloc[0]
        print(f"{name}: peak RMSE = {peak_rmse:.6e} m | RMS RMSE = {rms_rmse:.6e} m")

    # ----- 4. SAVE INDIVIDUAL RESULTS -----

    collected = []
    for folder, predictions, metrics, timing, provenance in completed:
        results = root / "models" / folder / "results" / TEST_NAME
        results.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(results / "predictions.csv", index=False)
        metrics.to_csv(results / "metrics.csv", index=False)
        pd.DataFrame([timing]).to_csv(results / "prediction_timing.csv", index=False)
        collected.append(pd.read_csv(results / "metrics.csv"))

    # ----- 5. SAVE COMPARISONS -----

    comparison = root / "comparison" / "results"
    comparison.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(collected, ignore_index=True)
    combined.to_csv(comparison / f"{TEST_NAME}_metrics.csv", index=False)
    rows = []
    for group_name in ("all", "Unseen length", "Unseen thickness"):
        for _, _, name, _ in shared.MODEL_SPECS:
            subset = combined[(combined.model == name) & (combined.test_group == group_name) & (combined.regime == "all") & (combined.extrapolation_side == "all")]
            row = {"model": name, "test_group": group_name, "n_test_samples": int(subset.n_samples.iloc[0])}
            for _, metric in subset.iterrows():
                for key in ("RMSE", "MAE", "R2", "P95_absolute_error", "maximum_absolute_error"):
                    row[f"{metric['output']}_{key}"] = metric[key]
            rows.append(row)
    summary = pd.DataFrame(rows)

    # Compare RMSE with the baseline on the same geometry group.
    for group_name in ("all", "Unseen length", "Unseen thickness"):
        mask = summary.test_group == group_name
        baseline_row = summary[mask & (summary.model == "Baseline MLP")].iloc[0]
        for output in shared.OUTPUT_COLUMNS[4:]:
            key = f"{output}_RMSE"
            baseline = float(baseline_row[key])
            improvement = 100.0 * (baseline - summary.loc[mask, key]) / baseline if baseline > 0 else np.nan
            summary.loc[mask, f"{output}_RMSE_improvement_vs_baseline_percent"] = improvement

    comparison_file = comparison / f"{TEST_NAME}_comparison.csv"
    summary.to_csv(comparison_file, index=False)
    pd.DataFrame([item[3] for item in completed]).to_csv(comparison / f"{TEST_NAME}_timing.csv", index=False)
    audit = test.groupby(["test_case", "extrapolation_side"])[list(TRAINING_RANGES)].agg(["min", "max"])
    audit.to_csv(comparison / f"{TEST_NAME}_dataset_audit.csv")
    metadata = {
        "test": TEST_NAME,
        "dataset": str(test_file),
        "dataset_sha256": shared.file_hash(test_file),
        "main_dataset_sha256": shared.file_hash(main_file),
        "split_sha256": shared.file_hash(split_file),
        "shared_preprocessing_sha256": shared.file_hash(Path(shared.__file__)),
        "training_sampling_ranges": TRAINING_RANGES,
        "test_case_counts": {key: int(value) for key, value in test.test_case.value_counts().items()},
        "error_definition": "prediction - true",
        "prediction_clipping": False,
        "timing": "Median warmed CPU batch time including preprocessing; amortised time is not single-query latency",
        "models": [item[4] for item in completed],
    }
    (comparison / f"{TEST_NAME}_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nComparison saved:", comparison_file)
    print("Compare unseen length and thickness separately before interpreting the overall average.")


if __name__ == "__main__":
    main()
