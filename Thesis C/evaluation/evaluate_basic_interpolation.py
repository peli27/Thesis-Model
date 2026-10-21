"""
TEST 1 - BASIC INTERPOLATION

Compare the five trained models on unseen samples drawn from the same parameter ranges as the training dataset. 

1. Load the original nonlinear beam dataset and saved train/validation/test split.
2. Select the same held-out test samples for all five models.
3. Load each model's saved weights, input scaler and output scaler.
4. Check that the saved model and scaler statistics match the training split.
5. Prepare each model's inputs using its original training-time calculations:
    - Model 1: seven raw inputs.
    - Model 2: seven raw inputs plus five physics-guided features.
    - Model 3: seven raw inputs plus estimated peak and RMS displacement.
    - Model 4: seven raw inputs; add the physics estimates to predicted residuals after converting the network outputs back to their original units.
    - Model 5: seven raw inputs; physics regularisation was applied during training.
6. Calculate RMSE, MAE, R2, 95th-percentile and maximum absolute error for all six outputs, both overall and below, near and above resonance.
7. Measure prediction time, including preprocessing and scaling.
8. Save each model's results and collate the metrics into comparison CSV files.

"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


# ----- MODEL SETTINGS -----

INPUT_COLUMNS = ["L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"]
OUTPUT_COLUMNS = ["f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"]
FEATURE_COLUMNS = ["physics_f1", "physics_modal_mass", "physics_modal_stiffness", "physics_frequency_ratio", "resonance_distance"]
ESTIMATE_COLUMNS = ["physics_peak_displacement", "physics_rms_displacement"]
TARGET_COLUMNS = OUTPUT_COLUMNS[:4] + ["peak_residual", "rms_residual"]
TEST_NAME = "test_1_basic_interpolation"

# Folder name, saved-file prefix, display name, number of network inputs.
MODEL_SPECS = [
    ("model_1_baseline", "baseline", "Baseline MLP", 7),
    ("model_2_physics_guided", "physics_guided", "Physics-Guided NN", 12),
    ("model_3_physics_estimate", "physics_estimate", "Physics-Estimate Hybrid", 9),
    ("model_4_residual", "residual", "Residual-Learning Hybrid", 7),
    ("model_5_physics_regularised", "physics_regularised", "Physics-Regularised NN", 7),
]


# ----- PHYSICS PREPROCESSING -----

def modal_properties(data, width=0.04):
    # First-mode mass and stiffness including rotary inertia.
    L, h, E, rho = (data[column].to_numpy(float) for column in INPUT_COLUMNS[:4])

    A = width * h
    I = width * h**3 / 12.0
    k1 = np.pi / L
    mass = rho * (A + I * k1**2) * L / 2.0
    stiffness = E * I * k1**4 * L / 2.0
    omega1 = np.sqrt(stiffness / mass)

    return mass, stiffness, omega1


def calculate_physics_features(data, width=0.04):
    mass, stiffness, omega1 = modal_properties(data, width)
    f1 = omega1 / (2.0 * np.pi)
    ratio = data["excitation_frequency"].to_numpy(float) / f1

    return np.column_stack((f1, mass, stiffness, ratio, np.abs(ratio - 1.0)))


def calculate_physics_estimates(data, width=0.04, location=0.65):
    mass, stiffness, omega1 = modal_properties(data, width)
    omega = 2.0 * np.pi * data["excitation_frequency"].to_numpy(float)
    zeta = data["zeta"].to_numpy(float)

    # Convert uniform pressure to the first-mode harmonic force.
    force = 2.0 * data["pressure_amplitude"].to_numpy(float) * width * data["L"].to_numpy(float) / np.pi
    denominator = np.sqrt((stiffness - mass * omega**2)**2 + (2.0 * zeta * mass * omega1 * omega)**2)

    if np.any(denominator <= 0):
        raise ValueError("The linear physics estimate has a zero response denominator.")

    # Reconstruct the response at x/L = 0.65
    peak = np.abs((force / denominator) * np.sin(np.pi * location))
    return np.column_stack((peak, peak / np.sqrt(2.0)))


def prepare_inputs(data, prefix, checkpoint):
    raw = data[INPUT_COLUMNS].to_numpy(float)
    width = float(checkpoint.get("fixed_beam_width", 0.04))
    location = float(checkpoint.get("response_location_ratio", 0.65))
    physics = None

    if prefix == "physics_guided":
        raw = np.column_stack((raw, calculate_physics_features(data, width)))
    elif prefix in ("physics_estimate", "residual"):
        physics = calculate_physics_estimates(data, width, location)
        if prefix == "physics_estimate":
            raw = np.column_stack((raw, physics))

    if not np.all(np.isfinite(raw)) or (physics is not None and not np.all(np.isfinite(physics))):
        raise ValueError(f"Non-finite physics preprocessing for {prefix}.")
    return raw, physics


# ----- NEURAL-NETWORK ARCHITECTURE -----

class SurrogateNN(nn.Module):
    def __init__(self, input_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_count, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.network(x)


# ----- LOAD AND VALIDATE EXISTING ARTIFACTS -----

def find_scaler(folder, prefix, letter):
    candidates = [folder / f"{prefix}_{letter}_scaler.pkl", folder / f"{prefix}_{letter.lower()}_scaler.pkl"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing scaler: expected {candidates[0]} or {candidates[1]}")


def load_split(path, sample_count):
    with np.load(path, allow_pickle=False) as split:
        parts = {name: split[name].copy() for name in ("train_indices", "validation_indices", "test_indices")}
    for name, indices in parts.items():
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices):
            raise ValueError(f"Invalid or empty split: {name}")
        if np.any(indices < 0) or np.any(indices >= sample_count):
            raise ValueError(f"Out-of-range indices in {name}; check the original dataset.")
    combined = np.concatenate(list(parts.values()))
    if len(combined) != sample_count or len(np.unique(combined)) != sample_count:
        raise ValueError("Split must partition the dataset exactly once, without overlap or duplicate indices.")
    return parts


def validate_checkpoint(checkpoint, prefix, input_count):
    expected = [input_count, 64, 64, 32, 6]
    if list(checkpoint["architecture"]) != expected:
        raise ValueError(f"Unexpected architecture for {prefix}: {checkpoint['architecture']}")
    raw_columns = checkpoint.get("raw_input_columns", checkpoint.get("input_columns"))
    final_columns = checkpoint.get("final_output_columns", checkpoint.get("output_columns"))
    if raw_columns != INPUT_COLUMNS or final_columns != OUTPUT_COLUMNS:
        raise ValueError(f"Input/output column order does not match the training scripts for {prefix}.")
    if prefix == "physics_guided" and checkpoint.get("physics_feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Unexpected physics-guided feature order.")
    if prefix == "physics_estimate" and checkpoint.get("physics_estimate_columns") != ESTIMATE_COLUMNS:
        raise ValueError("Unexpected physics-estimate feature order.")
    if prefix == "residual":
        if checkpoint.get("target_columns") != TARGET_COLUMNS:
            raise ValueError("Unexpected residual target order.")
        if checkpoint.get("residual_definition") != "true_response - physics_estimate":
            raise ValueError("Unexpected residual definition.")


def validate_scaler(scaler, values, label):
    # Compare saved training statistics without fitting or changing scaler.
    if scaler.n_features_in_ != values.shape[1]:
        raise ValueError(f"Wrong number of features in {label}.")
    if not np.allclose(scaler.mean_, values.mean(axis=0), rtol=1e-8, atol=1e-12):
        raise ValueError(f"{label} mean does not match this training split. Check the dataset, split and saved model.")
    if not np.allclose(scaler.var_, values.var(axis=0), rtol=1e-8, atol=1e-18):
        raise ValueError(f"{label} variance does not match this training split. Check the dataset, split and saved model.")


def predict(data, prefix, checkpoint, model, x_scaler, y_scaler):
    inputs, physics = prepare_inputs(data, prefix, checkpoint)
    scaled = x_scaler.transform(inputs)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.inference_mode():
        predictions = y_scaler.inverse_transform(model(tensor).numpy())

    # Frequencies are direct predictions, only displacement outputs are residuals.
    if prefix == "residual":
        predictions[:, 4:6] += physics

    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"Non-finite predictions from {prefix}.")
    return predictions


# ----- METRICS IN ORIGINAL PHYSICAL UNITS -----

def metric_rows(true, predicted, model_name, regime="all"):
    rows = []
    for i, output in enumerate(OUTPUT_COLUMNS):
        error = predicted[:, i] - true[:, i]
        absolute = np.abs(error)
        rows.append({
            "model": model_name,
            "regime": regime,
            "output": output,
            "unit": "Hz" if i < 4 else "m",
            "n_samples": len(true),
            "RMSE": float(np.sqrt(np.mean(error**2))),
            "MAE": float(np.mean(absolute)),
            "R2": float(r2_score(true[:, i], predicted[:, i], force_finite=False)) if len(true) >= 2 and np.ptp(true[:, i]) > 0 else np.nan,
            "P95_absolute_error": float(np.percentile(absolute, 95)),
            "maximum_absolute_error": float(np.max(absolute)),
            "negative_prediction_count": int(np.sum(predicted[:, i] < 0)),
        })
    return rows


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    # ----- 1. READ SETTINGS AND PROJECT PATHS -----

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timing-repeats", type=int, default=10)
    args = parser.parse_args()

    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be at least 1")
    root = args.project_root.resolve()
    dataset_file = root / "data" / "main" / "nonlinear_beam_dataset.csv"
    split_file = root / "data" / "splits" / "main_split_indices.npz"

    # ----- 2. LOAD AND CHECK THE ORIGINAL DATASET -----

    dataset = pd.read_csv(dataset_file)
    required = INPUT_COLUMNS + OUTPUT_COLUMNS + ["frequency_ratio"]
    missing = [column for column in required if column not in dataset]

    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if not np.all(np.isfinite(dataset[required].to_numpy(float))):
        raise ValueError("Dataset contains missing or non-finite values.")
    if (dataset[["L", "h", "E", "rho"]] <= 0).any().any() or (dataset["zeta"] < 0).any():
        raise ValueError("Geometry/material values must be positive and damping nonnegative.")

    # ----- 3. SELECT THE SHARED TEST SAMPLES AND LABEL RESPONSE REGIMES -----

    split = load_split(split_file, len(dataset))
    train = dataset.iloc[split["train_indices"]]
    test = dataset.iloc[split["test_indices"]].copy()
    true = test[OUTPUT_COLUMNS].to_numpy(float)
    ratios = test["frequency_ratio"].to_numpy(float)
    regimes = np.where(ratios < 0.85, "below_resonance", np.where(ratios <= 1.20, "near_resonance", "above_resonance"))
    print(f"TEST 1: BASIC INTERPOLATION | {len(test)} shared held-out samples")
    print("CPU evaluation; no retraining, resplitting, scaler fitting or prediction clipping.")

    # ----- 4. CHECK INPUT RANGES -----

    bounds = pd.DataFrame({"input": INPUT_COLUMNS})
    bounds["training_min"] = train[INPUT_COLUMNS].min().to_numpy()
    bounds["training_max"] = train[INPUT_COLUMNS].max().to_numpy()
    bounds["test_min"] = test[INPUT_COLUMNS].min().to_numpy()
    bounds["test_max"] = test[INPUT_COLUMNS].max().to_numpy()
    outside = (test[INPUT_COLUMNS] < train[INPUT_COLUMNS].min()) | (test[INPUT_COLUMNS] > train[INPUT_COLUMNS].max())
    bounds["test_count_outside_observed_training_min_max"] = outside.sum().to_numpy()
    if outside.any().any():
        print("Some test values exceed observed training extrema; see the range audit. All held-out samples are retained.")

    # ----- 5. LOAD AND EVALUATE EACH TRAINED MODEL -----

    completed = []
    torch.set_num_threads(1)

    for folder, prefix, name, input_count in MODEL_SPECS:
        # Load the saved checkpoint and scalers for this model.
        saved = root / "models" / folder / "saved_model"
        model_file = saved / f"{prefix}_nn_model.pt"
        x_file = find_scaler(saved, prefix, "X")
        y_file = find_scaler(saved, prefix, "Y")

        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, prefix, input_count)
        x_scaler, y_scaler = joblib.load(x_file), joblib.load(y_file)

        # Confirm that preprocessing and scalers match the original training data.
        train_inputs, physics = prepare_inputs(train, prefix, checkpoint)
        train_targets = train[OUTPUT_COLUMNS].to_numpy(float).copy()
        if prefix == "residual":
            train_targets[:, 4:6] -= physics
        validate_scaler(x_scaler, train_inputs, f"{prefix} input scaler")
        validate_scaler(y_scaler, train_targets, f"{prefix} output scaler")

        # Restore the trained network and predict the shared test samples.
        model = SurrogateNN(input_count)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        predicted = predict(test, prefix, checkpoint, model, x_scaler, y_scaler)

        # Warm up inference before measuring repeated batch prediction times.
        for _ in range(3):
            predict(test, prefix, checkpoint, model, x_scaler, y_scaler)

        elapsed = []
        for _ in range(args.timing_repeats):
            start = time.perf_counter()
            predict(test, prefix, checkpoint, model, x_scaler, y_scaler)
            elapsed.append(time.perf_counter() - start)

        # Includes feature construction, scaling, forward pass and residual addition; excludes model loading, file I/O, metrics and warm-up.
        batch_seconds = float(np.median(elapsed))
        timings = {
            "model": name,
            "device": "cpu",
            "torch_threads": 1,
            "n_samples": len(test),
            "timing_repeats": args.timing_repeats,
            "median_batch_seconds": batch_seconds,
            "amortised_seconds_per_sample": batch_seconds / len(test),
        }

        # Calculate overall metrics and separate metrics for each response regime.
        rows = metric_rows(true, predicted, name)
        for regime in ("below_resonance", "near_resonance", "above_resonance"):
            mask = regimes == regime
            if mask.any():
                rows.extend(metric_rows(true[mask], predicted[mask], name, regime))

        predictions = test[INPUT_COLUMNS + ["frequency_ratio"]].reset_index(drop=True)
        predictions.insert(0, "dataset_row_index", split["test_indices"])
        predictions["regime"] = regimes
        for i, output in enumerate(OUTPUT_COLUMNS):
            predictions[f"true_{output}"] = true[:, i]
            predictions[f"predicted_{output}"] = predicted[:, i]
            predictions[f"error_{output}"] = predicted[:, i] - true[:, i]
            predictions[f"absolute_error_{output}"] = np.abs(predicted[:, i] - true[:, i])

        provenance = {
            "model": name,
            "checkpoint": str(model_file),
            "checkpoint_sha256": file_hash(model_file),
            "input_scaler_sha256": file_hash(x_file),
            "output_scaler_sha256": file_hash(y_file),
        }
        completed.append((folder, predictions, pd.DataFrame(rows), timings, provenance))
        print(f"{name}: peak RMSE = {rows[4]['RMSE']:.6e} m | RMS RMSE = {rows[5]['RMSE']:.6e} m")

    # ----- 6. SAVE INDIVIDUAL MODEL RESULTS -----

    # Save after every model has loaded and evaluated successfully.
    individual_metrics = []
    for folder, predictions, metrics, timings, provenance in completed:
        results = root / "models" / folder / "results" / TEST_NAME
        results.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(results / "predictions.csv", index=False)
        metrics.to_csv(results / "metrics.csv", index=False)
        pd.DataFrame([timings]).to_csv(results / "prediction_timing.csv", index=False)
        individual_metrics.append(pd.read_csv(results / "metrics.csv"))

    # ----- 7. COLLATE RESULTS INTO THE FIVE-MODEL COMPARISON -----

    comparison = root / "comparison" / "results"
    comparison.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(individual_metrics, ignore_index=True)
    combined.to_csv(comparison / f"{TEST_NAME}_metrics.csv", index=False)

    summary = []
    for name in [spec[2] for spec in MODEL_SPECS]:
        row = {"model": name, "n_test_samples": len(test)}
        selected = combined[(combined["model"] == name) & (combined["regime"] == "all")]
        for _, metric in selected.iterrows():
            for key in ("RMSE", "MAE", "R2", "P95_absolute_error", "maximum_absolute_error"):
                row[f"{metric['output']}_{key}"] = metric[key]
        summary.append(row)
    summary = pd.DataFrame(summary)

    # Positive values mean a reduction in RMSE relative to Model 1.
    for output in OUTPUT_COLUMNS[4:]:
        key = f"{output}_RMSE"
        baseline = float(summary.iloc[0][key])
        summary[f"{output}_RMSE_improvement_vs_baseline_percent"] = 100.0 * (baseline - summary[key]) / baseline if baseline > 0 else np.nan

    comparison_file = comparison / f"{TEST_NAME}_comparison.csv"
    summary.to_csv(comparison_file, index=False)
    pd.DataFrame([item[3] for item in completed]).to_csv(comparison / f"{TEST_NAME}_timing.csv", index=False)
    bounds.to_csv(comparison / f"{TEST_NAME}_input_range_audit.csv", index=False)

    # ----- 8. SAVE RUN METADATA AND PRINT THE FINAL SUMMARY -----

    metadata = {
        "test": TEST_NAME,
        "dataset": str(dataset_file),
        "dataset_sha256": file_hash(dataset_file),
        "split": str(split_file),
        "split_sha256": file_hash(split_file),
        "split_sizes": {key: len(value) for key, value in split.items()},
        "regime_thresholds": {
            "below": "r < 0.85",
            "near": "0.85 <= r <= 1.20",
            "above": "r > 1.20",
        },
        "regime_ratio_source": "dataset frequency_ratio; same classification as baseline training",
        "error_definition": "prediction - true",
        "prediction_clipping": False,
        "timing": "CPU, one torch thread, median warmed batch time including preprocessing; amortised time is not single-query latency",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
        },
        "models": [item[4] for item in completed],
    }
    (comparison / f"{TEST_NAME}_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nComparison saved: {comparison_file}")
    print("Frequency errors: Hz. Displacement errors: m. R2: dimensionless.")

if __name__ == "__main__":
    main()
