"""
TEST 4 - REDUCED TRAINING DATA

Determine whether physics integration helps models maintain prediction accuracy
when fewer nonlinear simulations are available for training.

"""

import os

# Apply native-library limits before importing NumPy, scikit-learn or PyTorch.
for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_THREAD_LIMIT"):
    os.environ[thread_variable] = "1"

import argparse
import copy
import faulthandler
import json
import platform
import random
import time
from pathlib import Path

faulthandler.enable()

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import evaluate_basic_interpolation as shared


# ----- MATCH THE ORIGINAL TRAINING SETTINGS -----

TEST_NAME = "test_4_reduced_training_data"
DATA_PERCENTAGES = (100, 50, 25, 10)
TRAINING_SEEDS = (42, 43, 44)
SUBSET_SEED = 2026
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 500
PATIENCE = 30
PHYSICS_WEIGHT = 0.01
B = 0.04
RESPONSE_LOCATION_RATIO = 0.65


# ----- FIXED NESTED TRAINING SUBSETS -----

def make_subsets(data, train_indices):
    subsets = {100: np.sort(train_indices.copy())}
    current = subsets[100]
    for percentage in DATA_PERCENTAGES[1:]:
        count = max(1, int(np.floor(len(train_indices) * percentage / 100.0)))
        ratios = data.iloc[current]["frequency_ratio"].to_numpy(float)
        labels = np.where(ratios < 0.85, 0, np.where(ratios <= 1.20, 1, 2))
        selected, _ = train_test_split(current, train_size=count, random_state=SUBSET_SEED + percentage, stratify=labels)
        subsets[percentage] = np.sort(selected)
        current = subsets[percentage]
    return subsets


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ----- PREPARE TARGETS AND THE ORIGINAL LOSS FUNCTIONS -----

def build_training_arrays(data, prefix):
    checkpoint_settings = {"fixed_beam_width": B, "response_location_ratio": RESPONSE_LOCATION_RATIO}
    inputs, estimates = shared.prepare_inputs(data, prefix, checkpoint_settings)
    targets = data[shared.OUTPUT_COLUMNS].to_numpy(float).copy()
    if prefix == "residual":
        targets[:, 4:6] -= estimates
    if prefix == "physics_regularised":
        estimates = shared.calculate_physics_estimates(data, B, RESPONSE_LOCATION_RATIO)
    return inputs, targets, estimates


def loss_components(prediction, target, physics, prefix):
    data_loss = nn.functional.mse_loss(prediction, target)
    if prefix == "physics_regularised":
        physics_loss = nn.functional.mse_loss(prediction[:, 4:6], physics)
        total = data_loss + PHYSICS_WEIGHT * physics_loss
    else:
        physics_loss = data_loss.new_zeros(())
        total = data_loss
    return total, data_loss, physics_loss


def checkpoint_metadata(prefix, input_count, model, seed, percentage, best_epoch, best_loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": [input_count, 64, 64, 32, 6],
        "output_columns": shared.OUTPUT_COLUMNS,
        "seed": seed,
        "training_data_percent": percentage,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "fixed_beam_width": B,
        "response_location_ratio": RESPONSE_LOCATION_RATIO,
    }
    if prefix in ("physics_guided", "physics_estimate"):
        checkpoint["raw_input_columns"] = shared.INPUT_COLUMNS
        if prefix == "physics_guided":
            checkpoint["physics_feature_columns"] = shared.FEATURE_COLUMNS
        else:
            checkpoint["physics_estimate_columns"] = shared.ESTIMATE_COLUMNS
    else:
        checkpoint["input_columns"] = shared.INPUT_COLUMNS
    if prefix == "residual":
        checkpoint["target_columns"] = shared.TARGET_COLUMNS
        checkpoint["final_output_columns"] = shared.OUTPUT_COLUMNS
        checkpoint["residual_definition"] = "true_response - physics_estimate"
    if prefix == "physics_regularised":
        checkpoint["physics_weight"] = PHYSICS_WEIGHT
    return checkpoint


# ----- TRAIN ONE MODEL / BUDGET / SEED -----

def train_run(data, inputs, targets, estimates, split, indices, spec, percentage, seed, device, output_dir):
    _, prefix, name, input_count = spec
    seed_training(seed)
    x_scaler = StandardScaler().fit(inputs[indices])
    y_scaler = StandardScaler().fit(targets[indices])
    train_x = x_scaler.transform(inputs[indices])
    train_y = y_scaler.transform(targets[indices])
    validation_indices = split["validation_indices"]
    val_x = x_scaler.transform(inputs[validation_indices])
    val_y = y_scaler.transform(targets[validation_indices])

    # Only Model 5's loss uses these scaled linear-physics response targets.
    if prefix == "physics_regularised":
        scaled_physics = (estimates - y_scaler.mean_[4:6]) / y_scaler.scale_[4:6]
        train_physics = scaled_physics[indices]
        val_physics = scaled_physics[validation_indices]
    else:
        train_physics = np.zeros((len(indices), 2))
        val_physics = np.zeros((len(validation_indices), 2))

    training = torch.utils.data.TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
        torch.tensor(train_physics, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(training, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)
    val_physics_t = torch.tensor(val_physics, dtype=torch.float32, device=device)
    model = shared.SurrogateNN(input_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_loss = np.inf
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0
    start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batches = []
        for x_batch, y_batch, physics_batch in loader:
            x_batch, y_batch, physics_batch = (value.to(device) for value in (x_batch, y_batch, physics_batch))
            optimizer.zero_grad()
            total, data_loss, physics_loss = loss_components(model(x_batch), y_batch, physics_batch, prefix)
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite training loss: {name}, {percentage}%, seed {seed}.")
            total.backward()
            optimizer.step()
            batches.append((total.item(), data_loss.item(), physics_loss.item()))

        model.eval()
        with torch.no_grad():
            val_total, val_data, val_physics_loss = loss_components(model(val_x_t), val_y_t, val_physics_t, prefix)
        validation_loss = float(val_total.item())
        if not np.isfinite(validation_loss):
            raise RuntimeError(f"Non-finite validation loss: {name}, {percentage}%, seed {seed}.")
        means = np.mean(batches, axis=0)
        history.append({
            "epoch": epoch,
            "training_total_loss": means[0],
            "training_data_loss": means[1],
            "training_physics_loss": means[2] if prefix == "physics_regularised" else np.nan,
            "validation_total_loss": validation_loss,
            "validation_data_loss": float(val_data.item()),
            "validation_physics_loss": float(val_physics_loss.item()) if prefix == "physics_regularised" else np.nan,
        })

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | train = {means[0]:.6e} | validation = {validation_loss:.6e}")
        if epochs_without_improvement >= PATIENCE:
            break

    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("No valid checkpoint was retained.")
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    checkpoint = checkpoint_metadata(prefix, input_count, model, seed, percentage, best_epoch, best_loss)

    # Test rows are first used here, after selecting the best validation checkpoint.
    test = data.iloc[split["test_indices"]].copy()
    true = test[shared.OUTPUT_COLUMNS].to_numpy(float)
    predicted = shared.predict(test, prefix, checkpoint, model, x_scaler, y_scaler)
    rows = shared.metric_rows(true, predicted, name)
    ratios = test.frequency_ratio.to_numpy(float)
    regimes = np.where(ratios < 0.85, "below_resonance", np.where(ratios <= 1.20, "near_resonance", "above_resonance"))
    for regime in ("below_resonance", "near_resonance", "above_resonance"):
        mask = regimes == regime
        if mask.any():
            rows.extend(shared.metric_rows(true[mask], predicted[mask], name, regime))
    for row in rows:
        row.update(training_data_percent=percentage, training_seed=seed, n_training_samples=len(indices))

    predictions = test[shared.INPUT_COLUMNS + ["frequency_ratio"]].reset_index(drop=True)
    predictions.insert(0, "dataset_row_index", split["test_indices"])
    predictions["regime"] = regimes
    for i, output in enumerate(shared.OUTPUT_COLUMNS):
        predictions[f"true_{output}"] = true[:, i]
        predictions[f"predicted_{output}"] = predicted[:, i]
        predictions[f"error_{output}"] = predicted[:, i] - true[:, i]
        predictions[f"absolute_error_{output}"] = np.abs(predicted[:, i] - true[:, i])

    # Save completion metadata last so interrupted runs cannot appear complete.
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_dir / f"{prefix}_nn_model.pt")
    joblib.dump(x_scaler, output_dir / f"{prefix}_X_scaler.pkl")
    joblib.dump(y_scaler, output_dir / f"{prefix}_y_scaler.pkl")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "model": name,
        "training_data_percent": percentage,
        "training_seed": seed,
        "n_training_samples": len(indices),
        "n_validation_samples": len(validation_indices),
        "n_test_samples": len(test),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "training_seconds": training_seconds,
        "device": str(device),
    }
    (output_dir / "completed_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, pd.DataFrame(rows)


# ----- COLLATE PER-RUN RESULTS AND MEAN / STANDARD DEVIATION -----

def collate_results(run_results, comparison):
    summaries = []
    detailed = []
    for summary, metrics in run_results:
        wide = summary.copy()
        for _, metric in metrics[metrics.regime == "all"].iterrows():
            for key in ("RMSE", "MAE", "R2", "P95_absolute_error", "maximum_absolute_error"):
                wide[f"{metric['output']}_{key}"] = metric[key]
        summaries.append(wide)
        detailed.append(metrics)
    runs = pd.DataFrame(summaries)
    runs.to_csv(comparison / f"{TEST_NAME}_runs.csv", index=False)
    pd.concat(detailed, ignore_index=True).to_csv(comparison / f"{TEST_NAME}_regime_metrics.csv", index=False)

    means = []
    for (name, percentage), group in runs.groupby(["model", "training_data_percent"], sort=False):
        if len(group) != len(TRAINING_SEEDS) or set(group.training_seed) != set(TRAINING_SEEDS):
            raise ValueError("Comparison requires all three training seeds at every model/budget.")
        row = {
            "model": name,
            "training_data_percent": percentage,
            "n_training_samples": int(group.n_training_samples.iloc[0]),
            "n_test_samples": int(group.n_test_samples.iloc[0]),
            "n_seeds": len(group),
        }
        for output in shared.OUTPUT_COLUMNS:
            for key in ("RMSE", "MAE", "R2", "P95_absolute_error", "maximum_absolute_error"):
                column = f"{output}_{key}"
                row[f"{column}_mean"] = float(group[column].mean())
                row[f"{column}_std"] = float(group[column].std(ddof=1))
        row["training_seconds_mean"] = float(group.training_seconds.mean())
        means.append(row)
    comparison_frame = pd.DataFrame(means)
    for name, group in comparison_frame.groupby("model"):
        reference = group[group.training_data_percent == 100].iloc[0]
        for output in shared.OUTPUT_COLUMNS[4:]:
            column = f"{output}_RMSE_mean"
            reference_rmse = float(reference[column])
            comparison_frame.loc[group.index, f"{output}_RMSE_ratio_to_full_training"] = group[column] / reference_rmse if reference_rmse > 0 else np.nan
    for percentage, group in comparison_frame.groupby("training_data_percent"):
        baseline = group[group.model == "Baseline MLP"].iloc[0]
        for output in shared.OUTPUT_COLUMNS[4:]:
            column = f"{output}_RMSE_mean"
            baseline_rmse = float(baseline[column])
            improvement = 100.0 * (baseline_rmse - group[column]) / baseline_rmse if baseline_rmse > 0 else np.nan
            comparison_frame.loc[group.index, f"{output}_RMSE_improvement_vs_baseline_percent"] = improvement
    comparison_frame.to_csv(comparison / f"{TEST_NAME}_comparison.csv", index=False)


# ----- ORCHESTRATE ALL 60 TRAINING RUNS -----

def check_training_backend(device):
    # Use the same layer sizes, batch size and optimizer as the first real run.
    model = shared.SurrogateNN(7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x = torch.randn(BATCH_SIZE, 7, dtype=torch.float32, device=device)
    target = torch.randn(BATCH_SIZE, 6, dtype=torch.float32, device=device)
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Training backend check passed: forward, backward and Adam step.")
    # Each real run resets its seed, so this check does not change initialization.


def prepare_experiment_metadata(root, experiment, manifest):
    manifest_file = experiment / "experiment_metadata.json"
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        if previous == manifest:
            return False

        # A crash before completion may leave only the experiment settings.
        # Refresh those settings only when no completed-run records exist.
        completed_runs = []
        for folder, _, _, _ in shared.MODEL_SPECS:
            results = root / "models" / folder / "results" / TEST_NAME
            if results.exists():
                completed_runs.extend(results.rglob("completed_run.json"))
        if completed_runs:
            raise ValueError("Existing Test 4 settings differ and completed runs exist. Preserve this experiment and use a separate project copy for changed settings.")
        print("Refreshing settings from the failed attempt; no completed runs were found.")

    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    # Configure threading once, before any tensor/gradient work is started.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print("CPU threads:", torch.get_num_threads(), "| inter-op threads:", torch.get_num_interop_threads(), flush=True)

    root = args.project_root.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available. Use --device cpu or auto.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print("Checking training backend on", device, "...", flush=True)
    check_training_backend(device)
    dataset_file = root / "data" / "main" / "nonlinear_beam_dataset.csv"
    split_file = root / "data" / "splits" / "main_split_indices.npz"
    data = pd.read_csv(dataset_file)
    required = shared.INPUT_COLUMNS + shared.OUTPUT_COLUMNS + ["frequency_ratio"]
    missing = [column for column in required if column not in data]
    if missing or not np.all(np.isfinite(data[required].to_numpy(float))):
        raise ValueError(f"Dataset has missing columns or non-finite values. Missing: {missing}")
    split = shared.load_split(split_file, len(data))
    subsets = make_subsets(data, split["train_indices"])
    comparison = root / "comparison" / "results"
    experiment = comparison / TEST_NAME
    experiment.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_sha256": shared.file_hash(dataset_file),
        "split_sha256": shared.file_hash(split_file),
        "script_sha256": shared.file_hash(Path(__file__)),
        "shared_preprocessing_sha256": shared.file_hash(Path(shared.__file__)),
        "data_percentages": list(DATA_PERCENTAGES),
        "training_seeds": list(TRAINING_SEEDS),
        "subset_seed": SUBSET_SEED,
        "subset_sizes": {str(key): len(value) for key, value in subsets.items()},
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "physics_weight": PHYSICS_WEIGHT,
        "device": str(device),
        "cpu_threads": 1,
        "interop_threads": 1,
        "native_thread_limits": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_THREAD_LIMIT")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
        "variation": "Training initialization and minibatch ordering vary; nested subsets are fixed across models and seeds",
    }
    if prepare_experiment_metadata(root, experiment, manifest):
        np.savez(experiment / "training_subset_indices.npz", **{f"train_{key}_percent": value for key, value in subsets.items()})
        coverage = []
        for percentage, indices in subsets.items():
            ratios = data.iloc[indices].frequency_ratio.to_numpy(float)
            coverage.append({
                "training_data_percent": percentage,
                "n_samples": len(indices),
                "below_resonance": int(sum(ratios < 0.85)),
                "near_resonance": int(sum((ratios >= 0.85) & (ratios <= 1.20))),
                "above_resonance": int(sum(ratios > 1.20)),
            })
        pd.DataFrame(coverage).to_csv(experiment / "subset_regime_coverage.csv", index=False)

    print("TEST 4 - REDUCED TRAINING DATA")
    print("Device:", device)
    print("Subset sizes:", {key: len(value) for key, value in subsets.items()})
    print("60 runs: five models x four budgets x three training seeds.")
    print("Completed runs are reused; original saved_model folders are preserved.")
    run_results = []

    for spec in shared.MODEL_SPECS:
        folder, prefix, name, _ = spec
        inputs, targets, estimates = build_training_arrays(data, prefix)
        for percentage in DATA_PERCENTAGES:
            for seed in TRAINING_SEEDS:
                output_dir = root / "models" / folder / "results" / TEST_NAME / f"data_{percentage}_percent" / f"seed_{seed}"
                completion = output_dir / "completed_run.json"
                print(f"\n{len(run_results) + 1}/60 | {name} | {percentage}% data | seed {seed}")
                required_files = ["metrics.csv", "predictions.csv", "training_history.csv", f"{prefix}_nn_model.pt", f"{prefix}_X_scaler.pkl", f"{prefix}_y_scaler.pkl"]
                if completion.exists() and all((output_dir / filename).is_file() for filename in required_files):
                    summary = json.loads(completion.read_text())
                    if summary["model"] != name or summary["training_data_percent"] != percentage or summary["training_seed"] != seed or summary["n_training_samples"] != len(subsets[percentage]):
                        raise ValueError(f"Completion metadata disagrees with the experiment: {output_dir}")
                    metrics = pd.read_csv(output_dir / "metrics.csv")
                    print("  Completed run loaded.")
                else:
                    summary, metrics = train_run(data, inputs, targets, estimates, split, subsets[percentage], spec, percentage, seed, device, output_dir)
                    print(f"  Best epoch: {summary['best_epoch']} | validation loss: {summary['best_validation_loss']:.6e}")
                run_results.append((summary, metrics))

    collate_results(run_results, comparison)
    print("\nAll 60 runs complete.")
    print("Comparison saved:", comparison / f"{TEST_NAME}_comparison.csv")
    print("Interpret mean and seed variation together when assessing data efficiency.")


if __name__ == "__main__":
    main()