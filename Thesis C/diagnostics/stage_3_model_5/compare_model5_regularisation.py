"""Compare Model 5 with and without weak physics regularisation.

PURPOSE
-------
The original Model 5 lambda-sensitivity experiment showed that increasing the
weight of a linear first-mode physics target (lambda = 0.01, 0.1, 1.0) worsened
validation accuracy. I am now checking does the original weak penalty (lambda = 0.01) 
help compared with no physics penalty (lambda = 0)?


"""

import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ----- PATHS AND SETTINGS -----
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "main" / "nonlinear_beam_dataset.csv"
SPLIT = ROOT / "data" / "splits" / "main_split_indices.npz"
OUT = ROOT / "diagnostics" / "stage_3_model_5" / "results"
FIGURES = OUT / "figures"

SEEDS = [42, 123, 456]
LAMBDAS = [0.0, 0.01]
B = 0.04
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 500
PATIENCE = 30

INPUTS = ["L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"]
OUTPUTS = ["f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"]

# Keep the original interpolation-test resonance boundaries.
RESONANCE_LOW = 0.85
RESONANCE_HIGH = 1.20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----- SAME ANALYTICAL PHYSICS ESTIMATE AS ORIGINAL SCRIPT -----
def physics_estimate(data):
    L = data["L"].to_numpy(float)
    h = data["h"].to_numpy(float)
    E = data["E"].to_numpy(float)
    rho = data["rho"].to_numpy(float)
    zeta = data["zeta"].to_numpy(float)
    P = data["pressure_amplitude"].to_numpy(float)
    f = data["excitation_frequency"].to_numpy(float)

    A = B * h
    I = B * h**3 / 12
    k = np.pi / L
    M = rho * (A + I * k**2) * L / 2
    K = E * I * k**4 * L / 2
    omega_1 = np.sqrt(K / M)
    omega = 2 * np.pi * f
    Q = 2 * P * B * L / np.pi
    denominator = np.sqrt((K - M * omega**2)**2 + (2 * zeta * M * omega_1 * omega)**2)
    if np.any(denominator <= 0):
        raise ValueError("Analytical physics estimate contains a zero response denominator.")
    amplitude = Q / denominator
    peak = np.abs(amplitude * np.sin(0.65 * np.pi))
    rms = peak / np.sqrt(2)
    return np.column_stack((peak, rms))


# ----- ORIGINAL MODEL 5 NETWORK -----
class PhysicsRegularisedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 6),
        )

    def forward(self, x):
        return self.network(x)


def make_metrics(truth, prediction, lambda_value, seed, group_type, group_name):
    rows = []
    for index, name in enumerate(OUTPUTS[4:], start=4):
        error = prediction[:, index] - truth[:, index]
        rows.append({
            "lambda": lambda_value,
            "seed": seed,
            "group_type": group_type,
            "group": group_name,
            "output": name,
            "n_samples": len(truth),
            "rmse_mm": float(np.sqrt(np.mean(error**2)) * 1000),
            "mae_mm": float(np.mean(np.abs(error)) * 1000),
            "r2": float(r2_score(truth[:, index], prediction[:, index])) if len(truth) > 1 and np.ptp(truth[:, index]) > 0 else np.nan,
            "p95_absolute_error_mm": float(np.percentile(np.abs(error), 95) * 1000),
            "max_absolute_error_mm": float(np.max(np.abs(error)) * 1000),
            "negative_predictions": int(np.sum(prediction[:, index] < 0)),
        })
    return rows


def summarise(metrics, group_columns):
    measures = ["rmse_mm", "mae_mm", "r2", "p95_absolute_error_mm", "max_absolute_error_mm", "negative_predictions"]
    summary = metrics.groupby(group_columns, dropna=False)[measures].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    return summary.reset_index()


def plot_comparison(metrics, group_type, title, path):
    subset = metrics.loc[metrics["group_type"] == group_type]
    groups = ["all"] if group_type == "overall" else sorted(subset["group"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    x = np.arange(len(groups))
    width = 0.36
    for axis, output in zip(axes, OUTPUTS[4:]):
        for j, lam in enumerate(LAMBDAS):
            vals = []
            errs = []
            for group in groups:
                values = subset.loc[(subset["group"] == group) & (subset["lambda"] == lam) & (subset["output"] == output), "rmse_mm"].to_numpy()
                vals.append(values.mean() if len(values) else np.nan)
                errs.append(values.std(ddof=1) if len(values) > 1 else 0.0)
            axis.bar(x + (j - 0.5) * width, vals, width, label=f"λ = {lam:g}", yerr=errs, capsize=3)
        axis.set_title(output.replace("_", " ").title())
        axis.set_xticks(x, groups, rotation=20, ha="right")
        axis.set_ylabel("Validation RMSE (mm)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    print("MODEL 5: NO PHYSICS VS WEAK PHYSICS REGULARISATION")
    print("Validation-only comparison; original model checkpoints are not modified.")

    df = pd.read_csv(DATA)
    required = INPUTS + OUTPUTS
    if any(column not in df.columns for column in required):
        raise ValueError(f"Missing dataset columns: {sorted(set(required) - set(df.columns))}")
    if not np.isfinite(df[required].to_numpy(float)).all():
        raise ValueError("Dataset contains nonfinite input/output values.")

    with np.load(SPLIT, allow_pickle=False) as split:
        train_indices = split["train_indices"]
        validation_indices = split["validation_indices"]
        test_indices = split["test_indices"]
    all_indices = np.concatenate((train_indices, validation_indices, test_indices))
    if len(all_indices) != len(df) or len(np.unique(all_indices)) != len(df) or np.any(all_indices < 0) or np.any(all_indices >= len(df)):
        raise ValueError("Saved split is not a valid, disjoint partition of the dataset.")

    X = df[INPUTS].to_numpy(float)
    y = df[OUTPUTS].to_numpy(float)
    physics = physics_estimate(df)
    X_scaler = StandardScaler().fit(X[train_indices])
    y_scaler = StandardScaler().fit(y[train_indices])

    X_train = torch.tensor(X_scaler.transform(X[train_indices]), dtype=torch.float32)
    y_train = torch.tensor(y_scaler.transform(y[train_indices]), dtype=torch.float32)
    X_validation = torch.tensor(X_scaler.transform(X[validation_indices]), dtype=torch.float32)
    y_validation = torch.tensor(y_scaler.transform(y[validation_indices]), dtype=torch.float32)

    physics_scaled = (physics - y_scaler.mean_[4:6]) / y_scaler.scale_[4:6]
    physics_train = torch.tensor(physics_scaled[train_indices], dtype=torch.float32)
    physics_validation = torch.tensor(physics_scaled[validation_indices], dtype=torch.float32)

    # Compute frequency ratio from inputs, so no additional dataset column is required.
    dval = df.iloc[validation_indices].copy()
    length, h, young, density = (dval[name].to_numpy(float) for name in ("L", "h", "E", "rho"))
    area = B * h
    inertia = B * h**3 / 12
    k = np.pi / length
    mass = density * (area + inertia * k**2) * length / 2
    stiffness = young * inertia * k**4 * length / 2
    f1_physics = np.sqrt(stiffness / mass) / (2 * np.pi)
    ratios = dval["excitation_frequency"].to_numpy(float) / f1_physics
    regimes = np.where(ratios < RESONANCE_LOW, "below_resonance", np.where(ratios <= RESONANCE_HIGH, "near_resonance", "above_resonance"))

    # Training-only terciles ensure forcing thresholds do not use validation labels.
    pressure = df["pressure_amplitude"].to_numpy(float)
    forcing_cutoffs = np.quantile(pressure[train_indices], [1 / 3, 2 / 3])
    if not forcing_cutoffs[0] < forcing_cutoffs[1]:
        raise ValueError("Training forcing quantiles are identical; cannot make three distinct forcing bands.")
    forcing = np.where(pressure[validation_indices] <= forcing_cutoffs[0], "low", np.where(pressure[validation_indices] <= forcing_cutoffs[1], "medium", "high"))

    mse = nn.MSELoss()
    metrics = []
    losses = []
    predictions_rows = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            dataset = torch.utils.data.TensorDataset(X_train, y_train, physics_train)
            loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
            model = PhysicsRegularisedNN()
            optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            best_loss = float("inf")
            best_state = None
            best_epoch = 0
            epochs_without_improvement = 0
            start = time.perf_counter()

            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                for batch_X, batch_y, batch_physics in loader:
                    optimiser.zero_grad()
                    output = model(batch_X)
                    data_loss = mse(output, batch_y)
                    physics_loss = mse(output[:, 4:6], batch_physics)
                    (data_loss + lam * physics_loss).backward()
                    optimiser.step()

                model.eval()
                with torch.no_grad():
                    output = model(X_validation)
                    val_data_loss = float(mse(output, y_validation).item())
                    val_physics_loss = float(mse(output[:, 4:6], physics_validation).item())
                    val_total = val_data_loss + lam * val_physics_loss
                if val_total < best_loss:
                    best_loss = val_total
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    break

            duration = time.perf_counter() - start
            if best_state is None:
                raise RuntimeError("Training produced no best checkpoint.")
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                prediction_scaled = model(X_validation)
                val_data_loss = float(mse(prediction_scaled, y_validation).item())
                val_physics_loss = float(mse(prediction_scaled[:, 4:6], physics_validation).item())
            predicted = y_scaler.inverse_transform(prediction_scaled.numpy())
            truth = y[validation_indices]

            losses.append({
                "lambda": lam, "seed": seed, "best_epoch": best_epoch,
                "epochs_run": epoch, "validation_data_loss": val_data_loss,
                "validation_physics_loss": val_physics_loss,
                "weighted_validation_physics_loss": lam * val_physics_loss,
                "validation_total_loss": val_data_loss + lam * val_physics_loss,
                "weighted_physics_to_data_loss_ratio": lam * val_physics_loss / val_data_loss if val_data_loss else np.nan,
                "effective_target_physics_fraction": 3 * lam / (1 + 3 * lam),
                "training_seconds": duration,
            })
            metrics.extend(make_metrics(truth, predicted, lam, seed, "overall", "all"))
            for label in ("below_resonance", "near_resonance", "above_resonance"):
                indices = regimes == label
                if indices.any():
                    metrics.extend(make_metrics(truth[indices], predicted[indices], lam, seed, "resonance", label))
            for label in ("low", "medium", "high"):
                indices = forcing == label
                if indices.any():
                    metrics.extend(make_metrics(truth[indices], predicted[indices], lam, seed, "forcing", label))
            for position, row_index in enumerate(validation_indices):
                row = {"dataset_row_index": int(row_index), "lambda": lam, "seed": seed,
                       "frequency_ratio": float(ratios[position]), "regime": regimes[position],
                       "pressure_amplitude": float(pressure[row_index]), "forcing_band": forcing[position]}
                for j, output_name in enumerate(OUTPUTS):
                    row[f"true_{output_name}"] = float(truth[position, j])
                    row[f"predicted_{output_name}"] = float(predicted[position, j])
                predictions_rows.append(row)
            print(f"lambda={lam:.2f} seed={seed}: peak RMSE={np.sqrt(np.mean((predicted[:,4]-truth[:,4])**2))*1000:.4f} mm, "
                  f"RMS RMSE={np.sqrt(np.mean((predicted[:,5]-truth[:,5])**2))*1000:.4f} mm; best epoch={best_epoch}")

    metrics_df = pd.DataFrame(metrics)
    loss_df = pd.DataFrame(losses)
    metrics_df.to_csv(OUT / "run_metrics.csv", index=False)
    loss_df.to_csv(OUT / "loss_summary.csv", index=False)
    pd.DataFrame(predictions_rows).to_csv(OUT / "validation_predictions.csv", index=False)
    summarise(metrics_df[metrics_df.group_type == "overall"], ["lambda", "output"]).to_csv(OUT / "overall_summary.csv", index=False)
    summarise(metrics_df[metrics_df.group_type == "resonance"], ["lambda", "group", "output"]).to_csv(OUT / "regime_summary.csv", index=False)
    summarise(metrics_df[metrics_df.group_type == "forcing"], ["lambda", "group", "output"]).to_csv(OUT / "forcing_summary.csv", index=False)
    plot_comparison(metrics_df, "overall", "Model 5: With vs without physics regularisation", FIGURES / "overall_rmse.png")
    plot_comparison(metrics_df, "resonance", "Validation error by resonance regime", FIGURES / "resonance_rmse.png")
    plot_comparison(metrics_df, "forcing", "Validation error by forcing band", FIGURES / "forcing_rmse.png")
    settings = {
        "dataset": str(DATA.relative_to(ROOT)), "split": str(SPLIT.relative_to(ROOT)),
        "seeds": SEEDS, "lambdas": LAMBDAS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "architecture": [7, 64, 64, 32, 6], "width_m": B, "response_location_ratio": 0.65,
        "resonance_ratio_bounds": [RESONANCE_LOW, RESONANCE_HIGH],
        "forcing_training_quantile_cutoffs": forcing_cutoffs.tolist(),
        "forcing_band_note": "Low/medium/high pressure amplitude terciles; not a direct measure of nonlinear intensity.",
        "evaluation_split": "validation only",
        "checkpoint_selection": "minimum validation data loss + lambda*validation physics loss",
        "physics_loss_note": "Six-output data MSE and two-output displacement physics MSE; coefficient ratio per displacement = 3*lambda.",
    }
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("\nOVERALL VALIDATION RESULTS (MEAN ± SD OVER THREE SEEDS, IN mm)")
    view = summarise(metrics_df[metrics_df.group_type == "overall"], ["lambda", "output"])
    print(view[["lambda", "output", "rmse_mm_mean", "rmse_mm_std", "r2_mean"]].to_string(index=False))
    print(f"\nSaved results to: {OUT}")


if __name__ == "__main__":
    main()
