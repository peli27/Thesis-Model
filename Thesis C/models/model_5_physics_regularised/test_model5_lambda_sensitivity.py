import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ----- PATHS AND SETTINGS -----

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "main" / "nonlinear_beam_dataset.csv"
SPLIT = ROOT / "data" / "splits" / "main_split_indices.npz"
OUT = ROOT / "models" / "model_5_physics_regularised" / "results" / "lambda_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
LAMBDAS = [0.01, 0.1, 1.0]
B = 0.04

INPUTS = ["L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"]
OUTPUTS = ["f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"]


# ----- REPRODUCIBILITY -----

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# ----- SIMPLIFIED PHYSICS MODEL -----

def physics_estimate(data):
    L = data["L"].values
    h = data["h"].values
    E = data["E"].values
    rho = data["rho"].values
    zeta = data["zeta"].values
    P = data["pressure_amplitude"].values
    f = data["excitation_frequency"].values

    A = B * h
    I = B * h**3 / 12
    k = np.pi / L

    M = rho * (A + I * k**2) * L / 2
    K = E * I * k**4 * L / 2

    omega_1 = np.sqrt(K / M)
    omega = 2 * np.pi * f
    Q = 2 * P * B * L / np.pi

    amplitude = Q / np.sqrt((K - M * omega**2)**2 + (2 * zeta * M * omega_1 * omega)**2)

    peak = np.abs(amplitude * np.sin(0.65 * np.pi))
    rms = peak / np.sqrt(2)

    return np.column_stack((peak, rms))


# ----- NEURAL NETWORK -----

class PhysicsRegularisedNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.network(x)


# ----- LOAD DATA -----

df = pd.read_csv(DATA)

X = df[INPUTS].values
y = df[OUTPUTS].values
physics = physics_estimate(df)

split = np.load(SPLIT)
train_indices = split["train_indices"]
validation_indices = split["validation_indices"]

X_scaler = StandardScaler().fit(X[train_indices])
y_scaler = StandardScaler().fit(y[train_indices])

X_train = X_scaler.transform(X[train_indices])
X_validation = X_scaler.transform(X[validation_indices])

y_train = y_scaler.transform(y[train_indices])
y_validation = y_scaler.transform(y[validation_indices])

physics_train = np.column_stack((
    (physics[train_indices, 0] - y_scaler.mean_[4]) / y_scaler.scale_[4],
    (physics[train_indices, 1] - y_scaler.mean_[5]) / y_scaler.scale_[5]
))

physics_validation = np.column_stack((
    (physics[validation_indices, 0] - y_scaler.mean_[4]) / y_scaler.scale_[4],
    (physics[validation_indices, 1] - y_scaler.mean_[5]) / y_scaler.scale_[5]
))

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
physics_train = torch.tensor(physics_train, dtype=torch.float32)

X_validation = torch.tensor(X_validation, dtype=torch.float32)
y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32)
physics_validation = torch.tensor(physics_validation, dtype=torch.float32)

mse = nn.MSELoss()
results = []


# ----- TEST EACH PHYSICS WEIGHT -----

print("MODEL 5 PHYSICS-WEIGHT SENSITIVITY TEST")

for physics_weight in LAMBDAS:

    set_seed()

    generator = torch.Generator().manual_seed(SEED)

    dataset = torch.utils.data.TensorDataset(X_train, y_train, physics_train)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=generator
    )

    model = PhysicsRegularisedNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, 501):

        model.train()

        for X_batch, y_batch, physics_batch in loader:

            optimizer.zero_grad()

            prediction = model(X_batch)

            data_loss = mse(prediction, y_batch)
            physics_loss = mse(prediction[:, 4:6], physics_batch)

            total_loss = data_loss + physics_weight * physics_loss

            total_loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():

            prediction = model(X_validation)

            validation_data_loss = mse(
                prediction,
                y_validation_tensor
            ).item()

            validation_physics_loss = mse(
                prediction[:, 4:6],
                physics_validation
            ).item()

            validation_total_loss = (
                validation_data_loss
                + physics_weight * validation_physics_loss
            )

        if validation_total_loss < best_loss:

            best_loss = validation_total_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 30:
            break


    # ----- EVALUATE BEST MODEL -----

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():

        prediction = model(X_validation)

        validation_data_loss = mse(
            prediction,
            y_validation_tensor
        ).item()

        validation_physics_loss = mse(
            prediction[:, 4:6],
            physics_validation
        ).item()

    prediction_real = y_scaler.inverse_transform(prediction.numpy())
    truth = y[validation_indices]

    peak_error = prediction_real[:, 4] - truth[:, 4]
    rms_error = prediction_real[:, 5] - truth[:, 5]

    peak_rmse = np.sqrt(np.mean(peak_error**2))
    peak_mae = np.mean(np.abs(peak_error))

    rms_rmse = np.sqrt(np.mean(rms_error**2))
    rms_mae = np.mean(np.abs(rms_error))

    results.append([
        physics_weight,
        best_epoch,
        validation_data_loss,
        validation_physics_loss,
        peak_rmse,
        peak_mae,
        rms_rmse,
        rms_mae
    ])

    print(f"\nPhysics weight = {physics_weight}")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation data loss: {validation_data_loss:.6e}")
    print(f"Validation physics loss: {validation_physics_loss:.6e}")
    print(f"Peak RMSE: {peak_rmse:.6e}")
    print(f"Peak MAE: {peak_mae:.6e}")
    print(f"RMS RMSE: {rms_rmse:.6e}")
    print(f"RMS MAE: {rms_mae:.6e}")


# ----- SAVE RESULTS -----

results = pd.DataFrame(
    results,
    columns=[
        "physics_weight",
        "best_epoch",
        "validation_data_loss",
        "validation_physics_loss",
        "peak_rmse",
        "peak_mae",
        "rms_rmse",
        "rms_mae"
    ]
)

results.to_csv(
    OUT / "lambda_sensitivity_summary.csv",
    index=False
)

print("SENSITIVITY TEST COMPLETE")
print(results.to_string(index=False))