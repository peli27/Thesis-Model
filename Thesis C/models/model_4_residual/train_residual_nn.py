"""This script trains Model 4B: Residual-Learning Hybrid Neural Network."""

import copy
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ----- PROJECT PATHS -----

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = PROJECT_ROOT / "data" / "main" / "nonlinear_beam_dataset.csv"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "main_split_indices.npz"

MODEL_DIR = PROJECT_ROOT / "models" / "model_4_residual"
SAVED_MODEL_DIR = MODEL_DIR / "saved_model_4b"
TRAINING_RESULTS_DIR = MODEL_DIR / "results" / "training_4b"

SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = SAVED_MODEL_DIR / "residual_nn_model.pt"
X_SCALER_FILE = SAVED_MODEL_DIR / "residual_X_scaler.pkl"
Y_SCALER_FILE = SAVED_MODEL_DIR / "residual_y_scaler.pkl"
HISTORY_FILE = TRAINING_RESULTS_DIR / "residual_training_history.csv"
RESIDUAL_CHECK_FILE = TRAINING_RESULTS_DIR / "residual_target_check.csv"
PLOT_FILE = TRAINING_RESULTS_DIR / "residual_training_history.png"


# ----- SETTINGS -----

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 500
PATIENCE = 30

B = 0.04
RESPONSE_LOCATION_RATIO = 0.65

RAW_INPUT_COLUMNS = ["L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"]
PHYSICS_ESTIMATE_COLUMNS = ["physics_peak_displacement", "physics_rms_displacement"]
INPUT_COLUMNS = RAW_INPUT_COLUMNS + PHYSICS_ESTIMATE_COLUMNS
TARGET_COLUMNS = ["f1", "f2", "f3", "f4", "peak_residual", "rms_residual"]
FINAL_OUTPUT_COLUMNS = ["f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"]


# ----- REPRODUCIBILITY -----

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----- SIMPLIFIED PHYSICS ESTIMATE -----

def calculate_physics_estimates(data):
    L = data["L"].to_numpy(float)
    h = data["h"].to_numpy(float)
    E = data["E"].to_numpy(float)
    rho = data["rho"].to_numpy(float)
    zeta = data["zeta"].to_numpy(float)
    pressure = data["pressure_amplitude"].to_numpy(float)
    frequency = data["excitation_frequency"].to_numpy(float)

    A = B * h
    I = B * h**3 / 12.0
    k1 = np.pi / L
    M1 = rho * (A + I * k1**2) * L / 2.0
    K1 = E * I * k1**4 * L / 2.0
    omega1 = np.sqrt(K1 / M1)
    omega = 2.0 * np.pi * frequency
    q0 = pressure * B
    Q1 = 2.0 * q0 * L / np.pi
    denominator = np.sqrt((K1 - M1 * omega**2)**2 + (2.0 * zeta * M1 * omega1 * omega)**2)
    q_amp = Q1 / denominator
    peak = np.abs(q_amp * np.sin(np.pi * RESPONSE_LOCATION_RATIO))
    rms = peak / np.sqrt(2.0)
    estimates = np.column_stack((peak, rms))

    if not np.all(np.isfinite(estimates)):
        raise ValueError("Physics estimates contain NaN or infinite values.")
    return estimates


# ----- LOAD DATA -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("MODEL 4B: RESIDUAL LEARNING WITH PHYSICS INPUTS")
print("Device:", device)

dataset = pd.read_csv(DATASET_FILE)
required = RAW_INPUT_COLUMNS + FINAL_OUTPUT_COLUMNS
missing = [column for column in required if column not in dataset.columns]
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")
if not np.all(np.isfinite(dataset[required].to_numpy(float))):
    raise ValueError("Dataset contains missing or non-finite values.")


# ----- BUILD INPUTS AND RESIDUAL TARGETS -----

raw_inputs = dataset[RAW_INPUT_COLUMNS].to_numpy(float)
physics = calculate_physics_estimates(dataset)
X = np.column_stack((raw_inputs, physics))

true_displacements = dataset[["peak_displacement", "rms_displacement"]].to_numpy(float)
residuals = true_displacements - physics
y = np.column_stack((dataset[FINAL_OUTPUT_COLUMNS[:4]].to_numpy(float), residuals))

print("\nInput construction:")
print("Raw inputs:", raw_inputs.shape[1])
print("Physics inputs:", physics.shape[1])
print("Total inputs:", X.shape[1])
print("Training targets:", y.shape[1])
print("Samples:", X.shape[0])

print("\nResidual target summary:")
print("Peak residual mean:", np.mean(residuals[:, 0]))
print("Peak residual MAE:", np.mean(np.abs(residuals[:, 0])))
print("RMS residual mean:", np.mean(residuals[:, 1]))
print("RMS residual MAE:", np.mean(np.abs(residuals[:, 1])))

residual_check = pd.DataFrame({
    "physics_peak_displacement": physics[:, 0],
    "true_peak_displacement": true_displacements[:, 0],
    "peak_residual": residuals[:, 0],
    "physics_rms_displacement": physics[:, 1],
    "true_rms_displacement": true_displacements[:, 1],
    "rms_residual": residuals[:, 1]
})
residual_check.to_csv(RESIDUAL_CHECK_FILE, index=False)


# ----- LOAD FIXED DATA SPLIT -----

with np.load(SPLIT_FILE, allow_pickle=False) as split:
    train_indices = split["train_indices"]
    validation_indices = split["validation_indices"]
    test_indices = split["test_indices"]

print("\nFixed data split:")
print("Training:", len(train_indices))
print("Validation:", len(validation_indices))
print("Testing:", len(test_indices))


# ----- SCALE DATA USING TRAINING ROWS ONLY -----

X_train, X_val = X[train_indices], X[validation_indices]
y_train, y_val = y[train_indices], y[validation_indices]
X_scaler, y_scaler = StandardScaler(), StandardScaler()
X_train_s = X_scaler.fit_transform(X_train)
X_val_s = X_scaler.transform(X_val)
y_train_s = y_scaler.fit_transform(y_train)
y_val_s = y_scaler.transform(y_val)


# ----- PYTORCH DATA -----

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_s, dtype=torch.float32).to(device)

training_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
generator = torch.Generator().manual_seed(SEED)
loader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)


# ----- MODEL 4B RESIDUAL-LEARNING NETWORK -----

class ResidualLearningNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.network(x)


model = ResidualLearningNN().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ----- TRAINING -----

training_losses = []
validation_losses = []
best_validation_loss = np.inf
best_model_state = None
epochs_without_improvement = 0

print("\nTraining network...")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    batch_losses = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_function(predictions, y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    training_loss = float(np.mean(batch_losses))
    model.eval()
    with torch.no_grad():
        validation_predictions = model(X_val_t)
        validation_loss = float(loss_function(validation_predictions, y_val_t).item())

    training_losses.append(training_loss)
    validation_losses.append(validation_loss)

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train loss = {training_loss:.6e} | Validation loss = {validation_loss:.6e}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break


# ----- RESTORE AND SAVE BEST MODEL -----

if best_model_state is None:
    raise RuntimeError("No valid model state was recorded during training.")

model.load_state_dict(best_model_state)
best_epoch = int(np.argmin(validation_losses) + 1)

# Save only after successful training, preserving the original Model 4 artifacts.
joblib.dump(X_scaler, X_SCALER_FILE)
joblib.dump(y_scaler, Y_SCALER_FILE)
torch.save({
    "model_state_dict": model.state_dict(),
    "input_columns": INPUT_COLUMNS,
    "raw_input_columns": RAW_INPUT_COLUMNS,
    "physics_estimate_columns": PHYSICS_ESTIMATE_COLUMNS,
    "target_columns": TARGET_COLUMNS,
    "final_output_columns": FINAL_OUTPUT_COLUMNS,
    "architecture": [9, 64, 64, 32, 6],
    "best_validation_loss": best_validation_loss,
    "best_epoch": best_epoch,
    "seed": SEED,
    "fixed_beam_width": B,
    "response_location_ratio": RESPONSE_LOCATION_RATIO,
    "residual_definition": "true_response - physics_estimate",
    "variant": "4B_physics_inputs"
}, MODEL_FILE)

history = pd.DataFrame({"epoch": np.arange(1, len(training_losses) + 1), "training_loss": training_losses, "validation_loss": validation_losses})
history.to_csv(HISTORY_FILE, index=False)

print("\nTRAINING COMPLETE")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_validation_loss)
print("Saved model:", MODEL_FILE)
print("Saved input scaler:", X_SCALER_FILE)
print("Saved target scaler:", Y_SCALER_FILE)
print("Saved training history:", HISTORY_FILE)
print("Saved residual check:", RESIDUAL_CHECK_FILE)

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["training_loss"], label="Training")
plt.plot(history["epoch"], history["validation_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Scaled MSE loss")
plt.title("Model 4B Residual-Learning Neural Network Training")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=180)
plt.close()