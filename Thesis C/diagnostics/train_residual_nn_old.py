"""This script trains Model 4: Residual-Learning Hybrid Neural Network."""

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
SAVED_MODEL_DIR = MODEL_DIR / "saved_model"
TRAINING_RESULTS_DIR = MODEL_DIR / "results" / "training"

SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = SAVED_MODEL_DIR / "residual_nn_model.pt"
X_SCALER_FILE = SAVED_MODEL_DIR / "residual_X_scaler.pkl"
Y_SCALER_FILE = SAVED_MODEL_DIR / "residual_y_scaler.pkl"
HISTORY_FILE = TRAINING_RESULTS_DIR / "residual_training_history.csv"
RESIDUAL_CHECK_FILE = TRAINING_RESULTS_DIR / "residual_target_check.csv"


# ----- SETTINGS -----

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 500
PATIENCE = 30

B = 0.04
RESPONSE_LOCATION_RATIO = 0.65

INPUT_COLUMNS = ["L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"]

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

print("RESIDUAL-LEARNING HYBRID NEURAL NETWORK TRAINING")
print("Device:", device)

dataset = pd.read_csv(DATASET_FILE)

required = INPUT_COLUMNS + FINAL_OUTPUT_COLUMNS
missing = [c for c in required if c not in dataset.columns]

if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")

if dataset[required].isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

if np.isinf(dataset[required].to_numpy(float)).sum() > 0:
    raise ValueError("Dataset contains infinite values.")


# ----- BUILD RESIDUAL TARGETS -----

X = dataset[INPUT_COLUMNS].to_numpy(float)

physics = calculate_physics_estimates(dataset)
physics_peak = physics[:, 0]
physics_rms = physics[:, 1]

true_peak = dataset["peak_displacement"].to_numpy(float)
true_rms = dataset["rms_displacement"].to_numpy(float)

peak_residual = true_peak - physics_peak
rms_residual = true_rms - physics_rms

y = np.column_stack((
    dataset["f1"].to_numpy(float),
    dataset["f2"].to_numpy(float),
    dataset["f3"].to_numpy(float),
    dataset["f4"].to_numpy(float),
    peak_residual,
    rms_residual
))

print("\nInput construction:")
print("Raw inputs:", X.shape[1])
print("Training targets:", y.shape[1])
print("Samples:", X.shape[0])

print("\nResidual target summary:")
print("Peak residual mean:", np.mean(peak_residual))
print("Peak residual MAE:", np.mean(np.abs(peak_residual)))
print("RMS residual mean:", np.mean(rms_residual))
print("RMS residual MAE:", np.mean(np.abs(rms_residual)))

residual_check = pd.DataFrame({
    "physics_peak_displacement": physics_peak,
    "true_peak_displacement": true_peak,
    "peak_residual": peak_residual,
    "physics_rms_displacement": physics_rms,
    "true_rms_displacement": true_rms,
    "rms_residual": rms_residual
})

residual_check.to_csv(RESIDUAL_CHECK_FILE, index=False)


# ----- LOAD FIXED DATA SPLIT -----

split = np.load(SPLIT_FILE)

train_indices = split["train_indices"]
validation_indices = split["validation_indices"]
test_indices = split["test_indices"]

print("\nFixed data split:")
print("Training:", len(train_indices))
print("Validation:", len(validation_indices))
print("Testing:", len(test_indices))


# ----- SCALE DATA -----

X_train = X[train_indices]
X_val = X[validation_indices]

y_train = y[train_indices]
y_val = y[validation_indices]

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_s = X_scaler.fit_transform(X_train)
X_val_s = X_scaler.transform(X_val)

y_train_s = y_scaler.fit_transform(y_train)
y_val_s = y_scaler.transform(y_val)

joblib.dump(X_scaler, X_SCALER_FILE)
joblib.dump(y_scaler, Y_SCALER_FILE)


# ----- PYTORCH DATA -----

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)

X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_s, dtype=torch.float32).to(device)

training_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)

generator = torch.Generator()
generator.manual_seed(SEED)

loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator
)


# ----- RESIDUAL-LEARNING NETWORK -----

class ResidualLearningNN(nn.Module):

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

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

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
        validation_loss = float(
            loss_function(validation_predictions, y_val_t).item()
        )

    training_losses.append(training_loss)
    validation_losses.append(validation_loss)

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Train loss = {training_loss:.6e} | "
            f"Validation loss = {validation_loss:.6e}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break


# ----- SAVE MODEL -----

if best_model_state is None:
    raise RuntimeError("No valid model state was recorded during training.")

model.load_state_dict(best_model_state)
best_epoch = int(np.argmin(validation_losses) + 1)

torch.save({
    "model_state_dict": model.state_dict(),
    "input_columns": INPUT_COLUMNS,
    "target_columns": TARGET_COLUMNS,
    "final_output_columns": FINAL_OUTPUT_COLUMNS,
    "architecture": [7, 64, 64, 32, 6],
    "best_validation_loss": best_validation_loss,
    "best_epoch": best_epoch,
    "seed": SEED,
    "fixed_beam_width": B,
    "response_location_ratio": RESPONSE_LOCATION_RATIO,
    "residual_definition": "true_response - physics_estimate"
}, MODEL_FILE)


# ----- SAVE RESULTS -----

history = pd.DataFrame({
    "epoch": np.arange(1, len(training_losses) + 1),
    "training_loss": training_losses,
    "validation_loss": validation_losses
})

history.to_csv(HISTORY_FILE, index=False)

print("TRAINING COMPLETE")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_validation_loss)
print("Saved model:", MODEL_FILE)
print("Saved input scaler:", X_SCALER_FILE)
print("Saved target scaler:", Y_SCALER_FILE)
print("Saved training history:", HISTORY_FILE)
print("Saved residual check:", RESIDUAL_CHECK_FILE)


# ----- LOSS PLOT -----

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["training_loss"], label="Training")
plt.plot(history["epoch"], history["validation_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Scaled MSE loss")
plt.title("Residual-Learning Hybrid Neural Network Training")
plt.legend()
plt.grid()
plt.show()