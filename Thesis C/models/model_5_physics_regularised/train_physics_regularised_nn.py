"""This script trains Model 5: Physics-Regularised Neural Network."""

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

MODEL_DIR = PROJECT_ROOT / "models" / "model_5_physics_regularised"
SAVED_MODEL_DIR = MODEL_DIR / "saved_model"
TRAINING_RESULTS_DIR = MODEL_DIR / "results" / "training"

SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = SAVED_MODEL_DIR / "physics_regularised_nn_model.pt"
X_SCALER_FILE = SAVED_MODEL_DIR / "physics_regularised_X_scaler.pkl"
Y_SCALER_FILE = SAVED_MODEL_DIR / "physics_regularised_y_scaler.pkl"
HISTORY_FILE = TRAINING_RESULTS_DIR / "physics_regularised_training_history.csv"


# ----- SETTINGS -----

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 500
PATIENCE = 30

PHYSICS_WEIGHT = 0.01
B = 0.04
RESPONSE_LOCATION_RATIO = 0.65

INPUT_COLUMNS = ["L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"]

OUTPUT_COLUMNS = ["f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"]


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

    denominator = np.sqrt(
        (K1 - M1 * omega**2)**2
        + (2.0 * zeta * M1 * omega1 * omega)**2
    )

    q_amp = Q1 / denominator

    peak = np.abs(q_amp * np.sin(np.pi * RESPONSE_LOCATION_RATIO))
    rms = peak / np.sqrt(2.0)

    estimates = np.column_stack((peak, rms))

    if not np.all(np.isfinite(estimates)):
        raise ValueError("Physics estimates contain NaN or infinite values.")

    return estimates


# ----- LOAD DATA -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PHYSICS-REGULARISED NEURAL NETWORK TRAINING")
print("Device:", device)
print("Physics loss weight:", PHYSICS_WEIGHT)

dataset = pd.read_csv(DATASET_FILE)

required = INPUT_COLUMNS + OUTPUT_COLUMNS
missing = [c for c in required if c not in dataset.columns]

if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")

if dataset[required].isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

if np.isinf(dataset[required].to_numpy(float)).sum() > 0:
    raise ValueError("Dataset contains infinite values.")

X = dataset[INPUT_COLUMNS].to_numpy(float)
y = dataset[OUTPUT_COLUMNS].to_numpy(float)

physics = calculate_physics_estimates(dataset)


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

physics_train_s = np.column_stack((
    (physics[train_indices, 0] - y_scaler.mean_[4]) / y_scaler.scale_[4],
    (physics[train_indices, 1] - y_scaler.mean_[5]) / y_scaler.scale_[5]
))

physics_val_s = np.column_stack((
    (physics[validation_indices, 0] - y_scaler.mean_[4]) / y_scaler.scale_[4],
    (physics[validation_indices, 1] - y_scaler.mean_[5]) / y_scaler.scale_[5]
))

joblib.dump(X_scaler, X_SCALER_FILE)
joblib.dump(y_scaler, Y_SCALER_FILE)


# ----- PYTORCH DATA -----

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
physics_train_t = torch.tensor(physics_train_s, dtype=torch.float32)

X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_s, dtype=torch.float32).to(device)
physics_val_t = torch.tensor(physics_val_s, dtype=torch.float32).to(device)

training_dataset = torch.utils.data.TensorDataset(
    X_train_t, y_train_t, physics_train_t
)

generator = torch.Generator()
generator.manual_seed(SEED)

loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator
)


# ----- PHYSICS-REGULARISED NETWORK -----

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


model = PhysicsRegularisedNN().to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ----- PHYSICS-REGULARISED LOSS -----

def calculate_loss(predictions, true_values, physics_values):

    data_loss = loss_function(predictions, true_values)

    physics_loss = loss_function(
        predictions[:, 4:6],
        physics_values
    )

    total_loss = data_loss + PHYSICS_WEIGHT * physics_loss

    return total_loss, data_loss, physics_loss


# ----- TRAINING -----

history = []

best_validation_loss = np.inf
best_model_state = None
epochs_without_improvement = 0

print("\nTraining network...")

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    batch_total_losses = []
    batch_data_losses = []
    batch_physics_losses = []

    for X_batch, y_batch, physics_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        physics_batch = physics_batch.to(device)

        optimizer.zero_grad()

        predictions = model(X_batch)

        total_loss, data_loss, physics_loss = calculate_loss(
            predictions,
            y_batch,
            physics_batch
        )

        total_loss.backward()
        optimizer.step()

        batch_total_losses.append(total_loss.item())
        batch_data_losses.append(data_loss.item())
        batch_physics_losses.append(physics_loss.item())

    model.eval()

    with torch.no_grad():

        validation_predictions = model(X_val_t)

        val_total, val_data, val_physics = calculate_loss(
            validation_predictions,
            y_val_t,
            physics_val_t
        )

    row = [
        epoch,
        np.mean(batch_total_losses),
        float(val_total),
        np.mean(batch_data_losses),
        np.mean(batch_physics_losses),
        float(val_data),
        float(val_physics)
    ]

    history.append(row)

    if float(val_total) < best_validation_loss:
        best_validation_loss = float(val_total)
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Train total = {row[1]:.6e} | "
            f"Val total = {row[2]:.6e} | "
            f"Val data = {row[5]:.6e} | "
            f"Val physics = {row[6]:.6e}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break


# ----- SAVE MODEL -----

if best_model_state is None:
    raise RuntimeError("No valid model state was recorded during training.")

model.load_state_dict(best_model_state)
best_epoch = int(np.argmin([row[2] for row in history]) + 1)

torch.save({
    "model_state_dict": model.state_dict(),
    "input_columns": INPUT_COLUMNS,
    "output_columns": OUTPUT_COLUMNS,
    "architecture": [7, 64, 64, 32, 6],
    "best_validation_loss": best_validation_loss,
    "best_epoch": best_epoch,
    "seed": SEED,
    "physics_weight": PHYSICS_WEIGHT,
    "fixed_beam_width": B,
    "response_location_ratio": RESPONSE_LOCATION_RATIO,
    "loss_definition": "data_mse + physics_weight * response_physics_mse"
}, MODEL_FILE)


# ----- SAVE RESULTS -----

history = pd.DataFrame(
    history,
    columns=[
        "epoch",
        "training_total_loss",
        "validation_total_loss",
        "training_data_loss",
        "training_physics_loss",
        "validation_data_loss",
        "validation_physics_loss"
    ]
)

history.to_csv(HISTORY_FILE, index=False)

print("TRAINING COMPLETE")
print("Best epoch:", best_epoch)
print("Best validation total loss:", best_validation_loss)
print("Physics loss weight:", PHYSICS_WEIGHT)
print("Saved model:", MODEL_FILE)
print("Saved input scaler:", X_SCALER_FILE)
print("Saved output scaler:", Y_SCALER_FILE)
print("Saved training history:", HISTORY_FILE)


# ----- LOSS PLOT -----

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["training_total_loss"], label="Training")
plt.plot(history["epoch"], history["validation_total_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Scaled combined loss")
plt.title("Physics-Regularised Neural Network Training")
plt.legend()
plt.grid()
plt.show()