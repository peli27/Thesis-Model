"""This script trains Model 3: Physics-Estimate Hybrid Neural Network."""

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

MODEL_DIR = PROJECT_ROOT / "models" / "model_3_physics_estimate"
SAVED_MODEL_DIR = MODEL_DIR / "saved_model"
TRAINING_RESULTS_DIR = MODEL_DIR / "results" / "training"

SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = SAVED_MODEL_DIR / "physics_estimate_nn_model.pt"
X_SCALER_FILE = SAVED_MODEL_DIR / "physics_estimate_X_scaler.pkl"
Y_SCALER_FILE = SAVED_MODEL_DIR / "physics_estimate_y_scaler.pkl"
HISTORY_FILE = TRAINING_RESULTS_DIR / "physics_estimate_training_history.csv"
ESTIMATE_CHECK_FILE = TRAINING_RESULTS_DIR / "physics_estimate_check.csv"

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
OUTPUT_COLUMNS = ["f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"]

# ----- REPRODUCIBILITY -----

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----- PHYSICS RESPONSE ESTIMATOR -----

def calculate_physics_estimates(data):
    L = data["L"].to_numpy(dtype=np.float64)
    h = data["h"].to_numpy(dtype=np.float64)
    E = data["E"].to_numpy(dtype=np.float64)
    rho = data["rho"].to_numpy(dtype=np.float64)
    zeta = data["zeta"].to_numpy(dtype=np.float64)
    pressure_amplitude = data["pressure_amplitude"].to_numpy(dtype=np.float64)
    excitation_frequency = data["excitation_frequency"].to_numpy(dtype=np.float64)

    A = B * h
    I = B * h**3 / 12.0
    k1 = np.pi / L

    modal_mass = rho * (A + I * k1**2) * L / 2.0
    modal_stiffness = E * I * k1**4 * L / 2.0

    omega_1 = np.sqrt(modal_stiffness / modal_mass)
    omega = 2.0 * np.pi * excitation_frequency

    q0 = pressure_amplitude * B
    modal_force = 2.0 * q0 * L / np.pi

    denominator = np.sqrt(
        (modal_stiffness - modal_mass * omega**2)**2
        + (2.0 * zeta * modal_mass * omega_1 * omega)**2
    )

    modal_amplitude = modal_force / denominator
    mode_shape_at_response = np.sin(np.pi * RESPONSE_LOCATION_RATIO)

    physics_peak_displacement = np.abs(modal_amplitude * mode_shape_at_response)
    physics_rms_displacement = physics_peak_displacement / np.sqrt(2.0)

    physics_estimates = np.column_stack((physics_peak_displacement, physics_rms_displacement))

    if not np.all(np.isfinite(physics_estimates)):
        raise ValueError("Physics estimates contain NaN or infinite values.")

    return physics_estimates

# ----- DEVICE -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PHYSICS-ESTIMATE HYBRID NEURAL NETWORK TRAINING")
print("Device:", device)

# ----- LOAD DATA -----

dataset = pd.read_csv(DATASET_FILE)
required_columns = RAW_INPUT_COLUMNS + OUTPUT_COLUMNS
missing_columns = [column for column in required_columns if column not in dataset.columns]

if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

if dataset[required_columns].isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

if np.isinf(dataset[required_columns].to_numpy(dtype=float)).sum() > 0:
    raise ValueError("Dataset contains infinite values.")

# ----- BUILD PHYSICS-ESTIMATE INPUTS -----

raw_inputs = dataset[RAW_INPUT_COLUMNS].to_numpy(dtype=np.float64)
physics_estimates = calculate_physics_estimates(dataset)
X = np.column_stack((raw_inputs, physics_estimates))
y = dataset[OUTPUT_COLUMNS].to_numpy(dtype=np.float64)

print("\nInput construction:")
print("Raw inputs:", raw_inputs.shape[1])
print("Physics-estimate inputs:", physics_estimates.shape[1])
print("Total inputs:", X.shape[1])
print("Samples:", X.shape[0])

# ----- SAVE PHYSICS-ESTIMATE CHECK -----

estimate_check = pd.DataFrame({
    "physics_peak_displacement": physics_estimates[:, 0],
    "true_peak_displacement": dataset["peak_displacement"].to_numpy(dtype=np.float64),
    "physics_rms_displacement": physics_estimates[:, 1],
    "true_rms_displacement": dataset["rms_displacement"].to_numpy(dtype=np.float64)
})
estimate_check["peak_error"] = estimate_check["physics_peak_displacement"] - estimate_check["true_peak_displacement"]
estimate_check["rms_error"] = estimate_check["physics_rms_displacement"] - estimate_check["true_rms_displacement"]
estimate_check.to_csv(ESTIMATE_CHECK_FILE, index=False)

print("\nPhysics-estimate check:")
print("Peak estimate MAE:", np.mean(np.abs(estimate_check["peak_error"])))
print("RMS estimate MAE:", np.mean(np.abs(estimate_check["rms_error"])))

# ----- LOAD FIXED BASELINE SPLIT -----

split_data = np.load(SPLIT_FILE)
train_indices = split_data["train_indices"]
validation_indices = split_data["validation_indices"]
test_indices = split_data["test_indices"]

print("\nFixed data split:")
print("Training:", len(train_indices))
print("Validation:", len(validation_indices))
print("Testing:", len(test_indices))

# ----- ARRAYS -----

X_train = X[train_indices]
X_validation = X[validation_indices]
y_train = y[train_indices]
y_validation = y[validation_indices]

# ----- SCALE INPUTS AND OUTPUTS -----

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_validation_scaled = X_scaler.transform(X_validation)
y_train_scaled = y_scaler.fit_transform(y_train)
y_validation_scaled = y_scaler.transform(y_validation)

joblib.dump(X_scaler, X_SCALER_FILE)
joblib.dump(y_scaler, Y_SCALER_FILE)

# ----- PYTORCH TENSORS -----

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_validation_tensor = torch.tensor(X_validation_scaled, dtype=torch.float32).to(device)
y_validation_tensor = torch.tensor(y_validation_scaled, dtype=torch.float32).to(device)

# ----- DATA LOADER -----

training_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
generator = torch.Generator()
generator.manual_seed(SEED)

training_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator
)

# ----- PHYSICS-ESTIMATE HYBRID NETWORK -----

class PhysicsEstimateNN(nn.Module):
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

model = PhysicsEstimateNN().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----- TRAINING -----

print("\nTraining network...")
training_losses = []
validation_losses = []
best_validation_loss = np.inf
best_model_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    batch_losses = []

    for X_batch, y_batch in training_loader:
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
        validation_predictions = model(X_validation_tensor)
        validation_loss = float(loss_function(validation_predictions, y_validation_tensor).item())

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

torch.save({
    "model_state_dict": model.state_dict(),
    "raw_input_columns": RAW_INPUT_COLUMNS,
    "physics_estimate_columns": PHYSICS_ESTIMATE_COLUMNS,
    "output_columns": OUTPUT_COLUMNS,
    "architecture": [9, 64, 64, 32, 6],
    "best_validation_loss": best_validation_loss,
    "best_epoch": best_epoch,
    "seed": SEED,
    "fixed_beam_width": B,
    "response_location_ratio": RESPONSE_LOCATION_RATIO
}, MODEL_FILE)

# ----- SAVE TRAINING HISTORY -----

history = pd.DataFrame({
    "epoch": np.arange(1, len(training_losses) + 1),
    "training_loss": training_losses,
    "validation_loss": validation_losses
})
history.to_csv(HISTORY_FILE, index=False)

# ----- FINAL TRAINING SUMMARY -----

print("TRAINING COMPLETE")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_validation_loss)
print("Saved model:", MODEL_FILE)
print("Saved input scaler:", X_SCALER_FILE)
print("Saved output scaler:", Y_SCALER_FILE)
print("Saved training history:", HISTORY_FILE)
print("Saved physics-estimate check:", ESTIMATE_CHECK_FILE)

# ----- LOSS PLOT -----

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["training_loss"], label="Training")
plt.plot(history["epoch"], history["validation_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Scaled MSE loss")
plt.title("Physics-Estimate Hybrid Neural Network Training")
plt.legend()
plt.grid()
plt.show()
