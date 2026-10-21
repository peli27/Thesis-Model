"""
This script trains Model 2: Physics-Guided Neural Network.

The model uses the same seven raw inputs and six outputs as the baseline,
but adds five physics-guided input features calculated from beam theory.

Raw inputs:
L, h, E, rho, zeta, pressure amplitude, excitation frequency.

Physics-guided inputs:
first natural frequency, first modal mass, first modal stiffness,
excitation-frequency ratio, and distance from resonance.

Outputs:
f1, f2, f3, f4, peak displacement and RMS displacement.
"""

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

MODEL_DIR = PROJECT_ROOT / "models" / "model_2_physics_guided"
SAVED_MODEL_DIR = MODEL_DIR / "saved_model"
TRAINING_RESULTS_DIR = MODEL_DIR / "results" / "training"

SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = SAVED_MODEL_DIR / "physics_guided_nn_model.pt"
X_SCALER_FILE = SAVED_MODEL_DIR / "physics_guided_X_scaler.pkl"
Y_SCALER_FILE = SAVED_MODEL_DIR / "physics_guided_y_scaler.pkl"
HISTORY_FILE = TRAINING_RESULTS_DIR / "physics_guided_training_history.csv"


# ----- SETTINGS -----

SEED = 42

BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 500
PATIENCE = 30

# Beam width is fixed in the nonlinear dataset.
B = 0.04

RAW_INPUT_COLUMNS = [
    "L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"
]

PHYSICS_FEATURE_COLUMNS = [
    "physics_f1",
    "physics_modal_mass",
    "physics_modal_stiffness",
    "physics_frequency_ratio",
    "resonance_distance"
]

OUTPUT_COLUMNS = [
    "f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"
]


# ----- REPRODUCIBILITY -----

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----- PHYSICS FEATURES -----

def calculate_physics_features(data):
    L = data["L"].to_numpy(dtype=np.float64)
    h = data["h"].to_numpy(dtype=np.float64)
    E = data["E"].to_numpy(dtype=np.float64)
    rho = data["rho"].to_numpy(dtype=np.float64)
    excitation_frequency = data["excitation_frequency"].to_numpy(dtype=np.float64)

    # Beam section properties.
    A = B * h
    I = B * h**3 / 12.0

    # First simply-supported mode.
    k1 = np.pi / L

    # Same modal definitions used by the nonlinear beam simulator.
    modal_mass = rho * (A + I * k1**2) * L / 2.0
    modal_stiffness = E * I * k1**4 * L / 2.0

    omega_1 = np.sqrt(modal_stiffness / modal_mass)
    physics_f1 = omega_1 / (2.0 * np.pi)

    physics_frequency_ratio = excitation_frequency / physics_f1
    resonance_distance = np.abs(physics_frequency_ratio - 1.0)

    physics_features = np.column_stack(
        (
            physics_f1,
            modal_mass,
            modal_stiffness,
            physics_frequency_ratio,
            resonance_distance
        )
    )

    if not np.all(np.isfinite(physics_features)):
        raise ValueError("Physics features contain NaN or infinite values.")

    return physics_features


# ----- DEVICE -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- LOAD DATA -----

dataset = pd.read_csv(DATASET_FILE)

required_columns = RAW_INPUT_COLUMNS + OUTPUT_COLUMNS

missing_columns = [
    column for column in required_columns if column not in dataset.columns
]

if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

if dataset[required_columns].isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

if np.isinf(dataset[required_columns].to_numpy(dtype=float)).sum() > 0:
    raise ValueError("Dataset contains infinite values.")


# ----- BUILD PHYSICS-GUIDED INPUTS -----

raw_inputs = dataset[RAW_INPUT_COLUMNS].to_numpy(dtype=np.float64)
physics_inputs = calculate_physics_features(dataset)

X = np.column_stack((raw_inputs, physics_inputs))
y = dataset[OUTPUT_COLUMNS].to_numpy(dtype=np.float64)

print("\nInput construction:")
print("Raw inputs:", raw_inputs.shape[1])
print("Physics-guided inputs:", physics_inputs.shape[1])
print("Total inputs:", X.shape[1])
print("Samples:", X.shape[0])


# ----- VERIFY PHYSICS FEATURE CALCULATION -----

# The stored f1 is used here only as a one-off implementation check.
# It is NOT used to construct the physics-guided input features.
physics_f1 = physics_inputs[:, 0]
stored_f1 = dataset["f1"].to_numpy(dtype=np.float64)

maximum_f1_difference = np.max(np.abs(physics_f1 - stored_f1))
maximum_f1_relative_difference = np.max(
    np.abs(physics_f1 - stored_f1) / stored_f1 * 100.0
)

print("\nPhysics-feature verification:")
print(f"Maximum f1 absolute difference = {maximum_f1_difference:.6e} Hz")
print(f"Maximum f1 relative difference = {maximum_f1_relative_difference:.6e} %")


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

# Fit scalers using training data only.
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

X_validation_tensor = torch.tensor(
    X_validation_scaled, dtype=torch.float32
).to(device)

y_validation_tensor = torch.tensor(
    y_validation_scaled, dtype=torch.float32
).to(device)


# ----- DATA LOADER -----

training_dataset = torch.utils.data.TensorDataset(
    X_train_tensor, y_train_tensor
)

generator = torch.Generator()
generator.manual_seed(SEED)

training_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator
)


# ----- PHYSICS-GUIDED NETWORK -----

class PhysicsGuidedNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.network(x)


model = PhysicsGuidedNN().to(device)

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

    # ----- VALIDATION LOSS -----

    model.eval()

    with torch.no_grad():
        validation_predictions = model(X_validation_tensor)
        validation_loss = float(
            loss_function(
                validation_predictions,
                y_validation_tensor
            ).item()
        )

    training_losses.append(training_loss)
    validation_losses.append(validation_loss)

    # ----- EARLY STOPPING -----

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


# ----- RESTORE BEST MODEL -----

if best_model_state is None:
    raise RuntimeError("No valid model state was recorded during training.")

model.load_state_dict(best_model_state)

best_epoch = int(np.argmin(validation_losses) + 1)


# ----- SAVE MODEL -----

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "physics_feature_columns": PHYSICS_FEATURE_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "architecture": [12, 64, 64, 32, 6],
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "seed": SEED,
        "fixed_beam_width": B
    },
    MODEL_FILE
)


# ----- SAVE TRAINING HISTORY -----

history = pd.DataFrame(
    {
        "epoch": np.arange(1, len(training_losses) + 1),
        "training_loss": training_losses,
        "validation_loss": validation_losses
    }
)

history.to_csv(HISTORY_FILE, index=False)


# ----- FINAL TRAINING SUMMARY -----

print("TRAINING COMPLETE")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_validation_loss)
print("Saved model:", MODEL_FILE)
print("Saved input scaler:", X_SCALER_FILE)
print("Saved output scaler:", Y_SCALER_FILE)
print("Saved training history:", HISTORY_FILE)


# ----- LOSS PLOT -----

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["training_loss"], label="Training")
plt.plot(history["epoch"], history["validation_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Scaled MSE loss")
plt.title("Physics-Guided Neural Network Training")
plt.legend()
plt.grid()
plt.show()