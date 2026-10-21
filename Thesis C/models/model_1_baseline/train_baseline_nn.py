"""
This script trains the purely data-driven feedforward neural-network baseline.

Inputs:
L, h, E, rho, zeta, pressure amplitude and excitation frequency.

Outputs:
f1, f2, f3, f4, peak displacement and RMS displacement.

"""

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ----- PROJECT PATHS -----

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "model_1_baseline"

MAIN_DATA_DIR = DATA_DIR / "main"
SPLIT_DIR = DATA_DIR / "splits"

SAVED_MODEL_DIR = MODEL_DIR / "saved_model"
RESULTS_DIR = MODEL_DIR / "results"
TRAINING_RESULTS_DIR = RESULTS_DIR / "training"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----- SETTINGS -----

DATASET_FILE = MAIN_DATA_DIR / "nonlinear_beam_dataset.csv"

SEED = 42

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 500
PATIENCE = 30

MODEL_FILE = SAVED_MODEL_DIR / "baseline_nn_model.pt"
X_SCALER_FILE = SAVED_MODEL_DIR / "baseline_X_scaler.pkl"
Y_SCALER_FILE = SAVED_MODEL_DIR / "baseline_y_scaler.pkl"

SPLIT_FILE = SPLIT_DIR / "main_split_indices.npz"

HISTORY_FILE = TRAINING_RESULTS_DIR / "baseline_training_history.csv"

INPUT_COLUMNS = [
    "L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"
]

OUTPUT_COLUMNS = [
    "f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"
]


# ----- REPRODUCIBILITY -----

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----- DEVICE -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("BASELINE NEURAL NETWORK TRAINING")
print("Device:", device)


# ----- LOAD DATA -----

dataset = pd.read_csv(DATASET_FILE)

required_columns = INPUT_COLUMNS + OUTPUT_COLUMNS + ["frequency_ratio"]
missing_columns = [column for column in required_columns if column not in dataset.columns]

if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

if dataset[required_columns].isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

if np.isinf(dataset[required_columns].to_numpy(dtype=float)).sum() > 0:
    raise ValueError("Dataset contains infinite values.")


# ----- REGIME LABELS FOR STRATIFIED SPLITTING -----

def classify_regime(ratio):
    if ratio < 0.85:
        return 0
    elif ratio <= 1.20:
        return 1
    return 2

regime_labels = dataset["frequency_ratio"].apply(classify_regime).to_numpy()
all_indices = np.arange(len(dataset))


# ----- TRAIN / VALIDATION / TEST SPLIT -----

train_indices, temporary_indices = train_test_split(
    all_indices,
    test_size=(1 - TRAIN_FRACTION),
    random_state=SEED,
    stratify=regime_labels
)

temporary_labels = regime_labels[temporary_indices]
test_share_of_temporary = TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION)

validation_indices, test_indices = train_test_split(
    temporary_indices,
    test_size=test_share_of_temporary,
    random_state=SEED,
    stratify=temporary_labels
)

np.savez(
    SPLIT_FILE,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices
)

print("\nData split:")
print("Training:", len(train_indices))
print("Validation:", len(validation_indices))
print("Testing:", len(test_indices))


# ----- PREPARE ARRAYS -----

X = dataset[INPUT_COLUMNS].to_numpy(dtype=np.float64)
y = dataset[OUTPUT_COLUMNS].to_numpy(dtype=np.float64)

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

X_validation_tensor = torch.tensor(X_validation_scaled, dtype=torch.float32).to(device)
y_validation_tensor = torch.tensor(y_validation_scaled, dtype=torch.float32).to(device)


# ----- DATA LOADER -----

training_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

generator = torch.Generator()
generator.manual_seed(SEED)

training_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator
)


# ----- BASELINE NETWORK -----

class BaselineNN(nn.Module):

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


model = BaselineNN().to(device)

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
        validation_loss = float(loss_function(validation_predictions, y_validation_tensor).item())

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
        print(f"Epoch {epoch:3d} | Train loss = {training_loss:.6e} | Validation loss = {validation_loss:.6e}")


    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break


# ----- RESTORE BEST MODEL -----

if best_model_state is None:
    raise RuntimeError("No valid model state was recorded during training.")

model.load_state_dict(best_model_state)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_columns": INPUT_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "architecture": [7, 64, 64, 32, 6],
        "best_validation_loss": best_validation_loss,
        "seed": SEED
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

best_epoch = int(np.argmin(validation_losses) + 1)

print("TRAINING COMPLETE")

print("Best epoch:", best_epoch)
print("Best validation loss:", best_validation_loss)
print("Saved model:", MODEL_FILE)
print("Saved input scaler:", X_SCALER_FILE)
print("Saved output scaler:", Y_SCALER_FILE)
print("Saved split indices:", SPLIT_FILE)
print("Saved training history:", HISTORY_FILE)


# ----- LOSS PLOT -----

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["training_loss"], label="Training")
plt.plot(history["epoch"], history["validation_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Scaled MSE loss")
plt.title("Baseline Neural Network Training")
plt.legend()
plt.grid()
plt.show()