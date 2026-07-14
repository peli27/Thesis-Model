"""
Physics-guided neural network.

This script trains a feedforward neural network to predict beam
displacement response using raw beam/loading parameters and
physics-derived input features.

Inputs:
    Beam and loading parameters
        L
        b
        h
        E
        rho
        damping_ratio
        force_amplitude
        excitation_frequency

    Physics-guided features
        first_natural_frequency
        modal_stiffness
        modal_mass
        frequency_ratio
        resonance_proximity

Outputs:
    peak_displacement
    rms_displacement
"""

import copy
import random
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler


# ----- Configuration -----

SEED = 42

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

HYBRID_DATASET_DIR = PROJECT_DIR / "dataset" / "hybrid"

TRAIN_DATA_PATH = (HYBRID_DATASET_DIR / "train_hybrid_dataset.csv")

VALIDATION_DATA_PATH = (HYBRID_DATASET_DIR / "validation_hybrid_dataset.csv")

TEST_DATA_PATH = (HYBRID_DATASET_DIR / "test_hybrid_dataset.csv")

RESULTS_DIR = PROJECT_DIR / "results" / "Physics_Guided_NN"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = PROJECT_DIR / "figures" / "phase_5_plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_COLUMNS = [
    "L",
    "b",
    "h",
    "E",
    "rho",
    "damping_ratio",
    "force_amplitude",
    "excitation_frequency",
    "first_natural_frequency",
    "modal_stiffness",
    "modal_mass",
    "frequency_ratio",
    "resonance_proximity",
]

OUTPUT_COLUMNS = [
    "peak_displacement",
    "rms_displacement",
]


BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.001
PATIENCE = 50

DEVICE = torch.device("cpu")


# ----- Reproducibility -----

def set_seed(seed):
    """
    Sets random seeds so model training is reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ----- Neural network model -----

class PhysicsGuidedBeamNN(nn.Module):
    """
    Feedforward physics-guided neural network.

    Architecture:
        input layer
        -> 64 neurons -> ReLU
        -> 64 neurons -> ReLU
        -> 32 neurons -> ReLU
        -> output layer
    """

    def __init__(self, input_size, output_size):
        super(PhysicsGuidedBeamNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.network(x)


# ----- Load datasets -----

train_df = pd.read_csv(TRAIN_DATA_PATH)
validation_df = pd.read_csv(VALIDATION_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

print(f"Training dataset shape:   {train_df.shape}")
print(f"Validation dataset shape: {validation_df.shape}")
print(f"Test dataset shape:       {test_df.shape}")

# ----- Check required columns -----

def check_required_columns(df, dataset_name):

    missing_inputs = [column for column in INPUT_COLUMNS if column not in df.columns]

    missing_outputs = [column for column in OUTPUT_COLUMNS if column not in df.columns]

    if missing_inputs or missing_outputs:
        raise ValueError(
            f"Missing columns detected in {dataset_name}.\n"
            f"Missing input columns: {missing_inputs}\n"
            f"Missing output columns: {missing_outputs}"
        )

check_required_columns(train_df, "training dataset")

check_required_columns(validation_df, "validation dataset")

check_required_columns(test_df, "test dataset")

# ----- Select inputs and outputs -----

X_train = train_df[INPUT_COLUMNS].values
y_train = train_df[OUTPUT_COLUMNS].values

X_val = validation_df[INPUT_COLUMNS].values
y_val = validation_df[OUTPUT_COLUMNS].values

X_test = test_df[INPUT_COLUMNS].values
y_test = test_df[OUTPUT_COLUMNS].values

# ----- Scale inputs and outputs -----

input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)

X_val_scaled = input_scaler.transform(X_val)

X_test_scaled = input_scaler.transform(X_test)


y_train_scaled = output_scaler.fit_transform(y_train)

y_val_scaled = output_scaler.transform(y_val)

y_test_scaled = output_scaler.transform(y_test)


joblib.dump(input_scaler, MODELS_DIR / "physics_guided_input_scaler.pkl")

joblib.dump(output_scaler, MODELS_DIR / "physics_guided_output_scaler.pkl")

print("\nInput and output scalers saved.")


# ----- Convert data to PyTorch tensors -----

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)


X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# ----- Initialise model -----

input_size = len(INPUT_COLUMNS)
output_size = len(OUTPUT_COLUMNS)

model = PhysicsGuidedBeamNN(input_size=input_size, output_size=output_size).to(DEVICE)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nPhysics-guided neural network initialised.")

print(model)

# ----- Training with early stopping -----

training_losses = []
validation_losses = []

best_validation_loss = np.inf
best_model_state = None

epochs_without_improvement = 0

start_training_time = time.perf_counter()

for epoch in range(EPOCHS):

    model.train()

    batch_losses = []

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # Forward pass
        predictions = model(X_batch)

        # Calculate loss
        loss = loss_function(predictions, y_batch)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_losses.append(loss.item())

    mean_training_loss = np.mean(batch_losses)


    # Validation loss
    model.eval()

    with torch.no_grad():

        val_predictions = model(X_val_tensor.to(DEVICE))

        val_loss = loss_function(val_predictions, y_val_tensor.to(DEVICE)).item()

    training_losses.append(mean_training_loss)

    validation_losses.append(val_loss)


    # Early stopping check
    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Training Loss: "
            f"{mean_training_loss:.6f} "
            f"Validation Loss: "
            f"{val_loss:.6f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping triggered "
            f"at epoch {epoch + 1}."
        )

        break

end_training_time = time.perf_counter()

training_time_seconds = ( end_training_time - start_training_time )

# Restore best model

if best_model_state is None:
    raise RuntimeError("Training finished without saving a valid model state.")

model.load_state_dict(best_model_state)

print("\nTraining complete.")

print(f"Best validation loss: {best_validation_loss:.6f}")

print(f"Training time: {training_time_seconds:.4f} seconds")

# ----- Save trained model -----

model_save_path = (MODELS_DIR / "physics_guided_neural_network.pth")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_columns": INPUT_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "input_size": input_size,
        "output_size": output_size,
        "architecture":
            "13-64-64-32-2 physics-guided "
            "feedforward neural network",
    },
    model_save_path,
)

print(f"\nTrained model saved to: {model_save_path}")


# ----- Save training history -----

history_df = pd.DataFrame({
    "epoch": np.arange(
        1,
        len(training_losses) + 1,
    ),
    "training_loss": training_losses,
    "validation_loss": validation_losses,
})

history_df.to_csv(RESULTS_DIR / "physics_guided_NN_training_history.csv", index=False)

print("Training history saved.")


# ----- Plot loss curves -----

plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["training_loss"],
    label="Training Loss",
)

plt.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "Physics-Guided Neural Network "
    "Training and Validation Loss"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "physics_guided_NN_loss_curve.png", dpi=300)
plt.close()

print("Loss curve saved.")


# ----- Test set prediction -----

model.eval()

start_prediction_time = time.perf_counter()

with torch.no_grad():

    y_pred_scaled = model(X_test_tensor.to(DEVICE)).cpu().numpy()

end_prediction_time = time.perf_counter()

prediction_time_seconds = (end_prediction_time - start_prediction_time)


# Convert predictions back to physical units

y_pred = output_scaler.inverse_transform(y_pred_scaled)

print(f"Prediction time: {prediction_time_seconds:.6f} seconds")


# ----- Save predictions -----

predictions_df = pd.DataFrame()

for i, output_name in enumerate(OUTPUT_COLUMNS):

    predictions_df[
        f"actual_{output_name}"
    ] = y_test[:, i]

    predictions_df[
        f"predicted_{output_name}"
    ] = y_pred[:, i]

    predictions_df[
        f"error_{output_name}"
    ] = (
        y_test[:, i]
        - y_pred[:, i]
    )

predictions_df.to_csv(RESULTS_DIR / "physics_guided_NN_predictions.csv", index=False)

print("Predictions saved.")


# ----- Calculate test metrics -----

metrics = []

for i, output_name in enumerate(OUTPUT_COLUMNS):

    actual = y_test[:, i]

    predicted = y_pred[:, i]


    rmse = np.sqrt(mean_squared_error(actual, predicted))

    mae = mean_absolute_error(actual, predicted)

    r2 = r2_score(actual, predicted)

    metrics.append({
        "model": "Physics_Guided_Neural_Network",
        "output": output_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "training_time_seconds": training_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
    })

metrics_df = pd.DataFrame(metrics)


metrics_df.to_csv(RESULTS_DIR/ "physics_guided_NN_model_metrics.csv", index=False)

print("\nPhysics-guided neural network test metrics:")

print(metrics_df)


# ----- Predicted vs actual plots -----

for i, output_name in enumerate(OUTPUT_COLUMNS):

    actual = y_test[:, i]

    predicted = y_pred[:, i]


    min_value = min(actual.min(), predicted.min())

    max_value = max(actual.max(), predicted.max())

    plt.figure(figsize=(6, 6))

    plt.scatter(actual, predicted, alpha=0.6)

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    plt.xlabel(f"Actual {output_name}")

    plt.ylabel(f"Predicted {output_name}")

    plt.title(f"Physics-Guided NN Predicted vs Actual - {output_name}")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(PLOTS_DIR / ( "physics_guided_NN_predicted_vs_actual_" f"{output_name}.png"), dpi=300)

    plt.close()

print("Predicted vs actual plots saved.")


# ----- Error distribution plots -----

for i, output_name in enumerate(OUTPUT_COLUMNS):

    errors = (y_test[:, i] - y_pred[:, i])

    plt.figure(figsize=(7, 5))

    plt.hist(errors, bins=30, alpha=0.8)

    plt.xlabel(f"Prediction Error for {output_name}")

    plt.ylabel("Frequency")

    plt.title(f"Physics-Guided NN Error Distribution - {output_name}")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(PLOTS_DIR / ("physics_guided_NN_error_distribution_" f"{output_name}.png"), dpi=300)

    plt.close()