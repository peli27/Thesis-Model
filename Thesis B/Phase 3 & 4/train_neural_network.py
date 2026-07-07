"""
This script trains a feedforward neural network to predict beam dynamic response
quantities from input beam parameters.

Inputs:
    L, b, h, E, rho, damping_ratio

Outputs:
    f1, f2, f3, f4, peak_displacement, rms_displacement
"""

import time
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----- Configuration -----

SEED = 42

TRAIN_DATA_PATH = Path("dataset/train_dataset.csv")
VALIDATION_DATA_PATH = Path("dataset/validation_dataset.csv")
TEST_DATA_PATH = Path("dataset/test_dataset.csv")

RESULTS_DIR = Path("results")
MODELS_DIR = Path("Phase 3 & 4/models")
PLOTS_DIR = Path("figures/phase_4_plots")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

INPUT_COLUMNS = [
    "L",
    "b",
    "h",
    "E",
    "rho",
    "damping_ratio",
]

OUTPUT_COLUMNS = [
    "f1",
    "f2",
    "f3",
    "f4",
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)

#----- Neural Network Model -----

class BeamResponseNN(nn.Module):
    """
    Feedforward artificial neural network for multi-output regression.

    Architecture:
        input layer -> 64 neurons -> ReLU
                    -> 64 neurons -> ReLU
                    -> 32 neurons -> ReLU
                    -> output layer
    """

    def __init__(self, input_size, output_size):
        super(BeamResponseNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)

# ---- Load and Preprocess Data -----

for path in [TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "Check that your dataset folder and filenames are correct."
        )

train_df = pd.read_csv(TRAIN_DATA_PATH)
validation_df = pd.read_csv(VALIDATION_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

print("\nDatasets loaded successfully.")
print(f"Training dataset shape:   {train_df.shape}")
print(f"Validation dataset shape: {validation_df.shape}")
print(f"Test dataset shape:       {test_df.shape}")


def check_required_columns(df, dataset_name):
    missing_inputs = [col for col in INPUT_COLUMNS if col not in df.columns]
    missing_outputs = [col for col in OUTPUT_COLUMNS if col not in df.columns]

    if missing_inputs or missing_outputs:
        raise ValueError(
            f"Missing columns detected in {dataset_name}.\n"
            f"Missing input columns: {missing_inputs}\n"
            f"Missing output columns: {missing_outputs}"
        )

check_required_columns(train_df, "training dataset")
check_required_columns(validation_df, "validation dataset")
check_required_columns(test_df, "test dataset")


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

joblib.dump(input_scaler, MODELS_DIR / "input_scaler.pkl")
joblib.dump(output_scaler, MODELS_DIR / "output_scaler.pkl")

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
    shuffle=True
)

# ----- Initialise model, loss function and optimiser -----

input_size = len(INPUT_COLUMNS)
output_size = len(OUTPUT_COLUMNS)

model = BeamResponseNN(input_size=input_size, output_size=output_size).to(DEVICE)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam( # Changing the neural network's weights
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nNeural network initialised.")
print(model)


# ----- Training (with early stopping) -----

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
        val_loss = loss_function(
            val_predictions,
            y_val_tensor.to(DEVICE)
        ).item()

    training_losses.append(mean_training_loss)
    validation_losses.append(val_loss)

    # Early stopping check
    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        best_model_state = model.state_dict()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Training Loss: {mean_training_loss:.6f} "
            f"Validation Loss: {val_loss:.6f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
        break

end_training_time = time.perf_counter()
training_time_seconds = end_training_time - start_training_time

# Restore best model
model.load_state_dict(best_model_state)

print("\nTraining complete.")
print(f"Best validation loss: {best_validation_loss:.6f}")
print(f"Training time: {training_time_seconds:.4f} seconds")


# ----- Save trained model -----

model_save_path = MODELS_DIR / "neural_network_model.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_columns": INPUT_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "input_size": input_size,
        "output_size": output_size,
        "architecture": "6-64-64-32-6 feedforward neural network",
    },
    model_save_path
)

print(f"\nTrained neural network saved to: {model_save_path}")


# ----- Save training history -----

history_df = pd.DataFrame({
    "epoch": np.arange(1, len(training_losses) + 1),
    "training_loss": training_losses,
    "validation_loss": validation_losses,
})

history_df.to_csv(
    RESULTS_DIR / "NN_training_history.csv",
    index=False
)

print("Training history saved.")


# ----- Plot training and validation loss curves -----

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["training_loss"], label="Training Loss")
plt.plot(history_df["epoch"], history_df["validation_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Neural Network Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "NN_loss_curve.png", dpi=300)
plt.close()

print("Loss curve saved.")


# ----- Test set prediction -----

model.eval()

start_prediction_time = time.perf_counter()

with torch.no_grad():
    y_pred_scaled = model(X_test_tensor.to(DEVICE)).cpu().numpy()

end_prediction_time = time.perf_counter()
prediction_time_seconds = end_prediction_time - start_prediction_time

# Convert predictions back to real physical units
y_pred = output_scaler.inverse_transform(y_pred_scaled)

print(f"Prediction time: {prediction_time_seconds:.6f} seconds")


# ------ Save predictions -----

predictions_df = pd.DataFrame()

for i, output_name in enumerate(OUTPUT_COLUMNS):
    predictions_df[f"actual_{output_name}"] = y_test[:, i]
    predictions_df[f"predicted_{output_name}"] = y_pred[:, i]
    predictions_df[f"error_{output_name}"] = y_test[:, i] - y_pred[:, i]

predictions_df.to_csv(
    RESULTS_DIR / "NN_predictions.csv",
    index=False
)

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
        "model": "Neural_Network",
        "output": output_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "training_time_seconds": training_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
    })

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    RESULTS_DIR / "NN_model_metrics.csv",
    index=False
)

print("\nNeural network test metrics:")
print(metrics_df)


# ----- Predicted vs actual plots -----

for i, output_name in enumerate(OUTPUT_COLUMNS):
    actual = y_test[:, i]
    predicted = y_pred[:, i]

    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.6)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel(f"Actual {output_name}")
    plt.ylabel(f"Predicted {output_name}")
    plt.title(f"Predicted vs Actual - {output_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"NN_predicted_vs_actual_{output_name}.png",
        dpi=300
    )
    plt.close()

print("Predicted vs actual plots saved.")


# ----- Error distribution plots -----

for i, output_name in enumerate(OUTPUT_COLUMNS):
    errors = y_test[:, i] - y_pred[:, i]

    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=30, alpha=0.8)
    plt.xlabel(f"Prediction Error for {output_name}")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution - {output_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"NN_error_distribution_{output_name}.png",
        dpi=300
    )
    plt.close()

print("Error distribution plots saved.")


# ----- Compare against baseline ML models -----

baseline_metrics_path = RESULTS_DIR / "baseline_model_metrics.csv"

if baseline_metrics_path.exists():
    baseline_metrics_df = pd.read_csv(baseline_metrics_path)

    comparison_df = pd.concat(
        [baseline_metrics_df, metrics_df],
        ignore_index=True
    )

    comparison_df.to_csv(
        RESULTS_DIR / "comparison_baseline_vs_NN.csv",
        index=False
    )

    print("\nComparison table saved to results/comparison_baseline_vs_NN.csv")
else:
    print(
        "\nNo Phase 3 baseline metrics file found at results/baseline_model_metrics.csv."
    )
    print(
        "Neural network metrics were still saved to results/NN_model_metrics.csv."
    )