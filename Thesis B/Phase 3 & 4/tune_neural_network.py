"""
Phase 4.4: Controlled neural network hyperparameter tuning.

The script trains multiple neural network configurations using the
same dataset splits, scaling procedure and random seed.

Inputs:
    L, b, h, E, rho, damping_ratio,
    force_amplitude, excitation_frequency

Outputs:
    f1, f2, f3, f4,
    peak_displacement, rms_displacement
"""

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


# ----- Configuration -----

SEED = 42

TRAIN_DATA_PATH = Path("dataset/train_dataset.csv")
VALIDATION_DATA_PATH = Path("dataset/validation_dataset.csv")
TEST_DATA_PATH = Path("dataset/test_dataset.csv")

RESULTS_DIR = Path("results")
MODELS_DIR = Path("Phase 3 & 4/models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

INPUT_COLUMNS = [
    "L",
    "b",
    "h",
    "E",
    "rho",
    "damping_ratio",
    "force_amplitude",
    "excitation_frequency",
]

OUTPUT_COLUMNS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "peak_displacement",
    "rms_displacement",
]

EPOCHS = 500
PATIENCE = 50

DEVICE = torch.device("cpu")


# ----- Neural network configurations -----

CONFIGURATIONS = [
    {
        "name": "NN1_baseline",
        "hidden_layers": [64, 64, 32],
        "learning_rate": 0.001,
        "batch_size": 64,
    },
    {
        "name": "NN2_smaller",
        "hidden_layers": [32, 32],
        "learning_rate": 0.001,
        "batch_size": 64,
    },
    {
        "name": "NN3_wider",
        "hidden_layers": [128, 64, 32],
        "learning_rate": 0.001,
        "batch_size": 64,
    },
    {
        "name": "NN4_larger",
        "hidden_layers": [128, 128, 64],
        "learning_rate": 0.001,
        "batch_size": 64,
    },
    {
        "name": "NN5_low_learning_rate",
        "hidden_layers": [64, 64, 32],
        "learning_rate": 0.0001,
        "batch_size": 64,
    },
    {
        "name": "NN6_high_learning_rate",
        "hidden_layers": [64, 64, 32],
        "learning_rate": 0.01,
        "batch_size": 64,
    },
    {
        "name": "NN7_small_batch",
        "hidden_layers": [64, 64, 32],
        "learning_rate": 0.001,
        "batch_size": 32,
    },
    {
        "name": "NN8_large_batch",
        "hidden_layers": [64, 64, 32],
        "learning_rate": 0.001,
        "batch_size": 128,
    },
]


# ----- Reproducibility -----

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----- Neural network model -----

class BeamResponseNN(nn.Module):
    """
    Configurable feedforward neural network.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers,
    ):
        super().__init__()

        layers = []

        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(
                nn.Linear(
                    previous_size,
                    hidden_size,
                )
            )

            layers.append(nn.ReLU())

            previous_size = hidden_size

        layers.append(
            nn.Linear(
                previous_size,
                output_size,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ----- Load datasets -----

for path in [
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )


train_df = pd.read_csv(TRAIN_DATA_PATH)
validation_df = pd.read_csv(VALIDATION_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

print("\nDatasets loaded successfully.")
print(f"Training dataset shape:   {train_df.shape}")
print(f"Validation dataset shape: {validation_df.shape}")
print(f"Test dataset shape:       {test_df.shape}")


# ----- Extract inputs and outputs -----

X_train = train_df[INPUT_COLUMNS].values
y_train = train_df[OUTPUT_COLUMNS].values

X_val = validation_df[INPUT_COLUMNS].values
y_val = validation_df[OUTPUT_COLUMNS].values

X_test = test_df[INPUT_COLUMNS].values
y_test = test_df[OUTPUT_COLUMNS].values


# ----- Scale data -----

input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled = input_scaler.transform(X_val)
X_test_scaled = input_scaler.transform(X_test)

y_train_scaled = output_scaler.fit_transform(y_train)
y_val_scaled = output_scaler.transform(y_val)


# ----- Convert to tensors -----

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)


# ----- Train and evaluate one configuration -----

def train_and_evaluate(config):

    print("\n" + "=" * 60)
    print(f"Training {config['name']}")
    print(
        f"Architecture: "
        f"{config['hidden_layers']}"
    )
    print(
        f"Learning rate: "
        f"{config['learning_rate']}"
    )
    print(
        f"Batch size: "
        f"{config['batch_size']}"
    )
    print("=" * 60)

    # Reset seed so each configuration is reproducible
    set_seed(SEED)

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
    )

    model = BeamResponseNN(
        input_size=len(INPUT_COLUMNS),
        output_size=len(OUTPUT_COLUMNS),
        hidden_layers=config["hidden_layers"],
    ).to(DEVICE)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    best_validation_loss = np.inf
    best_model_state = None
    epochs_without_improvement = 0
    completed_epochs = 0

    start_training_time = time.perf_counter()

    for epoch in range(EPOCHS):

        model.train()

        batch_losses = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            predictions = model(X_batch)

            loss = loss_function(
                predictions,
                y_batch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        # Validation
        model.eval()

        with torch.no_grad():

            validation_predictions = model(
                X_val_tensor.to(DEVICE)
            )

            validation_loss = loss_function(
                validation_predictions,
                y_val_tensor.to(DEVICE),
            ).item()

        completed_epochs = epoch + 1

        # Early stopping
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (epoch + 1) % 25 == 0:

            mean_training_loss = np.mean(
                batch_losses
            )

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Training Loss: "
                f"{mean_training_loss:.6f} "
                f"Validation Loss: "
                f"{validation_loss:.6f}"
            )

        if epochs_without_improvement >= PATIENCE:

            print(
                f"Early stopping at "
                f"epoch {epoch + 1}."
            )

            break

    training_time_seconds = (
        time.perf_counter()
        - start_training_time
    )

    model.load_state_dict(best_model_state)

    # Test predictions
    model.eval()

    start_prediction_time = time.perf_counter()

    with torch.no_grad():

        y_pred_scaled = (
            model(
                X_test_tensor.to(DEVICE)
            )
            .cpu()
            .numpy()
        )

    prediction_time_seconds = (
        time.perf_counter()
        - start_prediction_time
    )

    y_pred = output_scaler.inverse_transform(
        y_pred_scaled
    )

    # Calculate metrics
    output_metrics = []

    for i, output_name in enumerate(
        OUTPUT_COLUMNS
    ):

        actual = y_test[:, i]
        predicted = y_pred[:, i]

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )

        mae = mean_absolute_error(
            actual,
            predicted,
        )

        r2 = r2_score(
            actual,
            predicted,
        )

        output_metrics.append({
            "model": config["name"],
            "architecture": "-".join(
                map(
                    str,
                    config["hidden_layers"],
                )
            ),
            "learning_rate": (
                config["learning_rate"]
            ),
            "batch_size": (
                config["batch_size"]
            ),
            "output": output_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "best_validation_loss": (
                best_validation_loss
            ),
            "epochs": completed_epochs,
            "training_time_seconds": (
                training_time_seconds
            ),
            "prediction_time_seconds": (
                prediction_time_seconds
            ),
        })

    metrics_df = pd.DataFrame(
        output_metrics
    )

    peak_r2 = metrics_df.loc[
        metrics_df["output"]
        == "peak_displacement",
        "r2",
    ].iloc[0]

    rms_r2 = metrics_df.loc[
        metrics_df["output"]
        == "rms_displacement",
        "r2",
    ].iloc[0]

    frequency_r2 = metrics_df[
        metrics_df["output"].isin(
            ["f1", "f2", "f3", "f4"]
        )
    ]["r2"].mean()

    displacement_r2 = np.mean([
        peak_r2,
        rms_r2,
    ])

    summary = {
        "model": config["name"],
        "architecture": "-".join(
            map(
                str,
                config["hidden_layers"],
            )
        ),
        "learning_rate": (
            config["learning_rate"]
        ),
        "batch_size": (
            config["batch_size"]
        ),
        "average_frequency_r2": (
            frequency_r2
        ),
        "peak_displacement_r2": peak_r2,
        "rms_displacement_r2": rms_r2,
        "average_displacement_r2": (
            displacement_r2
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "epochs": completed_epochs,
        "training_time_seconds": (
            training_time_seconds
        ),
        "prediction_time_seconds": (
            prediction_time_seconds
        ),
    }

    print("\nConfiguration complete.")
    print(
        f"Peak displacement R2: "
        f"{peak_r2:.6f}"
    )
    print(
        f"RMS displacement R2: "
        f"{rms_r2:.6f}"
    )
    print(
        f"Average displacement R2: "
        f"{displacement_r2:.6f}"
    )
    print(
        f"Training time: "
        f"{training_time_seconds:.4f} s"
    )

    return (
        model,
        metrics_df,
        summary,
    )


# ----- Run all configurations -----

all_metrics = []
all_summaries = []
trained_models = {}

for config in CONFIGURATIONS:

    model, metrics_df, summary = (
        train_and_evaluate(config)
    )

    all_metrics.append(metrics_df)
    all_summaries.append(summary)

    trained_models[
        config["name"]
    ] = copy.deepcopy(
        model.state_dict()
    )


# ----- Save full metrics -----

all_metrics_df = pd.concat(
    all_metrics,
    ignore_index=True,
)

all_metrics_df.to_csv(
    RESULTS_DIR
    / "NN_hyperparameter_metrics.csv",
    index=False,
)


# ----- Save summary table -----

summary_df = pd.DataFrame(
    all_summaries
)

summary_df = summary_df.sort_values(
    "average_displacement_r2",
    ascending=False,
)

summary_df.to_csv(
    RESULTS_DIR
    / "NN_hyperparameter_summary.csv",
    index=False,
)

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 60)

print(
    summary_df[
        [
            "model",
            "architecture",
            "learning_rate",
            "batch_size",
            "peak_displacement_r2",
            "rms_displacement_r2",
            "average_displacement_r2",
            "training_time_seconds",
            "prediction_time_seconds",
        ]
    ].to_string(index=False)
)


# ----- Select best configuration -----

best_result = summary_df.iloc[0]

best_model_name = best_result["model"]

best_config = next(
    config
    for config in CONFIGURATIONS
    if config["name"] == best_model_name
)

best_model_path = (
    MODELS_DIR
    / "best_tuned_neural_network.pth"
)

torch.save(
    {
        "model_state_dict": (
            trained_models[best_model_name]
        ),
        "model_name": best_model_name,
        "input_columns": INPUT_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "hidden_layers": (
            best_config["hidden_layers"]
        ),
        "learning_rate": (
            best_config["learning_rate"]
        ),
        "batch_size": (
            best_config["batch_size"]
        ),
        "selection_metric": (
            "average_displacement_r2"
        ),
        "average_displacement_r2": (
            best_result[
                "average_displacement_r2"
            ]
        ),
    },
    best_model_path,
)


print("\nBest configuration:")
print(
    f"Model: {best_model_name}"
)
print(
    f"Architecture: "
    f"{best_config['hidden_layers']}"
)
print(
    f"Learning rate: "
    f"{best_config['learning_rate']}"
)
print(
    f"Batch size: "
    f"{best_config['batch_size']}"
)
print(
    f"Average displacement R2: "
    f"{best_result['average_displacement_r2']:.6f}"
)

print(
    f"\nBest model saved to: "
    f"{best_model_path}"
)

print(
    "\nHyperparameter tuning complete."
)