"""
Phase 4.5: Testing displacement target transformation.

Compares original displacement targets to log-transformed displacement targets
"""

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
PLOTS_DIR = Path("figures/displacement_transformation")

RESULTS_DIR.mkdir(exist_ok=True)
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
]

OUTPUT_COLUMNS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "peak_displacement",
    "rms_displacement",
]

DISPLACEMENT_COLUMNS = [
    "peak_displacement",
    "rms_displacement",
]

HIDDEN_LAYERS = [64, 64, 32]

LEARNING_RATE = 0.001
BATCH_SIZE = 64

EPOCHS = 500
PATIENCE = 50

DEVICE = torch.device("cpu")


# ----- Reproducibility -----

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----- Neural network -----

class BeamResponseNN(nn.Module):

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


# ----- Load data -----

train_df = pd.read_csv(TRAIN_DATA_PATH)
validation_df = pd.read_csv(VALIDATION_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

print("\nDatasets loaded successfully.")


X_train = train_df[INPUT_COLUMNS].values
X_val = validation_df[INPUT_COLUMNS].values
X_test = test_df[INPUT_COLUMNS].values

y_train_original = train_df[OUTPUT_COLUMNS].values
y_val_original = validation_df[OUTPUT_COLUMNS].values
y_test = test_df[OUTPUT_COLUMNS].values


# ----- Scale inputs -----

input_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled = input_scaler.transform(X_val)
X_test_scaled = input_scaler.transform(X_test)


X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
)

X_val_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32,
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
)


# ----- Target transformations -----

def transform_targets(y, use_log_transform):
    """
    Applies log1p transformation to displacement outputs only.
    """

    transformed = y.copy()

    if use_log_transform:

        for column in DISPLACEMENT_COLUMNS:

            index = OUTPUT_COLUMNS.index(column)

            transformed[:, index] = np.log1p(
                transformed[:, index]
            )

    return transformed


def inverse_transform_targets(
    y,
    use_log_transform,
):
    """
    Converts displacement outputs back to physical units.
    """

    restored = y.copy()

    if use_log_transform:

        for column in DISPLACEMENT_COLUMNS:

            index = OUTPUT_COLUMNS.index(column)

            restored[:, index] = np.expm1(
                restored[:, index]
            )

    return restored


# ----- Train one model -----

def train_model(
    model_name,
    use_log_transform,
):

    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)

    set_seed(SEED)

    # Transform targets
    y_train = transform_targets(
        y_train_original,
        use_log_transform,
    )

    y_val = transform_targets(
        y_val_original,
        use_log_transform,
    )

    # Scale outputs
    output_scaler = StandardScaler()

    y_train_scaled = output_scaler.fit_transform(
        y_train
    )

    y_val_scaled = output_scaler.transform(
        y_val
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val_scaled,
        dtype=torch.float32,
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    # Initialise model
    model = BeamResponseNN(
        input_size=len(INPUT_COLUMNS),
        output_size=len(OUTPUT_COLUMNS),
        hidden_layers=HIDDEN_LAYERS,
    ).to(DEVICE)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = np.inf
    best_model_state = None

    epochs_without_improvement = 0
    completed_epochs = 0

    start_training_time = time.perf_counter()

    # ----- Training loop -----

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

            val_predictions = model(
                X_val_tensor.to(DEVICE)
            )

            val_loss = loss_function(
                val_predictions,
                y_val_tensor.to(DEVICE),
            ).item()

        completed_epochs = epoch + 1

        # Early stopping
        if val_loss < best_validation_loss:

            best_validation_loss = val_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (epoch + 1) % 25 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Training Loss: "
                f"{np.mean(batch_losses):.6f} "
                f"Validation Loss: "
                f"{val_loss:.6f}"
            )

        if epochs_without_improvement >= PATIENCE:

            print(
                f"Early stopping at epoch "
                f"{epoch + 1}."
            )

            break


    training_time = (
        time.perf_counter()
        - start_training_time
    )

    model.load_state_dict(best_model_state)


    # ----- Test predictions -----

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

    prediction_time = (
        time.perf_counter()
        - start_prediction_time
    )

    # Remove standard scaling
    y_pred_transformed = (
        output_scaler.inverse_transform(
            y_pred_scaled
        )
    )

    # Remove log transformation
    y_pred = inverse_transform_targets(
        y_pred_transformed,
        use_log_transform,
    )


    # ----- Metrics -----

    metrics = []

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

        metrics.append({
            "model": model_name,
            "output": output_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "best_validation_loss": (
                best_validation_loss
            ),
            "epochs": completed_epochs,
            "training_time_seconds": (
                training_time
            ),
            "prediction_time_seconds": (
                prediction_time
            ),
        })


    metrics_df = pd.DataFrame(metrics)


    # ----- Plots -----

    for output_name in DISPLACEMENT_COLUMNS:

        index = OUTPUT_COLUMNS.index(
            output_name
        )

        actual = y_test[:, index]
        predicted = y_pred[:, index]

        minimum = min(
            actual.min(),
            predicted.min(),
        )

        maximum = max(
            actual.max(),
            predicted.max(),
        )

        # Predicted vs actual
        plt.figure(figsize=(6, 6))

        plt.scatter(
            actual,
            predicted,
            alpha=0.6,
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )

        plt.xlabel(
            f"Actual {output_name}"
        )

        plt.ylabel(
            f"Predicted {output_name}"
        )

        plt.title(
            f"{model_name}: Predicted vs Actual"
        )

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR
            / (
                f"{model_name}_"
                f"{output_name}_predicted_vs_actual.png"
            ),
            dpi=300,
        )

        plt.close()


        # Error distribution
        errors = actual - predicted

        plt.figure(figsize=(7, 5))

        plt.hist(
            errors,
            bins=30,
        )

        plt.xlabel(
            f"Prediction Error for {output_name}"
        )

        plt.ylabel("Frequency")

        plt.title(
            f"{model_name}: Error Distribution"
        )

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR
            / (
                f"{model_name}_"
                f"{output_name}_error_distribution.png"
            ),
            dpi=300,
        )

        plt.close()


    return metrics_df


# ----- Run both models -----

original_metrics = train_model(
    model_name="Original_Targets",
    use_log_transform=False,
)

log_metrics = train_model(
    model_name="Log_Transformed_Targets",
    use_log_transform=True,
)


# ----- Combine results -----

comparison_df = pd.concat(
    [
        original_metrics,
        log_metrics,
    ],
    ignore_index=True,
)

comparison_df.to_csv(
    RESULTS_DIR
    / "NN_target_transformation_comparison.csv",
    index=False,
)


# ----- Displacement comparison -----

displacement_comparison = comparison_df[
    comparison_df["output"].isin(
        DISPLACEMENT_COLUMNS
    )
]

print("\n" + "=" * 60)
print("TARGET TRANSFORMATION COMPARISON")
print("=" * 60)

print(
    displacement_comparison[
        [
            "model",
            "output",
            "rmse",
            "mae",
            "r2",
            "training_time_seconds",
            "prediction_time_seconds",
        ]
    ].to_string(index=False)
)


# ----- Skewness comparison -----

skewness_results = []

for output_name in DISPLACEMENT_COLUMNS:

    index = OUTPUT_COLUMNS.index(
        output_name
    )

    original_values = (
        y_train_original[:, index]
    )

    transformed_values = np.log1p(
        original_values
    )

    skewness_results.append({
        "output": output_name,
        "original_skewness": (
            pd.Series(
                original_values
            ).skew()
        ),
        "transformed_skewness": (
            pd.Series(
                transformed_values
            ).skew()
        ),
    })


skewness_df = pd.DataFrame(
    skewness_results
)

skewness_df.to_csv(
    RESULTS_DIR
    / "displacement_transformation_skewness.csv",
    index=False,
)

print("\nSkewness comparison:")
print(skewness_df.to_string(index=False))

print("\nTarget transformation analysis complete.")