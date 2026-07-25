"""
Experiment 1: Random parametric error and generalisation study.

This script trains two neural networks:

1. Standard neural network.
2. Physics-guided neural network with five physics-guided features.

Both models are trained once using known perturbations up to +/-2.5%. They are
then evaluated on known and hidden perturbation test datasets at 0%, +/-1%,
+/-2.5%, +/-5% and +/-10% uncertainty.

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ----- Project directories -----

PHASE_6_DIR = Path(__file__).resolve().parent
DATASET_DIR = PHASE_6_DIR / "Phase_6_datasets"

RESULTS_DIR = PHASE_6_DIR / "Experiment_1_results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"


# ----- Experiment settings -----

SEED = 42

BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.001
PATIENCE = 50

DEVICE = torch.device("cpu")

TRAIN_UNCERTAINTY = 0.025
TEST_UNCERTAINTY_LEVELS = [0.0, 0.01, 0.025, 0.05, 0.10]
PERTURBATION_TYPES = ["known", "hidden"]

STANDARD_INPUT_COLUMNS = [
    "L",
    "b",
    "h",
    "E",
    "rho",
    "damping_ratio",
    "force_amplitude",
    "excitation_frequency",
]

PHYSICS_FEATURE_COLUMNS = [
    "first_natural_frequency",
    "modal_stiffness",
    "modal_mass",
    "frequency_ratio",
    "resonance_proximity",
]

HYBRID_INPUT_COLUMNS = STANDARD_INPUT_COLUMNS + PHYSICS_FEATURE_COLUMNS

OUTPUT_COLUMNS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "peak_displacement",
    "rms_displacement",
]


# ----- Reproducibility -----

def set_seed(seed):
    """Makes the neural-network results repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----- Neural-network model -----

class BeamResponseNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

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


# ----- Dataset helpers -----

def level_label(level):
    """Converts 0.025 into the filename label 2p5pct."""

    percentage = f"{level * 100:g}".replace(".", "p")
    return f"{percentage}pct"


def load_dataset(path, input_columns):
    """Loads one CSV and checks that the required columns are available."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run generate_uncertainty_datasets.py before Experiment 1."
        )

    dataframe = pd.read_csv(path)
    required_columns = input_columns + OUTPUT_COLUMNS
    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing these columns: {missing_columns}"
        )

    return dataframe


def check_experiment_files(model_settings):
    """Checks that every required uncertainty dataset exists before training."""

    required_paths = []

    for settings in model_settings:
        required_paths.append(settings["train_path"])
        required_paths.append(settings["validation_path"])

        for perturbation_type in PERTURBATION_TYPES:
            for uncertainty_level in TEST_UNCERTAINTY_LEVELS:
                test_level = level_label(uncertainty_level)
                test_filename = (
                    f"test_{perturbation_type}_{test_level}"
                    f"{settings['hybrid_suffix']}.csv"
                )
                required_paths.append(DATASET_DIR / test_filename)

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_list = "\n".join(f"  - {path.name}" for path in missing_paths)
        raise FileNotFoundError(
            "Experiment 1 cannot start because these datasets are missing:\n"
            f"{missing_list}\n"
            "Run generate_uncertainty_datasets.py first."
        )


def create_tensors(dataframe, input_columns, input_scaler, output_scaler):
    """Selects model columns, scales them and converts them to tensors."""

    inputs = dataframe[input_columns].to_numpy(dtype=float)
    outputs = dataframe[OUTPUT_COLUMNS].to_numpy(dtype=float)

    scaled_inputs = input_scaler.transform(inputs)
    scaled_outputs = output_scaler.transform(outputs)

    input_tensor = torch.tensor(scaled_inputs, dtype=torch.float32)
    output_tensor = torch.tensor(scaled_outputs, dtype=torch.float32)

    return input_tensor, output_tensor


# ----- Model training -----

def train_model(model_name, input_columns, train_path, validation_path):
    """Trains one model and returns its model, scalers and training summary."""

    print(f"\nTraining {model_name}...")

    set_seed(SEED)

    train_dataframe = load_dataset(train_path, input_columns)
    validation_dataframe = load_dataset(validation_path, input_columns)

    print(f"Training samples:   {len(train_dataframe)}")
    print(f"Validation samples: {len(validation_dataframe)}")

    # Scalers are fitted only to the training data.
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    input_scaler.fit(train_dataframe[input_columns])
    output_scaler.fit(train_dataframe[OUTPUT_COLUMNS])

    X_train, y_train = create_tensors(
        train_dataframe,
        input_columns,
        input_scaler,
        output_scaler,
    )

    X_validation, y_validation = create_tensors(
        validation_dataframe,
        input_columns,
        input_scaler,
        output_scaler,
    )

    training_dataset = TensorDataset(X_train, y_train)

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(SEED)

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=data_loader_generator,
    )

    model = BeamResponseNN(
        input_size=len(input_columns),
        output_size=len(OUTPUT_COLUMNS),
    ).to(DEVICE)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    training_losses = []
    validation_losses = []

    best_validation_loss = np.inf
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []

        for X_batch, y_batch in training_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        mean_training_loss = float(np.mean(batch_losses))

        model.eval()

        with torch.no_grad():
            validation_predictions = model(X_validation.to(DEVICE))
            validation_loss = loss_function(
                validation_predictions,
                y_validation.to(DEVICE),
            ).item()

        training_losses.append(mean_training_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"training loss: {mean_training_loss:.6f}, "
                f"validation loss: {validation_loss:.6f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    training_time = time.perf_counter() - start_time

    if best_model_state is None:
        raise RuntimeError(f"Training failed for {model_name}.")

    model.load_state_dict(best_model_state)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Training time: {training_time:.2f} seconds")

    history_dataframe = pd.DataFrame(
        {
            "epoch": np.arange(1, len(training_losses) + 1),
            "training_loss": training_losses,
            "validation_loss": validation_losses,
        }
    )

    history_dataframe.to_csv(
        RESULTS_DIR / f"{model_name}_training_history.csv",
        index=False,
    )

    save_loss_plot(history_dataframe, model_name)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_columns": input_columns,
            "output_columns": OUTPUT_COLUMNS,
            "input_size": len(input_columns),
            "output_size": len(OUTPUT_COLUMNS),
            "architecture": (
                f"{len(input_columns)}-64-64-32-{len(OUTPUT_COLUMNS)}"
            ),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        MODELS_DIR / f"{model_name}.pth",
    )

    joblib.dump(
        input_scaler,
        MODELS_DIR / f"{model_name}_input_scaler.pkl",
    )
    joblib.dump(
        output_scaler,
        MODELS_DIR / f"{model_name}_output_scaler.pkl",
    )

    training_summary = {
        "model": model_name,
        "number_of_inputs": len(input_columns),
        "training_samples": len(train_dataframe),
        "validation_samples": len(validation_dataframe),
        "epochs_completed": len(training_losses),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_time_seconds": training_time,
    }

    return model, input_scaler, output_scaler, training_summary


# ----- Model evaluation -----

def evaluate_model(
    model,
    model_name,
    input_columns,
    input_scaler,
    output_scaler,
    test_path,
    perturbation_type,
    uncertainty_level,
):
    """Evaluates one trained model on one uncertainty test dataset."""

    test_dataframe = load_dataset(test_path, input_columns)

    X_test = test_dataframe[input_columns].to_numpy(dtype=float)
    y_test = test_dataframe[OUTPUT_COLUMNS].to_numpy(dtype=float)

    X_test_scaled = input_scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    model.eval()
    start_time = time.perf_counter()

    with torch.no_grad():
        predictions_scaled = model(X_test_tensor.to(DEVICE)).cpu().numpy()

    prediction_time = time.perf_counter() - start_time
    predictions = output_scaler.inverse_transform(predictions_scaled)

    metric_records = []

    for output_index, output_name in enumerate(OUTPUT_COLUMNS):
        actual = y_test[:, output_index]
        predicted = predictions[:, output_index]

        metric_records.append(
            {
                "model": model_name,
                "perturbation_type": perturbation_type,
                "uncertainty_level": uncertainty_level,
                "uncertainty_percent": uncertainty_level * 100,
                "output": output_name,
                "rmse": np.sqrt(mean_squared_error(actual, predicted)),
                "mae": mean_absolute_error(actual, predicted),
                "r2": r2_score(actual, predicted),
                "number_of_test_samples": len(test_dataframe),
                "prediction_time_seconds": prediction_time,
            }
        )

    prediction_dataframe = pd.DataFrame(
        {
            "model": model_name,
            "perturbation_type": perturbation_type,
            "uncertainty_level": uncertainty_level,
            "uncertainty_percent": uncertainty_level * 100,
            "sample_id": (
                test_dataframe["sample_id"].to_numpy()
                if "sample_id" in test_dataframe.columns
                else np.arange(len(test_dataframe))
            ),
        }
    )

    for output_index, output_name in enumerate(OUTPUT_COLUMNS):
        prediction_dataframe[f"actual_{output_name}"] = y_test[:, output_index]
        prediction_dataframe[f"predicted_{output_name}"] = (
            predictions[:, output_index]
        )
        prediction_dataframe[f"error_{output_name}"] = (
            y_test[:, output_index] - predictions[:, output_index]
        )

    print(
        f"Evaluated {model_name}: {perturbation_type}, "
        f"+/-{uncertainty_level * 100:g}%"
    )

    return metric_records, prediction_dataframe


# ----- Plotting -----

def save_loss_plot(history_dataframe, model_name):
    """Saves the training and validation loss curve for one model."""

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["training_loss"],
        label="Training loss",
    )
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["validation_loss"],
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Scaled MSE loss")
    plt.title(f"Training history - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"{model_name}_loss_curve.png",
        dpi=300,
    )
    plt.close()


def save_uncertainty_plots(metrics_dataframe):
    """Plots RMSE and R2 against uncertainty for each predicted output."""

    colours = {
        ("standard_NN", "known"): "tab:blue",
        ("standard_NN", "hidden"): "tab:cyan",
        ("physics_guided_NN", "known"): "tab:orange",
        ("physics_guided_NN", "hidden"): "tab:red",
    }

    line_styles = {
        "known": "-",
        "hidden": "--",
    }

    for output_name in OUTPUT_COLUMNS:
        output_metrics = metrics_dataframe[
            metrics_dataframe["output"] == output_name
        ]

        for metric_name, axis_label in [
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("r2", "RÂ²"),
        ]:
            plt.figure(figsize=(8, 5))

            for model_name in ["standard_NN", "physics_guided_NN"]:
                for perturbation_type in PERTURBATION_TYPES:
                    condition = output_metrics[
                        (output_metrics["model"] == model_name)
                        & (
                            output_metrics["perturbation_type"]
                            == perturbation_type
                        )
                    ].sort_values("uncertainty_percent")

                    label = (
                        f"{model_name.replace('_', ' ').title()} - "
                        f"{perturbation_type.title()}"
                    )

                    plt.plot(
                        condition["uncertainty_percent"],
                        condition[metric_name],
                        marker="o",
                        linestyle=line_styles[perturbation_type],
                        color=colours[(model_name, perturbation_type)],
                        label=label,
                    )

            plt.xlabel("Maximum uncertainty (%)")
            plt.ylabel(axis_label)
            plt.title(f"{axis_label} against uncertainty - {output_name}")
            plt.legend(fontsize=8)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                PLOTS_DIR / f"experiment_1_{metric_name}_{output_name}.png",
                dpi=300,
            )
            plt.close()


# ----- Run Experiment 1 -----

if __name__ == "__main__":

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_level = level_label(TRAIN_UNCERTAINTY)

    model_settings = [
        {
            "name": "standard_NN",
            "input_columns": STANDARD_INPUT_COLUMNS,
            "train_path": DATASET_DIR / f"train_known_{train_level}.csv",
            "validation_path": (
                DATASET_DIR / f"validation_known_{train_level}.csv"
            ),
            "hybrid_suffix": "",
        },
        {
            "name": "physics_guided_NN",
            "input_columns": HYBRID_INPUT_COLUMNS,
            "train_path": (
                DATASET_DIR / f"train_known_{train_level}_hybrid.csv"
            ),
            "validation_path": (
                DATASET_DIR / f"validation_known_{train_level}_hybrid.csv"
            ),
            "hybrid_suffix": "_hybrid",
        },
    ]

    # Stop before training if any required dataset has not been generated.
    check_experiment_files(model_settings)

    all_metrics = []
    all_predictions = []
    training_summaries = []

    for settings in model_settings:
        model, input_scaler, output_scaler, training_summary = train_model(
            model_name=settings["name"],
            input_columns=settings["input_columns"],
            train_path=settings["train_path"],
            validation_path=settings["validation_path"],
        )

        training_summaries.append(training_summary)

        # The trained model is reused for all known and hidden test conditions.
        for perturbation_type in PERTURBATION_TYPES:
            for uncertainty_level in TEST_UNCERTAINTY_LEVELS:
                test_level = level_label(uncertainty_level)
                test_filename = (
                    f"test_{perturbation_type}_{test_level}"
                    f"{settings['hybrid_suffix']}.csv"
                )

                metric_records, prediction_dataframe = evaluate_model(
                    model=model,
                    model_name=settings["name"],
                    input_columns=settings["input_columns"],
                    input_scaler=input_scaler,
                    output_scaler=output_scaler,
                    test_path=DATASET_DIR / test_filename,
                    perturbation_type=perturbation_type,
                    uncertainty_level=uncertainty_level,
                )

                all_metrics.extend(metric_records)
                all_predictions.append(prediction_dataframe)

    metrics_dataframe = pd.DataFrame(all_metrics)
    predictions_dataframe = pd.concat(all_predictions, ignore_index=True)
    training_summary_dataframe = pd.DataFrame(training_summaries)

    metrics_dataframe.to_csv(
        RESULTS_DIR / "experiment_1_metrics.csv",
        index=False,
    )

    predictions_dataframe.to_csv(
        RESULTS_DIR / "experiment_1_predictions.csv",
        index=False,
    )

    training_summary_dataframe.to_csv(
        RESULTS_DIR / "experiment_1_training_summary.csv",
        index=False,
    )

    save_uncertainty_plots(metrics_dataframe)

    print("\nExperiment 1 completed successfully.")
    print(f"Results saved in: {RESULTS_DIR}")