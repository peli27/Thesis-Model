"""
Experiment 3: reduced-training-data study.

The standard and physics-guided neural networks are trained with progressively
smaller portions of the same original training pool. Every run is evaluated on
the same fixed validation and test datasets.

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
DATASET_DIR = PHASE_6_DIR / "Experiment_3_datasets"

RESULTS_DIR = PHASE_6_DIR / "Experiment_3_results"
RUNS_DIR = RESULTS_DIR / "runs"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"


# ----- Experiment settings -----

SEED = 42
NUMBER_OF_REPEATS = 3
TRAINING_PERCENTAGES = [5, 10, 25, 50, 75, 100]

BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.001
PATIENCE = 50

DEVICE = torch.device("cpu")

SKIP_COMPLETED_RUNS = True # Completed runs skipped when re-running the experiment.

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

FREQUENCY_OUTPUTS = ["f1", "f2", "f3", "f4"]
DISPLACEMENT_OUTPUTS = ["peak_displacement", "rms_displacement"]


# ----- Reproducibility -----

def set_seed(seed):
    """Makes one model run repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----- Neural-network model -----

class BeamResponseNN(nn.Module):
    """The feedforward architecture used in Phases 4, 5, and 6."""

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

def training_filename(training_percentage, repeat_number, hybrid=False):
    """Creates the filename used by the preparation script."""

    suffix = "_hybrid" if hybrid else ""

    return (
        f"train_{training_percentage}pct_repeat_{repeat_number}"
        f"{suffix}.csv"
    )


def load_dataset(path, input_columns):
    """Loads a CSV and verifies every required model column."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run prepare_experiment_3_datasets.py first."
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

    if dataframe[required_columns].isna().any().any():
        raise ValueError(f"{path.name} contains missing model values.")

    return dataframe


def check_experiment_files(model_settings):
    """Checks every required file before training begins."""

    required_paths = []

    for settings in model_settings:
        required_paths.extend(
            [
                settings["validation_path"],
                settings["test_path"],
            ]
        )

        for repeat_number in range(1, NUMBER_OF_REPEATS + 1):
            for training_percentage in TRAINING_PERCENTAGES:
                required_paths.append(
                    DATASET_DIR
                    / training_filename(
                        training_percentage,
                        repeat_number,
                        hybrid=settings["hybrid"],
                    )
                )

    missing_paths = [
        path for path in dict.fromkeys(required_paths)
        if not path.exists()
    ]

    if missing_paths:
        missing_list = "\n".join(f"  - {path.name}" for path in missing_paths)
        raise FileNotFoundError(
            "Experiment 3 cannot start because these datasets are missing:\n"
            f"{missing_list}\n"
            "Run prepare_experiment_3_datasets.py first."
        )


def create_tensors(dataframe, input_columns, input_scaler, output_scaler):
    """Scales selected columns and converts them to PyTorch tensors."""

    inputs = dataframe[input_columns].to_numpy(dtype=float)
    outputs = dataframe[OUTPUT_COLUMNS].to_numpy(dtype=float)

    scaled_inputs = input_scaler.transform(inputs)
    scaled_outputs = output_scaler.transform(outputs)

    return (
        torch.tensor(scaled_inputs, dtype=torch.float32),
        torch.tensor(scaled_outputs, dtype=torch.float32),
    )


# ----- Individual run paths and restart support -----

def run_label(model_name, training_percentage, repeat_number):
    """Creates a unique label for one trained model."""

    return (
        f"{model_name}_{training_percentage}pct_repeat_{repeat_number}"
    )


def run_result_paths(label):
    """Returns every output path required for one completed run."""

    run_directory = RUNS_DIR / label

    return {
        "directory": run_directory,
        "metrics": run_directory / "metrics.csv",
        "predictions": run_directory / "predictions.csv",
        "consistency": run_directory / "physical_consistency.csv",
        "history": run_directory / "training_history.csv",
        "training_summary": run_directory / "training_summary.csv",
        "model": MODELS_DIR / f"{label}.pth",
        "input_scaler": MODELS_DIR / f"{label}_input_scaler.pkl",
        "output_scaler": MODELS_DIR / f"{label}_output_scaler.pkl",
    }


def completed_run_exists(paths):
    """Checks whether all outputs from one run already exist."""

    required_keys = [
        "metrics",
        "predictions",
        "consistency",
        "history",
        "training_summary",
        "model",
        "input_scaler",
        "output_scaler",
    ]

    return all(paths[key].exists() for key in required_keys)


def load_completed_run(paths):
    """Loads tabular results from a previously completed run."""

    return {
        "metrics": pd.read_csv(paths["metrics"]),
        "predictions": pd.read_csv(paths["predictions"]),
        "consistency": pd.read_csv(paths["consistency"]),
        "history": pd.read_csv(paths["history"]),
        "training_summary": pd.read_csv(paths["training_summary"]),
    }


# ----- Training and evaluation -----

def train_and_evaluate(
    model_name,
    input_columns,
    train_path,
    validation_path,
    test_path,
    training_percentage,
    repeat_number,
):
    """Trains and evaluates one model at one training-data size."""

    label = run_label(
        model_name,
        training_percentage,
        repeat_number,
    )
    paths = run_result_paths(label)

    if SKIP_COMPLETED_RUNS and completed_run_exists(paths):
        print(f"\nSkipping completed run: {label}")
        return load_completed_run(paths)

    print(f"\nTraining {label}...")

    model_seed = SEED + repeat_number - 1
    set_seed(model_seed)

    train_dataframe = load_dataset(train_path, input_columns)
    validation_dataframe = load_dataset(validation_path, input_columns)
    test_dataframe = load_dataset(test_path, input_columns)

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
    data_loader_generator.manual_seed(model_seed)

    training_loader = DataLoader(
        training_dataset,
        batch_size=min(BATCH_SIZE, len(training_dataset)),
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

    start_training_time = time.perf_counter()

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
            validation_predictions = model(
                X_validation.to(DEVICE)
            )
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

    training_time = time.perf_counter() - start_training_time

    if best_model_state is None:
        raise RuntimeError(f"Training failed for {label}.")

    model.load_state_dict(best_model_state)

    history_dataframe = pd.DataFrame(
        {
            "model": model_name,
            "training_percentage": training_percentage,
            "training_samples": len(train_dataframe),
            "repeat": repeat_number,
            "epoch": np.arange(1, len(training_losses) + 1),
            "training_loss": training_losses,
            "validation_loss": validation_losses,
        }
    )

    X_test = test_dataframe[input_columns].to_numpy(dtype=float)
    y_test = test_dataframe[OUTPUT_COLUMNS].to_numpy(dtype=float)

    X_test_scaled = input_scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    model.eval()
    start_prediction_time = time.perf_counter()

    with torch.no_grad():
        predictions_scaled = model(
            X_test_tensor.to(DEVICE)
        ).cpu().numpy()

    prediction_time = time.perf_counter() - start_prediction_time
    predictions = output_scaler.inverse_transform(predictions_scaled)

    metric_records = []
    consistency_records = []

    for output_index, output_name in enumerate(OUTPUT_COLUMNS):
        actual = y_test[:, output_index]
        predicted = predictions[:, output_index]

        metric_records.append(
            {
                "model": model_name,
                "training_percentage": training_percentage,
                "training_samples": len(train_dataframe),
                "repeat": repeat_number,
                "output": output_name,
                "rmse": np.sqrt(
                    mean_squared_error(actual, predicted)
                ),
                "mae": mean_absolute_error(actual, predicted),
                "r2": r2_score(actual, predicted),
                "number_of_test_samples": len(test_dataframe),
                "prediction_time_seconds": prediction_time,
            }
        )

        if output_name in DISPLACEMENT_OUTPUTS:
            negative_count = int(np.sum(predicted < 0.0))

            consistency_records.append(
                {
                    "model": model_name,
                    "training_percentage": training_percentage,
                    "training_samples": len(train_dataframe),
                    "repeat": repeat_number,
                    "output": output_name,
                    "negative_prediction_count": negative_count,
                    "number_of_test_samples": len(test_dataframe),
                    "negative_prediction_percent": (
                        100.0 * negative_count / len(test_dataframe)
                    ),
                }
            )

    prediction_dataframe = pd.DataFrame(
        {
            "model": model_name,
            "training_percentage": training_percentage,
            "training_samples": len(train_dataframe),
            "repeat": repeat_number,
            "source_row_index": (
                test_dataframe["source_row_index"].to_numpy()
                if "source_row_index" in test_dataframe.columns
                else np.arange(len(test_dataframe))
            ),
        }
    )

    for output_index, output_name in enumerate(OUTPUT_COLUMNS):
        prediction_dataframe[f"actual_{output_name}"] = (
            y_test[:, output_index]
        )
        prediction_dataframe[f"predicted_{output_name}"] = (
            predictions[:, output_index]
        )
        prediction_dataframe[f"error_{output_name}"] = (
            y_test[:, output_index] - predictions[:, output_index]
        )

    training_summary_dataframe = pd.DataFrame(
        [
            {
                "model": model_name,
                "number_of_inputs": len(input_columns),
                "training_percentage": training_percentage,
                "training_samples": len(train_dataframe),
                "validation_samples": len(validation_dataframe),
                "test_samples": len(test_dataframe),
                "repeat": repeat_number,
                "random_seed": model_seed,
                "epochs_completed": len(training_losses),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "training_time_seconds": training_time,
            }
        ]
    )

    metrics_dataframe = pd.DataFrame(metric_records)
    consistency_dataframe = pd.DataFrame(consistency_records)

    paths["directory"].mkdir(parents=True, exist_ok=True)

    metrics_dataframe.to_csv(paths["metrics"], index=False)
    prediction_dataframe.to_csv(paths["predictions"], index=False)
    consistency_dataframe.to_csv(paths["consistency"], index=False)
    history_dataframe.to_csv(paths["history"], index=False)
    training_summary_dataframe.to_csv(
        paths["training_summary"],
        index=False,
    )

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
            "training_percentage": training_percentage,
            "training_samples": len(train_dataframe),
            "repeat": repeat_number,
            "random_seed": model_seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        paths["model"],
    )

    joblib.dump(input_scaler, paths["input_scaler"])
    joblib.dump(output_scaler, paths["output_scaler"])

    print(
        f"Completed {label}: best epoch {best_epoch}, "
        f"test prediction time {prediction_time:.4f} seconds"
    )

    return {
        "metrics": metrics_dataframe,
        "predictions": prediction_dataframe,
        "consistency": consistency_dataframe,
        "history": history_dataframe,
        "training_summary": training_summary_dataframe,
    }


# ----- Result summaries -----

def create_metric_summary(metrics_dataframe):
    """Calculates mean and standard deviation across repeated runs."""

    summary = (
        metrics_dataframe
        .groupby(
            [
                "model",
                "training_percentage",
                "training_samples",
                "output",
            ],
            as_index=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            repeats=("repeat", "nunique"),
        )
    )

    standard_deviation_columns = [
        "rmse_std",
        "mae_std",
        "r2_std",
    ]
    summary[standard_deviation_columns] = (
        summary[standard_deviation_columns].fillna(0.0)
    )

    return summary


def create_training_summary(training_dataframe):
    """Summarises training cost and convergence across repeats."""

    summary = (
        training_dataframe
        .groupby(
            [
                "model",
                "training_percentage",
                "training_samples",
            ],
            as_index=False,
        )
        .agg(
            training_time_mean_seconds=("training_time_seconds", "mean"),
            training_time_std_seconds=("training_time_seconds", "std"),
            best_validation_loss_mean=("best_validation_loss", "mean"),
            best_validation_loss_std=("best_validation_loss", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            epochs_completed_mean=("epochs_completed", "mean"),
            repeats=("repeat", "nunique"),
        )
    )

    standard_deviation_columns = [
        "training_time_std_seconds",
        "best_validation_loss_std",
    ]
    summary[standard_deviation_columns] = (
        summary[standard_deviation_columns].fillna(0.0)
    )

    return summary


def create_consistency_summary(consistency_dataframe):
    """Summarises negative displacement predictions across repeats."""

    summary = (
        consistency_dataframe
        .groupby(
            [
                "model",
                "training_percentage",
                "training_samples",
                "output",
            ],
            as_index=False,
        )
        .agg(
            negative_prediction_percent_mean=(
                "negative_prediction_percent",
                "mean",
            ),
            negative_prediction_percent_std=(
                "negative_prediction_percent",
                "std",
            ),
            repeats=("repeat", "nunique"),
        )
    )

    summary["negative_prediction_percent_std"] = (
        summary["negative_prediction_percent_std"].fillna(0.0)
    )

    return summary


def create_model_comparison(metric_summary):
    """Compares the mean physics-guided result with the standard result."""

    records = []

    for training_percentage in TRAINING_PERCENTAGES:
        percentage_data = metric_summary[
            metric_summary["training_percentage"] == training_percentage
        ]

        for output_name in OUTPUT_COLUMNS:
            output_data = percentage_data[
                percentage_data["output"] == output_name
            ].set_index("model")

            if not {
                "standard_NN",
                "physics_guided_NN",
            }.issubset(output_data.index):
                continue

            standard = output_data.loc["standard_NN"]
            physics_guided = output_data.loc["physics_guided_NN"]

            standard_rmse = float(standard["rmse_mean"])
            physics_rmse = float(physics_guided["rmse_mean"])

            records.append(
                {
                    "training_percentage": training_percentage,
                    "training_samples": int(
                        standard["training_samples"]
                    ),
                    "output": output_name,
                    "standard_rmse_mean": standard_rmse,
                    "physics_guided_rmse_mean": physics_rmse,
                    "rmse_reduction_percent": (
                        100.0
                        * (standard_rmse - physics_rmse)
                        / standard_rmse
                        if standard_rmse != 0.0
                        else np.nan
                    ),
                    "standard_mae_mean": float(standard["mae_mean"]),
                    "physics_guided_mae_mean": float(
                        physics_guided["mae_mean"]
                    ),
                    "standard_r2_mean": float(standard["r2_mean"]),
                    "physics_guided_r2_mean": float(
                        physics_guided["r2_mean"]
                    ),
                    "r2_difference": (
                        float(physics_guided["r2_mean"])
                        - float(standard["r2_mean"])
                    ),
                }
            )

    return pd.DataFrame(records)


# ----- Report-ready plots -----

MODEL_STYLES = {
    "standard_NN": {
        "label": "Standard NN",
        "colour": "tab:blue",
    },
    "physics_guided_NN": {
        "label": "Physics-guided NN",
        "colour": "tab:orange",
    },
}


def plot_mean_with_band(
    axis,
    dataframe,
    x_column,
    mean_column,
    std_column,
    label,
    colour,
):
    """Plots a repeated-run mean with a one-standard-deviation band."""

    sorted_data = dataframe.sort_values(x_column)
    x_values = sorted_data[x_column].to_numpy(dtype=float)
    means = sorted_data[mean_column].to_numpy(dtype=float)
    standard_deviations = sorted_data[std_column].to_numpy(dtype=float)

    axis.plot(
        x_values,
        means,
        marker="o",
        linewidth=2,
        label=label,
        color=colour,
    )
    axis.fill_between(
        x_values,
        means - standard_deviations,
        means + standard_deviations,
        color=colour,
        alpha=0.15,
    )


def save_displacement_learning_curves(metric_summary):
    """Plots displacement RMSE, MAE, and R2 against training data."""

    metric_settings = [
        ("rmse", "RMSE (mm)"),
        ("mae", "MAE (mm)"),
        ("r2", "R²"),
    ]

    for metric_name, axis_label in metric_settings:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))

        for axis, output_name in zip(axes, DISPLACEMENT_OUTPUTS):
            output_data = metric_summary[
                metric_summary["output"] == output_name
            ].copy()

            mean_column = f"{metric_name}_mean"
            std_column = f"{metric_name}_std"

            if metric_name in {"rmse", "mae"}:
                output_data[mean_column] *= 1000.0
                output_data[std_column] *= 1000.0

            for model_name, style in MODEL_STYLES.items():
                model_data = output_data[
                    output_data["model"] == model_name
                ]

                plot_mean_with_band(
                    axis=axis,
                    dataframe=model_data,
                    x_column="training_percentage",
                    mean_column=mean_column,
                    std_column=std_column,
                    label=style["label"],
                    colour=style["colour"],
                )

            axis.set_xlabel("Available training data (%)")
            axis.set_ylabel(axis_label)
            axis.set_title(output_name.replace("_", " ").title())
            axis.grid(True, alpha=0.3)
            axis.legend()

        figure.suptitle(
            f"Displacement {axis_label.split(' ')[0]} "
            "as training data are reduced"
        )
        figure.tight_layout()
        figure.savefig(
            PLOTS_DIR
            / f"experiment_3_displacement_{metric_name}.png",
            dpi=300,
        )
        plt.close(figure)


def save_frequency_r2_plot(metrics_dataframe):
    """Plots mean R2 across all four natural frequencies."""

    frequency_run_data = metrics_dataframe[
        metrics_dataframe["output"].isin(FREQUENCY_OUTPUTS)
    ]

    per_run_mean = (
        frequency_run_data
        .groupby(
            [
                "model",
                "training_percentage",
                "training_samples",
                "repeat",
            ],
            as_index=False,
        )
        .agg(mean_frequency_r2=("r2", "mean"))
    )

    frequency_summary = (
        per_run_mean
        .groupby(
            [
                "model",
                "training_percentage",
                "training_samples",
            ],
            as_index=False,
        )
        .agg(
            mean_frequency_r2=("mean_frequency_r2", "mean"),
            std_frequency_r2=("mean_frequency_r2", "std"),
        )
    )
    frequency_summary["std_frequency_r2"] = (
        frequency_summary["std_frequency_r2"].fillna(0.0)
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    for model_name, style in MODEL_STYLES.items():
        model_data = frequency_summary[
            frequency_summary["model"] == model_name
        ]

        plot_mean_with_band(
            axis=axis,
            dataframe=model_data,
            x_column="training_percentage",
            mean_column="mean_frequency_r2",
            std_column="std_frequency_r2",
            label=style["label"],
            colour=style["colour"],
        )

    axis.set_xlabel("Available training data (%)")
    axis.set_ylabel("Mean R² across f1-f4")
    axis.set_title("Natural-frequency accuracy as training data are reduced")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        PLOTS_DIR / "experiment_3_frequency_r2.png",
        dpi=300,
    )
    plt.close(figure)


def save_training_time_plot(training_summary):
    """Plots mean training time against training-set size."""

    figure, axis = plt.subplots(figsize=(8, 5))

    for model_name, style in MODEL_STYLES.items():
        model_data = training_summary[
            training_summary["model"] == model_name
        ]

        plot_mean_with_band(
            axis=axis,
            dataframe=model_data,
            x_column="training_percentage",
            mean_column="training_time_mean_seconds",
            std_column="training_time_std_seconds",
            label=style["label"],
            colour=style["colour"],
        )

    axis.set_xlabel("Available training data (%)")
    axis.set_ylabel("Training time (seconds)")
    axis.set_title("Training cost as training data are reduced")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        PLOTS_DIR / "experiment_3_training_time.png",
        dpi=300,
    )
    plt.close(figure)


def save_negative_prediction_plot(consistency_summary):
    """Plots physically invalid displacement predictions."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis, output_name in zip(axes, DISPLACEMENT_OUTPUTS):
        output_data = consistency_summary[
            consistency_summary["output"] == output_name
        ]

        for model_name, style in MODEL_STYLES.items():
            model_data = output_data[
                output_data["model"] == model_name
            ]

            plot_mean_with_band(
                axis=axis,
                dataframe=model_data,
                x_column="training_percentage",
                mean_column="negative_prediction_percent_mean",
                std_column="negative_prediction_percent_std",
                label=style["label"],
                colour=style["colour"],
            )

        axis.set_xlabel("Available training data (%)")
        axis.set_ylabel("Negative predictions (%)")
        axis.set_title(output_name.replace("_", " ").title())
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle(
        "Physical consistency as training data are reduced"
    )
    figure.tight_layout()
    figure.savefig(
        PLOTS_DIR / "experiment_3_negative_predictions.png",
        dpi=300,
    )
    plt.close(figure)


# ----- Main experiment -----

if __name__ == "__main__":

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model_settings = [
        {
            "name": "standard_NN",
            "input_columns": STANDARD_INPUT_COLUMNS,
            "hybrid": False,
            "validation_path": DATASET_DIR / "validation_fixed.csv",
            "test_path": DATASET_DIR / "test_fixed.csv",
        },
        {
            "name": "physics_guided_NN",
            "input_columns": HYBRID_INPUT_COLUMNS,
            "hybrid": True,
            "validation_path": (
                DATASET_DIR / "validation_fixed_hybrid.csv"
            ),
            "test_path": DATASET_DIR / "test_fixed_hybrid.csv",
        },
    ]

    check_experiment_files(model_settings)

    all_metrics = []
    all_predictions = []
    all_consistency = []
    all_histories = []
    all_training_summaries = []

    total_runs = (
        len(model_settings)
        * len(TRAINING_PERCENTAGES)
        * NUMBER_OF_REPEATS
    )
    completed_runs = 0

    print(
        f"\nStarting Experiment 3: {total_runs} model runs "
        f"({NUMBER_OF_REPEATS} repeats)."
    )

    for training_percentage in TRAINING_PERCENTAGES:
        for repeat_number in range(1, NUMBER_OF_REPEATS + 1):
            for settings in model_settings:
                train_path = (
                    DATASET_DIR
                    / training_filename(
                        training_percentage,
                        repeat_number,
                        hybrid=settings["hybrid"],
                    )
                )

                run_results = train_and_evaluate(
                    model_name=settings["name"],
                    input_columns=settings["input_columns"],
                    train_path=train_path,
                    validation_path=settings["validation_path"],
                    test_path=settings["test_path"],
                    training_percentage=training_percentage,
                    repeat_number=repeat_number,
                )

                all_metrics.append(run_results["metrics"])
                all_predictions.append(run_results["predictions"])
                all_consistency.append(run_results["consistency"])
                all_histories.append(run_results["history"])
                all_training_summaries.append(
                    run_results["training_summary"]
                )

                completed_runs += 1
                print(f"Progress: {completed_runs}/{total_runs} runs")

    metrics_dataframe = pd.concat(all_metrics, ignore_index=True)
    predictions_dataframe = pd.concat(
        all_predictions,
        ignore_index=True,
    )
    consistency_dataframe = pd.concat(
        all_consistency,
        ignore_index=True,
    )
    histories_dataframe = pd.concat(
        all_histories,
        ignore_index=True,
    )
    training_dataframe = pd.concat(
        all_training_summaries,
        ignore_index=True,
    )

    metric_summary = create_metric_summary(metrics_dataframe)
    training_summary = create_training_summary(training_dataframe)
    consistency_summary = create_consistency_summary(
        consistency_dataframe
    )
    model_comparison = create_model_comparison(metric_summary)

    metrics_dataframe.to_csv(
        RESULTS_DIR / "experiment_3_metrics_all_runs.csv",
        index=False,
    )
    metric_summary.to_csv(
        RESULTS_DIR / "experiment_3_metrics_summary.csv",
        index=False,
    )
    model_comparison.to_csv(
        RESULTS_DIR / "experiment_3_model_comparison.csv",
        index=False,
    )
    predictions_dataframe.to_csv(
        RESULTS_DIR / "experiment_3_predictions.csv",
        index=False,
    )
    consistency_dataframe.to_csv(
        RESULTS_DIR / "experiment_3_physical_consistency_all_runs.csv",
        index=False,
    )
    consistency_summary.to_csv(
        RESULTS_DIR / "experiment_3_physical_consistency_summary.csv",
        index=False,
    )
    histories_dataframe.to_csv(
        RESULTS_DIR / "experiment_3_training_histories.csv",
        index=False,
    )
    training_dataframe.to_csv(
        RESULTS_DIR / "experiment_3_training_all_runs.csv",
        index=False,
    )
    training_summary.to_csv(
        RESULTS_DIR / "experiment_3_training_summary.csv",
        index=False,
    )

    save_displacement_learning_curves(metric_summary)
    save_frequency_r2_plot(metrics_dataframe)
    save_training_time_plot(training_summary)
    save_negative_prediction_plot(consistency_summary)