"""
Phase 6, Experiment 2: unseen-resonance generalisation study.

This script trains:

1. Standard neural network.
2. Physics-guided neural network with five physics-guided features.

Both models are trained once using only off-resonance data. They are then
evaluated on paired test beams at fixed excitation-to-natural frequency ratios,
including near-resonance ratios that were excluded from training.

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
DATASET_DIR = PHASE_6_DIR / "Experiment_2_datasets"

RESULTS_DIR = PHASE_6_DIR / "Experiment_2_results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"


# ----- Experiment settings -----

SEED = 42

BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.001
PATIENCE = 50

DEVICE = torch.device("cpu")

TEST_FREQUENCY_RATIOS = [
    0.50,
    0.80,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.20,
    1.50,
]

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

DISPLACEMENT_OUTPUTS = [
    "peak_displacement",
    "rms_displacement",
]


# ----- Reproducibility -----

def set_seed(seed):
    """Makes training results repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----- Neural-network model -----

class BeamResponseNN(nn.Module):
    """The same feedforward architecture used in Phases 4 and 5."""

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

def ratio_label(frequency_ratio):
    """Converts 0.95 into the filename label 0p95."""

    return f"{frequency_ratio:.2f}".replace(".", "p")


def classify_ratio(frequency_ratio):
    """Labels whether a ratio was excluded from the training data."""

    if 0.80 < frequency_ratio < 1.20:
        return "unseen_near_resonance"

    return "seen_off_resonance"


def load_dataset(path, input_columns):
    """Loads one CSV and checks that all model columns exist."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run generate_resonance_datasets.py before Experiment 2."
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
    """Checks all required files before either model starts training."""

    required_paths = []

    for settings in model_settings:
        required_paths.extend(
            [
                settings["train_path"],
                settings["validation_path"],
            ]
        )

        for frequency_ratio in TEST_FREQUENCY_RATIOS:
            filename = (
                f"test_ratio_{ratio_label(frequency_ratio)}"
                f"{settings['hybrid_suffix']}.csv"
            )
            required_paths.append(DATASET_DIR / filename)

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_list = "\n".join(f"  - {path.name}" for path in missing_paths)
        raise FileNotFoundError(
            "Experiment 2 cannot start because these datasets are missing:\n"
            f"{missing_list}\n"
            "Run generate_resonance_datasets.py first."
        )


def create_tensors(dataframe, input_columns, input_scaler, output_scaler):
    """Scales selected model columns and converts them to PyTorch tensors."""

    inputs = dataframe[input_columns].to_numpy(dtype=float)
    outputs = dataframe[OUTPUT_COLUMNS].to_numpy(dtype=float)

    scaled_inputs = input_scaler.transform(inputs)
    scaled_outputs = output_scaler.transform(outputs)

    input_tensor = torch.tensor(scaled_inputs, dtype=torch.float32)
    output_tensor = torch.tensor(scaled_outputs, dtype=torch.float32)

    return input_tensor, output_tensor


# ----- Plotting helpers -----

def save_loss_plot(history_dataframe, model_name):
    """Saves the training and validation loss curves for one model."""

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
    plt.axvline(
        history_dataframe.loc[
            history_dataframe["validation_loss"].idxmin(),
            "epoch",
        ],
        color="black",
        linestyle=":",
        label="Best validation epoch",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Scaled MSE loss")
    plt.title(f"Training history - {model_name.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"{model_name}_loss_curve.png",
        dpi=300,
    )
    plt.close()


def save_displacement_metric_plots(metrics_dataframe):
    """Plots displacement RMSE, MAE and R2 against frequency ratio."""

    model_styles = {
        "standard_NN": {
            "label": "Standard NN",
            "colour": "tab:blue",
        },
        "physics_guided_NN": {
            "label": "Physics-guided NN",
            "colour": "tab:orange",
        },
    }

    for metric_name, axis_label in [
        ("rmse", "RMSE (mm)"),
        ("mae", "MAE (mm)"),
        ("r2", "R²"),
    ]:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))

        for axis, output_name in zip(axes, DISPLACEMENT_OUTPUTS):
            output_metrics = metrics_dataframe[
                metrics_dataframe["output"] == output_name
            ]

            for model_name, style in model_styles.items():
                model_metrics = output_metrics[
                    output_metrics["model"] == model_name
                ].sort_values("frequency_ratio")

                values = model_metrics[metric_name].to_numpy()

                if metric_name in {"rmse", "mae"}:
                    values = values * 1000.0

                axis.plot(
                    model_metrics["frequency_ratio"],
                    values,
                    marker="o",
                    linewidth=2,
                    label=style["label"],
                    color=style["colour"],
                )

            axis.axvspan(
                0.80,
                1.20,
                color="grey",
                alpha=0.15,
                label="Excluded from training",
            )
            axis.axvline(
                1.0,
                color="black",
                linestyle=":",
                linewidth=1.2,
            )
            axis.set_xlabel("Frequency ratio, excitation frequency / f1")
            axis.set_ylabel(axis_label)
            axis.set_title(output_name.replace("_", " ").title())
            axis.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
        )
        figure.suptitle(
            f"Displacement {axis_label.split(' ')[0]} across first-mode resonance",
            y=1.02,
        )
        figure.tight_layout()
        figure.savefig(
            PLOTS_DIR / f"experiment_2_displacement_{metric_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def save_resonance_scatter_plot(predictions_dataframe):
    """Plots actual versus predicted displacement at exact resonance."""

    resonance_predictions = predictions_dataframe[
        np.isclose(predictions_dataframe["frequency_ratio"], 1.0)
    ]

    figure, axes = plt.subplots(2, 2, figsize=(11, 10))

    model_order = ["standard_NN", "physics_guided_NN"]

    for row_index, output_name in enumerate(DISPLACEMENT_OUTPUTS):
        for column_index, model_name in enumerate(model_order):
            axis = axes[row_index, column_index]

            condition = resonance_predictions[
                resonance_predictions["model"] == model_name
            ]

            actual = condition[f"actual_{output_name}"] * 1000.0
            predicted = condition[f"predicted_{output_name}"] * 1000.0

            lower = min(actual.min(), predicted.min())
            upper = max(actual.max(), predicted.max())

            axis.scatter(
                actual,
                predicted,
                s=18,
                alpha=0.6,
            )
            axis.plot(
                [lower, upper],
                [lower, upper],
                color="black",
                linestyle="--",
                linewidth=1,
            )
            axis.set_xlabel("Actual displacement (mm)")
            axis.set_ylabel("Predicted displacement (mm)")
            axis.set_title(
                f"{model_name.replace('_', ' ').title()}\n"
                f"{output_name.replace('_', ' ').title()}"
            )
            axis.grid(True, alpha=0.3)

    figure.suptitle(
        "Actual versus predicted displacement at exact resonance (r = 1.00)"
    )
    figure.tight_layout()
    figure.savefig(
        PLOTS_DIR / "experiment_2_resonance_predictions.png",
        dpi=300,
    )
    plt.close(figure)


def save_negative_prediction_plot(consistency_dataframe):
    """Plots physically invalid negative displacement predictions."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis, output_name in zip(axes, DISPLACEMENT_OUTPUTS):
        output_data = consistency_dataframe[
            consistency_dataframe["output"] == output_name
        ]

        for model_name, colour in [
            ("standard_NN", "tab:blue"),
            ("physics_guided_NN", "tab:orange"),
        ]:
            condition = output_data[
                output_data["model"] == model_name
            ].sort_values("frequency_ratio")

            axis.plot(
                condition["frequency_ratio"],
                condition["negative_prediction_percent"],
                marker="o",
                linewidth=2,
                color=colour,
                label=model_name.replace("_", " ").title(),
            )

        axis.axvspan(0.80, 1.20, color="grey", alpha=0.15)
        axis.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
        axis.set_xlabel("Frequency ratio, excitation frequency / f1")
        axis.set_ylabel("Negative predictions (%)")
        axis.set_title(output_name.replace("_", " ").title())
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("Physical consistency across first-mode resonance")
    figure.tight_layout()
    figure.savefig(
        PLOTS_DIR / "experiment_2_negative_predictions.png",
        dpi=300,
    )
    plt.close(figure)


# ----- Model training -----

def train_model(model_name, input_columns, train_path, validation_path):
    """Trains one model and returns it with its fitted scalers."""

    print(f"\nTraining {model_name}...")

    set_seed(SEED)

    train_dataframe = load_dataset(train_path, input_columns)
    validation_dataframe = load_dataset(validation_path, input_columns)

    print(f"Training samples:   {len(train_dataframe)}")
    print(f"Validation samples: {len(validation_dataframe)}")

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
    frequency_ratio,
):
    """Evaluates one trained model at one fixed frequency ratio."""

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
    consistency_records = []

    for output_index, output_name in enumerate(OUTPUT_COLUMNS):
        actual = y_test[:, output_index]
        predicted = predictions[:, output_index]

        metric_records.append(
            {
                "model": model_name,
                "frequency_ratio": frequency_ratio,
                "distance_from_resonance": abs(1.0 - frequency_ratio),
                "test_region": classify_ratio(frequency_ratio),
                "output": output_name,
                "rmse": np.sqrt(mean_squared_error(actual, predicted)),
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
                    "frequency_ratio": frequency_ratio,
                    "distance_from_resonance": abs(1.0 - frequency_ratio),
                    "test_region": classify_ratio(frequency_ratio),
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
            "frequency_ratio": frequency_ratio,
            "distance_from_resonance": abs(1.0 - frequency_ratio),
            "test_region": classify_ratio(frequency_ratio),
            "sample_id": (
                test_dataframe["sample_id"].to_numpy()
                if "sample_id" in test_dataframe.columns
                else np.arange(len(test_dataframe))
            ),
            "damping_ratio": test_dataframe["damping_ratio"].to_numpy(),
            "force_amplitude": test_dataframe["force_amplitude"].to_numpy(),
            "excitation_frequency": (
                test_dataframe["excitation_frequency"].to_numpy()
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
        f"Evaluated {model_name} at frequency ratio "
        f"r = {frequency_ratio:.2f}"
    )

    return metric_records, consistency_records, prediction_dataframe


def create_resonance_comparison(metrics_dataframe):
    """Creates a concise standard-versus-physics comparison at r = 1.00."""

    resonance_metrics = metrics_dataframe[
        np.isclose(metrics_dataframe["frequency_ratio"], 1.0)
        & metrics_dataframe["output"].isin(DISPLACEMENT_OUTPUTS)
    ]

    records = []

    for output_name in DISPLACEMENT_OUTPUTS:
        output_metrics = resonance_metrics[
            resonance_metrics["output"] == output_name
        ].set_index("model")

        standard = output_metrics.loc["standard_NN"]
        physics_guided = output_metrics.loc["physics_guided_NN"]

        records.append(
            {
                "output": output_name,
                "standard_rmse": standard["rmse"],
                "physics_guided_rmse": physics_guided["rmse"],
                "rmse_reduction_percent": (
                    100.0
                    * (standard["rmse"] - physics_guided["rmse"])
                    / standard["rmse"]
                ),
                "standard_mae": standard["mae"],
                "physics_guided_mae": physics_guided["mae"],
                "mae_reduction_percent": (
                    100.0
                    * (standard["mae"] - physics_guided["mae"])
                    / standard["mae"]
                ),
                "standard_r2": standard["r2"],
                "physics_guided_r2": physics_guided["r2"],
                "r2_change": physics_guided["r2"] - standard["r2"],
            }
        )

    return pd.DataFrame.from_records(records)


# ----- Run Experiment 2 -----

if __name__ == "__main__":

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model_settings = [
        {
            "name": "standard_NN",
            "input_columns": STANDARD_INPUT_COLUMNS,
            "train_path": DATASET_DIR / "train_off_resonance.csv",
            "validation_path": (
                DATASET_DIR / "validation_off_resonance.csv"
            ),
            "hybrid_suffix": "",
        },
        {
            "name": "physics_guided_NN",
            "input_columns": HYBRID_INPUT_COLUMNS,
            "train_path": (
                DATASET_DIR / "train_off_resonance_hybrid.csv"
            ),
            "validation_path": (
                DATASET_DIR / "validation_off_resonance_hybrid.csv"
            ),
            "hybrid_suffix": "_hybrid",
        },
    ]

    check_experiment_files(model_settings)

    all_metrics = []
    all_consistency_records = []
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

        for frequency_ratio in TEST_FREQUENCY_RATIOS:
            test_filename = (
                f"test_ratio_{ratio_label(frequency_ratio)}"
                f"{settings['hybrid_suffix']}.csv"
            )

            (
                metric_records,
                consistency_records,
                prediction_dataframe,
            ) = evaluate_model(
                model=model,
                model_name=settings["name"],
                input_columns=settings["input_columns"],
                input_scaler=input_scaler,
                output_scaler=output_scaler,
                test_path=DATASET_DIR / test_filename,
                frequency_ratio=frequency_ratio,
            )

            all_metrics.extend(metric_records)
            all_consistency_records.extend(consistency_records)
            all_predictions.append(prediction_dataframe)

    metrics_dataframe = pd.DataFrame.from_records(all_metrics)
    consistency_dataframe = pd.DataFrame.from_records(
        all_consistency_records
    )
    predictions_dataframe = pd.concat(
        all_predictions,
        ignore_index=True,
    )
    training_summary_dataframe = pd.DataFrame.from_records(
        training_summaries
    )

    resonance_comparison = create_resonance_comparison(metrics_dataframe)

    metrics_dataframe.to_csv(
        RESULTS_DIR / "experiment_2_metrics.csv",
        index=False,
    )
    consistency_dataframe.to_csv(
        RESULTS_DIR / "experiment_2_physical_consistency.csv",
        index=False,
    )
    predictions_dataframe.to_csv(
        RESULTS_DIR / "experiment_2_predictions.csv",
        index=False,
    )
    training_summary_dataframe.to_csv(
        RESULTS_DIR / "experiment_2_training_summary.csv",
        index=False,
    )
    resonance_comparison.to_csv(
        RESULTS_DIR / "experiment_2_resonance_comparison.csv",
        index=False,
    )

    save_displacement_metric_plots(metrics_dataframe)
    save_resonance_scatter_plot(predictions_dataframe)
    save_negative_prediction_plot(consistency_dataframe)

    print("\nExperiment 2 completed successfully.")
    print(f"Results saved in: {RESULTS_DIR}")
    print("\nExact-resonance comparison:")
    print(resonance_comparison.to_string(index=False))