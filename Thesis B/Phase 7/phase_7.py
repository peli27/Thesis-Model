"""
Physical consistency assessment.

Runs three controlled physical-consistency studies:
    - force-amplitude sweep
    - excitation-frequency / resonance sweep
    - beam-thickness sweep at a fixed frequency ratio.
"""

import copy
import random
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Project paths

PHASE_7_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_7_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
PHYSICS_DIR = PROJECT_ROOT / "Phase 1 & 2"
PHASE_5_DIR = PROJECT_ROOT / "Phase 5"

RESULTS_DIR = PHASE_7_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"

sys.path.insert(0, str(PHYSICS_DIR))
sys.path.insert(0, str(PHASE_5_DIR))

from frequency_analysis import natural_frequencies
from multi_mode_response import simulate_harmonic_beam_response
from physics_features import calculate_physics_features


# Settings

SEED = 42
DEVICE = torch.device("cpu")

BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.001
PATIENCE = 50

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

FORCE_RATIOS = [0.60, 0.90, 1.00]
FORCE_VALUES = np.linspace(0.10, 10.00, 30)

FREQUENCY_RATIOS = np.unique(
    np.concatenate(
        [
            np.linspace(0.50, 0.80, 7),
            np.linspace(0.82, 1.18, 37),
            np.linspace(1.20, 1.50, 7),
        ]
    )
)

THICKNESS_VALUES = np.linspace(0.005, 0.020, 30)
THICKNESS_FREQUENCY_RATIO = 0.80

MODEL_STYLES = {
    "simulator": {
        "label": "Physics simulator",
        "colour": "black",
        "linestyle": "-",
    },
    "standard_NN": {
        "label": "Standard NN",
        "colour": "tab:blue",
        "linestyle": "--",
    },
    "physics_guided_NN": {
        "label": "Physics-guided NN",
        "colour": "tab:orange",
        "linestyle": "-.",
    },
}


# ---------------------------------------------------------------------
# Reproducibility and model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BeamResponseNN(nn.Module):
    """Feedforward architecture used in Thesis B."""

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(OUTPUT_COLUMNS)),
        )

    def forward(self, inputs):
        return self.network(inputs)


# Dataset and feature helpers

def load_dataset(filename):
    path = DATASET_DIR / filename
    dataframe = pd.read_csv(path)

    required_columns = STANDARD_INPUT_COLUMNS + OUTPUT_COLUMNS
    missing = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")

    return dataframe.reset_index(drop=True)


def add_physics_features(dataframe):
    """Adds the five Phase 5 physics features to every row."""

    if all(
        column in dataframe.columns
        for column in PHYSICS_FEATURE_COLUMNS
    ):
        return dataframe.copy()

    records = []

    for _, row in dataframe.iterrows():
        features = calculate_physics_features(
            L=float(row["L"]),
            b=float(row["b"]),
            h=float(row["h"]),
            E=float(row["E"]),
            rho=float(row["rho"]),
            excitation_frequency=float(row["excitation_frequency"]),
        )
        records.append(features)

    feature_dataframe = pd.DataFrame(records, index=dataframe.index)

    return pd.concat(
        [
            dataframe.copy(),
            feature_dataframe[PHYSICS_FEATURE_COLUMNS],
        ],
        axis=1,
    )


def representative_case(test_dataframe):
    """
    Creates one representative in-range beam using median test-set values.

    Fixed numerical settings are taken from the test dataset when present.
    """

    case = {
        column: float(test_dataframe[column].median())
        for column in [
            "L",
            "b",
            "h",
            "E",
            "rho",
            "damping_ratio",
            "force_amplitude",
        ]
    }

    case["observation_point"] = (
        float(test_dataframe["observation_point"].median())
        if "observation_point" in test_dataframe
        else 0.65
    )
    case["simulation_time"] = (
        float(test_dataframe["simulation_time"].median())
        if "simulation_time" in test_dataframe
        else 2.0
    )
    case["time_step"] = (
        float(test_dataframe["time_step"].median())
        if "time_step" in test_dataframe
        else 0.002
    )
    case["number_of_modes"] = (
        int(round(test_dataframe["number_of_modes"].median()))
        if "number_of_modes" in test_dataframe
        else 4
    )

    return case


def build_beam(case):
    beam = {
        "L": float(case["L"]),
        "b": float(case["b"]),
        "h": float(case["h"]),
        "E": float(case["E"]),
        "rho": float(case["rho"]),
    }

    beam["A"] = beam["b"] * beam["h"]
    beam["I"] = beam["b"] * beam["h"] ** 3 / 12

    return beam


# Training

def model_paths(model_name):
    return {
        "model": MODELS_DIR / f"{model_name}.pth",
        "input_scaler": MODELS_DIR / f"{model_name}_input_scaler.pkl",
        "output_scaler": MODELS_DIR / f"{model_name}_output_scaler.pkl",
        "training": MODELS_DIR / f"{model_name}_training_summary.csv",
    }


def train_or_load_model(
    model_name,
    input_columns,
    training_dataframe,
    validation_dataframe,
):
    """
    Loads a cached Phase 7 model when available.
    Otherwise trains the model on the full Thesis B training dataset.
    """

    paths = model_paths(model_name)

    if all(path.exists() for path in paths.values()):
        model = BeamResponseNN(len(input_columns)).to(DEVICE)
        model.load_state_dict(
            torch.load(paths["model"], map_location=DEVICE)
        )
        model.eval()

        return {
            "model": model,
            "input_scaler": joblib.load(paths["input_scaler"]),
            "output_scaler": joblib.load(paths["output_scaler"]),
            "training_summary": pd.read_csv(paths["training"]),
        }

    print(f"\nTraining {model_name} on the full training dataset...")
    set_seed(SEED)

    input_scaler = StandardScaler().fit(
        training_dataframe[input_columns].to_numpy(dtype=float)
    )
    output_scaler = StandardScaler().fit(
        training_dataframe[OUTPUT_COLUMNS].to_numpy(dtype=float)
    )

    X_train = input_scaler.transform(
        training_dataframe[input_columns].to_numpy(dtype=float)
    )
    y_train = output_scaler.transform(
        training_dataframe[OUTPUT_COLUMNS].to_numpy(dtype=float)
    )
    X_validation = input_scaler.transform(
        validation_dataframe[input_columns].to_numpy(dtype=float)
    )
    y_validation = output_scaler.transform(
        validation_dataframe[OUTPUT_COLUMNS].to_numpy(dtype=float)
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_validation = torch.tensor(X_validation, dtype=torch.float32)
    y_validation = torch.tensor(y_validation, dtype=torch.float32)

    generator = torch.Generator().manual_seed(SEED)

    training_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = BeamResponseNN(len(input_columns)).to(DEVICE)
    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_state = None
    best_validation_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    training_losses = []
    validation_losses = []

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []

        for X_batch, y_batch in training_loader:
            predictions = model(X_batch.to(DEVICE))
            loss = loss_function(
                predictions,
                y_batch.to(DEVICE),
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            batch_losses.append(loss.item())

        training_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                X_validation.to(DEVICE)
            )
            validation_loss = loss_function(
                validation_prediction,
                y_validation.to(DEVICE),
            ).item()

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch}: "
                f"training loss={training_loss:.6f}, "
                f"validation loss={validation_loss:.6f}"
            )

        if epochs_without_improvement >= PATIENCE:
            break

    training_time = time.perf_counter() - start_time

    model.load_state_dict(best_state)
    model.eval()

    training_summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "training_samples": len(training_dataframe),
                "validation_samples": len(validation_dataframe),
                "epochs_completed": len(training_losses),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "training_time_seconds": training_time,
            }
        ]
    )

    history = pd.DataFrame(
        {
            "model": model_name,
            "epoch": range(1, len(training_losses) + 1),
            "training_loss": training_losses,
            "validation_loss": validation_losses,
        }
    )

    torch.save(model.state_dict(), paths["model"])
    joblib.dump(input_scaler, paths["input_scaler"])
    joblib.dump(output_scaler, paths["output_scaler"])
    training_summary.to_csv(paths["training"], index=False)
    history.to_csv(
        MODELS_DIR / f"{model_name}_training_history.csv",
        index=False,
    )

    return {
        "model": model,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "training_summary": training_summary,
    }


def predict(model_bundle, dataframe, input_columns):
    scaled_inputs = model_bundle["input_scaler"].transform(
        dataframe[input_columns].to_numpy(dtype=float)
    )

    input_tensor = torch.tensor(
        scaled_inputs,
        dtype=torch.float32,
    )

    model_bundle["model"].eval()

    with torch.no_grad():
        scaled_predictions = (
            model_bundle["model"](input_tensor.to(DEVICE))
            .cpu()
            .numpy()
        )

    return model_bundle["output_scaler"].inverse_transform(
        scaled_predictions
    )


# ---------------------------------------------------------------------
# Physics simulation

def simulate_case(case):
    beam = build_beam(case)
    number_of_modes = int(case["number_of_modes"])

    frequencies = natural_frequencies(
        beam,
        modes=number_of_modes,
    )

    time_array = np.arange(
        0.0,
        float(case["simulation_time"]) + float(case["time_step"]),
        float(case["time_step"]),
    )

    response = simulate_harmonic_beam_response(
        beam=beam,
        frequencies=frequencies,
        damping_ratio=float(case["damping_ratio"]),
        force_amplitude=float(case["force_amplitude"]),
        forcing_frequency=float(case["excitation_frequency"]),
        t=time_array,
        x_position=float(case["observation_point"]),
        number_of_modes=number_of_modes,
    )

    row = {
        "L": beam["L"],
        "b": beam["b"],
        "h": beam["h"],
        "E": beam["E"],
        "rho": beam["rho"],
        "damping_ratio": float(case["damping_ratio"]),
        "force_amplitude": float(case["force_amplitude"]),
        "excitation_frequency": float(case["excitation_frequency"]),
        "observation_point": float(case["observation_point"]),
        "simulation_time": float(case["simulation_time"]),
        "time_step": float(case["time_step"]),
        "number_of_modes": number_of_modes,
        "f1": float(frequencies[0]),
        "f2": float(frequencies[1]),
        "f3": float(frequencies[2]),
        "f4": float(frequencies[3]),
        "peak_displacement": float(response["peak_displacement"]),
        "rms_displacement": float(response["rms_displacement"]),
    }

    row.update(
        calculate_physics_features(
            L=beam["L"],
            b=beam["b"],
            h=beam["h"],
            E=beam["E"],
            rho=beam["rho"],
            excitation_frequency=float(case["excitation_frequency"]),
        )
    )

    return row


def add_model_predictions(
    simulator_dataframe,
    standard_bundle,
    physics_bundle,
):
    standard_predictions = predict(
        standard_bundle,
        simulator_dataframe,
        STANDARD_INPUT_COLUMNS,
    )
    physics_predictions = predict(
        physics_bundle,
        simulator_dataframe,
        HYBRID_INPUT_COLUMNS,
    )

    records = []

    for row_index, row in simulator_dataframe.iterrows():
        simulator_record = row.to_dict()

        for model_name, values in [
            ("simulator", row[OUTPUT_COLUMNS].to_numpy(dtype=float)),
            ("standard_NN", standard_predictions[row_index]),
            ("physics_guided_NN", physics_predictions[row_index]),
        ]:
            record = {
                key: value
                for key, value in simulator_record.items()
                if key not in OUTPUT_COLUMNS
            }
            record["model"] = model_name

            for output_index, output_name in enumerate(OUTPUT_COLUMNS):
                record[output_name] = float(values[output_index])

            records.append(record)

    return pd.DataFrame(records)


# Sweep generation

def run_force_sweep(base_case):
    rows = []

    beam = build_beam(base_case)
    f1 = natural_frequencies(beam, modes=1)[0]

    for ratio in FORCE_RATIOS:
        for force in FORCE_VALUES:
            case = base_case.copy()
            case["force_amplitude"] = float(force)
            case["excitation_frequency"] = float(ratio * f1)

            row = simulate_case(case)
            row["experiment"] = "force_sweep"
            row["case"] = f"frequency_ratio_{ratio:.2f}"
            row["sweep_value"] = float(force)
            row["frequency_ratio_control"] = float(ratio)
            rows.append(row)

    return pd.DataFrame(rows)


def run_frequency_sweep(base_case):
    rows = []

    beam = build_beam(base_case)
    f1 = natural_frequencies(beam, modes=1)[0]

    for ratio in FREQUENCY_RATIOS:
        case = base_case.copy()
        case["excitation_frequency"] = float(ratio * f1)

        row = simulate_case(case)
        row["experiment"] = "frequency_sweep"
        row["case"] = "first_mode_resonance"
        row["sweep_value"] = float(ratio)
        row["frequency_ratio_control"] = float(ratio)
        rows.append(row)

    return pd.DataFrame(rows)


def run_thickness_sweep(base_case):
    rows = []

    for thickness in THICKNESS_VALUES:
        case = base_case.copy()
        case["h"] = float(thickness)

        beam = build_beam(case)
        f1 = natural_frequencies(beam, modes=1)[0]

        case["excitation_frequency"] = float(
            THICKNESS_FREQUENCY_RATIO * f1
        )

        row = simulate_case(case)
        row["experiment"] = "thickness_sweep"
        row["case"] = (
            f"constant_frequency_ratio_"
            f"{THICKNESS_FREQUENCY_RATIO:.2f}"
        )
        row["sweep_value"] = float(thickness)
        row["frequency_ratio_control"] = float(
            THICKNESS_FREQUENCY_RATIO
        )
        rows.append(row)

    return pd.DataFrame(rows)


# Metrics

def rmse(actual, predicted):
    return float(
        np.sqrt(mean_squared_error(actual, predicted))
    )


def monotonicity_violation_rate(values, expected_direction):
    differences = np.diff(np.asarray(values, dtype=float))

    if expected_direction == "increasing":
        violations = np.sum(differences < 0)
    elif expected_direction == "decreasing":
        violations = np.sum(differences > 0)
    else:
        return np.nan

    return float(violations / len(differences))


def create_metrics(predictions):
    metrics = []

    simulator = predictions[
        predictions["model"] == "simulator"
    ]

    for experiment in predictions["experiment"].unique():
        experiment_simulator = simulator[
            simulator["experiment"] == experiment
        ]

        cases = experiment_simulator["case"].unique()

        for case_name in cases:
            true_case = (
                experiment_simulator[
                    experiment_simulator["case"] == case_name
                ]
                .sort_values("sweep_value")
                .reset_index(drop=True)
            )

            for model_name in [
                "standard_NN",
                "physics_guided_NN",
            ]:
                predicted_case = (
                    predictions[
                        (predictions["experiment"] == experiment)
                        & (predictions["case"] == case_name)
                        & (predictions["model"] == model_name)
                    ]
                    .sort_values("sweep_value")
                    .reset_index(drop=True)
                )

                for output_name in OUTPUT_COLUMNS:
                    actual = true_case[output_name].to_numpy(dtype=float)
                    predicted = predicted_case[
                        output_name
                    ].to_numpy(dtype=float)

                    record = {
                        "experiment": experiment,
                        "case": case_name,
                        "model": model_name,
                        "output": output_name,
                        "rmse": rmse(actual, predicted),
                        "mae": float(
                            mean_absolute_error(actual, predicted)
                        ),
                        "maximum_absolute_error": float(
                            np.max(np.abs(predicted - actual))
                        ),
                        "monotonicity_violation_rate": np.nan,
                        "force_proportionality_error": np.nan,
                        "near_resonance_rmse": np.nan,
                        "resonance_location_error": np.nan,
                        "resonance_amplitude_relative_error": np.nan,
                    }

                    if experiment == "force_sweep":
                        if output_name in DISPLACEMENT_OUTPUTS:
                            record["monotonicity_violation_rate"] = (
                                monotonicity_violation_rate(
                                    predicted,
                                    "increasing",
                                )
                            )

                            force = true_case[
                                "sweep_value"
                            ].to_numpy(dtype=float)
                            true_gain = actual / force
                            predicted_gain = predicted / force

                            record["force_proportionality_error"] = float(
                                np.mean(
                                    np.abs(
                                        predicted_gain - true_gain
                                    )
                                    / np.maximum(
                                        np.abs(true_gain),
                                        1e-12,
                                    )
                                )
                            )

                    elif experiment == "thickness_sweep":
                        if output_name in [
                            "f1",
                            "f2",
                            "f3",
                            "f4",
                        ]:
                            expected_direction = "increasing"
                        else:
                            expected_direction = "decreasing"

                        record["monotonicity_violation_rate"] = (
                            monotonicity_violation_rate(
                                predicted,
                                expected_direction,
                            )
                        )

                    elif experiment == "frequency_sweep":
                        ratios = true_case[
                            "sweep_value"
                        ].to_numpy(dtype=float)
                        near_mask = (
                            (ratios >= 0.90)
                            & (ratios <= 1.10)
                        )

                        record["near_resonance_rmse"] = rmse(
                            actual[near_mask],
                            predicted[near_mask],
                        )

                        true_peak_index = int(np.argmax(actual))
                        predicted_peak_index = int(np.argmax(predicted))

                        record["resonance_location_error"] = float(
                            abs(
                                ratios[predicted_peak_index]
                                - ratios[true_peak_index]
                            )
                        )

                        record[
                            "resonance_amplitude_relative_error"
                        ] = float(
                            abs(
                                predicted[predicted_peak_index]
                                - actual[true_peak_index]
                            )
                            / max(
                                abs(actual[true_peak_index]),
                                1e-12,
                            )
                        )

                    metrics.append(record)

    return pd.DataFrame(metrics)


def create_model_comparison(metrics):
    standard = metrics[
        metrics["model"] == "standard_NN"
    ]
    physics = metrics[
        metrics["model"] == "physics_guided_NN"
    ]

    comparison = standard.merge(
        physics,
        on=["experiment", "case", "output"],
        suffixes=("_standard", "_physics"),
    )

    comparison["rmse_reduction_percent"] = (
        100
        * (
            comparison["rmse_standard"]
            - comparison["rmse_physics"]
        )
        / comparison["rmse_standard"].replace(0, np.nan)
    )

    comparison["better_rmse_model"] = np.where(
        comparison["rmse_physics"]
        < comparison["rmse_standard"],
        "physics_guided_NN",
        "standard_NN",
    )

    return comparison


# Plotting

def plot_model_lines(axis, dataframe, x_column, y_column):
    for model_name in [
        "simulator",
        "standard_NN",
        "physics_guided_NN",
    ]:
        model_data = (
            dataframe[dataframe["model"] == model_name]
            .sort_values(x_column)
        )
        style = MODEL_STYLES[model_name]

        axis.plot(
            model_data[x_column],
            model_data[y_column],
            label=style["label"],
            color=style["colour"],
            linestyle=style["linestyle"],
            linewidth=2,
        )

    axis.grid(True, alpha=0.3)


def save_force_plots(predictions):
    force_data = predictions[
        predictions["experiment"] == "force_sweep"
    ]

    for output_name in DISPLACEMENT_OUTPUTS:
        figure, axes = plt.subplots(
            1,
            len(FORCE_RATIOS),
            figsize=(15, 4.8),
            sharey=False,
        )

        for axis, ratio in zip(axes, FORCE_RATIOS):
            case_name = f"frequency_ratio_{ratio:.2f}"
            case_data = force_data[
                force_data["case"] == case_name
            ]

            plot_model_lines(
                axis,
                case_data,
                "sweep_value",
                output_name,
            )

            axis.set_xlabel("Force amplitude (N/m)")
            axis.set_ylabel("Displacement (m)")
            axis.set_title(f"Frequency ratio = {ratio:.2f}")

        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
        )
        figure.suptitle(
            output_name.replace("_", " ").title(),
            y=1.02,
        )
        figure.tight_layout()
        figure.savefig(
            PLOTS_DIR / f"force_sweep_{output_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def save_frequency_plots(predictions):
    frequency_data = predictions[
        predictions["experiment"] == "frequency_sweep"
    ]

    for output_name in DISPLACEMENT_OUTPUTS:
        figure, axis = plt.subplots(figsize=(8, 5))

        plot_model_lines(
            axis,
            frequency_data,
            "sweep_value",
            output_name,
        )

        axis.axvline(
            1.0,
            color="grey",
            linestyle=":",
            linewidth=1.2,
        )
        axis.axvspan(
            0.90,
            1.10,
            color="grey",
            alpha=0.12,
        )
        axis.set_xlabel(
            "Frequency ratio, excitation frequency / f1"
        )
        axis.set_ylabel("Displacement (m)")
        axis.set_title(
            f"First-mode resonance: "
            f"{output_name.replace('_', ' ').title()}"
        )
        axis.legend()

        figure.tight_layout()
        figure.savefig(
            PLOTS_DIR / f"frequency_sweep_{output_name}.png",
            dpi=300,
        )
        plt.close(figure)


def save_thickness_plots(predictions):
    thickness_data = predictions[
        predictions["experiment"] == "thickness_sweep"
    ]

    settings = [
        ("f1", "First natural frequency (Hz)"),
        ("peak_displacement", "Peak displacement (m)"),
        ("rms_displacement", "RMS displacement (m)"),
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.8),
    )

    for axis, (output_name, ylabel) in zip(axes, settings):
        plot_model_lines(
            axis,
            thickness_data,
            "sweep_value",
            output_name,
        )
        axis.set_xlabel("Beam thickness (m)")
        axis.set_ylabel(ylabel)
        axis.set_title(output_name.replace("_", " ").title())

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        f"Thickness sweep at constant frequency ratio "
        f"{THICKNESS_FREQUENCY_RATIO:.2f}",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(
        PLOTS_DIR / "thickness_sweep.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# Main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading Thesis B datasets...")

    training = load_dataset("train_dataset.csv")
    validation = load_dataset("validation_dataset.csv")
    test = load_dataset("test_dataset.csv")

    training_hybrid = add_physics_features(training)
    validation_hybrid = add_physics_features(validation)

    standard_bundle = train_or_load_model(
        model_name="standard_NN",
        input_columns=STANDARD_INPUT_COLUMNS,
        training_dataframe=training,
        validation_dataframe=validation,
    )

    physics_bundle = train_or_load_model(
        model_name="physics_guided_NN",
        input_columns=HYBRID_INPUT_COLUMNS,
        training_dataframe=training_hybrid,
        validation_dataframe=validation_hybrid,
    )

    base_case = representative_case(test)

    pd.DataFrame([base_case]).to_csv(
        RESULTS_DIR / "phase_7_representative_case.csv",
        index=False,
    )

    force_simulator = run_force_sweep(base_case)

    frequency_simulator = run_frequency_sweep(base_case)

    thickness_simulator = run_thickness_sweep(base_case)

    simulator_data = pd.concat(
        [
            force_simulator,
            frequency_simulator,
            thickness_simulator,
        ],
        ignore_index=True,
    )

    predictions = add_model_predictions(
        simulator_dataframe=simulator_data,
        standard_bundle=standard_bundle,
        physics_bundle=physics_bundle,
    )

    metrics = create_metrics(predictions)
    comparison = create_model_comparison(metrics)

    training_summary = pd.concat(
        [
            standard_bundle["training_summary"],
            physics_bundle["training_summary"],
        ],
        ignore_index=True,
    )

    predictions.to_csv(
        RESULTS_DIR / "phase_7_predictions.csv",
        index=False,
    )
    metrics.to_csv(
        RESULTS_DIR / "phase_7_metrics_summary.csv",
        index=False,
    )
    comparison.to_csv(
        RESULTS_DIR / "phase_7_model_comparison.csv",
        index=False,
    )
    training_summary.to_csv(
        RESULTS_DIR / "phase_7_training_summary.csv",
        index=False,
    )

    save_force_plots(predictions)
    save_frequency_plots(predictions)
    save_thickness_plots(predictions)


if __name__ == "__main__":
    main()