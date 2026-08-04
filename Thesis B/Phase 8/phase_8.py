"""
Digital Twin Assessment

- Loads the final standard and physics-guided neural networks from Phase 7.
- Evaluates their accuracy on the fixed test dataset.
- Measures prediction time for both neural networks and the physics simulator.
- Calculates speed-up relative to the simulator.

"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Paths

PHASE_8_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_8_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
PHYSICS_DIR = PROJECT_ROOT / "Phase 1 & 2"
PHASE_5_DIR = PROJECT_ROOT / "Phase 5"
PHASE_7_RESULTS_DIR = PROJECT_ROOT / "Phase 7" / "results"
PHASE_7_MODELS_DIR = PHASE_7_RESULTS_DIR / "models"

RESULTS_DIR = PHASE_8_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

sys.path.insert(0, str(PHYSICS_DIR))
sys.path.insert(0, str(PHASE_5_DIR))

from frequency_analysis import natural_frequencies
from multi_mode_response import simulate_harmonic_beam_response
from physics_features import calculate_physics_features


# Settings

DEVICE = torch.device("cpu")

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

HYBRID_INPUT_COLUMNS = (
    STANDARD_INPUT_COLUMNS + PHYSICS_FEATURE_COLUMNS
)

OUTPUT_COLUMNS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "peak_displacement",
    "rms_displacement",
]

TIMING_REPEATS = 20
SIMULATOR_TIMING_CASES = 100


# Neural network

class BeamResponseNN(nn.Module):
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


def load_model_bundle(model_name, input_size):
    model = BeamResponseNN(input_size).to(DEVICE)

    model_path = PHASE_7_MODELS_DIR / f"{model_name}.pth"

    try:
        state = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            model_path,
            map_location=DEVICE,
        )

    model.load_state_dict(state)
    model.eval()

    return {
        "model": model,
        "input_scaler": joblib.load(
            PHASE_7_MODELS_DIR
            / f"{model_name}_input_scaler.pkl"
        ),
        "output_scaler": joblib.load(
            PHASE_7_MODELS_DIR
            / f"{model_name}_output_scaler.pkl"
        ),
    }


# Data preparation

def load_test_dataset():
    path = DATASET_DIR / "test_dataset.csv"
    dataframe = pd.read_csv(path)

    required = STANDARD_INPUT_COLUMNS + OUTPUT_COLUMNS
    missing = [
        column for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"test_dataset.csv is missing columns: {missing}"
        )

    return dataframe.reset_index(drop=True)


def add_physics_features(dataframe):
    if all(
        column in dataframe.columns
        for column in PHYSICS_FEATURE_COLUMNS
    ):
        return dataframe.copy()

    feature_rows = []

    for _, row in dataframe.iterrows():
        feature_rows.append(
            calculate_physics_features(
                L=float(row["L"]),
                b=float(row["b"]),
                h=float(row["h"]),
                E=float(row["E"]),
                rho=float(row["rho"]),
                excitation_frequency=float(
                    row["excitation_frequency"]
                ),
            )
        )

    feature_dataframe = pd.DataFrame(
        feature_rows,
        index=dataframe.index,
    )

    return pd.concat(
        [
            dataframe.copy(),
            feature_dataframe[PHYSICS_FEATURE_COLUMNS],
        ],
        axis=1,
    )


# ---------------------------------------------------------------------
# Neural-network prediction

def predict(model_bundle, dataframe, input_columns):
    scaled_inputs = model_bundle[
        "input_scaler"
    ].transform(
        dataframe[input_columns].to_numpy(dtype=float)
    )

    input_tensor = torch.tensor(
        scaled_inputs,
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.no_grad():
        scaled_predictions = (
            model_bundle["model"](input_tensor)
            .cpu()
            .numpy()
        )

    return model_bundle[
        "output_scaler"
    ].inverse_transform(scaled_predictions)


# Accuracy

def calculate_accuracy(
    test_dataframe,
    standard_bundle,
    physics_bundle,
):
    actual = test_dataframe[
        OUTPUT_COLUMNS
    ].to_numpy(dtype=float)

    hybrid_test = add_physics_features(test_dataframe)

    predictions = {
        "standard_NN": predict(
            standard_bundle,
            test_dataframe,
            STANDARD_INPUT_COLUMNS,
        ),
        "physics_guided_NN": predict(
            physics_bundle,
            hybrid_test,
            HYBRID_INPUT_COLUMNS,
        ),
    }

    rows = []

    for model_name, predicted in predictions.items():
        for index, output_name in enumerate(OUTPUT_COLUMNS):
            y_true = actual[:, index]
            y_pred = predicted[:, index]

            rmse = np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
            output_range = y_true.max() - y_true.min()

            rows.append(
                {
                    "model": model_name,
                    "output": output_name,
                    "rmse": float(rmse),
                    "mae": float(
                        mean_absolute_error(
                            y_true,
                            y_pred,
                        )
                    ),
                    "r2": float(
                        r2_score(
                            y_true,
                            y_pred,
                        )
                    ),
                    "nrmse": float(
                        rmse / output_range
                        if output_range > 0
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# Physics simulator

def row_to_case(row):
    return {
        "L": float(row["L"]),
        "b": float(row["b"]),
        "h": float(row["h"]),
        "E": float(row["E"]),
        "rho": float(row["rho"]),
        "damping_ratio": float(
            row["damping_ratio"]
        ),
        "force_amplitude": float(
            row["force_amplitude"]
        ),
        "excitation_frequency": float(
            row["excitation_frequency"]
        ),
        "observation_point": (
            float(row["observation_point"])
            if "observation_point" in row.index
            else 0.65
        ),
        "simulation_time": (
            float(row["simulation_time"])
            if "simulation_time" in row.index
            else 2.0
        ),
        "time_step": (
            float(row["time_step"])
            if "time_step" in row.index
            else 0.002
        ),
        "number_of_modes": (
            int(round(row["number_of_modes"]))
            if "number_of_modes" in row.index
            else 4
        ),
    }


def run_simulator(case):
    beam = {
        "L": case["L"],
        "b": case["b"],
        "h": case["h"],
        "E": case["E"],
        "rho": case["rho"],
    }

    beam["A"] = beam["b"] * beam["h"]
    beam["I"] = beam["b"] * beam["h"] ** 3 / 12

    frequencies = natural_frequencies(
        beam,
        modes=case["number_of_modes"],
    )

    time_array = np.arange(
        0.0,
        case["simulation_time"] + case["time_step"],
        case["time_step"],
    )

    simulate_harmonic_beam_response(
        beam=beam,
        frequencies=frequencies,
        damping_ratio=case["damping_ratio"],
        force_amplitude=case["force_amplitude"],
        forcing_frequency=case[
            "excitation_frequency"
        ],
        t=time_array,
        x_position=case["observation_point"],
        number_of_modes=case["number_of_modes"],
    )


# Timing

def median_runtime(function, repeats):
    runtimes = []

    for _ in range(repeats):
        start = time.perf_counter()
        function()
        runtimes.append(
            time.perf_counter() - start
        )

    return float(np.median(runtimes))


def benchmark_models(
    test_dataframe,
    standard_bundle,
    physics_bundle,
):
    hybrid_test = add_physics_features(test_dataframe)

    predict(
        standard_bundle,
        test_dataframe.iloc[:1],
        STANDARD_INPUT_COLUMNS,
    )
    predict(
        physics_bundle,
        hybrid_test.iloc[:1],
        HYBRID_INPUT_COLUMNS,
    )

    standard_total_time = median_runtime(
        lambda: predict(
            standard_bundle,
            test_dataframe,
            STANDARD_INPUT_COLUMNS,
        ),
        TIMING_REPEATS,
    )

    physics_total_time = median_runtime(
        lambda: predict(
            physics_bundle,
            hybrid_test,
            HYBRID_INPUT_COLUMNS,
        ),
        TIMING_REPEATS,
    )

    number_of_test_cases = len(test_dataframe)

    simulator_cases = test_dataframe.iloc[
        : min(
            SIMULATOR_TIMING_CASES,
            number_of_test_cases,
        )
    ]

    simulator_case_times = []

    for _, row in simulator_cases.iterrows():
        case = row_to_case(row)

        start = time.perf_counter()
        run_simulator(case)
        simulator_case_times.append(
            time.perf_counter() - start
        )

    simulator_time_per_case = float(
        np.median(simulator_case_times)
    )

    rows = [
        {
            "model": "physics_simulator",
            "number_of_cases": len(simulator_cases),
            "median_total_time_seconds": (
                simulator_time_per_case
                * len(simulator_cases)
            ),
            "median_time_per_case_seconds": (
                simulator_time_per_case
            ),
            "throughput_cases_per_second": (
                1 / simulator_time_per_case
            ),
        },
        {
            "model": "standard_NN",
            "number_of_cases": number_of_test_cases,
            "median_total_time_seconds": (
                standard_total_time
            ),
            "median_time_per_case_seconds": (
                standard_total_time
                / number_of_test_cases
            ),
            "throughput_cases_per_second": (
                number_of_test_cases
                / standard_total_time
            ),
        },
        {
            "model": "physics_guided_NN",
            "number_of_cases": number_of_test_cases,
            "median_total_time_seconds": (
                physics_total_time
            ),
            "median_time_per_case_seconds": (
                physics_total_time
                / number_of_test_cases
            ),
            "throughput_cases_per_second": (
                number_of_test_cases
                / physics_total_time
            ),
        },
    ]

    timing = pd.DataFrame(rows)

    simulator_time = timing.loc[
        timing["model"] == "physics_simulator",
        "median_time_per_case_seconds",
    ].iloc[0]

    timing["speedup_over_simulator"] = (
        simulator_time
        / timing["median_time_per_case_seconds"]
    )

    return timing


# Training time and final comparison

def load_training_times():
    path = (
        PHASE_7_RESULTS_DIR
        / "phase_7_training_summary.csv"
    )

    if not path.exists():
        return pd.DataFrame(
            {
                "model": [
                    "standard_NN",
                    "physics_guided_NN",
                ],
                "training_time_seconds": [
                    np.nan,
                    np.nan,
                ],
            }
        )

    summary = pd.read_csv(path)

    return summary[
        [
            "model",
            "training_time_seconds",
        ]
    ]


def create_comparison_table(
    accuracy,
    timing,
    training_times,
):
    displacement_accuracy = (
        accuracy[
            accuracy["output"].isin(
                [
                    "peak_displacement",
                    "rms_displacement",
                ]
            )
        ]
        .groupby("model", as_index=False)
        .agg(
            mean_displacement_rmse=("rmse", "mean"),
            mean_displacement_nrmse=("nrmse", "mean"),
            mean_displacement_r2=("r2", "mean"),
        )
    )

    comparison = (
        timing[
            timing["model"].isin(
                [
                    "standard_NN",
                    "physics_guided_NN",
                ]
            )
        ]
        .merge(
            displacement_accuracy,
            on="model",
            how="left",
        )
        .merge(
            training_times,
            on="model",
            how="left",
        )
    )

    return comparison


# Plots

def plot_prediction_time(timing):
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        timing["model"],
        timing["median_time_per_case_seconds"],
    )

    axis.set_yscale("log")
    axis.set_ylabel(
        "Median prediction time per case (s)"
    )
    axis.set_title(
        "Physics simulator and surrogate prediction time"
    )
    axis.tick_params(
        axis="x",
        rotation=15,
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        PLOTS_DIR / "prediction_time_comparison.png",
        dpi=300,
    )
    plt.close(figure)


def plot_accuracy_efficiency(comparison):
    figure, axis = plt.subplots(figsize=(8, 5))

    for _, row in comparison.iterrows():
        axis.scatter(
            row["median_time_per_case_seconds"],
            row["mean_displacement_nrmse"],
            s=90,
        )

        axis.annotate(
            row["model"].replace("_", " "),
            (
                row["median_time_per_case_seconds"],
                row["mean_displacement_nrmse"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
        )

    axis.set_xscale("log")
    axis.set_xlabel(
        "Median prediction time per case (s)"
    )
    axis.set_ylabel(
        "Mean displacement NRMSE"
    )
    axis.set_title(
        "Accuracy-efficiency comparison"
    )
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        PLOTS_DIR / "accuracy_efficiency_comparison.png",
        dpi=300,
    )
    plt.close(figure)


# Main

def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading test dataset and Phase 7 models...")

    test = load_test_dataset()

    standard_bundle = load_model_bundle(
        "standard_NN",
        len(STANDARD_INPUT_COLUMNS),
    )

    physics_bundle = load_model_bundle(
        "physics_guided_NN",
        len(HYBRID_INPUT_COLUMNS),
    )

    print("Calculating test-set accuracy...")
    accuracy = calculate_accuracy(
        test,
        standard_bundle,
        physics_bundle,
    )

    print("Measuring prediction time...")
    timing = benchmark_models(
        test,
        standard_bundle,
        physics_bundle,
    )

    training_times = load_training_times()

    comparison = create_comparison_table(
        accuracy,
        timing,
        training_times,
    )

    accuracy.to_csv(
        RESULTS_DIR / "phase_8_accuracy_summary.csv",
        index=False,
    )

    timing.to_csv(
        RESULTS_DIR / "phase_8_timing_summary.csv",
        index=False,
    )

    comparison.to_csv(
        RESULTS_DIR
        / "phase_8_accuracy_efficiency_comparison.csv",
        index=False,
    )

    plot_prediction_time(timing)
    plot_accuracy_efficiency(comparison)

    print("Phase 8 complete.")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()