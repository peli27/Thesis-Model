"""
Experiment 1: Random parametric error dataset generation.

This script applies random errors to Young's modulus E, density rho and beam
thickness h. The beam frequencies, displacement responses and physics-guided
features are then recalculated.

Training and validation uncertainty: +/-2.5%
Test uncertainty levels: 0%, +/-1%, +/-2.5%, +/-5% and +/-10%

Two test versions are created:

1. Known perturbation: The model receives the perturbed E, rho and h values.

2. Hidden perturbation: The model receives the original E, rho and h values, 
    while the target responses come from the perturbed beam.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----- Project directories -----

PHASE_6_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_6_DIR.parent

PHYSICS_DIR = PROJECT_ROOT / "Phase 1 & 2"
PHASE_5_DIR = PROJECT_ROOT / "Phase 5"
DATASET_DIR = PROJECT_ROOT / "dataset"

OUTPUT_DIR = PHASE_6_DIR / "Phase_6_datasets"

sys.path.insert(0, str(PHYSICS_DIR))
sys.path.insert(0, str(PHASE_5_DIR))

from frequency_analysis import natural_frequencies
from multi_mode_response import simulate_harmonic_beam_response
from physics_features import calculate_physics_features


# ----- Experiment configuration -----

RANDOM_SEED = 42
TRAIN_UNCERTAINTY = 0.025
TEST_UNCERTAINTY_LEVELS = (0.0, 0.01, 0.025, 0.05, 0.10)

PHYSICS_FEATURE_COLUMNS = [
    "first_natural_frequency",
    "modal_stiffness",
    "modal_mass",
    "frequency_ratio",
    "resonance_proximity",
]

OUTPUT_COLUMNS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "peak_displacement",
    "rms_displacement",
]

REQUIRED_COLUMNS = [
    "L",
    "b",
    "h",
    "E",
    "rho",
    "damping_ratio",
    "force_amplitude",
    "excitation_frequency",
    "observation_point",
    "simulation_time",
    "time_step",
    "number_of_modes",
] + OUTPUT_COLUMNS


# ----- Dataset loading -----

def load_source_dataset(filename):
    """Loads one of the original train, validation or test datasets."""

    path = DATASET_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{filename} is missing these columns: {missing_columns}"
        )

    return dataframe.reset_index(drop=True)


def generate_error_directions(number_of_samples, random_seed):
    """
    Generates one random error direction for E, rho and h for every sample.

    Each number is between -1 and 1. It is multiplied by the selected
    uncertainty level when the beam sample is created.
    """

    rng = np.random.default_rng(random_seed)

    error_directions = rng.uniform(
        -1.0,
        1.0,
        size=(number_of_samples, 3),
    )

    return error_directions


def simulate_perturbed_sample(
    original_row,
    sample_id,
    source_split,
    uncertainty_level,
    error_direction,
):
    """
    Perturbs one original beam sample and returns known and hidden versions.
    """

    # Calculate the random errors applied to this beam.
    error_E = uncertainty_level * error_direction[0]
    error_rho = uncertainty_level * error_direction[1]
    error_h = uncertainty_level * error_direction[2]

    # Original beam values.
    E_nominal = float(original_row["E"])
    rho_nominal = float(original_row["rho"])
    h_nominal = float(original_row["h"])

    # Perturbed beam values.
    E_perturbed = E_nominal * (1.0 + error_E)
    rho_perturbed = rho_nominal * (1.0 + error_rho)
    h_perturbed = h_nominal * (1.0 + error_h)

    # Build the perturbed beam required by the simulator.
    perturbed_beam = {
        "L": float(original_row["L"]),
        "b": float(original_row["b"]),
        "h": h_perturbed,
        "E": E_perturbed,
        "rho": rho_perturbed,
    }

    perturbed_beam["A"] = perturbed_beam["b"] * perturbed_beam["h"]
    perturbed_beam["I"] = (
        perturbed_beam["b"] * perturbed_beam["h"] ** 3 / 12
    )

    # Recalculate the natural frequencies of the perturbed beam.
    number_of_modes = int(original_row["number_of_modes"])

    frequencies = natural_frequencies(
        perturbed_beam,
        modes=number_of_modes,
    )

    # Recalculate the displacement response of the perturbed beam.
    simulation_time = float(original_row["simulation_time"])
    time_step = float(original_row["time_step"])

    time_array = np.arange(
        0.0,
        simulation_time + time_step,
        time_step,
    )

    response = simulate_harmonic_beam_response(
        beam=perturbed_beam,
        frequencies=frequencies,
        damping_ratio=float(original_row["damping_ratio"]),
        force_amplitude=float(original_row["force_amplitude"]),
        forcing_frequency=float(original_row["excitation_frequency"]),
        t=time_array,
        x_position=float(original_row["observation_point"]),
        number_of_modes=number_of_modes,
    )

    # Physics features calculated from the perturbed values.
    perturbed_physics_features = calculate_physics_features(
        L=float(original_row["L"]),
        b=float(original_row["b"]),
        h=h_perturbed,
        E=E_perturbed,
        rho=rho_perturbed,
        excitation_frequency=float(original_row["excitation_frequency"]),
    )

    # Physics features calculated from the original values.
    nominal_physics_features = calculate_physics_features(
        L=float(original_row["L"]),
        b=float(original_row["b"]),
        h=h_nominal,
        E=E_nominal,
        rho=rho_nominal,
        excitation_frequency=float(original_row["excitation_frequency"]),
    )

    # Information saved in both versions for checking the experiment.
    experiment_information = {
        "sample_id": sample_id,
        "source_split": source_split,
        "uncertainty_level": uncertainty_level,
        "error_E": error_E,
        "error_rho": error_rho,
        "error_h": error_h,
        "E_nominal": E_nominal,
        "rho_nominal": rho_nominal,
        "h_nominal": h_nominal,
        "E_perturbed": E_perturbed,
        "rho_perturbed": rho_perturbed,
        "h_perturbed": h_perturbed,
    }

    # Responses calculated from the perturbed beam.
    perturbed_outputs = {
        "f1": float(frequencies[0]),
        "f2": float(frequencies[1]),
        "f3": float(frequencies[2]),
        "f4": float(frequencies[3]),
        "peak_displacement": float(response["peak_displacement"]),
        "rms_displacement": float(response["rms_displacement"]),
    }

    # Known version: the model sees the perturbed beam properties.
    known_sample = original_row.to_dict()
    known_sample.update(experiment_information)
    known_sample.update(
        {
            "E": E_perturbed,
            "rho": rho_perturbed,
            "h": h_perturbed,
        }
    )
    known_sample.update(perturbed_outputs)
    known_sample.update(perturbed_physics_features)

    # Hidden version: the model still sees the original beam properties.
    # The correct outputs nevertheless come from the perturbed physical beam.
    hidden_sample = original_row.to_dict()
    hidden_sample.update(experiment_information)
    hidden_sample.update(perturbed_outputs)
    hidden_sample.update(nominal_physics_features)

    return known_sample, hidden_sample


def generate_uncertainty_dataset(
    original_dataframe,
    source_split,
    uncertainty_level,
    error_directions,
):
    """Generates the known and hidden datasets for one uncertainty level."""

    if len(original_dataframe) != len(error_directions):
        raise ValueError(
            "The dataset and random error arrays must have the same length."
        )

    known_data = []
    hidden_data = []

    for index, row in original_dataframe.iterrows():
        known_sample, hidden_sample = simulate_perturbed_sample(
            original_row=row,
            sample_id=index,
            source_split=source_split,
            uncertainty_level=uncertainty_level,
            error_direction=error_directions[index],
        )

        known_data.append(known_sample)
        hidden_data.append(hidden_sample)

        if (index + 1) % 100 == 0 or index + 1 == len(original_dataframe):
            print(
                f"Generated {index + 1} {source_split} samples "
                f"at +/-{uncertainty_level * 100:g}%"
            )

    known_dataframe = pd.DataFrame.from_records(known_data)
    hidden_dataframe = pd.DataFrame.from_records(hidden_data)

    return known_dataframe, hidden_dataframe


# ----- Dataset saving -----

def save_dataset(dataframe, filename):
    """Saves one dataset in the Phase 6 output folder."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename
    dataframe.to_csv(output_path, index=False)

    print(f"Saved: {output_path} ({len(dataframe)} samples)")

    return output_path


def save_raw_and_hybrid_datasets(dataframe, dataset_name):
    """Saves raw and physics-guided versions of the same dataset."""

    raw_dataframe = dataframe.drop(
        columns=PHYSICS_FEATURE_COLUMNS,
        errors="ignore",
    )

    save_dataset(
        raw_dataframe,
        f"{dataset_name}.csv",
    )

    save_dataset(
        dataframe,
        f"{dataset_name}_hybrid.csv",
    )


# ----- 0% uncertainty check -----

def check_zero_uncertainty(original_dataframe, zero_uncertainty_dataframe):
    """Checks that the 0% responses match the original test responses."""

    outputs_match = True

    for column in OUTPUT_COLUMNS:
        if not np.allclose(
            original_dataframe[column],
            zero_uncertainty_dataframe[column],
        ):
            outputs_match = False

    if outputs_match:
        print("0% uncertainty check passed.")
    else:
        print("Warning: 0% uncertainty responses do not match the original data.")


# ----- Run Experiment 1 -----

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the original datasets.
    train_dataframe = load_source_dataset("train_dataset.csv")
    validation_dataframe = load_source_dataset("validation_dataset.csv")
    test_dataframe = load_source_dataset("test_dataset.csv")

    # Generate separate random errors for training and validation.
    train_error_directions = generate_error_directions(
        len(train_dataframe),
        RANDOM_SEED,
    )

    validation_error_directions = generate_error_directions(
        len(validation_dataframe),
        RANDOM_SEED + 1,
    )

    # Create known-perturbation training data at +/-2.5%.
    train_known, _ = generate_uncertainty_dataset(
        train_dataframe,
        source_split="train",
        uncertainty_level=TRAIN_UNCERTAINTY,
        error_directions=train_error_directions,
    )

    validation_known, _ = generate_uncertainty_dataset(
        validation_dataframe,
        source_split="validation",
        uncertainty_level=TRAIN_UNCERTAINTY,
        error_directions=validation_error_directions,
    )

    train_level = f"{TRAIN_UNCERTAINTY * 100:g}".replace(".", "p")

    save_raw_and_hybrid_datasets(
        train_known,
        f"train_known_{train_level}pct",
    )

    save_raw_and_hybrid_datasets(
        validation_known,
        f"validation_known_{train_level}pct",
    )

    # Generate the test error directions once. Reusing them at every level
    # creates the paired uncertainty comparison.
    test_error_directions = generate_error_directions(
        len(test_dataframe),
        RANDOM_SEED + 2,
    )

    # Create known and hidden test datasets at every uncertainty level.
    for uncertainty_level in TEST_UNCERTAINTY_LEVELS:

        test_known, test_hidden = generate_uncertainty_dataset(
            test_dataframe,
            source_split="test",
            uncertainty_level=uncertainty_level,
            error_directions=test_error_directions,
        )

        level = f"{uncertainty_level * 100:g}".replace(".", "p")

        save_raw_and_hybrid_datasets(
            test_known,
            f"test_known_{level}pct",
        )

        save_raw_and_hybrid_datasets(
            test_hidden,
            f"test_hidden_{level}pct",
        )

        if np.isclose(uncertainty_level, 0.0):
            check_zero_uncertainty(test_dataframe, test_known)

    print("\nExperiment 1 uncertainty datasets generated successfully.")
    print(f"Files saved in: {OUTPUT_DIR}")