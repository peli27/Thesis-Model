"""
Experiment 2: unseen-resonance dataset generation

The standard and physics-guided neural networks will be trained and validated
using only off-resonance cases:

    0.20 <= excitation_frequency / f1 <= 0.80
    1.20 <= excitation_frequency / f1 <= 2.00

The test datasets reuse the same beam cases at fixed frequency ratios on both
sides of first-mode resonance. Ratios between 0.80 and 1.20 are unseen during training.

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
SOURCE_DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PHASE_6_DIR / "Experiment_2_datasets"

sys.path.insert(0, str(PHYSICS_DIR))
sys.path.insert(0, str(PHASE_5_DIR))

from frequency_analysis import natural_frequencies
from multi_mode_response import simulate_harmonic_beam_response
from physics_features import calculate_physics_features


# ----- Experiment settings -----

RANDOM_SEED = 42

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

PHYSICS_FEATURE_COLUMNS = [
    "first_natural_frequency",
    "modal_stiffness",
    "modal_mass",
    "frequency_ratio",
    "resonance_proximity",
]


# ----- Dataset helpers -----

def load_dataset(filename):
    return pd.read_csv(
        SOURCE_DATASET_DIR / filename
    ).reset_index(drop=True)


def generate_off_resonance_ratios(number_of_samples, random_seed):
    """
    Generates ratios from 0.20 to 0.80 and 1.20 to 2.00.

    Ratios between 0.80 and 1.20 are excluded.
    """

    rng = np.random.default_rng(random_seed)
    ratios = rng.uniform(0.20, 1.60, number_of_samples)

    # Move values above 0.80 past the excluded resonance range.
    ratios[ratios > 0.80] += 0.40

    return ratios


def ratio_label(frequency_ratio):
    return f"{frequency_ratio:.2f}".replace(".", "p")


def build_beam(row):
    beam = {
        "L": float(row["L"]),
        "b": float(row["b"]),
        "h": float(row["h"]),
        "E": float(row["E"]),
        "rho": float(row["rho"]),
    }

    beam["A"] = beam["b"] * beam["h"]
    beam["I"] = beam["b"] * beam["h"] ** 3 / 12

    return beam


def simulate_sample(row, sample_id, frequency_ratio):
    beam = build_beam(row)
    number_of_modes = int(row["number_of_modes"])

    frequencies = natural_frequencies(
        beam,
        modes=number_of_modes,
    )

    excitation_frequency = frequency_ratio * frequencies[0]

    time_array = np.arange(
        0.0,
        float(row["simulation_time"]) + float(row["time_step"]),
        float(row["time_step"]),
    )

    response = simulate_harmonic_beam_response(
        beam=beam,
        frequencies=frequencies,
        damping_ratio=float(row["damping_ratio"]),
        force_amplitude=float(row["force_amplitude"]),
        forcing_frequency=excitation_frequency,
        t=time_array,
        x_position=float(row["observation_point"]),
        number_of_modes=number_of_modes,
    )

    physics_features = calculate_physics_features(
        L=beam["L"],
        b=beam["b"],
        h=beam["h"],
        E=beam["E"],
        rho=beam["rho"],
        excitation_frequency=excitation_frequency,
    )

    sample = row.to_dict()

    sample.update(
        {
            "sample_id": sample_id,
            "excitation_frequency": excitation_frequency,
            "f1": frequencies[0],
            "f2": frequencies[1],
            "f3": frequencies[2],
            "f4": frequencies[3],
            "peak_displacement": response["peak_displacement"],
            "rms_displacement": response["rms_displacement"],
        }
    )

    sample.update(physics_features)

    return sample


def generate_dataset(source_dataframe, frequency_ratios, dataset_name):
    records = []

    for index, row in source_dataframe.iterrows():
        records.append(
            simulate_sample(
                row=row,
                sample_id=index,
                frequency_ratio=frequency_ratios[index],
            )
        )

        if (index + 1) % 100 == 0 or index + 1 == len(source_dataframe):
            print(
                f"Generated {index + 1}/{len(source_dataframe)} "
                f"{dataset_name} samples"
            )

    return pd.DataFrame(records)


def save_raw_and_hybrid(dataframe, dataset_name):
    raw_dataframe = dataframe.drop(
        columns=PHYSICS_FEATURE_COLUMNS
    )

    raw_path = OUTPUT_DIR / f"{dataset_name}.csv"
    hybrid_path = OUTPUT_DIR / f"{dataset_name}_hybrid.csv"

    raw_dataframe.to_csv(raw_path, index=False)
    dataframe.to_csv(hybrid_path, index=False)

    print(f"Saved: {raw_path}")
    print(f"Saved: {hybrid_path}")


# ----- Generate datasets -----

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_source = load_dataset("train_dataset.csv")
    validation_source = load_dataset("validation_dataset.csv")
    test_source = load_dataset("test_dataset.csv")

    train_ratios = generate_off_resonance_ratios(
        len(train_source),
        RANDOM_SEED,
    )
    validation_ratios = generate_off_resonance_ratios(
        len(validation_source),
        RANDOM_SEED + 1,
    )

    print("\nGenerating off-resonance training data...")
    train_dataframe = generate_dataset(
        train_source,
        train_ratios,
        "training",
    )
    save_raw_and_hybrid(
        train_dataframe,
        "train_off_resonance",
    )

    print("\nGenerating off-resonance validation data...")
    validation_dataframe = generate_dataset(
        validation_source,
        validation_ratios,
        "validation",
    )
    save_raw_and_hybrid(
        validation_dataframe,
        "validation_off_resonance",
    )

    for test_ratio in TEST_FREQUENCY_RATIOS:
        print(f"\nGenerating test data at r = {test_ratio:.2f}...")

        test_ratios = np.full(
            len(test_source),
            test_ratio,
        )

        test_dataframe = generate_dataset(
            test_source,
            test_ratios,
            f"test r={test_ratio:.2f}",
        )

        save_raw_and_hybrid(
            test_dataframe,
            f"test_ratio_{ratio_label(test_ratio)}",
        )

    print("\nExperiment 2 dataset generation complete.")