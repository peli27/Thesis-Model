"""
Experiment 3: reduced-training-data dataset preparation.

This script creates matched raw and physics-guided training subsets at: 5%, 10%, 25%, 50%, 75%, and 100%

Three repeated random orderings are used by default. Within each repeat, the
subsets are nested: every row in the 5% subset is also in the 10% subset, and
so on. The validation and test datasets remain fixed for every model run.

"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----- Project directories -----

PHASE_6_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_6_DIR.parent

SOURCE_DATASET_DIR = PROJECT_ROOT / "dataset"
PHASE_5_DIR = PROJECT_ROOT / "Phase 5"
OUTPUT_DIR = PHASE_6_DIR / "Experiment_3_datasets"

sys.path.insert(0, str(PHASE_5_DIR))

from physics_features import calculate_physics_features


# ----- Experiment settings -----

RANDOM_SEED = 42
NUMBER_OF_REPEATS = 3
TRAINING_PERCENTAGES = [5, 10, 25, 50, 75, 100]

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

OUTPUT_COLUMNS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "peak_displacement",
    "rms_displacement",
]

REQUIRED_COLUMNS = STANDARD_INPUT_COLUMNS + OUTPUT_COLUMNS


# ----- Dataset helpers -----

def load_source_dataset(filename):
    return pd.read_csv(
        SOURCE_DATASET_DIR / filename
    ).reset_index(drop=True)


def add_physics_features(dataframe):
    physics_dataframe = pd.DataFrame([
        calculate_physics_features(
            L=row.L,
            b=row.b,
            h=row.h,
            E=row.E,
            rho=row.rho,
            excitation_frequency=row.excitation_frequency,
        )
        for row in dataframe.itertuples(index=False)
    ])

    return pd.concat(
        [dataframe.reset_index(drop=True), physics_dataframe],
        axis=1,
    )


def save_dataset(dataframe, filename):
    path = OUTPUT_DIR / filename
    dataframe.to_csv(path, index=False)
    print(f"Saved: {path} ({len(dataframe)} samples)")


def create_nested_subsets(raw_dataframe, hybrid_dataframe, repeat_number):
    rng = np.random.default_rng(RANDOM_SEED + repeat_number - 1)
    shuffled_indices = rng.permutation(len(raw_dataframe))

    for percentage in TRAINING_PERCENTAGES:
        subset_size = max(
            1,
            round(len(raw_dataframe) * percentage / 100),
        )
        selected_indices = shuffled_indices[:subset_size]

        raw_subset = raw_dataframe.iloc[selected_indices].reset_index(drop=True)
        hybrid_subset = hybrid_dataframe.iloc[selected_indices].reset_index(
            drop=True
        )

        filename = f"train_{percentage}pct_repeat_{repeat_number}"

        save_dataset(raw_subset, f"{filename}.csv")
        save_dataset(hybrid_subset, f"{filename}_hybrid.csv")


# ----- Main experiment preparation -----

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPreparing Experiment 3 datasets...")

    train_dataframe = load_source_dataset("train_dataset.csv")
    validation_dataframe = load_source_dataset("validation_dataset.csv")
    test_dataframe = load_source_dataset("test_dataset.csv")

    print(f"Training samples: {len(train_dataframe)}")
    print(f"Validation samples: {len(validation_dataframe)}")
    print(f"Test samples: {len(test_dataframe)}")

    train_hybrid_dataframe = add_physics_features(train_dataframe)
    validation_hybrid_dataframe = add_physics_features(validation_dataframe)
    test_hybrid_dataframe = add_physics_features(test_dataframe)

    save_dataset(validation_dataframe, "validation_fixed.csv")
    save_dataset(
        validation_hybrid_dataframe,
        "validation_fixed_hybrid.csv",
    )
    save_dataset(test_dataframe, "test_fixed.csv")
    save_dataset(
        test_hybrid_dataframe,
        "test_fixed_hybrid.csv",
    )

    for repeat_number in range(1, NUMBER_OF_REPEATS + 1):
        print(f"\nCreating repeat {repeat_number}...")

        create_nested_subsets(
            train_dataframe,
            train_hybrid_dataframe,
            repeat_number,
        )