"""
Generate physics-guided datasets for the hybrid neural network.

1. Loads the existing train, validation and test datasets.
2. Calculates physics-derived features for every beam sample.
3. Adds the new features as dataset columns.
4. Saves new hybrid datasets.
"""

from pathlib import Path

import pandas as pd

from physics_features import calculate_physics_features


# ----- File paths -----

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "dataset"

HYBRID_DATASET_DIR = DATASET_DIR / "hybrid"

HYBRID_DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATASET_FILES = {
    "train": DATASET_DIR / "train_dataset.csv",
    "validation": DATASET_DIR / "validation_dataset.csv",
    "test": DATASET_DIR / "test_dataset.csv",
}


REQUIRED_COLUMNS = [
    "L",
    "b",
    "h",
    "E",
    "rho",
    "excitation_frequency",
]

# ----- Add physics-derived features -----

def add_physics_features(df):
    """
    Calculates and adds physics-derived features to a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Original beam dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset containing the original columns and the new physics-guided feature columns.
    """

    # Check all columns exist
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )

    physics_features = []

    for _, row in df.iterrows():

        features = calculate_physics_features(
            L=row["L"],
            b=row["b"],
            h=row["h"],
            E=row["E"],
            rho=row["rho"],
            excitation_frequency=row["excitation_frequency"],
        )
        physics_features.append(features)

    physics_features_df = pd.DataFrame(
        physics_features
    )

    hybrid_df = pd.concat([df.reset_index(drop=True), physics_features_df.reset_index(drop=True)], axis=1)

    return hybrid_df


# ----- Process one dataset -----

def process_dataset(
    dataset_name,
    input_path,
):
    """
    Loads one dataset, adds physics features and saves the hybrid dataset.
    """

    print(
        f"\nProcessing {dataset_name} dataset..."
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    print(
        f"Original dataset shape: {df.shape}"
    )

    hybrid_df = add_physics_features(df)

    print(
        f"Hybrid dataset shape: {hybrid_df.shape}"
    )

    output_path = (
        HYBRID_DATASET_DIR
        / f"{dataset_name}_hybrid_dataset.csv"
    )

    hybrid_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path}"
    )


# ----- Main script -----

if __name__ == "__main__":

    for dataset_name, input_path in DATASET_FILES.items():

        process_dataset(
            dataset_name=dataset_name,
            input_path=input_path,
        )

    print(
        "\nPhysics-guided dataset generation complete."
    )