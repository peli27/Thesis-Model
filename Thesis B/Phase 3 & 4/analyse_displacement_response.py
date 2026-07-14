"""
Phase 4.2: Analysing the distribution of peak and RMS displacement responses.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ----- Paths -----

FULL_DATA_PATH = Path("dataset/full_dataset.csv")
TEST_DATA_PATH = Path("dataset/test_dataset.csv")
PREDICTIONS_PATH = Path("results/NN_predictions.csv")

RESULTS_DIR = Path("results/displacement_analysis")
PLOTS_DIR = Path("figures/displacement_analysis")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ----- Configuration -----

DISPLACEMENT_COLUMNS = [
    "peak_displacement",
    "rms_displacement",
]

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

HIGH_RESPONSE_PERCENTILE = 0.95


# ----- Load data -----

full_df = pd.read_csv(FULL_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
predictions_df = pd.read_csv(PREDICTIONS_PATH)

# Frequency ratio indicates proximity to first-mode resonance
full_df["frequency_ratio"] = (
    full_df["excitation_frequency"] / full_df["f1"]
)

test_df["frequency_ratio"] = (
    test_df["excitation_frequency"] / test_df["f1"]
)

ANALYSIS_INPUT_COLUMNS = INPUT_COLUMNS + ["frequency_ratio"]


# ----- Distribution statistics -----

statistics = []

for output in DISPLACEMENT_COLUMNS:

    values = full_df[output]

    statistics.append({
        "output": output,
        "mean": values.mean(),
        "std": values.std(),
        "minimum": values.min(),
        "25th_percentile": values.quantile(0.25),
        "median": values.median(),
        "75th_percentile": values.quantile(0.75),
        "90th_percentile": values.quantile(0.90),
        "95th_percentile": values.quantile(0.95),
        "99th_percentile": values.quantile(0.99),
        "maximum": values.max(),
        "skewness": values.skew(),
    })

statistics_df = pd.DataFrame(statistics)

statistics_df.to_csv(
    RESULTS_DIR / "displacement_statistics.csv",
    index=False
)

print("\n--- Displacement Statistics ---")
print(statistics_df)


# ----- Distribution plots -----

for output in DISPLACEMENT_COLUMNS:

    plt.figure(figsize=(8, 5))
    plt.hist(full_df[output], bins=50)

    plt.xlabel(output)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {output}")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / f"{output}_distribution.png",
        dpi=300
    )

    plt.close()


# ----- Analyse high-response cases -----

parameter_comparisons = []

for output in DISPLACEMENT_COLUMNS:

    threshold = full_df[output].quantile(
        HIGH_RESPONSE_PERCENTILE
    )

    high_response_df = full_df[
        full_df[output] >= threshold
    ]

    print(f"\n--- High {output} Cases ---")
    print(f"95th percentile threshold: {threshold:.6e}")
    print(f"Number of high-response samples: {len(high_response_df)}")

    high_response_df.to_csv(
        RESULTS_DIR / f"high_{output}_samples.csv",
        index=False
    )

    for parameter in ANALYSIS_INPUT_COLUMNS:

        overall_mean = full_df[parameter].mean()
        high_response_mean = high_response_df[parameter].mean()

        parameter_comparisons.append({
            "output": output,
            "parameter": parameter,
            "overall_mean": overall_mean,
            "high_response_mean": high_response_mean,
            "percentage_difference": (
                (high_response_mean - overall_mean)
                / overall_mean
                * 100
            ),
        })


parameter_comparison_df = pd.DataFrame(parameter_comparisons)

parameter_comparison_df.to_csv(
    RESULTS_DIR / "high_response_parameter_comparison.csv",
    index=False
)

print("\n--- High Response Parameter Comparison ---")
print(parameter_comparison_df)


# ----- Link prediction errors to test inputs -----

if len(test_df) != len(predictions_df):
    raise ValueError(
        "Test dataset and prediction file have different row counts."
    )

analysis_df = pd.concat(
    [
        test_df.reset_index(drop=True),
        predictions_df.reset_index(drop=True),
    ],
    axis=1,
)


for output in DISPLACEMENT_COLUMNS:

    actual_column = f"actual_{output}"
    predicted_column = f"predicted_{output}"
    error_column = f"error_{output}"

    high_threshold = analysis_df[
        actual_column
    ].quantile(HIGH_RESPONSE_PERCENTILE)

    high_response_errors = analysis_df[
        (analysis_df[actual_column] >= high_threshold)
        & (analysis_df[error_column] > 0)
    ].copy()

    high_response_errors = high_response_errors.sort_values(
        error_column,
        ascending=False
    )

    high_response_errors.to_csv(
        RESULTS_DIR / f"underpredicted_high_{output}.csv",
        index=False
    )

    print(f"\n--- Underpredicted High {output} Cases ---")

    columns_to_print = (
        ANALYSIS_INPUT_COLUMNS
        + [
            actual_column,
            predicted_column,
            error_column,
        ]
    )

    print(
        high_response_errors[
            columns_to_print
        ].head(10)
    )