"""
Checks the nonlinear beam dataset before ML training.

Checks:
1. dataset size and required columns
2. missing and infinite values
3. duplicate rows
4. parameter/output ranges
5. resonance-region coverage
6. displacement-response distributions
7. extreme-response cases
8. input-output correlations

"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- PROJECT PATHS -----

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MAIN_DATA_DIR = DATA_DIR / "main"
ANALYSIS_DIR = DATA_DIR / "analysis"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILE = MAIN_DATA_DIR / "nonlinear_beam_dataset.csv"

DATASET_SUMMARY_FILE = ANALYSIS_DIR / "dataset_summary.csv"
LARGEST_CASES_FILE = ANALYSIS_DIR / "largest_response_cases.csv"

EXPECTED_SAMPLES = 5000

REQUIRED_COLUMNS = [
    "L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency",
    "frequency_ratio", "f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"
]

INPUT_COLUMNS = [
    "L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"
]

OUTPUT_COLUMNS = [
    "f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"
]

print("NONLINEAR BEAM DATASET AUDIT")

dataset = pd.read_csv(DATASET_FILE)

## ----- CHECK 1: DATASET STRUCTURE -----

print("\n========================================")
print("CHECK 1: DATASET STRUCTURE")

print("Rows:", len(dataset))
print("Columns:", len(dataset.columns))

missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]

if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

if len(dataset) != EXPECTED_SAMPLES:
    print(f"WARNING: Expected {EXPECTED_SAMPLES} samples but found {len(dataset)}")
else:
    print("Sample count is correct.")

## ----- CHECK 2: INVALID VALUES -----

print("\n========================================")
print("CHECK 2: INVALID VALUES")

missing_values = int(dataset[REQUIRED_COLUMNS].isnull().sum().sum())
numeric_values = dataset[REQUIRED_COLUMNS].to_numpy(dtype=float)
infinite_values = int(np.isinf(numeric_values).sum())
duplicate_rows = int(dataset.duplicated().sum())

print("Missing values:", missing_values)
print("Infinite values:", infinite_values)
print("Duplicate rows:", duplicate_rows)

if missing_values > 0 or infinite_values > 0:
    raise ValueError("Dataset contains invalid values and should not be used for ML training.")

## ----- CHECK 3: PHYSICAL SANITY CHECKS -----

print("\n========================================")
print("CHECK 3: PHYSICAL SANITY CHECKS")

non_positive_counts = {}

for column in REQUIRED_COLUMNS:
    non_positive_counts[column] = int((dataset[column] <= 0).sum())

for column, count in non_positive_counts.items():
    if count > 0:
        print(f"{column}: {count} non-positive values")

if sum(non_positive_counts.values()) == 0:
    print("All physical quantities are positive.")

frequency_order_ok = (
    (dataset["f1"] < dataset["f2"])
    & (dataset["f2"] < dataset["f3"])
    & (dataset["f3"] < dataset["f4"])
)

rms_peak_ok = dataset["rms_displacement"] <= dataset["peak_displacement"]

print("Rows with ordered frequencies f1 < f2 < f3 < f4:", int(frequency_order_ok.sum()), "/", len(dataset))
print("Rows with RMS <= peak displacement:", int(rms_peak_ok.sum()), "/", len(dataset))

if not frequency_order_ok.all():
    print("WARNING: Some samples do not have ordered natural frequencies.")

if not rms_peak_ok.all():
    print("WARNING: Some samples have RMS displacement greater than peak displacement.")

## ----- CHECK 4: FREQUENCY-RATIO CONSISTENCY -----

print("\n========================================")
print("CHECK 4: FREQUENCY-RATIO CONSISTENCY")

calculated_ratio = dataset["excitation_frequency"] / dataset["f1"]
ratio_error = np.max(np.abs(calculated_ratio - dataset["frequency_ratio"]))

print("Maximum stored frequency-ratio error:", ratio_error)

## ----- CHECK 5: PARAMETER AND OUTPUT RANGES -----

print("\n========================================")
print("CHECK 5: PARAMETER AND OUTPUT RANGES")

range_columns = INPUT_COLUMNS + ["frequency_ratio"] + OUTPUT_COLUMNS

for column in range_columns:
    print(f"{column}: {dataset[column].min():.6e} to {dataset[column].max():.6e}")

## ----- CHECK 6: RESONANCE-REGION COVERAGE -----

print("\n========================================")
print("CHECK 6: RESONANCE-REGION COVERAGE")

def classify_regime(ratio):
    if ratio < 0.85:
        return "Below resonance"
    elif ratio <= 1.20:
        return "Near resonance"
    return "Above resonance"

dataset["regime"] = dataset["frequency_ratio"].apply(classify_regime)

regime_counts = dataset["regime"].value_counts()
regime_percentages = dataset["regime"].value_counts(normalize=True) * 100

for regime in ["Below resonance", "Near resonance", "Above resonance"]:
    count = int(regime_counts.get(regime, 0))
    percentage = float(regime_percentages.get(regime, 0.0))
    print(f"{regime}: {count} samples ({percentage:.1f} %)")

## ----- CHECK 7: RESPONSE DISTRIBUTIONS -----

print("\n========================================")
print("CHECK 7: RESPONSE DISTRIBUTIONS")

for column in ["peak_displacement", "rms_displacement"]:
    values = dataset[column]
    print(f"\n{column}:")
    print(f"Minimum = {values.min():.6e}")
    print(f"25th percentile = {values.quantile(0.25):.6e}")
    print(f"Median = {values.median():.6e}")
    print(f"75th percentile = {values.quantile(0.75):.6e}")
    print(f"95th percentile = {values.quantile(0.95):.6e}")
    print(f"Maximum = {values.max():.6e}")

## ----- CHECK 8: EXTREME RESPONSE CASES -----

print("\n========================================")
print("CHECK 8: EXTREME RESPONSE CASES")

largest_cases = dataset.nlargest(10, "peak_displacement")[
    [
        "L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency",
        "frequency_ratio", "peak_displacement", "rms_displacement"
    ]
]

print("\n10 largest peak-displacement cases:")
print(largest_cases.to_string(index=False))

## ----- CHECK 9: INPUT-OUTPUT CORRELATIONS -----

print("\n========================================")
print("CHECK 9: INPUT-OUTPUT CORRELATIONS")

correlation_columns = INPUT_COLUMNS + ["frequency_ratio", "peak_displacement", "rms_displacement"]
correlations = dataset[correlation_columns].corr()

print("\nCorrelation with peak displacement:")
print(correlations["peak_displacement"].drop(["peak_displacement", "rms_displacement"]).sort_values(ascending=False))

print("\nCorrelation with RMS displacement:")
print(correlations["rms_displacement"].drop(["peak_displacement", "rms_displacement"]).sort_values(ascending=False))

summary = dataset[range_columns].describe().T
summary.to_csv(DATASET_SUMMARY_FILE)
largest_cases.to_csv(LARGEST_CASES_FILE, index=False)

## ----- PLOTS -----

plt.figure(figsize=(8, 5))
plt.hist(dataset["frequency_ratio"], bins=40)
plt.xlabel("Excitation frequency / first natural frequency")
plt.ylabel("Number of samples")
plt.title("Frequency-Ratio Distribution")
plt.grid()

plt.figure(figsize=(8, 5))
plt.hist(dataset["peak_displacement"], bins=50)
plt.xlabel("Peak displacement [m]")
plt.ylabel("Number of samples")
plt.title("Peak-Displacement Distribution")
plt.grid()

plt.figure(figsize=(8, 5))
plt.scatter(dataset["frequency_ratio"], dataset["peak_displacement"], s=8, alpha=0.5)
plt.xlabel("Excitation frequency / first natural frequency")
plt.ylabel("Peak displacement [m]")
plt.title("Peak Response Across Frequency Regimes")
plt.grid()

plt.figure(figsize=(8, 5))
plt.scatter(dataset["pressure_amplitude"], dataset["peak_displacement"], s=8, alpha=0.5)
plt.xlabel("Pressure amplitude [Pa]")
plt.ylabel("Peak displacement [m]")
plt.title("Peak Response vs Forcing Amplitude")
plt.grid()

plt.show()

print("DATASET CHECK COMPLETE")