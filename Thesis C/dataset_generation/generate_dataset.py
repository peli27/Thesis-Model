"""
Generating the nonlinear beam dataset used to train the baseline ML surrogate.

Inputs:
L, h, E, rho, zeta, pressure amplitude and excitation frequency.

Outputs:
first four natural frequencies, peak displacement and RMS displacement.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ----- PROJECT PATHS -----

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from beam_model.nonlinear_beam import simulate_beam

DATA_DIR = PROJECT_ROOT / "data"
MAIN_DATA_DIR = DATA_DIR / "main"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

MAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ----- DATASET SETTINGS -----

N_SAMPLES = 5000
SEED = 42

# Fixed beam width
b = 0.04

# Number of modes
N_MODES = 7

# Simulation settings
T_END = 10.0
DT = 0.002

# Save locations
OUTPUT_FILE = MAIN_DATA_DIR / "nonlinear_beam_dataset.csv"
CHECKPOINT_FILE = CHECKPOINT_DIR / "nonlinear_beam_dataset_checkpoint.csv"

CHECKPOINT_INTERVAL = 100


# ----- PARAMETER RANGES -----

L_range = (1.6, 2.4)                 # m
h_range = (0.008, 0.012)             # m
E_range = (185e9, 225e9)             # Pa
rho_range = (7450, 8250)             # kg/m^3
zeta_range = (0.005, 0.020)
pressure_range = (0.1, 30.0)         # Pa


# ----- CALCULATE FIRST NATURAL FREQUENCY -----
def calculate_f1(L, h, E, rho):

    A = b * h
    I = b * h**3 / 12
    k1 = np.pi / L

    modal_mass = rho * (A + I * k1**2) * L / 2
    modal_stiffness = E * I * k1**4 * L / 2

    omega_1 = np.sqrt(modal_stiffness / modal_mass)

    return omega_1 / (2 * np.pi)


# ----- LATIN HYPERCUBE SAMPLING -----

print("GENERATING PARAMETER SAMPLES")

sampler = qmc.LatinHypercube(d=6, seed=SEED)
lhs_samples = sampler.random(n=N_SAMPLES)

lower_bounds = np.array([
    L_range[0],
    h_range[0],
    E_range[0],
    rho_range[0],
    zeta_range[0],
    pressure_range[0]
])

upper_bounds = np.array([
    L_range[1],
    h_range[1],
    E_range[1],
    rho_range[1],
    zeta_range[1],
    pressure_range[1]
])

parameter_samples = qmc.scale(lhs_samples, lower_bounds, upper_bounds)


# ----- FREQUENCY-RATIO SAMPLING -----
# 60% near resonance
# 20% below resonance
# 20% above resonance

rng = np.random.default_rng(SEED + 1)

n_near = int(0.60 * N_SAMPLES)
n_below = int(0.20 * N_SAMPLES)
n_above = N_SAMPLES - n_near - n_below

near_ratios = rng.uniform(0.85, 1.20, n_near)
below_ratios = rng.uniform(0.50, 0.85, n_below)
above_ratios = rng.uniform(1.20, 1.50, n_above)

frequency_ratios = np.concatenate((near_ratios, below_ratios, above_ratios))

# Shuffle so operating regimes are mixed throughout the dataset
rng.shuffle(frequency_ratios)


# ----- GENERATE DATASET -----

print("\nGENERATING NONLINEAR BEAM DATASET")
print("Samples:", N_SAMPLES)
print("Modes:", N_MODES)
print("Simulation time:", T_END, "s")
print("Output timestep:", DT, "s")

start_time = time.time()

rows = []


for i in range(N_SAMPLES):

    L = parameter_samples[i, 0]
    h = parameter_samples[i, 1]
    E = parameter_samples[i, 2]
    rho = parameter_samples[i, 3]
    zeta = parameter_samples[i, 4]
    pressure_amplitude = parameter_samples[i, 5]

    frequency_ratio = frequency_ratios[i]

    # Calculate first natural frequency for this particular beam
    expected_f1 = calculate_f1(L, h, E, rho)

    # Convert frequency ratio into the actual excitation frequency
    excitation_frequency = frequency_ratio * expected_f1


    # ----- RUN NONLINEAR BEAM SIMULATION -----

    try:

        result = simulate_beam(
            L=L,
            b=b,
            h=h,
            E=E,
            rho=rho,
            zeta=zeta,
            pressure_amplitude=pressure_amplitude,
            excitation_frequency=excitation_frequency,
            N=N_MODES,
            use_nonlinearity=True,
            t_end=T_END,
            dt=DT
        )

    except Exception as error:

        # Save everything completed so far before stopping
        pd.DataFrame(rows).to_csv(CHECKPOINT_FILE, index=False)

        raise RuntimeError(
            f"Simulation failed at sample {i+1}. "
            f"Completed data saved to {CHECKPOINT_FILE}. "
            f"Original error: {error}"
        )


    # ----- VERIFY FREQUENCY INPUT -----

    used_frequency = result["excitation_frequency"]

    if abs(used_frequency - excitation_frequency) > 1e-10:
        raise ValueError(
            f"Excitation frequency mismatch at sample {i+1}: "
            f"requested {excitation_frequency}, used {used_frequency}"
        )


    # ----- SAVE SAMPLE -----

    row = {
        "L": L,
        "h": h,
        "E": E,
        "rho": rho,
        "zeta": zeta,
        "pressure_amplitude": pressure_amplitude,
        "excitation_frequency": excitation_frequency,
        "frequency_ratio": frequency_ratio,
        "f1": result["frequencies"][0],
        "f2": result["frequencies"][1],
        "f3": result["frequencies"][2],
        "f4": result["frequencies"][3],
        "peak_displacement": result["peak_displacement"],
        "rms_displacement": result["rms_displacement"]
    }

    rows.append(row)


    # ----- PROGRESS -----

    if (i + 1) % CHECKPOINT_INTERVAL == 0 or i == 0:

        elapsed = time.time() - start_time

        print(
            f"Completed {i+1}/{N_SAMPLES} samples "
            f"({100 * (i+1) / N_SAMPLES:.1f} %) - "
            f"Elapsed: {elapsed / 60:.1f} min"
        )


    # ----- CHECKPOINT -----

    if (i + 1) % CHECKPOINT_INTERVAL == 0:
        pd.DataFrame(rows).to_csv(CHECKPOINT_FILE, index=False)


# ----- CREATE FINAL DATAFRAME -----

dataset = pd.DataFrame(rows)


# ----- DATASET CHECKS -----

print("\n========================================")
print("DATASET CHECKS")

# Check for missing values
missing_values = dataset.isnull().sum().sum()

# Check for infinite values
numeric_data = dataset.select_dtypes(include=[np.number])
infinite_values = np.isinf(numeric_data.to_numpy()).sum()

# Verify stored frequency ratio
calculated_ratio = dataset["excitation_frequency"] / dataset["f1"]
ratio_error = np.max(np.abs(calculated_ratio - dataset["frequency_ratio"]))


print("Number of samples:", len(dataset))
print("Missing values:", missing_values)
print("Infinite values:", infinite_values)
print("Maximum frequency-ratio error:", ratio_error)


# ----- CHECK OPERATING-REGIME COVERAGE -----

below_count = np.sum(dataset["frequency_ratio"] < 0.85)

near_count = np.sum(
    (dataset["frequency_ratio"] >= 0.85)
    & (dataset["frequency_ratio"] <= 1.20)
)

above_count = np.sum(dataset["frequency_ratio"] > 1.20)


print("\nFrequency-region coverage:")

print(
    f"Below resonance: {below_count} "
    f"({100 * below_count / N_SAMPLES:.1f} %)"
)

print(
    f"Near resonance: {near_count} "
    f"({100 * near_count / N_SAMPLES:.1f} %)"
)

print(
    f"Above resonance: {above_count} "
    f"({100 * above_count / N_SAMPLES:.1f} %)"
)


# ----- DISPLAY PARAMETER RANGES -----

print("\nDataset ranges:")

columns_to_check = [
    "L",
    "h",
    "E",
    "rho",
    "zeta",
    "pressure_amplitude",
    "excitation_frequency",
    "frequency_ratio",
    "peak_displacement",
    "rms_displacement"
]


for column in columns_to_check:

    print(
        f"{column}: "
        f"{dataset[column].min():.6e} to "
        f"{dataset[column].max():.6e}"
    )


# ----- SAVE FINAL DATASET -----

dataset.to_csv(OUTPUT_FILE, index=False)

elapsed = time.time() - start_time


print("\n========================================")
print("DATASET GENERATION COMPLETE")

print("Saved to:", OUTPUT_FILE)
print("Total samples:", len(dataset))
print(f"Total runtime: {elapsed / 60:.2f} minutes")