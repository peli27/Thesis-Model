"""
TEST 3 - PARAMETER UNCERTAINTY AND HIDDEN ERROR DATASET GENERATION

Create paired uncertainty cases for Test 3A (known properties) and Test 3B
(hidden property errors), using the same nonlinear response targets for both.

"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ----- EXPERIMENT AND SIMULATOR SETTINGS -----

SEED = 42
N_NOMINAL_CASES = 500
UNCERTAINTY_LEVELS = (0.025, 0.05, 0.10)
PERTURBED_COLUMNS = ("E", "rho", "h")
INPUT_COLUMNS = ["L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"]
OUTPUT_COLUMNS = ["f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"]
B = 0.04
N_MODES = 7
T_END = 10.0
DT = 0.002
CHECKPOINT_INTERVAL = 50


# ----- SHARED NOMINAL CASES AND PAIRED PROPERTY ERRORS -----

def load_nominal_cases(dataset_file, split_file):
    data = pd.read_csv(dataset_file)
    required = INPUT_COLUMNS + OUTPUT_COLUMNS + ["frequency_ratio"]
    missing = [column for column in required if column not in data]
    if missing or not np.all(np.isfinite(data[required].to_numpy(float))):
        raise ValueError(f"Main dataset has missing columns or non-finite values. Missing: {missing}")

    with np.load(split_file, allow_pickle=False) as split:
        parts = [split[name].copy() for name in ("train_indices", "validation_indices", "test_indices")]
    for indices in parts:
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("The original split must contain one-dimensional integer row indices.")
    combined = np.concatenate(parts)
    if len(combined) != len(data) or not np.array_equal(np.sort(combined), np.arange(len(data))):
        raise ValueError("The original split does not partition the dataset without overlap.")

    test_indices = parts[2]
    if len(test_indices) < N_NOMINAL_CASES:
        raise ValueError(f"Need at least {N_NOMINAL_CASES} held-out samples.")
    ratios = data.iloc[test_indices]["frequency_ratio"].to_numpy(float)
    regimes = np.where(ratios < 0.85, 0, np.where(ratios <= 1.20, 1, 2))
    if len(test_indices) == N_NOMINAL_CASES:
        selected = test_indices
    else:
        selected, _ = train_test_split(test_indices, train_size=N_NOMINAL_CASES, random_state=SEED, stratify=regimes)
    selected = np.sort(selected)
    nominal = data.iloc[selected][required].reset_index(drop=True).copy()
    nominal.insert(0, "nominal_row_index", selected)
    nominal.insert(0, "nominal_case_id", np.arange(len(nominal)))
    return nominal


def first_frequency(data):
    L, h, E, rho = (data[column].to_numpy(float) for column in ("L", "h", "E", "rho"))
    A = B * h
    I = B * h**3 / 12.0
    k1 = np.pi / L
    mass = rho * (A + I * k1**2) * L / 2.0
    stiffness = E * I * k1**4 * L / 2.0
    return np.sqrt(stiffness / mass) / (2.0 * np.pi)


def build_cases(nominal):
    rng = np.random.default_rng(SEED + 100)
    factors = rng.uniform(-1.0, 1.0, size=(len(nominal), len(PERTURBED_COLUMNS)))
    levels = []

    for band in UNCERTAINTY_LEVELS:
        cases = nominal.copy()
        cases.rename(columns={c: f"nominal_{c}" for c in INPUT_COLUMNS + OUTPUT_COLUMNS + ["frequency_ratio"]}, inplace=True)
        cases["uncertainty_level_percent"] = 100.0 * band

        for column in INPUT_COLUMNS:
            cases[column] = cases[f"nominal_{column}"]
        for j, column in enumerate(PERTURBED_COLUMNS):
            error = band * factors[:, j]
            cases[column] = cases[f"nominal_{column}"].to_numpy(float) * (1.0 + error)
            cases[f"{column}_error_percent"] = 100.0 * error

        cases["frequency_ratio"] = cases["excitation_frequency"].to_numpy(float) / first_frequency(cases)
        levels.append(cases)

    cases = pd.concat(levels, ignore_index=True)
    cases.insert(0, "sample_id", np.arange(len(cases)))
    return cases


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(rows, path):
    temporary = path.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


# ----- GENERATE OR RESUME NONLINEAR RESPONSES -----

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cases-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    main_file = root / "data" / "main" / "nonlinear_beam_dataset.csv"
    split_file = root / "data" / "splits" / "main_split_indices.npz"
    output_dir = root / "data" / "test_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "parameter_uncertainty_dataset.csv"
    checkpoint_file = output_dir / "parameter_uncertainty_dataset_checkpoint.csv"
    manifest_file = output_dir / "parameter_uncertainty_generation_metadata.json"

    nominal = load_nominal_cases(main_file, split_file)
    cases = build_cases(nominal)
    print("TEST 3 - PAIRED PARAMETER UNCERTAINTY")
    print(f"Nominal beams: {len(nominal)} | New simulations: {len(cases)}")
    print("Bands: +/-2.5%, +/-5%, +/-10%; independent errors in E, rho and h.")
    print("Absolute excitation frequency remains fixed for each nominal beam.")

    if args.cases_only:
        path = output_dir / "parameter_uncertainty_case_definitions.csv"
        cases.to_csv(path, index=False)
        print("Case definitions saved:", path)
        return

    simulator_file = root / "beam_model" / "nonlinear_beam.py"
    manifest = {
        "seed": SEED,
        "n_nominal_cases": len(nominal),
        "uncertainty_levels": list(UNCERTAINTY_LEVELS),
        "distribution": "independent uniform [-band, band]; paired factors across bands",
        "fixed_excitation_frequency": True,
        "main_dataset_sha256": file_hash(main_file),
        "split_sha256": file_hash(split_file),
        "simulator_sha256": file_hash(simulator_file),
        "case_definitions_sha256": hashlib.sha256(cases.to_csv(index=False).encode()).hexdigest(),
        "simulation": {"b": B, "N": N_MODES, "t_end": T_END, "dt": DT, "use_nonlinearity": True},
    }

    if output_file.exists():
        raise FileExistsError(f"Completed dataset already exists: {output_file}. Use it for evaluation.")
    completed = []

    if args.resume:
        if not checkpoint_file.exists() or not manifest_file.exists():
            raise FileNotFoundError("Resume needs both the checkpoint CSV and generation metadata.")
        if json.loads(manifest_file.read_text()) != manifest:
            raise ValueError("Resume metadata differs from the current dataset, split, case definitions or simulator.")
        previous = pd.read_csv(checkpoint_file)
        if len(previous) > len(cases) or not set(OUTPUT_COLUMNS).issubset(previous.columns):
            raise ValueError("Invalid checkpoint contents.")
        if not np.array_equal(previous["sample_id"].to_numpy(), cases.iloc[:len(previous)]["sample_id"].to_numpy()):
            raise ValueError("Checkpoint rows do not match the case sequence.")
        if not np.all(np.isfinite(previous[OUTPUT_COLUMNS].to_numpy(float))):
            raise ValueError("Checkpoint contains non-finite responses.")
        # Also compare stored definitions against regenerated deterministic cases.
        for column in cases.columns:
            if not np.allclose(previous[column], cases.iloc[:len(previous)][column], rtol=1e-12, atol=1e-14):
                raise ValueError(f"Checkpoint case definitions differ in {column}.")
        completed = previous.to_dict("records")
        print(f"Resuming after {len(completed)} completed simulations.")
    elif checkpoint_file.exists():
        raise FileExistsError("A checkpoint exists. Use --resume to continue it.")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sys.path.insert(0, str(root))
    from beam_model.nonlinear_beam import simulate_beam

    start = time.perf_counter()
    try:
        for _, row in cases.iloc[len(completed):].iterrows():
            result = simulate_beam(
                L=row["L"], b=B, h=row["h"], E=row["E"], rho=row["rho"],
                zeta=row["zeta"], pressure_amplitude=row["pressure_amplitude"],
                excitation_frequency=row["excitation_frequency"], N=N_MODES,
                use_nonlinearity=True, t_end=T_END, dt=DT,
            )
            saved = row.to_dict()
            for mode in range(4):
                saved[f"f{mode + 1}"] = result["frequencies"][mode]
            saved["peak_displacement"] = result["peak_displacement"]
            saved["rms_displacement"] = result["rms_displacement"]
            if not np.all(np.isfinite([saved[c] for c in OUTPUT_COLUMNS])):
                raise ValueError("Simulator returned non-finite responses.")
            if not np.isclose(result["excitation_frequency"], row["excitation_frequency"], rtol=0, atol=1e-10):
                raise ValueError("Simulator changed the prescribed excitation frequency.")
            completed.append(saved)

            count = len(completed)
            if count % 10 == 0:
                print(f"Completed {count}/{len(cases)} | elapsed this run = {(time.perf_counter() - start) / 60:.1f} min")
            if count % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(completed, checkpoint_file)

    except BaseException:
        if completed:
            save_checkpoint(completed, checkpoint_file)
            print("Completed responses saved. Continue with --resume.")
        raise

    dataset = pd.DataFrame(completed)
    if not np.allclose(dataset["excitation_frequency"] / dataset["f1"], dataset["frequency_ratio"], rtol=1e-10, atol=1e-12):
        raise ValueError("Actual frequency ratios disagree with simulated f1.")
    dataset.to_csv(output_file, index=False)
    print("\nUncertainty dataset complete:", output_file)
    print("Both Test 3A and Test 3B use this single dataset.")


if __name__ == "__main__":
    main()