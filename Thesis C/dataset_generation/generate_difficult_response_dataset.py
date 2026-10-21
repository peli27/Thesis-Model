"""
TEST 5 - DIFFICULT NONLINEAR RESPONSE: DATASET GENERATION


1. Sample 500 new beams/loading conditions within the original training ranges.
2. Restrict excitation-frequency ratio to 0.85-1.20, the near-resonance band
   used in the original data split and the earlier evaluations.
3. Run the original seven-mode nonlinear simulator for 10 s at dt = 0.002 s.
4. Save the usual six outputs: f1-f4, peak displacement and RMS displacement.
5. From the modal displacement histories, calculate the nonlinear addition to
   first-mode restoring stiffness r58ikelative to the original linear stiffness:

       eta_1(t) = A / (4*I*k1**2) * sum(k_n**2 * q_n(t)**2)
                = 3 / h**2 * sum(n**2 * q_n(t)**2)

   This follows directly from the simulator's von Karman restoring force.
   It avoids dividing force by q1 at zero crossings.
6. Measure max(eta_1) over the second half of the simulation, matching the
   peak/RMS response window. Define strong cases by max(eta_1) >= 0.10.

"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


# ----- EXPERIMENT SETTINGS -----

N_SAMPLES = 500
SEED = 1425
B = 0.04
N_MODES = 7
T_END = 10.0
DT = 0.002
STRONG_THRESHOLD = 0.10
CHECKPOINT_INTERVAL = 50
RANGES = {
    "L": (1.6, 2.4),
    "h": (0.008, 0.012),
    "E": (185e9, 225e9),
    "rho": (7450.0, 8250.0),
    "zeta": (0.005, 0.020),
    "pressure_amplitude": (0.1, 30.0),
    "frequency_ratio": (0.85, 1.20),
}
INPUT_COLUMNS = ["L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"]
OUTPUT_COLUMNS = ["f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"]
DIAGNOSTIC_COLUMNS = ["nonlinear_stiffness_ratio_peak", "nonlinear_stiffness_ratio_mean", "first_modal_peak", "peak_displacement_to_thickness"]


# ----- NEW NEAR-RESONANCE CASE DEFINITIONS -----

def calculate_f1(data):
    L, h, E, rho = (data[column].to_numpy(float) for column in ("L", "h", "E", "rho"))
    A = B * h
    I = B * h**3 / 12.0
    k1 = np.pi / L
    mass = rho * (A + I * k1**2) * L / 2.0
    stiffness = E * I * k1**4 * L / 2.0
    return np.sqrt(stiffness / mass) / (2.0 * np.pi)


def build_cases():
    columns = list(RANGES)
    sampler = qmc.LatinHypercube(d=len(columns), seed=SEED)
    values = qmc.scale(sampler.random(n=N_SAMPLES), [RANGES[c][0] for c in columns], [RANGES[c][1] for c in columns])
    cases = pd.DataFrame(values, columns=columns)
    cases["excitation_frequency"] = cases.frequency_ratio.to_numpy(float) * calculate_f1(cases)
    cases.insert(0, "sample_id", np.arange(N_SAMPLES))
    return cases


def nonlinear_indicators(result, h):
    q = np.asarray(result["q_history"], dtype=float)
    modes = np.asarray(result["modes"], dtype=float)
    if q.ndim != 2 or q.shape[0] != len(modes) or not np.all(np.isfinite(q)):
        raise ValueError("Invalid modal displacement histories.")
    q_steady = q[:, q.shape[1] // 2:]
    eta = 3.0 / h**2 * np.sum(modes[:, None]**2 * q_steady**2, axis=0)
    return {
        "nonlinear_stiffness_ratio_peak": float(np.max(eta)),
        "nonlinear_stiffness_ratio_mean": float(np.mean(eta)),
        "first_modal_peak": float(np.max(np.abs(q_steady[0]))),
        "peak_displacement_to_thickness": float(result["peak_displacement"] / h),
    }


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(rows, path):
    temporary = path.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


# ----- GENERATE AND CLASSIFY SIMULATED RESPONSES -----

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cases-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = root / "data" / "test_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "difficult_response_dataset.csv"
    checkpoint_file = output_dir / "difficult_response_dataset_checkpoint.csv"
    metadata_file = output_dir / "difficult_response_generation_metadata.json"
    cases = build_cases()
    print("TEST 5 - DIFFICULT NONLINEAR RESPONSE")
    print("500 near-resonance cases; the simulator determines the strong subset.")

    if args.cases_only:
        cases_file = output_dir / "difficult_response_case_definitions.csv"
        cases.to_csv(cases_file, index=False)
        print("Case definitions saved:", cases_file)
        return

    main_file = root / "data" / "main" / "nonlinear_beam_dataset.csv"
    original = pd.read_csv(main_file)
    if any(column not in original for column in INPUT_COLUMNS):
        raise ValueError("Original dataset is missing required inputs.")
    # Reject any exact reuse of training/validation/test inputs.
    duplicates = cases.merge(original[INPUT_COLUMNS], on=INPUT_COLUMNS, how="inner")
    if len(duplicates):
        raise ValueError("Some new cases duplicate original dataset inputs.")

    metadata = {
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "ranges": {key: list(value) for key, value in RANGES.items()},
        "strong_threshold": STRONG_THRESHOLD,
        "criterion": "max eta_1 over second half of simulation >= strong_threshold",
        "main_dataset_sha256": file_hash(main_file),
        "simulator_sha256": file_hash(root / "beam_model" / "nonlinear_beam.py"),
        "case_definitions_sha256": hashlib.sha256(cases.to_csv(index=False).encode()).hexdigest(),
        "simulation": {"b": B, "N": N_MODES, "t_end": T_END, "dt": DT, "use_nonlinearity": True},
    }
    if output_file.exists():
        raise FileExistsError(f"Dataset already complete: {output_file}. Proceed to evaluation.")
    completed = []
    if args.resume:
        if not metadata_file.exists() or not checkpoint_file.exists():
            raise FileNotFoundError("Resume requires the checkpoint CSV and generation metadata.")
        if json.loads(metadata_file.read_text()) != metadata:
            raise ValueError("Dataset, simulator or generation settings changed since the checkpoint.")
        previous = pd.read_csv(checkpoint_file)
        required = list(cases.columns) + OUTPUT_COLUMNS + DIAGNOSTIC_COLUMNS
        if len(previous) > len(cases) or any(c not in previous for c in required):
            raise ValueError("Invalid checkpoint columns or length.")
        if not np.all(np.isfinite(previous[required].to_numpy(float))):
            raise ValueError("Checkpoint contains non-finite values.")
        for column in cases.columns:
            if not np.allclose(previous[column], cases.iloc[:len(previous)][column], rtol=1e-12, atol=1e-14):
                raise ValueError(f"Checkpoint definitions differ in {column}.")
        if not np.array_equal(previous.sample_id, cases.iloc[:len(previous)].sample_id):
            raise ValueError("Checkpoint sample IDs do not match the case sequence.")
        completed = previous.to_dict("records")
        print("Resuming after", len(completed), "simulations.")
    elif checkpoint_file.exists():
        raise FileExistsError("A checkpoint exists. Use --resume to continue.")
    else:
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sys.path.insert(0, str(root))
    from beam_model.nonlinear_beam import simulate_beam

    start = time.perf_counter()
    try:
        for _, row in cases.iloc[len(completed):].iterrows():
            result = simulate_beam(
                L=row.L, b=B, h=row.h, E=row.E, rho=row.rho, zeta=row.zeta,
                pressure_amplitude=row.pressure_amplitude, excitation_frequency=row.excitation_frequency,
                N=N_MODES, use_nonlinearity=True, t_end=T_END, dt=DT,
            )
            saved = row.to_dict()
            for mode in range(4):
                saved[f"f{mode + 1}"] = result["frequencies"][mode]
            saved["peak_displacement"] = result["peak_displacement"]
            saved["rms_displacement"] = result["rms_displacement"]
            saved.update(nonlinear_indicators(result, row.h))
            saved["strong_nonlinearity"] = saved["nonlinear_stiffness_ratio_peak"] >= STRONG_THRESHOLD
            if not np.all(np.isfinite([saved[c] for c in OUTPUT_COLUMNS + DIAGNOSTIC_COLUMNS])):
                raise ValueError("Simulator returned non-finite responses or indicators.")
            if not np.isclose(result["excitation_frequency"], row.excitation_frequency, rtol=0, atol=1e-10):
                raise ValueError("Simulator changed the prescribed excitation frequency.")
            completed.append(saved)
            count = len(completed)
            if count % 10 == 0:
                strong_count = sum(r["nonlinear_stiffness_ratio_peak"] >= STRONG_THRESHOLD for r in completed)
                print(f"Completed {count}/{N_SAMPLES} | strong cases = {strong_count} | elapsed this run = {(time.perf_counter() - start) / 60:.1f} min")
            if count % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(completed, checkpoint_file)
    except BaseException:
        if completed:
            save_checkpoint(completed, checkpoint_file)
            print("Completed responses saved. Continue with --resume.")
        raise

    dataset = pd.DataFrame(completed)
    if not np.allclose(dataset.excitation_frequency / dataset.f1, dataset.frequency_ratio, rtol=1e-10, atol=1e-12):
        raise ValueError("Stored frequency ratios disagree with the simulated first frequencies.")
    dataset.to_csv(output_file, index=False)
    strong_count = int((dataset.nonlinear_stiffness_ratio_peak >= STRONG_THRESHOLD).sum())
    print("\nDataset saved:", output_file)
    print(f"Test 5A: {len(dataset)} near-resonance cases | Test 5B: {strong_count} strong cases")
    if strong_count < 30:
        print("The strong subset has fewer than 30 cases; report its size and treat its results as preliminary.")


if __name__ == "__main__":
    main()