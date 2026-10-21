"""
UNSEEN GEOMETRY DATASET GENERATION FOR TEST 2

Generate 500 new nonlinear simulations with 250 unseen-length cases and 250 unseen-thickness cases. 
One geometry input is outside the original sampling range at a time.

- Sample length outside 1.6-2.4m or thickness outside 0.008-0.012m.
- The other geometry input, material properties, damping and pressure are kept inside their original sampling ranges.
- 20% below-resonance, 60% near-resonance and 20% above-resonance cases in each geometry group, with frequency ratio restricted to 0.50-1.50.

"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


# ----- SETTINGS (MATCH THE MAIN DATASET GENERATOR) -----

SEED = 42
N_PER_GROUP = 250
B = 0.04
N_MODES = 7
T_END = 10.0
DT = 0.002
CHECKPOINT_INTERVAL = 50

TRAINING_RANGES = {
    "L": (1.6, 2.4),
    "h": (0.008, 0.012),
    "E": (185e9, 225e9),
    "rho": (7450.0, 8250.0),
    "zeta": (0.005, 0.020),
    "pressure_amplitude": (0.1, 30.0),
}

EXTRAPOLATION_RANGES = {
    "L": ((1.4, 1.6), (2.4, 2.6)),
    "h": ((0.007, 0.008), (0.012, 0.013)),
}


# ----- FIXED CASE DEFINITIONS -----

def calculate_f1(data):
    L, h, E, rho = (data[c].to_numpy(float) for c in ("L", "h", "E", "rho"))
    A = B * h
    I = B * h**3 / 12.0
    k1 = np.pi / L
    mass = rho * (A + I * k1**2) * L / 2.0
    stiffness = E * I * k1**4 * L / 2.0
    return np.sqrt(stiffness / mass) / (2.0 * np.pi)


def build_cases():
    groups = []
    columns = list(TRAINING_RANGES)

    for group_index, (case_name, variable) in enumerate((("Unseen length", "L"), ("Unseen thickness", "h"))):
        rng = np.random.default_rng(SEED + 10 + group_index)
        sampler = qmc.LatinHypercube(d=len(columns), seed=SEED + group_index)
        lower = [TRAINING_RANGES[c][0] for c in columns]
        upper = [TRAINING_RANGES[c][1] for c in columns]
        values = qmc.scale(sampler.random(n=N_PER_GROUP), lower, upper)
        group = pd.DataFrame(values, columns=columns)

        # Exactly 125 cases on each side of the original geometry range.
        half = N_PER_GROUP // 2
        low_range, high_range = EXTRAPOLATION_RANGES[variable]
        low = rng.uniform(*low_range, half)
        high = rng.uniform(*high_range, N_PER_GROUP - half)
        # Keep extrapolation strict even if an RNG draw lands at a boundary.
        low = np.minimum(low, np.nextafter(TRAINING_RANGES[variable][0], -np.inf))
        high = np.maximum(high, np.nextafter(TRAINING_RANGES[variable][1], np.inf))
        group[variable] = np.concatenate((low, high))
        group["test_case"] = case_name
        group["extrapolation_side"] = ["Below training range"] * half + ["Above training range"] * (N_PER_GROUP - half)

        # Exactly 50 below, 150 near and 50 above resonance in each group.
        ratios = np.concatenate((
            rng.uniform(0.50, 0.85, 50),
            rng.uniform(0.85, 1.20, 150),
            rng.uniform(np.nextafter(1.20, np.inf), 1.50, 50),
        ))
        rng.shuffle(ratios)
        group["frequency_ratio"] = ratios
        group["response_regime"] = np.where(ratios < 0.85, "Below resonance", np.where(ratios <= 1.20, "Near resonance", "Above resonance"))
        group["excitation_frequency"] = ratios * calculate_f1(group)
        groups.append(group)

    cases = pd.concat(groups, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    cases.insert(0, "sample_id", np.arange(len(cases)))
    return cases


# ----- RUN AND SAVE NONLINEAR SIMULATIONS -----

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cases-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = root / "data" / "test_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "unseen_geometry_dataset.csv"
    checkpoint_file = output_dir / "unseen_geometry_dataset_checkpoint.csv"

    cases = build_cases()
    print("TEST 2 - UNSEEN GEOMETRY DATASET")
    print(cases.groupby(["test_case", "extrapolation_side"]).size().to_string())
    print(cases.groupby(["test_case", "response_regime"]).size().to_string())

    if args.cases_only:
        cases_file = output_dir / "unseen_geometry_case_definitions.csv"
        cases.to_csv(cases_file, index=False)
        print("Case definitions saved:", cases_file)
        return

    sys.path.insert(0, str(root))
    from beam_model.nonlinear_beam import simulate_beam

    completed = []
    start = time.perf_counter()

    for i, row in cases.iterrows():
        try:
            simulation = simulate_beam(
                L=row["L"], b=B, h=row["h"], E=row["E"], rho=row["rho"],
                zeta=row["zeta"], pressure_amplitude=row["pressure_amplitude"],
                excitation_frequency=row["excitation_frequency"], N=N_MODES,
                use_nonlinearity=True, t_end=T_END, dt=DT,
            )
            if not np.isclose(simulation["excitation_frequency"], row["excitation_frequency"], rtol=0, atol=1e-10):
                raise ValueError("Simulator did not use the requested excitation frequency.")

            result = row.to_dict()
            for mode in range(4):
                result[f"f{mode + 1}"] = simulation["frequencies"][mode]
            result["peak_displacement"] = simulation["peak_displacement"]
            result["rms_displacement"] = simulation["rms_displacement"]
            outputs = [result[c] for c in ("f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement")]
            if not np.all(np.isfinite(outputs)):
                raise ValueError("Simulator returned non-finite outputs.")
            completed.append(result)

        except Exception as error:
            pd.DataFrame(completed).to_csv(checkpoint_file, index=False)
            raise RuntimeError(f"Simulation failed at sample {i}; completed responses saved to {checkpoint_file}.") from error

        count = len(completed)
        if count % 10 == 0:
            print(f"Completed {count}/{len(cases)} | elapsed = {(time.perf_counter() - start) / 60:.1f} min")
        if count % CHECKPOINT_INTERVAL == 0:
            pd.DataFrame(completed).to_csv(checkpoint_file, index=False)

    dataset = pd.DataFrame(completed)
    calculated_ratio = dataset["excitation_frequency"] / dataset["f1"]
    if not np.allclose(calculated_ratio, dataset["frequency_ratio"], rtol=1e-10, atol=1e-12):
        raise ValueError("Stored frequency ratios do not match the simulated first frequencies.")
    dataset.to_csv(output_file, index=False)
    print("\nDataset complete:", output_file)
    print(f"Total runtime: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
