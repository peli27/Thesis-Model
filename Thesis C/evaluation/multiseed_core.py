"""Shared five-model experiment engine. Train with 3 matched seeds; evaluate Tests 1 to 5."""

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456)
PERCENTAGES = (100, 50, 25, 10)
INPUTS = ["L", "h", "E", "rho", "zeta", "pressure_amplitude", "excitation_frequency"]
OUTPUTS = ["f1", "f2", "f3", "f4", "peak_displacement", "rms_displacement"]
NAMES = {"baseline": "Model 1 Baseline", "physics_guided": "Model 2 Physics-Guided", "physics_estimate": "Model 3 Physics-Estimate", "residual": "Model 4B Residual", "physics_regularised": "Model 5 Physics-Regularised"}
SIZES = {"baseline": 7, "physics_guided": 12, "physics_estimate": 9, "residual": 9, "physics_regularised": 7}
RESULTS = ROOT / "comparison" / "results" / "multiseed_final"
WEIGHT = 0.01


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_data():
    path = ROOT / "data/main/nonlinear_beam_dataset.csv"
    split_path = ROOT / "data/splits/main_split_indices.npz"
    df = pd.read_csv(path)
    required = INPUTS + OUTPUTS + ["frequency_ratio"]
    if any(c not in df.columns for c in required) or not np.isfinite(df[required].to_numpy(float)).all():
        raise ValueError("Missing or invalid main dataset columns")
    with np.load(split_path, allow_pickle=False) as z:
        split = {k: z[k].copy() for k in ("train_indices", "validation_indices", "test_indices")}
    indices = np.concatenate(list(split.values()))
    if len(indices) != len(df) or not np.array_equal(np.sort(indices), np.arange(len(df))):
        raise ValueError("Saved split does not partition the dataset")
    return df, split, {"dataset_sha256": hash_file(path), "split_sha256": hash_file(split_path)}


def physics_features(df):
    L, h, E, rho = [df[c].to_numpy(float) for c in INPUTS[:4]]
    b = 0.04
    A, I, k = b * h, b * h**3 / 12, np.pi / L
    M = rho * (A + I*k**2) * L / 2
    K = E * I * k**4 * L / 2
    w1 = np.sqrt(K/M)
    ratio = 2*np.pi*df.excitation_frequency.to_numpy(float)/w1
    guidance = np.column_stack((w1/(2*np.pi), M, K, ratio, np.abs(ratio-1)))
    w = 2*np.pi*df.excitation_frequency.to_numpy(float)
    Q = 2*df.pressure_amplitude.to_numpy(float)*b*L/np.pi
    damping = 2*df.zeta.to_numpy(float)*M*w1*w
    denominator = np.sqrt((K-M*w**2)**2+damping**2)
    if np.any(denominator <= 0):
        raise ValueError("Invalid linear physics denominator")
    peak = np.abs(Q*np.sin(np.pi*0.65)/denominator)
    estimates = np.column_stack((peak, peak/np.sqrt(2)))
    return guidance, estimates


def arrays(df, kind, *, include_targets=True):
    raw = df[INPUTS].to_numpy(float)
    # Inference (including hidden-uncertainty inputs) needs no response labels.
    target = df[OUTPUTS].to_numpy(float).copy() if include_targets else None
    guided, estimate = physics_features(df)
    if kind == "physics_guided":
        raw = np.column_stack((raw, guided))
    if kind in ("physics_estimate", "residual"):
        raw = np.column_stack((raw, estimate))
    if kind == "residual" and target is not None:
        target[:, 4:6] -= estimate
    return raw, target, estimate


def network(kind):
    n = SIZES[kind]
    return torch.nn.Sequential(torch.nn.Linear(n, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 6))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def folder(kind, seed, percentage):
    return RESULTS / "trained" / kind / f"data_{percentage}" / f"seed_{seed}"


def train(df, split, kind, seed, train_indices, percentage):
    target_folder = folder(kind, seed, percentage)
    target_folder.mkdir(parents=True, exist_ok=True)
    complete = target_folder / "complete.json"
    files = [target_folder / f for f in ("model.pt", "x_scaler.pkl", "y_scaler.pkl")]
    if complete.is_file() and all(p.is_file() for p in files):
        info = json.loads(complete.read_text())
        if (info["seed"], info["percentage"], info["n_train"]) != (seed, percentage, len(train_indices)):
            raise ValueError("Existing training run differs: " + str(target_folder))
        return
    seed_everything(seed)
    x, y, physics = arrays(df, kind)
    tr, va = train_indices, split["validation_indices"]
    xs = StandardScaler().fit(x[tr]); ys = StandardScaler().fit(y[tr])
    xt = torch.tensor(xs.transform(x[tr]), dtype=torch.float32)
    yt = torch.tensor(ys.transform(y[tr]), dtype=torch.float32)
    xv = torch.tensor(xs.transform(x[va]), dtype=torch.float32)
    yv = torch.tensor(ys.transform(y[va]), dtype=torch.float32)
    p = (physics - ys.mean_[4:6]) / ys.scale_[4:6]
    pt = torch.tensor(p[tr], dtype=torch.float32)
    pv = torch.tensor(p[va], dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt, yt, pt), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = network(kind)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    def objective(pred, truth, linear):
        loss = torch.nn.functional.mse_loss(pred, truth)
        if kind == "physics_regularised":
            loss = loss + WEIGHT*torch.nn.functional.mse_loss(pred[:, 4:6], linear)
        return loss
    best, patience, state, epoch_best = np.inf, 0, None, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, 501):
        model.train()
        for xb, yb, pb in loader:
            opt.zero_grad()
            loss = objective(model(xb), yb, pb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            validation = float(objective(model(xv), yv, pv).item())
        if not np.isfinite(validation):
            raise ValueError("Nonfinite validation objective")
        history.append({"epoch": epoch, "validation_loss": validation})
        if validation < best:
            best, state, epoch_best, patience = validation, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            patience += 1
        if patience >= 30:
            break
    model.load_state_dict(state)
    torch.save({"state_dict": model.state_dict(), "kind": kind, "seed": seed, "percentage": percentage, "n_inputs": SIZES[kind], "best_epoch": epoch_best}, files[0])
    joblib.dump(xs, files[1]); joblib.dump(ys, files[2])
    pd.DataFrame(history).to_csv(target_folder / "history.csv", index=False)
    complete.write_text(json.dumps({"seed": seed, "percentage": percentage, "n_train": len(tr), "best_epoch": epoch_best, "training_seconds": time.perf_counter()-start}, indent=2))
    print(f"Trained {kind}, {percentage}%, seed {seed}, best epoch {epoch_best}")


def predict(df, kind, seed, percentage):
    target_folder = folder(kind, seed, percentage)
    checkpoint = torch.load(target_folder / "model.pt", map_location="cpu", weights_only=False)
    if (checkpoint["kind"], checkpoint["seed"], checkpoint["percentage"], checkpoint["n_inputs"]) != (kind, seed, percentage, SIZES[kind]):
        raise ValueError("Checkpoint metadata mismatch")
    xs = joblib.load(target_folder / "x_scaler.pkl")
    ys = joblib.load(target_folder / "y_scaler.pkl")
    x, _, physics = arrays(df, kind, include_targets=False)
    model = network(kind)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        scaled = model(torch.tensor(xs.transform(x), dtype=torch.float32)).numpy()
    pred = ys.inverse_transform(scaled)
    if kind == "residual":
        pred[:, 4:6] += physics
    elapsed = time.perf_counter() - t0
    return pred, elapsed


def groups_for(test, data):
    if test == 1:
        r = data.frequency_ratio.to_numpy(float)
        return {"all": np.ones(len(data), bool), "below_resonance": r < .85, "near_resonance": (r >= .85) & (r <= 1.20), "above_resonance": r > 1.20}
    if test == 2:
        return {"all": np.ones(len(data), bool), "unseen_length": (data.test_case == "Unseen length").to_numpy(), "unseen_thickness": (data.test_case == "Unseen thickness").to_numpy()}
    if test == 5:
        return {"5A_near_resonance": np.ones(len(data), bool), "5B_strong_nonlinearity": (data.nonlinear_stiffness_ratio_peak.to_numpy(float) >= .10)}
    return {"all": np.ones(len(data), bool)}


def metrics(truth, pred, test, group, kind, seed, percentage, timing):
    rows = []
    for j, label in enumerate(OUTPUTS):
        error = pred[:, j]-truth[:, j]
        rmse = float(np.sqrt(np.mean(error**2)))
        denominator = np.sum((truth[:, j]-truth[:, j].mean())**2)
        r2 = float(1-np.sum(error**2)/denominator) if denominator > 0 else np.nan
        rows.append({"test": test, "group": group, "model": NAMES[kind], "seed": seed, "percentage": percentage, "output": label, "n_cases": len(truth), "rmse": rmse, "mae": float(np.mean(np.abs(error))), "r2": r2, "p95_error": float(np.percentile(np.abs(error), 95)), "max_error": float(np.max(np.abs(error))), "negative_predictions": int((pred[:, j] < 0).sum()), "prediction_seconds": timing})
    return rows


def test_data(test, original, split):
    if test == 1 or test == 4:
        return original.iloc[split["test_indices"]].copy()
    paths = {2: "unseen_geometry_dataset.csv", 3: "parameter_uncertainty_dataset.csv", 5: "difficult_response_dataset.csv"}
    data = pd.read_csv(ROOT / "data" / "test_datasets" / paths[test])
    required = INPUTS + OUTPUTS
    if any(c not in data for c in required) or not np.isfinite(data[required].to_numpy(float)).all():
        raise ValueError(f"Invalid Test {test} data")
    if test == 2 and (len(data) != 500 or data.test_case.value_counts().to_dict() != {"Unseen length": 250, "Unseen thickness": 250}):
        raise ValueError("Test 2 dataset counts differ from original")
    if test == 3 and (len(data) != 1500 or set(data.uncertainty_level_percent.unique()) != {2.5, 5., 10.}):
        raise ValueError("Test 3 dataset counts/bands differ from original")
    if test == 5 and (len(data) != 500 or "nonlinear_stiffness_ratio_peak" not in data or not (data.nonlinear_stiffness_ratio_peak >= .10).any()):
        raise ValueError("Test 5 dataset/strong subset invalid")
    # Reuse the original experiment validators, leaving those files unchanged.
    if test == 2:
        from evaluate_unseen_geometry import validate_dataset
        validate_dataset(data)
    if test == 3:
        from evaluate_parameter_uncertainty import validate_dataset
        original, split, source = fixed_data()
        validate_dataset(data, original, split)
        metadata = json.loads((ROOT / "data/test_datasets/parameter_uncertainty_generation_metadata.json").read_text())
        if metadata["main_dataset_sha256"] != source["dataset_sha256"] or metadata["split_sha256"] != source["split_sha256"]:
            raise ValueError("Test 3 generation provenance mismatch")
    if test == 5:
        from evaluate_difficult_response import validate_dataset
        original, _, source = fixed_data()
        validate_dataset(data, original)
        metadata = json.loads((ROOT / "data/test_datasets/difficult_response_generation_metadata.json").read_text())
        if metadata["main_dataset_sha256"] != source["dataset_sha256"] or metadata["strong_threshold"] != .10:
            raise ValueError("Test 5 generation provenance mismatch")
    return data


def evaluate(test):
    torch.set_num_threads(1)
    original, split, source = fixed_data()
    data = test_data(test, original, split)
    rows = []
    budgets = PERCENTAGES if test == 4 else (100,)
    training_pool = split["train_indices"]
    subsets = {100: training_pool}
    if test == 4:
        current = training_pool
        for pct in PERCENTAGES[1:]:
            count = int(np.floor(len(training_pool)*pct/100))
            ratio = original.iloc[current].frequency_ratio.to_numpy(float)
            labels = np.where(ratio < .85, 0, np.where(ratio <= 1.2, 1, 2))
            current, _ = train_test_split(current, train_size=count, stratify=labels, random_state=2026+pct)
            subsets[pct] = np.sort(current)
    for percentage in budgets:
        for kind in NAMES:
            for seed in SEEDS:
                train(original, split, kind, seed, subsets[percentage], percentage)
                if test == 3:
                    for level in (2.5, 5., 10.):
                        cases = data.loc[data.uncertainty_level_percent == level].sort_values("nominal_case_id")
                        truth = cases[OUTPUTS].to_numpy(float)
                        for case in ("3A_known", "3B_hidden"):
                            inputs = cases
                            if case == "3B_hidden":
                                inputs = cases[[f"nominal_{c}" for c in INPUTS]].copy()
                                inputs.columns = INPUTS
                            pred, seconds = predict(inputs, kind, seed, percentage)
                            rows.extend(metrics(truth, pred, test, f"{case}_{level:g}pct", kind, seed, percentage, seconds))
                else:
                    pred, seconds = predict(data, kind, seed, percentage)
                    truth = data[OUTPUTS].to_numpy(float)
                    for group, mask in groups_for(test, data).items():
                        if mask.any():
                            rows.extend(metrics(truth[mask], pred[mask], test, group, kind, seed, percentage, seconds))
    out = RESULTS / f"test_{test}"
    out.mkdir(parents=True, exist_ok=True)
    detailed = pd.DataFrame(rows)
    detailed.to_csv(out / "seed_metrics.csv", index=False)
    keys = ["test", "group", "model", "percentage", "output"]
    summary = detailed.groupby(keys, as_index=False).agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"), r2_mean=("r2", "mean"), n_seeds=("seed", "nunique"), n_cases=("n_cases", "first"), prediction_seconds_mean=("prediction_seconds", "mean"))
    # Paired by seed; positive improvement means lower RMSE than baseline.
    baseline = detailed.loc[detailed.model == NAMES["baseline"], ["group", "percentage", "output", "seed", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    paired = detailed.merge(baseline, on=["group", "percentage", "output", "seed"], validate="many_to_one")
    paired["improvement_pct"] = 100*(paired.baseline_rmse-paired.rmse)/paired.baseline_rmse
    benefit = paired.groupby(keys, as_index=False).agg(improvement_pct_mean=("improvement_pct", "mean"), improvement_pct_sd=("improvement_pct", "std"), seeds_better=("improvement_pct", lambda x: int((x > 0).sum())))
    summary = summary.merge(benefit, on=keys, validate="one_to_one")
    summary.to_csv(out / "comparison_summary.csv", index=False)
    paired.to_csv(out / "paired_seed_comparison.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({**source, "seeds": SEEDS, "budgets": budgets, "model_4": "4B: nine inputs, physics estimates supplied and residuals reconstructed", "test": test, "physics_weight": WEIGHT, "note": "All variants fixed before these test evaluations; prior held-out tests had been inspected"}, indent=2))
    print(f"Test {test} complete: {out / 'comparison_summary.csv'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, choices=range(1, 6), required=True)
    evaluate(parser.parse_args().test)
