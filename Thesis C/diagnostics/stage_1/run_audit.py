"""Audit existing artifacts without importing training scripts or retraining."""
import ast
import importlib.util
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def main():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("shared", ROOT / "evaluation/evaluate_basic_interpolation.py")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    protected = [p for folder in ("models", "data", "comparison", "beam_model", "evaluation", "dataset_generation", "regime_maps")
                 for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    before = {str(p.relative_to(ROOT)): shared.file_hash(p) for p in protected}
    df = pd.read_csv(ROOT / "data/main/nonlinear_beam_dataset.csv")
    split = shared.load_split(ROOT / "data/splits/main_split_indices.npz", len(df))
    X, y = df[shared.INPUT_COLUMNS].to_numpy(float), df[shared.OUTPUT_COLUMNS].to_numpy(float)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    paths = {4: ROOT / "models/model_4_residual/train_residual_nn.py",
             5: ROOT / "models/model_5_physics_regularised/train_physics_regularised_nn.py"}
    physics = [extract(p, "calculate_physics_estimates", dict(np=np, B=.04, RESPONSE_LOCATION_RATIO=.65))(df) for p in paths.values()]
    np.testing.assert_array_equal(*physics)
    p = physics[0]
    np.testing.assert_allclose(p, shared.calculate_physics_estimates(df), rtol=1e-14, atol=1e-16)
    sensitivity_path = ROOT / "models/model_5_physics_regularised/test_model5_lambda_sensitivity.py"
    sensitivity_physics = extract(sensitivity_path, "physics_estimate", dict(np=np, B=.04))(df)
    np.testing.assert_allclose(p, sensitivity_physics, rtol=1e-14, atol=1e-16)
    residual = y.copy()
    residual[:, 4:6] -= p
    np.testing.assert_allclose(residual[:, 4:6] + p, y[:, 4:6], rtol=1e-12, atol=1e-16)
    # Check Hz labels against the simulator's rotary-inertia modal formula.
    k = np.arange(1, 5)[None, :] * np.pi / df.L.to_numpy()[:, None]
    area = .04 * df.h.to_numpy()[:, None]
    inertia = .04 * df.h.to_numpy()[:, None]**3 / 12
    frequencies = np.sqrt(df.E.to_numpy()[:, None] * inertia * k**4 /
                          (df.rho.to_numpy()[:, None] * (area + inertia*k**2))) / (2*np.pi)
    np.testing.assert_allclose(y[:, :4], frequencies, rtol=1e-12, atol=1e-12)
    cached_residual = pd.read_csv(ROOT / "models/model_4_residual/results/training/residual_target_check.csv")
    np.testing.assert_allclose(cached_residual[["peak_residual", "rms_residual"]], residual[:, 4:6], rtol=1e-10, atol=1e-16)
    provenance = json.loads((ROOT / "comparison/results/test_1_basic_interpolation_run_metadata.json").read_text())
    assert provenance["dataset_sha256"] == before["data/main/nonlinear_beam_dataset.csv"]
    assert provenance["split_sha256"] == before["data/splits/main_split_indices.npz"]
    torch.set_num_threads(1)
    metrics, scaler_rows, checkpoints = [], [], {}
    predictions = {}
    for folder, prefix, name, count in (shared.MODEL_SPECS[i] for i in (0, 3, 4)):
        base = ROOT / "models" / folder
        saved = base / "saved_model"
        checkpoint = torch.load(saved / f"{prefix}_nn_model.pt", map_location="cpu", weights_only=False)
        shared.validate_checkpoint(checkpoint, prefix, count)
        xs = joblib.load(shared.find_scaler(saved, prefix, "X"))
        ys = joblib.load(shared.find_scaler(saved, prefix, "Y"))
        targets = residual if prefix == "residual" else y
        tr = split["train_indices"]
        shared.validate_scaler(xs, X[tr], prefix + " inputs")
        shared.validate_scaler(ys, targets[tr], prefix + " targets")
        assert xs.n_samples_seen_ == ys.n_samples_seen_ == len(tr)
        for j, output in enumerate(shared.OUTPUT_COLUMNS):
            scaler_rows.append(dict(model=prefix, output=output, mean=ys.mean_[j], scale=ys.scale_[j], variance=ys.var_[j]))
        saved_provenance = next(item for item in provenance["models"] if item["model"] == name)
        for label, path in (("checkpoint", saved / f"{prefix}_nn_model.pt"),
                            ("input_scaler", shared.find_scaler(saved, prefix, "X")),
                            ("output_scaler", shared.find_scaler(saved, prefix, "Y"))):
            assert shared.file_hash(path) == saved_provenance[label + "_sha256"]
        model = shared.SurrogateNN(count)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        checkpoints[prefix] = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        for partition, idx in split.items():
            pred = shared.predict(df.iloc[idx], prefix, checkpoint, model, xs, ys)
            predictions[prefix, partition] = pred
            for j, output in enumerate(shared.OUTPUT_COLUMNS):
                err = pred[:, j] - y[idx, j]
                metrics.append(dict(model=prefix, split=partition, output=output,
                                    rmse=float(np.sqrt(np.mean(err**2))), mae=float(np.mean(np.abs(err))),
                                    negative_count=int((pred[:, j] < 0).sum())))
            if partition == "validation_indices":
                with torch.inference_mode():
                    z = model(torch.tensor(xs.transform(X[idx]), dtype=torch.float32))
                t = torch.tensor(ys.transform(targets[idx]), dtype=torch.float32)
                loss = float(nn.functional.mse_loss(z, t))
                history = pd.read_csv(base / "results/training" / f"{prefix}_training_history.csv")
                if prefix == "physics_regularised":
                    pt = torch.tensor((p[idx] - ys.mean_[4:6]) / ys.scale_[4:6], dtype=torch.float32)
                    total, data, phys = extract(paths[5], "calculate_loss", dict(loss_function=nn.MSELoss(), PHYSICS_WEIGHT=checkpoint["physics_weight"]))(z, t, pt)
                    loss = float(total)
                    checkpoints[prefix].update(recomputed_validation_data=float(data), recomputed_validation_physics=float(phys),
                                              weighted_physics_to_data=float(checkpoint["physics_weight"] * phys / data),
                                              best_data_epoch=int(history.loc[history.validation_data_loss.idxmin(), "epoch"]),
                                              best_logged_data_loss=float(history.validation_data_loss.min()))
                    column = "validation_total_loss"
                else:
                    column = "validation_loss"
                np.testing.assert_allclose(loss, checkpoint["best_validation_loss"], rtol=2e-5, atol=1e-7)
                np.testing.assert_allclose(loss, history[column].min(), rtol=2e-5, atol=1e-7)
                if "best_epoch" in checkpoint:
                    assert checkpoint["best_epoch"] == int(history.loc[history[column].idxmin(), "epoch"])
                checkpoints[prefix]["recomputed_validation_objective"] = loss
                checkpoints[prefix]["history_epochs"] = len(history)
            if partition == "test_indices":
                cached = pd.read_csv(base / "results/test_1_basic_interpolation/predictions.csv")
                np.testing.assert_array_equal(cached.dataset_row_index.to_numpy(), idx)
                previous = cached[["predicted_" + c for c in shared.OUTPUT_COLUMNS]].to_numpy()
                np.testing.assert_allclose(pred, previous, rtol=2e-5, atol=1e-7)
                checkpoints[prefix]["cached_prediction_max_abs_difference"] = float(np.max(np.abs(pred - previous)))
                checkpoints[prefix]["cached_displacement_max_abs_difference_m"] = float(np.max(np.abs(pred[:, 4:6] - previous[:, 4:6])))
    tr = split["train_indices"]
    stats = []
    for partition, idx in split.items():
        for j, output in enumerate(shared.OUTPUT_COLUMNS[4:]):
            r = residual[idx, j + 4]
            sq = np.sort(r**2)
            stats.append(dict(split=partition, output=output, true_std=float(y[idx, j+4].std()),
                              residual_std=float(r.std()), physics_max=float(p[idx, j].max()),
                              true_max=float(y[idx, j+4].max()), residual_min=float(r.min()), residual_max=float(r.max()),
                              largest_one_percent_residual_energy_fraction=float(sq[-max(1, int(np.ceil(.01*len(sq)))):].sum()/sq.sum()),
                              residual_vs_direct_dimensional_error_weight=float(y[tr, j+4].var()/residual[tr, j+4].var())))
    # Verify the exact gradient implied by the two different MSE reductions.
    z = torch.tensor([[.2, -.1, .5, .3, .8, -.4]], dtype=torch.float64, requires_grad=True)
    t, ph = torch.zeros_like(z), torch.tensor([[2., 1.]], dtype=torch.float64)
    total, _, _ = extract(paths[5], "calculate_loss", dict(loss_function=nn.MSELoss(), PHYSICS_WEIGHT=.01))(z, t, ph)
    grad = torch.autograd.grad(total, z)[0]
    expected = 2*(z-t)/6
    expected[:, 4:6] += .01*(z[:, 4:6]-ph)
    torch.testing.assert_close(grad, expected)
    after = {str(p.relative_to(ROOT)): shared.file_hash(p) for p in protected}
    assert before == after, "Original project artifacts changed during audit"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(scaler_rows).to_csv(OUT / "scalers.csv", index=False)
    pd.DataFrame(stats).to_csv(OUT / "residual_statistics.csv", index=False)
    metadata = dict(dataset_rows=len(df), split_sizes={k: len(v) for k,v in split.items()}, checkpoints=checkpoints,
                    checks="split partition; finite data; train-only scaler statistics/counts; estimator parity; residual identity/cache; frequency formula; checkpoint metadata/state; validation/history agreement; cached predictions; provenance hashes; loss gradient",
                    artifacts_unchanged=True, protected_artifact_count=len(before), hashes=before,
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__, sklearn=sklearn.__version__))
    (OUT / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k:v for k,v in metadata.items() if k != "hashes"}, indent=2))


if __name__ == "__main__":
    main()
