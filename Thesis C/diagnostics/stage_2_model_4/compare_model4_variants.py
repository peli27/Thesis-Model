""" MODEL 4 RESIDUAL-LEARNING COMPARISON

This script investigates whether changing how analytical physics is used or how
residual errors are weighted can improve Model 4. All three versions predict
the correction between the nonlinear beam response and a simplified linear
first-mode physics estimate. Final displacement predictions are reconstructed
by adding the physics estimate back to the predicted correction.

"""
from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

COLUMN_ALIASES = {
    'L': ['L', 'length', 'beam_length', 'length_m'],
    'h': ['h', 'thickness', 'beam_thickness', 'thickness_m'],
    'E': ['E', 'youngs_modulus', 'young_modulus', 'elastic_modulus', 'youngs_modulus_pa'],
    'rho': ['rho', 'density', 'density_kg_m3'],
    'zeta': ['zeta', 'damping', 'damping_ratio'],
    'pressure': ['pressure_amplitude', 'pressure', 'load_amplitude', 'forcing_amplitude', 'pressure_pa'],
    'fexc': ['excitation_frequency', 'excitation_frequency_hz', 'forcing_frequency', 'forcing_frequency_hz', 'excitation_freq'],
    'f1': ['f1', 'f_1', 'frequency_1', 'natural_frequency_1', 'f1_hz'],
    'f2': ['f2', 'f_2', 'frequency_2', 'natural_frequency_2', 'f2_hz'],
    'f3': ['f3', 'f_3', 'frequency_3', 'natural_frequency_3', 'f3_hz'],
    'f4': ['f4', 'f_4', 'frequency_4', 'natural_frequency_4', 'f4_hz'],
    'peak': ['peak_displacement', 'peak_displacement_m', 'peak_disp', 'peak', 'max_displacement'],
    'rms': ['rms_displacement', 'rms_displacement_m', 'rms_disp', 'rms'],
}
INPUTS = ['L', 'h', 'E', 'rho', 'zeta', 'pressure', 'fexc']
OUTPUTS = ['f1', 'f2', 'f3', 'f4', 'peak', 'rms']


def resolve_columns(df, override):
    mapping = {}
    for k in INPUTS + OUTPUTS:
        if k in override:
            selected = override[k]
        else:
            matches = [s for s in COLUMN_ALIASES[k] if s in df.columns]
            if not matches:
                raise ValueError(f"Missing column {k!r}. Available: {df.columns.tolist()}\n"
                                 f"Supply --column-map path/to/map.json, e.g. {{\"{k}\":\"actual_column_name\"}}")
            selected = matches[0]
        if selected not in df.columns:
            raise ValueError(f'Column {selected!r} not present for {k}')
        mapping[k] = selected
    return mapping


def get_split(npz_path, n):
    obj = np.load(npz_path, allow_pickle=False)
    aliases = {'train': ['train_idx', 'train_indices', 'train'],
               'val': ['val_idx', 'val_indices', 'validation_idx', 'validation_indices', 'valid_idx', 'val'],
               'test': ['test_idx', 'test_indices', 'test']}
    out = {}
    for split, options in aliases.items():
        keys = [k for k in options if k in obj.files]
        if not keys:
            raise ValueError(f'Missing {split} index key in {obj.files}. Update get_split aliases.')
        out[split] = np.asarray(obj[keys[0]], dtype=int)
    combined = np.concatenate(list(out.values()))
    if len(combined) != n or len(np.unique(combined)) != n or combined.min() < 0 or combined.max() >= n:
        raise ValueError('Saved splits do not cover exactly the CSV rows without overlap')
    return out


def linear_physics(df, col, width, observation_x):
    v = lambda key: df[col[key]].to_numpy(dtype=np.float64)
    L, h, E, rho, zeta, pressure, fexc = (v(k) for k in INPUTS)
    if np.any(L <= 0) or np.any(h <= 0) or np.any(E <= 0) or np.any(rho <= 0):
        raise ValueError('Nonpositive beam dimensions or properties')
    A = width * h
    I = width * h**3 / 12.0
    k = np.pi / L
    M = rho * (A + I*k*k) * L / 2.0
    K = E * I*k**4*L/2.0
    C = 2.0*zeta*M*np.sqrt(K/M)
    Q = 2.0*pressure*width*L/np.pi
    omega = 2*np.pi*fexc
    denominator = np.sqrt((K - M*omega**2)**2 + (C*omega)**2)
    if np.any(denominator == 0):
        raise ValueError('Zero analytical-response denominator')
    peak = np.abs(Q*np.sin(np.pi*observation_x)/denominator)
    return np.column_stack((peak, peak/np.sqrt(2.0)))


class MLP(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(nin,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(),
                                     nn.Linear(64,32), nn.ReLU(), nn.Linear(32,6))
    def forward(self, x):
        return self.network(x)


def metrics(ytrue, ypred):
    out = {}
    for j, label in [(4, 'peak'), (5, 'rms')]:
        error = (ypred[:,j] - ytrue[:,j]) * 1000.0
        out[f'{label}_rmse_mm'] = float(np.sqrt(np.mean(error**2)))
        out[f'{label}_mae_mm'] = float(np.mean(np.abs(error)))
        denom = float(np.sum((ytrue[:,j] - np.mean(ytrue[:,j]))**2))
        out[f'{label}_r2'] = float(1 - np.sum((ypred[:,j]-ytrue[:,j])**2)/denom) if denom > 0 else None
        out[f'{label}_negative_n'] = int(np.sum(ypred[:,j] < 0))
    return out


def run_variant(name, seed, Xraw, Y, physics, split, args, out_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    tr, va = split['train'], split['val']
    X = np.column_stack((Xraw, physics)) if name == '4B' else Xraw
    xs = StandardScaler().fit(X[tr])
    Xs = xs.transform(X).astype(np.float32)
    targets = Y.copy()
    targets[:,4:6] -= physics
    target_mean = targets[tr].mean(axis=0)
    target_std = targets[tr].std(axis=0, ddof=0)
    if name == '4C':
        # Hold the importance of a physical displacement error equal to the
        # baseline (direct-response StandardScaler) rather than residual SD.
        target_std[4:6] = Y[tr,4:6].std(axis=0, ddof=0)
    if np.any(target_std <= 0):
        raise ValueError('Zero target standard deviation')
    Z = ((targets - target_mean) / target_std).astype(np.float32)
    def tensors(idx):
        return torch.tensor(Xs[idx]), torch.tensor(Z[idx])
    Xtr, Ztr = tensors(tr)
    Xva, Zva = tensors(va)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ztr), batch_size=args.batch_size, shuffle=True, generator=generator)
    net = MLP(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best, best_epoch, bad = float('inf'), 0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.max_epochs+1):
        net.train()
        for xb, zb in loader:
            opt.zero_grad()
            loss = torch.mean((net(xb)-zb)**2)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = float(torch.mean((net(Xva)-Zva)**2).item())
        history.append({'epoch': epoch, 'validation_objective': vloss})
        if vloss < best - 1e-10:
            best, best_epoch, bad = vloss, epoch, 0
            best_weights = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            break
    net.load_state_dict(best_weights)
    net.eval()
    metrics_rows = []
    for part in ('train', 'val'):
        idx = split[part]
        with torch.no_grad():
            zout = net(torch.tensor(Xs[idx])).cpu().numpy().astype(float)
        reconstructed = zout*target_std + target_mean
        reconstructed[:,4:6] += physics[idx]
        m = metrics(Y[idx], reconstructed)
        metrics_rows.append({'variant': name, 'seed':seed, 'split':part, 'best_epoch':best_epoch,
                             'best_validation_objective':best, 'training_seconds':round(time.perf_counter()-start,2), **m})
    # Save full reproduction information in a new directory, not original models.
    dest = out_dir/name/str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict':best_weights, 'variant':name, 'seed':seed, 'input_mean':xs.mean_,
                'input_std':xs.scale_, 'target_mean':target_mean, 'target_std':target_std,
                'width':args.width, 'observation_x':args.observation_x}, dest/'checkpoint.pt')
    pd.DataFrame(history).to_csv(dest/'history.csv', index=False)
    return metrics_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data/main/nonlinear_beam_dataset.csv')
    parser.add_argument('--split', default='data/splits/main_split_indices.npz')
    parser.add_argument('--output', default='diagnostics/stage_2_model_4/variant_comparison')
    parser.add_argument('--column-map', help='Optional JSON mapping canonical keys to CSV headings')
    parser.add_argument('--width', type=float, default=0.04)
    parser.add_argument('--observation-x', type=float, default=0.65)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,123,456])
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    overrides = json.loads(Path(args.column_map).read_text()) if args.column_map else {}
    col = resolve_columns(df, overrides)
    split = get_split(args.split, len(df))
    X = df[[col[k] for k in INPUTS]].to_numpy(dtype=np.float64)
    Y = df[[col[k] for k in OUTPUTS]].to_numpy(dtype=np.float64)
    physics = linear_physics(df, col, args.width, args.observation_x)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)) or not np.all(np.isfinite(physics)):
        raise ValueError('Non-finite input, target or physics value')
    print('Column mapping:', col)
    print('Split sizes:', {k:len(v) for k,v in split.items()})
    print('Analytical estimate max peak (mm):', round(1000*physics[:,0].max(),4))
    print('Physics formula, width and response location verified against original scripts.')
    rows = []
    for seed in args.seeds:
        for variant in ('4A','4B','4C'):
            print(f'Training {variant}, seed {seed}', flush=True)
            result = run_variant(variant,seed,X,Y,physics,split,args,out)
            rows.extend(result)
            pd.DataFrame(rows).to_csv(out/'all_metrics.csv',index=False)
    result = pd.DataFrame(rows)
    val = result[result.split=='val']
    summary = val.groupby('variant').agg(peak_rmse_mm_mean=('peak_rmse_mm','mean'),
                                         peak_rmse_mm_sd=('peak_rmse_mm','std'),
                                         rms_rmse_mm_mean=('rms_rmse_mm','mean'),
                                         rms_rmse_mm_sd=('rms_rmse_mm','std'),
                                         seeds=('seed','count')).reset_index()
    summary.to_csv(out/'validation_summary.csv',index=False)
    settings = vars(args).copy()
    settings.update(columns=col, split_sizes={k:len(v) for k,v in split.items()},
                    variants={'4A':'raw inputs + residual target / residual SD',
                              '4B':'raw + analytical inputs + residual target / residual SD',
                              '4C':'raw inputs + residual target / direct-response SD'},
                    note='Only training and validation rows evaluated. Held-out test set is not used for variant selection.')
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    print('\nVALIDATION SUMMARY (mean across seeds; lower RMSE is better)')
    print(summary.to_string(index=False))
    print('\nSaved:',out)

if __name__ == '__main__':
    main()
