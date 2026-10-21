"""Isolated, deterministic one-mode diagnostic. Run from any directory."""

import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
sys.path.insert(0, str(ROOT))
from beam_model.nonlinear_beam import simulate_beam

OUT = HERE / 'results'
INPUTS = ['L', 'h', 'E', 'rho', 'zeta', 'pressure_amplitude', 'excitation_frequency']
TARGETS = ['peak_displacement', 'rms_displacement']
SETTINGS = dict(b=0.04, t_end=10., dt=0.002, rtol=1e-8, atol=1e-10)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_estimator(model):
    # Extract only the existing pure function: never import a training script.
    path = ROOT / 'models' / model
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate_physics_estimates')
    namespace = dict(np=np, B=0.04, RESPONSE_LOCATION_RATIO=0.65)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['calculate_physics_estimates']

def simulate(row, n, nonlinear=True):
    return simulate_beam(**{k: float(row[k]) for k in INPUTS}, **SETTINGS,
                         N=n, use_nonlinearity=nonlinear)

def main():
    OUT.mkdir(exist_ok=True)
    paths = [ROOT / 'data/main/nonlinear_beam_dataset.csv',
             ROOT / 'data/test_datasets/difficult_response_dataset.csv']
    protected = [p for directory in ('models', 'data', 'comparison', 'beam_model', 'evaluation', 'dataset_generation', 'regime_maps')
                 for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    metadata = json.loads((paths[1].parent / 'difficult_response_generation_metadata.json').read_text())
    assert metadata['simulator_sha256'] == digest(ROOT / 'beam_model/nonlinear_beam.py')
    assert metadata['main_dataset_sha256'] == digest(paths[0])
    main_data, difficult = [pd.read_csv(p) for p in paths]
    for frame, source in ((main_data, 'main'), (difficult, 'difficult')):
        frame['source'] = source
        frame['source_row'] = np.arange(len(frame))
    rng = np.random.default_rng(20261019)
    groups = []
    used = set()
    def take(pool, n, label):
        pool = pool.loc[[ (r.source, r.source_row) not in used for r in pool.itertuples()]]
        assert len(pool) >= n, (label, len(pool))
        part = pool.iloc[rng.choice(len(pool), n, replace=False)].copy()
        used.update(zip(part.source, part.source_row))
        part['selection_group'] = label
        groups.append(part)
    take(main_data[main_data.pressure_amplitude <= 3], 40, 'low_forcing')
    take(main_data[main_data.frequency_ratio < .85], 30, 'below_resonance')
    take(main_data[main_data.frequency_ratio > 1.2], 30, 'above_resonance')
    take(main_data[main_data.frequency_ratio.between(.85, 1.2)], 50, 'near_resonance')
    strong = difficult[difficult.nonlinear_stiffness_ratio_peak >= .10]
    # Cover both moderate and the strongest nonlinear cases without selecting by approximation error.
    threshold = strong.nonlinear_stiffness_ratio_peak.quantile(.75)
    take(strong[strong.nonlinear_stiffness_ratio_peak >= threshold], 25, 'strong_upper_quartile')
    take(strong[strong.nonlinear_stiffness_ratio_peak < threshold], 25, 'strong_other')
    cases = pd.concat(groups, ignore_index=True)
    cases.insert(0, 'case_id', np.arange(len(cases)))
    cases.to_csv(OUT / 'selected_cases.csv', index=False)
    linear = load_estimator('model_4_residual/train_residual_nn.py')
    linear5 = load_estimator('model_5_physics_regularised/train_physics_regularised_nn.py')
    np.testing.assert_array_equal(linear(cases), linear5(cases))
    linear(cases)  # warm up
    batch_times = []
    for _ in range(1000):
        start = time.perf_counter(); estimates = linear(cases); batch_times.append(time.perf_counter()-start)
    records = []
    timing = []
    histories = []
    # Time seven-mode reruns on five cases per sampling group (30 total), reusing stored labels elsewhere.
    timed_ids = set(cases.groupby('selection_group', sort=False).head(5).case_id)
    simulate(cases.iloc[0], 1)  # solver warm up, excluded from timing
    for i, row in cases.iterrows():
        record = row.to_dict()
        for j, target in enumerate(TARGETS):
            record['linear_' + target] = estimates[i, j]
        start = time.perf_counter(); one = simulate(row, 1); elapsed = time.perf_counter()-start
        record['one_mode_seconds'] = elapsed
        for target in TARGETS:
            record['one_mode_' + target] = one[target]
        q = one['q_history'][0, 2500:]
        record['one_mode_eta_peak'] = 3 / row.h**2 * np.max(q*q)
        if i in timed_ids:
            # Alternate ordering to reduce systematic timing bias.
            if i % 2:
                start = time.perf_counter(); one = simulate(row, 1); elapsed = time.perf_counter()-start
            start = time.perf_counter(); seven = simulate(row, 7); seven_time = time.perf_counter()-start
            if not i % 2:
                start = time.perf_counter(); one = simulate(row, 1); elapsed = time.perf_counter()-start
            linear_td = simulate(row, 1, False)
            entry = dict(case_id=i, selection_group=row.selection_group,
                         one_mode_seconds=elapsed, seven_mode_seconds=seven_time)
            for target in TARGETS:
                entry['rerun_' + target] = seven[target]
                entry['stored_delta_' + target] = seven[target]-row[target]
                record['linear_time_domain_' + target] = linear_td[target]
                np.testing.assert_allclose(seven[target], row[target], rtol=1e-5, atol=1e-10)
            timing.append(entry)
            histories.append((i, row, one, seven, linear_td))
        records.append(record)
        if (i+1) % 20 == 0:
            print(f'Completed {i+1}/200; seven-mode timing reruns {len(timing)}/30', flush=True)
            pd.DataFrame(records).to_csv(OUT / 'case_results.csv', index=False)
    result = pd.DataFrame(records)
    metrics = []
    for group, frame in [('all', result), *list(result.groupby('selection_group'))]:
        for method in ('linear', 'one_mode', 'linear_time_domain'):
            for target in TARGETS:
                sub = frame.dropna(subset=[method+'_'+target])
                if not len(sub): continue
                error = sub[method+'_'+target]-sub[target]
                absolute = np.abs(error)
                relative = absolute/sub[target]*100
                metrics.append(dict(group=group, method=method, target=target, n=len(sub),
                    rmse_mm=np.sqrt(np.mean(error**2))*1000, mae_mm=absolute.mean()*1000,
                    p95_abs_mm=absolute.quantile(.95)*1000, max_abs_mm=absolute.max()*1000,
                    median_relative_percent=relative.median(), p95_relative_percent=relative.quantile(.95),
                    max_relative_percent=relative.max(), over_5_percent=int((relative>5).sum()),
                    over_10_percent=int((relative>10).sum())))
    pd.DataFrame(metrics).to_csv(OUT/'accuracy_metrics.csv', index=False)
    result.to_csv(OUT/'case_results.csv', index=False)
    times = pd.DataFrame(timing); times.to_csv(OUT/'paired_solver_timing.csv', index=False)
    worst = []
    for method in ('linear','one_mode'):
        for target in TARGETS:
            for ranking in ('absolute','relative'):
                frame = result.copy()
                frame['absolute_error_mm'] = np.abs(frame[method+'_'+target]-frame[target])*1000
                frame['relative_error_percent'] = frame.absolute_error_mm/(frame[target]*1000)*100
                frame = frame.nlargest(10, 'absolute_error_mm' if ranking=='absolute' else 'relative_error_percent')
                frame['method'] = method; frame['target'] = target; frame['ranking'] = ranking
                worst.append(frame)
    pd.concat(worst).to_csv(OUT/'largest_errors.csv', index=False)
    timing_summary = []
    for method, values, scope in (
        ('linear', np.array(batch_times)/200, '1000 vectorised 200-case calls; amortised per case'),
        ('one_mode', times.one_mode_seconds, '30 paired cases'),
        ('seven_mode', times.seven_mode_seconds, '30 paired cases')):
        timing_summary.append(dict(method=method, scope=scope, mean_seconds=np.mean(values),
                                   median_seconds=np.median(values), p95_seconds=np.quantile(values,.95)))
    pd.DataFrame(timing_summary).to_csv(OUT/'timing_summary.csv', index=False)
    fig, axes = plt.subplots(2,2,figsize=(11,9))
    for j,target in enumerate(TARGETS):
        for k,method in enumerate(('linear','one_mode')):
            ax = axes[j,k]
            for group, frame in result.groupby('selection_group'):
                ax.scatter(frame[target]*1000, frame[method+'_'+target]*1000, s=15, alpha=.7, label=group)
            bounds = [min(ax.get_xlim()[0],ax.get_ylim()[0]), max(ax.get_xlim()[1],ax.get_ylim()[1])]
            ax.plot(bounds,bounds,'k--',lw=1); ax.set(xlabel='Seven-mode reference [mm]', ylabel='Estimate [mm]',title=f'{method}: {target}')
    axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT/'parity.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax,target in zip(axes,TARGETS):
        for method in ('linear','one_mode'):
            err = np.abs(result[method+'_'+target]-result[target])*1000
            ax.scatter(result.frequency_ratio, err, label=method, s=18, alpha=.65)
        ax.set(yscale='log',xlabel='Excitation / first natural frequency',ylabel='Absolute error [mm]',title=target); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'errors_by_frequency.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].bar([r['method'] for r in timing_summary],[r['median_seconds'] for r in timing_summary]); axes[0].set(yscale='log',ylabel='Median seconds per case',title='Linear timing amortised over 200 cases')
    axes[1].scatter(times.seven_mode_seconds,times.one_mode_seconds); axes[1].set(xlabel='Seven-mode seconds',ylabel='One-mode seconds',title='Paired solver timings')
    fig.tight_layout(); fig.savefig(OUT/'timing.png',dpi=180); plt.close(fig)
    # Display low forcing, strongest one-mode error among reruns, and strongest nonlinear rerun.
    chosen = [histories[0], max(histories,key=lambda x:abs(x[2]['peak_displacement']-x[3]['peak_displacement'])), max(histories,key=lambda x: result.loc[x[0],'one_mode_eta_peak'])]
    fig, axes = plt.subplots(3,1,figsize=(12,9))
    for ax,(i,row,one,seven,lin) in zip(axes,chosen):
        for label, sim in [('seven-mode',seven),('one-mode',one),('linear time-domain',lin)]:
            ax.plot(sim['time'][2500:],sim['displacement'][2500:]*1000,label=label,lw=1,alpha=.8)
        ax.set(xlabel='Time [s]',ylabel='Displacement [mm]',title=f'Case {i}, {row.selection_group}, frequency ratio {row.frequency_ratio:.3f}'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'response_histories.png',dpi=180); plt.close(fig)
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert before == after, 'An existing project artifact changed during the diagnostic.'
    manifest = dict(seed=20261019, n=200, settings=SETTINGS, solver='DOP853', initial_conditions='zero',
        response='x/L=0.65; peak abs and sampled RMS over t=5..10 s inclusive',
        selection='40 low forcing <=3 Pa; 30 below; 30 above; 50 near; 50 strong eta>=0.1 (25 upper quartile)',
        sampling_caution='Purposive regime-balanced diagnostic, not a population-weighted test set; strong selection uses reference outputs.',
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},
        protected_artifacts_unchanged=True, protected_file_count=len(protected),
        python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
        linear_batch_median_seconds=float(np.median(batch_times)),
        paired_median_speedup=float(np.median(times.seven_mode_seconds/times.one_mode_seconds)))
    (OUT/'metadata.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(metrics).query("group == 'all'").to_string(index=False))
    print(pd.DataFrame(timing_summary).to_string(index=False))

if __name__ == '__main__':
    main()
