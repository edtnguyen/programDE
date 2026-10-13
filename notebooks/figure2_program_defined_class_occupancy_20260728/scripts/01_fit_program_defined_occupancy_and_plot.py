#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.stats import norm
from statsmodels.api import GLM, families
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')


def read_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    base = path.parent.parent.resolve()
    for key in ['export_dir', 'output_dir']:
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str((base / p).resolve())
    return cfg


def clean_design(day_df: pd.DataFrame):
    x = pd.DataFrame(index=day_df.index)
    x['intercept'] = 1.0
    for col in ['total_counts', 'guides_per_cell', 'guide_umi_counts', 'n_genes_by_counts']:
        v = np.log1p(pd.to_numeric(day_df[col], errors='coerce').fillna(0).astype(float))
        sd = v.std(ddof=0)
        x[f'z_log1p_{col}'] = (v - v.mean()) / sd if sd > 0 else 0.0
    v = pd.to_numeric(day_df['pct_counts_mt'], errors='coerce').fillna(day_df['pct_counts_mt'].median()).astype(float)
    sd = v.std(ddof=0)
    x['z_pct_counts_mt'] = (v - v.mean()) / sd if sd > 0 else 0.0
    reps = pd.get_dummies(day_df['biological_replicate'].astype(str), prefix='sample', drop_first=True, dtype=float)
    return pd.concat([x, reps], axis=1)


def fit_glm(y, exposure, base_x):
    x = base_x.copy()
    x.insert(1, 'target_carrier', np.asarray(exposure, dtype=float))
    try:
        fit = GLM(np.asarray(y, dtype=float), x.to_numpy(float), family=families.Binomial()).fit(maxiter=150, disp=0)
        beta = float(fit.params[1])
        se = float(fit.bse[1])
        z = beta / se if se > 0 else np.nan
        p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        converged = bool(getattr(fit, 'converged', True))
        return beta, se, z, p, converged, ''
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan, False, str(e)


def combine_fixed_effect(rep_df: pd.DataFrame):
    d = rep_df.dropna(subset=['log_odds_ratio', 'standard_error']).copy()
    d = d[d['standard_error'] > 0]
    if len(d) == 0:
        return np.nan, np.nan, np.nan, np.nan
    w = 1 / np.square(d['standard_error'].to_numpy())
    b = d['log_odds_ratio'].to_numpy()
    beta = float(np.sum(w * b) / np.sum(w))
    se = float(np.sqrt(1 / np.sum(w)))
    z = beta / se
    p = float(2 * norm.sf(abs(z)))
    return beta, se, z, p


def run_models(cell, assignment, program_defs, class_membership, cfg, outdir: Path, population_name: str, use_all_day_cells: bool):
    target_cells = assignment.groupby('target_name')['cell_id'].apply(set).to_dict()
    class_membership = class_membership.rename(columns={'gene': 'target_name'})
    class_membership = class_membership[class_membership['target_name'].isin(target_cells)].copy()

    min_carriers = int(cfg.get('minimum_carrier_cells_per_day', 25))
    min_state_cells = int(cfg.get('minimum_program_state_cells_per_day', 25))
    min_carrier_state = int(cfg.get('minimum_carrier_state_cells', 3))
    min_carrier_nonstate = int(cfg.get('minimum_carrier_nonstate_cells', 3))
    min_noncarrier_state = int(cfg.get('minimum_noncarrier_state_cells', 10))
    fdr_threshold = float(cfg.get('fdr_threshold', 0.05))

    primary_defs = program_defs[program_defs['use_in_primary']].copy().sort_values('plot_order')
    states_by_day = {
        day: sub['program_id'].astype(str).tolist()
        for day, sub in primary_defs.groupby('expected_day', sort=False)
    }

    day_data = {}
    coverage_rows = []
    tasks = []

    for day in ['D0', 'D1', 'D2', 'D3']:
        ddf_all = cell[cell['day'].astype(str) == day].copy()
        if len(ddf_all) == 0:
            continue
        if use_all_day_cells:
            ddf = ddf_all.copy()
        else:
            ddf = ddf_all[ddf_all['is_primary_program_state'].astype(bool)].copy()
        ddf = ddf.reset_index(drop=True)
        if len(ddf) == 0:
            continue
        day_data[day] = (ddf, clean_design(ddf))
        day_cell_ids = set(ddf['cell_id'])
        valid_states = []
        for state in states_by_day.get(day, []):
            n_state = int((ddf['assigned_state'].astype(str) == state).sum())
            if n_state >= min_state_cells:
                valid_states.append(state)

        for target, cset in target_cells.items():
            n_carrier = len(day_cell_ids & cset)
            coverage_rows.append({
                'analysis_population': population_name,
                'day': day,
                'target_name': target,
                'n_analysis_cells': len(ddf),
                'n_carrier_cells': n_carrier,
                'passes_carrier_threshold': n_carrier >= min_carriers,
                'n_evaluable_program_states': len(valid_states),
            })
            if n_carrier < min_carriers:
                continue
            for state in valid_states:
                tasks.append((day, target, state))

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(outdir / f'coverage_and_filtering_{population_name}.tsv', sep='\t', index=False)

    def run_task(task):
        day, target, state = task
        ddf, base_x = day_data[day]
        carriers = ddf['cell_id'].isin(target_cells[target]).to_numpy()
        y = (ddf['assigned_state'].astype(str) == str(state)).to_numpy(int)
        n_carrier_state = int((carriers & (y == 1)).sum())
        n_carrier_nonstate = int((carriers & (y == 0)).sum())
        n_noncarrier_state = int(((~carriers) & (y == 1)).sum())
        sparse_reason = ''
        if n_carrier_state < min_carrier_state:
            sparse_reason = 'too_few_carrier_state_cells'
        elif n_carrier_nonstate < min_carrier_nonstate:
            sparse_reason = 'too_few_carrier_nonstate_cells'
        elif n_noncarrier_state < min_noncarrier_state:
            sparse_reason = 'too_few_noncarrier_state_cells'

        if sparse_reason:
            beta = se = z = p = np.nan
            converged = False
            error = sparse_reason
        else:
            beta, se, z, p, converged, error = fit_glm(y, carriers, base_x)

        return {
            'analysis_population': population_name,
            'day': day,
            'target_name': target,
            'state': state,
            'log_odds_ratio': beta,
            'standard_error': se,
            'z_value': z,
            'p_value': p,
            'ci_lower': beta - 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan,
            'ci_upper': beta + 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan,
            'n_analysis_cells': int(len(ddf)),
            'n_carrier_cells': int(carriers.sum()),
            'n_noncarrier_cells': int((~carriers).sum()),
            'n_state_cells': int(y.sum()),
            'n_carrier_state_cells': n_carrier_state,
            'n_carrier_nonstate_cells': n_carrier_nonstate,
            'n_noncarrier_state_cells': n_noncarrier_state,
            'converged': converged,
            'model_error': error,
            'covariates': 'biological_sample + log1p(total_counts) + log1p(guides_per_cell) + log1p(guide_umi_counts) + log1p(n_genes_by_counts) + pct_counts_mt',
        }

    n_jobs = int(cfg.get('n_jobs', 8))
    effects = pd.DataFrame(Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_task)(t) for t in tasks))
    if len(effects) == 0:
        return effects, pd.DataFrame(), pd.DataFrame(), coverage

    effects['fdr'] = np.nan
    for day, idx in effects.groupby('day').groups.items():
        idx = np.asarray(list(idx))
        valid = effects.loc[idx, 'p_value'].notna().to_numpy()
        if valid.any():
            effects.loc[idx[valid], 'fdr'] = multipletests(effects.loc[idx[valid], 'p_value'], method='fdr_bh')[1]
    effects['significant'] = effects['fdr'] < fdr_threshold

    # Replicate-specific sensitivity.
    rep_rows = []
    for day, (ddf_all, _) in day_data.items():
        for rep, ddf in ddf_all.groupby('biological_replicate', sort=True):
            ddf = ddf.reset_index(drop=True)
            base = clean_design(ddf)
            base = base[[c for c in base.columns if not c.startswith('sample_')]]
            valid_states = [s for s in states_by_day.get(day, []) if int((ddf['assigned_state'].astype(str) == s).sum()) >= max(10, min_state_cells // 2)]
            eligible_targets = coverage.query('day == @day and passes_carrier_threshold')['target_name'].unique()
            for target in eligible_targets:
                carriers = ddf['cell_id'].isin(target_cells[target]).to_numpy()
                if carriers.sum() < max(8, min_carriers // 3):
                    continue
                for state in valid_states:
                    y = (ddf['assigned_state'].astype(str) == str(state)).to_numpy(int)
                    if (carriers & (y == 1)).sum() < 2 or (carriers & (y == 0)).sum() < 2:
                        continue
                    beta, se, z, p, conv, err = fit_glm(y, carriers, base)
                    rep_rows.append({
                        'analysis_population': population_name,
                        'day': day,
                        'biological_replicate': rep,
                        'target_name': target,
                        'state': state,
                        'log_odds_ratio': beta,
                        'standard_error': se,
                        'z_value': z,
                        'p_value': p,
                        'n_carrier_cells': int(carriers.sum()),
                        'n_state_cells': int(y.sum()),
                        'converged': conv,
                        'model_error': err,
                    })
    rep_df = pd.DataFrame(rep_rows)

    if len(rep_df):
        comb = []
        for key, sub in rep_df.groupby(['day', 'target_name', 'state']):
            b, se, z, p = combine_fixed_effect(sub)
            signs = np.sign(sub['log_odds_ratio'].dropna())
            concord = float((signs == np.sign(b)).mean()) if len(signs) and np.isfinite(b) else np.nan
            comb.append(dict(zip(['day', 'target_name', 'state'], key)) | {
                'replicate_meta_log_odds_ratio': b,
                'replicate_meta_standard_error': se,
                'replicate_meta_p_value': p,
                'replicate_sign_concordance': concord,
                'n_replicates_evaluable': int(sub['biological_replicate'].nunique()),
            })
        effects = effects.merge(pd.DataFrame(comb), on=['day', 'target_name', 'state'], how='left')

    # Stable estimates only for class summaries; raw target estimates remain exported.
    max_se = float(cfg.get('maximum_standard_error_for_class_summary', 3.0))
    max_abs = float(cfg.get('maximum_abs_log_odds_for_class_summary', 8.0))
    effects['stable_for_class_summary'] = (
        effects['converged'].fillna(False)
        & effects['log_odds_ratio'].notna()
        & effects['standard_error'].notna()
        & (effects['standard_error'] <= max_se)
        & (effects['log_odds_ratio'].abs() <= max_abs)
    )

    joined = effects[effects['stable_for_class_summary']].merge(
        class_membership[['target_name', 'regulator_class']], on='target_name', how='inner'
    )
    class_rows = []
    for key, sub in joined.groupby(['regulator_class', 'day', 'state'], sort=True):
        cls, day, state = key
        n = sub['target_name'].nunique()
        if n < int(cfg.get('minimum_targets_per_class', 4)):
            continue
        med = float(sub['log_odds_ratio'].median())
        direction = 1 if med >= 0 else -1
        sig_dir = sub['significant'] & (np.sign(sub['log_odds_ratio']) == direction)
        class_rows.append({
            'analysis_population': population_name,
            'regulator_class': cls,
            'day': day,
            'state': state,
            'n_evaluable_targets': n,
            'median_log_odds_ratio': med,
            'fraction_significant_in_displayed_direction': float(sig_dir.mean()),
            'n_significant_in_displayed_direction': int(sig_dir.sum()),
            'displayed_direction': 'enriched' if direction > 0 else 'depleted',
        })
    class_summary = pd.DataFrame(class_rows)
    return effects, rep_df, class_summary, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = read_cfg(Path(args.config).resolve())
    export_dir = Path(cfg['export_dir'])
    outdir = Path(cfg['output_dir'])
    outdir.mkdir(parents=True, exist_ok=True)

    cell = pd.read_csv(export_dir / 'occupancy_cell_metadata.tsv.gz', sep='\t')
    assignment = pd.read_csv(export_dir / 'occupancy_cell_target_assignments.tsv.gz', sep='\t')
    class_membership = pd.read_csv(export_dir / 'class_membership.tsv', sep='\t')
    program_defs = pd.read_csv(export_dir / 'program_state_definitions.tsv', sep='\t')
    program_defs['use_in_primary'] = program_defs['use_in_primary'].astype(str).str.lower().isin(['true', '1', 'yes'])

    # Primary analysis: among cells whose global dominant program is a primary curated
    # program expected for the observed day.
    effects, rep_df, class_summary, coverage = run_models(
        cell, assignment, program_defs, class_membership, cfg, outdir,
        population_name='primary_curated_dominant_expected_day',
        use_all_day_cells=False,
    )
    effects.to_csv(outdir / 'target_program_state_occupancy_effects.tsv', sep='\t', index=False)
    rep_df.to_csv(outdir / 'target_program_state_occupancy_by_replicate.tsv', sep='\t', index=False)
    class_summary.to_csv(outdir / 'class_program_state_occupancy_summary.tsv', sep='\t', index=False)
    coverage.to_csv(outdir / 'coverage_and_filtering.tsv', sep='\t', index=False)

    # Sensitivity: same program-state outcomes versus all other cells from that observed day.
    if bool(cfg.get('run_all_day_cells_sensitivity', True)):
        e2, r2, c2, cov2 = run_models(
            cell, assignment, program_defs, class_membership, cfg, outdir,
            population_name='all_day_cells_sensitivity',
            use_all_day_cells=True,
        )
        e2.to_csv(outdir / 'target_program_state_occupancy_effects_all_day_cells_sensitivity.tsv', sep='\t', index=False)
        c2.to_csv(outdir / 'class_program_state_occupancy_summary_all_day_cells_sensitivity.tsv', sep='\t', index=False)

    # Descriptive raw target-class distribution among primary program-state cells.
    cm = class_membership.rename(columns={'gene': 'target_name'})[['target_name', 'regulator_class']]
    cell_primary = cell[cell['is_primary_program_state'].astype(bool)][['cell_id', 'day', 'assigned_state']]
    desc = assignment.merge(cell_primary, on='cell_id', how='inner').merge(cm, on='target_name', how='inner')
    desc = desc.drop_duplicates(['cell_id', 'regulator_class', 'day', 'assigned_state'])
    raw = (
        desc.groupby(['regulator_class', 'day', 'assigned_state'], as_index=False)
        .agg(n_unique_carrier_cells=('cell_id', 'nunique'), n_targets_observed=('target_name', 'nunique'))
    )
    denom = cell_primary.groupby(['day', 'assigned_state'], as_index=False).agg(n_state_cells=('cell_id', 'nunique'))
    raw = raw.merge(denom, on=['day', 'assigned_state'], how='left')
    raw['raw_fraction_of_state_cells_with_class_target'] = raw['n_unique_carrier_cells'] / raw['n_state_cells']
    raw.to_csv(outdir / 'target_class_program_state_raw_distribution.tsv', sep='\t', index=False)

    # Copy source definition tables.
    class_membership.to_csv(outdir / 'class_membership.tsv', sep='\t', index=False)
    program_defs.to_csv(outdir / 'program_state_definitions.tsv', sep='\t', index=False)
    pd.read_csv(export_dir / 'program_state_counts_by_day_and_replicate.tsv', sep='\t').to_csv(
        outdir / 'program_state_counts_by_day_and_replicate.tsv', sep='\t', index=False
    )

    # --------------------------------------------------------------
    # Plot main bubble matrix.
    # --------------------------------------------------------------
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    if len(class_summary) == 0:
        raise RuntimeError('No class-level results passed coverage/stability filters. Inspect coverage_and_filtering.tsv.')

    state_meta = program_defs[program_defs['use_in_primary']].copy().sort_values('plot_order')
    state_meta['key'] = list(zip(state_meta['expected_day'].astype(str), state_meta['program_id'].astype(str)))
    available = set(zip(class_summary['day'].astype(str), class_summary['state'].astype(str)))
    col_keys = [k for k in state_meta['key'] if k in available]
    xmap = {k: i for i, k in enumerate(col_keys)}
    label_map = dict(zip(state_meta['program_id'].astype(str), state_meta['program_name'].astype(str)))

    preferred_class_order = [
        'TGF/BMP/Activin', 'WNT / mesoderm regulators', 'VEGF / EC specification',
        'Hippo / YAP', 'Chromatin regulation', 'Transcription / RNA processing',
        'Translation / proteostasis', 'Apoptosis / cell cycle / stress',
        'Metabolism / trafficking / adhesion', 'Other reviewed regulators',
    ]
    present = set(class_summary['regulator_class'].astype(str))
    classes = [c for c in preferred_class_order if c in present] + sorted(present - set(preferred_class_order))
    ymap = {c: i for i, c in enumerate(classes)}

    class_coverage = (
        class_summary.groupby('regulator_class', as_index=False)
        .agg(
            minimum_evaluable_targets=('n_evaluable_targets', 'min'),
            maximum_evaluable_targets=('n_evaluable_targets', 'max'),
            n_day_program_combinations=('state', 'size'),
        )
    )
    class_coverage.to_csv(outdir / 'class_coverage_summary.tsv', sep='\t', index=False)

    q95 = float(np.nanquantile(np.abs(class_summary['median_log_odds_ratio']), 0.95))
    clip = max(0.5, min(q95, float(cfg.get('maximum_abs_log_odds_for_class_summary', 8.0))))
    norm2 = TwoSlopeNorm(vmin=-clip, vcenter=0, vmax=clip)
    cmap = mpl.colormaps['RdBu_r']

    fig_w = max(10.0, 0.78 * len(col_keys) + 4.5)
    fig_h = max(5.3, 0.58 * len(classes) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for _, r in class_summary.iterrows():
        key = (str(r['day']), str(r['state']))
        if key not in xmap or r['regulator_class'] not in ymap:
            continue
        size = 24 + 390 * float(r['fraction_significant_in_displayed_direction'])
        ax.scatter(
            xmap[key], ymap[r['regulator_class']], s=size,
            c=[cmap(norm2(float(r['median_log_odds_ratio'])))],
            edgecolors='#3F3F3F', linewidths=0.55, zorder=3
        )

    day_order = ['D0', 'D1', 'D2', 'D3']
    ordered_days = [d for d in day_order if any(k[0] == d for k in col_keys)]
    for d1, d2 in zip(ordered_days[:-1], ordered_days[1:]):
        last_x = max(xmap[k] for k in col_keys if k[0] == d1)
        next_x = min(xmap[k] for k in col_keys if k[0] == d2)
        ax.axvline((last_x + next_x) / 2, color='#BDBDBD', linewidth=0.8, zorder=0)

    def wrap_label(text: str) -> str:
        replacements = {
            'Primitive streak, BMP-repressed 1': 'Primitive streak,\nBMP-repressed 1',
            'Primitive streak, BMP-repressed 2': 'Primitive streak,\nBMP-repressed 2',
            'Posterior primitive streak': 'Posterior\nprimitive streak',
            'Lateral mesoderm, YAP/TAZ-induced': 'Lateral mesoderm,\nYAP/TAZ-induced',
            'Mesenchyme, YAP/TAZ-induced': 'Mesenchyme,\nYAP/TAZ-induced',
            'Anterior-like mesoderm': 'Anterior-like\nmesoderm',
            'Endothelial function 1': 'Endothelial\nfunction 1',
            'Endothelial function 2': 'Endothelial\nfunction 2',
            'Artery endothelium 1': 'Artery\nendothelium 1',
            'Artery endothelium 2': 'Artery\nendothelium 2',
        }
        return replacements.get(text, text)

    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels([
        f'{d}\n{state}\n{wrap_label(label_map.get(state, state))}' for d, state in col_keys
    ], rotation=42, ha='right')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([
        f"{c}  (n≥{int(class_summary[class_summary.regulator_class == c].n_evaluable_targets.min())})"
        for c in classes
    ])
    ax.invert_yaxis()
    ax.set_title('Perturbation classes redistribute cells among program-defined developmental states', loc='left', weight='bold', pad=14)
    ax.text(
        0, 1.015,
        'Cell state = global dominant cNMF program when that program is in the curated day-specific set; no target→program differential effect is used',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=9, color='#555555'
    )
    ax.set_xlabel('Observed day and dominant curated program state')
    ax.set_ylabel('Reviewed regulator class')
    ax.set_axisbelow(True)
    ax.grid(color='#E8E8E8', linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label('Median adjusted log odds ratio')
    for frac in [0, 0.25, 0.5, 1.0]:
        ax.scatter([], [], s=24 + 390 * frac, facecolor='white', edgecolor='#3F3F3F', label=f'{int(frac * 100)}%')
    ax.legend(
        title='Targets significant\nin median direction', frameon=False,
        bbox_to_anchor=(1.02, 0.02), loc='lower left', labelspacing=1.0, borderaxespad=0
    )
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(outdir / f'Fig2_perturbation_class_program_state_occupancy.{ext}', dpi=350 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)

    # --------------------------------------------------------------
    # Supplementary target-level heatmap.
    # --------------------------------------------------------------
    eff = effects.merge(cm, on='target_name', how='left')
    eff['col'] = eff['day'].astype(str) + ' | ' + eff['state'].astype(str)
    target_order = (
        eff[['target_name', 'regulator_class']].drop_duplicates()
        .sort_values(['regulator_class', 'target_name'], na_position='last')['target_name'].tolist()
    )
    col_order = [f'{d} | {state}' for d, state in col_keys]
    mat = eff.pivot_table(index='target_name', columns='col', values='log_odds_ratio', aggfunc='first').reindex(index=target_order, columns=col_order)
    qmat = eff.pivot_table(index='target_name', columns='col', values='fdr', aggfunc='first').reindex(index=target_order, columns=col_order)
    finite = np.abs(mat.to_numpy(dtype=float))
    finite = finite[np.isfinite(finite)]
    hclip = max(0.5, min(float(np.quantile(finite, 0.98)) if len(finite) else 1.0, 8.0))
    fig, ax = plt.subplots(figsize=(max(9, .58 * len(col_order) + 3), max(10, .12 * len(mat) + 2)))
    im = ax.imshow(mat.to_numpy(), aspect='auto', cmap='RdBu_r', vmin=-hclip, vmax=hclip, interpolation='nearest')
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([f'{d} | {s}' for d, s in col_keys], rotation=45, ha='right')
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=5)
    ax.set_title('Supplementary: complete target-level program-state occupancy effects', loc='left', weight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label('Adjusted log odds ratio')
    yy, xx = np.where(qmat.to_numpy() < float(cfg.get('fdr_threshold', 0.05)))
    ax.scatter(xx, yy, s=2.0, c='black')
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(outdir / f'Fig2_target_program_state_occupancy_heatmap_supplementary.{ext}', dpi=350 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)

    legend = f'''Figure 2. Perturbation classes redistribute cells among program-defined developmental states. Each cell was assigned its global dominant cNMF program across all 50 programs. The primary analysis retained cells only when that globally dominant program belonged to the curated program set and matched the observed collection day; cells were not forced into a curated state. For each biological target and each day-specific dominant-program state, adjusted one-versus-rest logistic regression was performed using a target-union carrier indicator and biological sample, pct_counts_mt, log1p(total_counts), log1p(guides_per_cell), log1p(guide_umi_counts), and log1p(n_genes_by_counts) as covariates. Dot color denotes the median stable target-level adjusted log odds ratio in each reviewed regulator class. Dot size denotes the fraction of evaluable targets with FDR < {float(cfg.get('fdr_threshold',0.05)):g} in the direction of the class median. Class guides were not pooled before target-level modeling. No target→program differential-effect estimate was used to define or test these states.\n'''
    (outdir / 'draft_panel_legend.md').write_text(legend)

    methods = '''# Methods note\n\nProgram-defined states were created from cell-level cNMF usage rather than Leiden clusters. For each cell, the globally highest-usage program among all 50 cNMF programs was identified. A cell entered the primary analysis only when that global dominant program was in the curated primary program list derived from the Hunter/Kyle annotation table and its expected stage matched the cell's observed collection day. Cells whose dominant program was not curated were not reassigned to the best curated program. This makes the primary estimand the relative occupancy of curated program-defined states among cells confidently represented by the curated state space. A sensitivity model compares each curated dominant-program state with all cells collected on the same day.\n\nTarget carriage was defined as at least one assigned guide mapping to the reviewed intended_target_name; multiple guides targeting the same gene were collapsed to one target-union indicator. Cells with guides targeting multiple genes contributed to each corresponding target model. Cells carrying both control and real-targeting guides were retained as target carriers and were not considered pure controls. Per-target one-versus-rest logistic models were fitted separately by day with biological sample and approved technical covariates. Benjamini-Hochberg correction was applied within day across target-by-program-state tests. Class summaries were calculated only after target-level modeling.\n'''
    (outdir / 'methods_note.md').write_text(methods)

    audit = {
        'n_all_cells': int(len(cell)),
        'n_primary_program_state_cells': int(cell['is_primary_program_state'].astype(bool).sum()),
        'n_targets_with_assignments': int(assignment['target_name'].nunique()),
        'n_target_program_state_tests': int(len(effects)),
        'n_classes_displayed': int(class_summary['regulator_class'].nunique()),
        'state_definition': 'global dominant cNMF program; primary curated program list; expected-day match',
        'target_to_program_differential_effects_used': False,
    }
    (outdir / 'analysis_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
