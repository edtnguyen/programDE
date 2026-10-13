#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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
    return yaml.safe_load(path.read_text())


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
        fit = GLM(np.asarray(y, dtype=float), x.to_numpy(float), family=families.Binomial()).fit(maxiter=100, disp=0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--export-dir', required=True)
    ap.add_argument('--output-dir', default='occupancy_analysis_results')
    ap.add_argument('--n-jobs', type=int, default=4)
    args = ap.parse_args()

    cfg = read_cfg(Path(args.config))
    export_dir = Path(args.export_dir).resolve()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cell = pd.read_csv(export_dir / 'occupancy_cell_metadata.tsv.gz', sep='\t')
    assignment = pd.read_csv(export_dir / 'occupancy_cell_target_assignments.tsv.gz', sep='\t')
    class_membership = pd.read_csv(export_dir / 'class_membership.tsv', sep='\t')
    state_defs = pd.read_csv(export_dir / 'state_definitions.tsv', sep='\t')

    # Build carrier sets.
    target_cells = assignment.groupby('target_name')['cell_id'].apply(set).to_dict()
    carrier_counts = assignment.groupby('target_name')['cell_id'].nunique().to_dict()

    # Preserve only reviewed classes that occur in the experiment.
    class_membership = class_membership.rename(columns={'gene': 'target_name'})
    class_membership = class_membership[class_membership['target_name'].isin(target_cells)].copy()
    class_membership.to_csv(outdir / 'class_membership.tsv', sep='\t', index=False)
    state_defs.to_csv(outdir / 'state_definitions.tsv', sep='\t', index=False)

    min_carriers = int(cfg.get('minimum_carrier_cells_per_day', 25))
    min_state_cells = int(cfg.get('minimum_state_cells_per_day', 25))
    fdr_threshold = float(cfg.get('fdr_threshold', 0.05))

    tasks = []
    coverage_rows = []
    for day, ddf in cell.groupby('day', sort=True):
        day_cell_ids = set(ddf['cell_id'])
        states = ddf['assigned_state'].value_counts()
        evaluable_states = states[states >= min_state_cells].index.tolist()
        for target, cset in target_cells.items():
            n_carrier = len(day_cell_ids & cset)
            coverage_rows.append({'day': day, 'target_name': target, 'n_carrier_cells': n_carrier, 'passes_carrier_threshold': n_carrier >= min_carriers})
            if n_carrier < min_carriers:
                continue
            for state in evaluable_states:
                tasks.append((day, target, state))

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(outdir / 'coverage_and_filtering.tsv', sep='\t', index=False)

    # Precompute day data and designs.
    day_data = {}
    for day, ddf in cell.groupby('day', sort=True):
        ddf = ddf.reset_index(drop=True)
        day_data[day] = (ddf, clean_design(ddf))

    def run_task(task):
        day, target, state = task
        ddf, base_x = day_data[day]
        carriers = ddf['cell_id'].isin(target_cells[target]).to_numpy()
        y = (ddf['assigned_state'].astype(str) == str(state)).to_numpy(int)
        beta, se, z, p, converged, error = fit_glm(y, carriers, base_x)
        n_carrier = int(carriers.sum())
        n_noncarrier = int((~carriers).sum())
        n_state = int(y.sum())
        n_carrier_state = int((carriers & (y == 1)).sum())
        return {
            'day': day, 'target_name': target, 'state': state,
            'log_odds_ratio': beta, 'standard_error': se, 'z_value': z, 'p_value': p,
            'ci_lower': beta - 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan,
            'ci_upper': beta + 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan,
            'n_carrier_cells': n_carrier, 'n_noncarrier_cells': n_noncarrier,
            'n_state_cells': n_state, 'n_carrier_state_cells': n_carrier_state,
            'converged': converged, 'model_error': error,
            'analysis_population': 'all cells within day; marginal target-union indicator',
            'covariates': 'biological_sample + log1p(total_counts) + log1p(guides_per_cell) + log1p(guide_umi_counts) + log1p(n_genes_by_counts) + pct_counts_mt',
        }

    effects = pd.DataFrame(Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(run_task)(t) for t in tasks))
    effects['fdr'] = np.nan
    for day, idx in effects.groupby('day').groups.items():
        valid = effects.loc[idx, 'p_value'].notna()
        if valid.any():
            effects.loc[np.asarray(idx)[valid.to_numpy()], 'fdr'] = multipletests(effects.loc[np.asarray(idx)[valid.to_numpy()], 'p_value'], method='fdr_bh')[1]
    effects['significant'] = effects['fdr'] < fdr_threshold
    effects.to_csv(outdir / 'target_state_occupancy_effects.tsv', sep='\t', index=False)

    # Replicate-specific sensitivity.
    rep_rows = []
    for day, ddf_all in cell.groupby('day', sort=True):
        for rep, ddf in ddf_all.groupby('biological_replicate', sort=True):
            ddf = ddf.reset_index(drop=True)
            # no sample dummies within replicate
            base = clean_design(ddf)
            base = base[[c for c in base.columns if not c.startswith('sample_')]]
            states = ddf['assigned_state'].value_counts()
            for target in coverage.query('day == @day and passes_carrier_threshold')['target_name']:
                carriers = ddf['cell_id'].isin(target_cells[target]).to_numpy()
                if carriers.sum() < max(8, min_carriers // 3):
                    continue
                for state in states[states >= max(10, min_state_cells // 2)].index:
                    y = (ddf['assigned_state'].astype(str) == str(state)).to_numpy(int)
                    beta, se, z, p, conv, err = fit_glm(y, carriers, base)
                    rep_rows.append({
                        'day': day, 'biological_replicate': rep, 'target_name': target, 'state': state,
                        'log_odds_ratio': beta, 'standard_error': se, 'z_value': z, 'p_value': p,
                        'n_carrier_cells': int(carriers.sum()), 'n_state_cells': int(y.sum()),
                        'converged': conv, 'model_error': err,
                    })
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(outdir / 'target_state_occupancy_by_replicate.tsv', sep='\t', index=False)

    # Add replicate consistency summary to target-level effects.
    if len(rep_df):
        comb = []
        for key, sub in rep_df.groupby(['day', 'target_name', 'state']):
            b, se, z, p = combine_fixed_effect(sub)
            signs = np.sign(sub['log_odds_ratio'].dropna())
            concord = float((signs == np.sign(b)).mean()) if len(signs) and np.isfinite(b) else np.nan
            comb.append(dict(zip(['day', 'target_name', 'state'], key)) | {
                'replicate_meta_log_odds_ratio': b, 'replicate_meta_standard_error': se,
                'replicate_meta_p_value': p, 'replicate_sign_concordance': concord,
                'n_replicates_evaluable': int(sub['biological_replicate'].nunique()),
            })
        comb = pd.DataFrame(comb)
        effects = effects.merge(comb, on=['day', 'target_name', 'state'], how='left')
        effects.to_csv(outdir / 'target_state_occupancy_effects.tsv', sep='\t', index=False)

    # Class summary, target-level estimates are not pooled before fitting.
    joined = effects.merge(class_membership[['target_name', 'regulator_class']], on='target_name', how='inner')
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
            'regulator_class': cls, 'day': day, 'state': state,
            'n_evaluable_targets': n,
            'median_log_odds_ratio': med,
            'fraction_significant_in_displayed_direction': float(sig_dir.mean()),
            'n_significant_in_displayed_direction': int(sig_dir.sum()),
            'displayed_direction': 'enriched' if direction > 0 else 'depleted',
        })
    class_summary = pd.DataFrame(class_rows)
    class_summary.to_csv(outdir / 'class_state_occupancy_summary.tsv', sep='\t', index=False)

    # Soft-state sensitivity only if actual non-one-hot probabilities are supplied separately.
    soft_note = 'Soft assignment sensitivity not run: the legacy author-defined atlas supplies hard states/one-hot probabilities only.'
    (outdir / 'soft_state_sensitivity_note.txt').write_text(soft_note)

    # Plotting.
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    # Column ordering from observed day and state definition order.
    state_order_map = dict(zip(state_defs['assigned_state'].astype(str), state_defs['state_order']))
    state_label_map = dict(zip(state_defs['assigned_state'].astype(str), state_defs['state_label'].astype(str)))
    cols = class_summary[['day', 'state']].drop_duplicates().copy()
    day_order = {d: i for i, d in enumerate(['D0', 'D1', 'D2', 'D3'])}
    cols['day_i'] = cols['day'].map(day_order).fillna(99)
    cols['state_i'] = cols['state'].map(state_order_map).fillna(99)
    cols = cols.sort_values(['day_i', 'state_i'])
    col_keys = list(map(tuple, cols[['day', 'state']].to_records(index=False)))

    preferred_class_order = [
        'TGF/BMP/Activin',
        'WNT / mesoderm regulators',
        'VEGF / EC specification',
        'Hippo / YAP',
        'Chromatin regulation',
        'Transcription / RNA processing',
        'Translation / proteostasis',
        'Apoptosis / cell cycle / stress',
        'Metabolism / trafficking / adhesion',
        'Other reviewed regulators',
    ]
    present = set(class_summary['regulator_class'].astype(str))
    classes = [c for c in preferred_class_order if c in present]
    classes += sorted(present - set(classes))
    ymap = {c: i for i, c in enumerate(classes)}
    xmap = {k: i for i, k in enumerate(col_keys)}

    # Save class coverage explicitly.
    class_coverage = (
        class_summary.groupby('regulator_class', as_index=False)
        .agg(
            minimum_evaluable_targets=('n_evaluable_targets', 'min'),
            maximum_evaluable_targets=('n_evaluable_targets', 'max'),
            n_day_state_combinations=('state', 'size'),
        )
    )
    class_coverage.to_csv(outdir / 'class_coverage_summary.tsv', sep='\t', index=False)

    clip = max(0.5, float(np.nanquantile(np.abs(class_summary['median_log_odds_ratio']), 0.95)))
    norm2 = TwoSlopeNorm(vmin=-clip, vcenter=0, vmax=clip)
    cmap = mpl.colormaps['RdBu_r']

    fig_w = max(9.0, 0.72 * len(col_keys) + 4.2)
    fig_h = max(5.0, 0.55 * len(classes) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for _, r in class_summary.iterrows():
        if r['regulator_class'] not in ymap:
            continue
        x = xmap[(r['day'], r['state'])]
        y = ymap[r['regulator_class']]
        size = 28 + 380 * r['fraction_significant_in_displayed_direction']
        ax.scatter(
            x, y, s=size,
            c=[cmap(norm2(r['median_log_odds_ratio']))],
            edgecolors='#3F3F3F', linewidths=0.55, zorder=3
        )

    # Light vertical separators between collection days.
    ordered_days = [d for d in ['D0', 'D1', 'D2', 'D3'] if any(k[0] == d for k in col_keys)]
    for d1, d2 in zip(ordered_days[:-1], ordered_days[1:]):
        last_x = max(xmap[k] for k in col_keys if k[0] == d1)
        next_x = min(xmap[k] for k in col_keys if k[0] == d2)
        ax.axvline((last_x + next_x) / 2, color='#BDBDBD', linewidth=0.8, zorder=0)

    def wrap_state_label(state):
        label = state_label_map.get(str(state), str(state))
        replacements = {
            'Neuroectoderm-like iPSC': 'Neuroectoderm-like\niPSC',
            'Pre-EC / vascular progenitor': 'Pre-EC / vascular\nprogenitor',
            'Fibro-mesenchymal alternative': 'Fibro-mesenchymal\nalternative',
            'Neural-mesenchymal': 'Neural-\nmesenchymal',
            'Other / unresolved': 'Other /\nunresolved',
        }
        return replacements.get(label, label)

    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels([f'{d}\n{wrap_state_label(s)}' for d, s in col_keys], rotation=40, ha='right')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([
        f"{c}  (n≥{int(class_summary[class_summary.regulator_class==c].n_evaluable_targets.min())})"
        for c in classes
    ])
    ax.invert_yaxis()
    ax.set_title('Perturbation classes redistribute cells among distinct developmental states', loc='left', weight='bold', pad=14)
    ax.text(
        0, 1.015,
        'Adjusted target-level one-vs-rest occupancy effects; state boundaries fixed before target testing',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=9, color='#555555'
    )
    ax.set_xlabel('Day-specific coarse state')
    ax.set_ylabel('Reviewed regulator class')
    ax.set_axisbelow(True)
    ax.grid(color='#E8E8E8', linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label('Median adjusted log odds ratio')
    for frac in [0, 0.25, 0.5, 1.0]:
        ax.scatter([], [], s=28 + 380*frac, facecolor='white', edgecolor='#3F3F3F', label=f'{int(frac*100)}%')
    ax.legend(
        title='Targets significant\nin median direction',
        frameon=False, bbox_to_anchor=(1.02, 0.02), loc='lower left',
        labelspacing=1.0, borderaxespad=0
    )
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(outdir / f'Fig2_perturbation_class_state_occupancy.{ext}', dpi=350 if ext=='png' else None, bbox_inches='tight')
    plt.close(fig)

    # Supplementary target-level heatmap.
    eff = effects.merge(class_membership[['target_name', 'regulator_class']], on='target_name', how='left')
    eff['col'] = eff['day'].astype(str) + ' | ' + eff['state'].astype(str)
    target_order = (
        eff[['target_name', 'regulator_class']].drop_duplicates()
        .sort_values(['regulator_class', 'target_name'], na_position='last')['target_name'].tolist()
    )
    col_order = [f'{d} | {state_label_map.get(str(s), str(s))}' for d, s in col_keys]
    mat = eff.pivot_table(index='target_name', columns='col', values='log_odds_ratio', aggfunc='first').reindex(index=target_order, columns=col_order)
    qmat = eff.pivot_table(index='target_name', columns='col', values='fdr', aggfunc='first').reindex(index=target_order, columns=col_order)
    hclip = max(0.5, float(np.nanquantile(np.abs(mat.to_numpy()), 0.98)))
    fig, ax = plt.subplots(figsize=(max(8, .55*len(col_order)+3), max(10, .12*len(mat)+2)))
    im = ax.imshow(mat.to_numpy(), aspect='auto', cmap='RdBu_r', vmin=-hclip, vmax=hclip, interpolation='nearest')
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha='right')
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=5)
    ax.set_title('Supplementary: complete target-level coarse-state occupancy effects', loc='left', weight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label('Adjusted log odds ratio')
    # significance dots
    yy, xx = np.where(qmat.to_numpy() < fdr_threshold)
    ax.scatter(xx, yy, s=2.0, c='black')
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(outdir / f'Fig2_target_state_occupancy_heatmap_supplementary.{ext}', dpi=350 if ext=='png' else None, bbox_inches='tight')
    plt.close(fig)

    # Legend and methods.
    legend = f'''Figure 2. Perturbation classes redistribute cells among distinct developmental states.\nRows denote reviewed regulator classes and columns denote day-specific author-defined coarse states. For each target and state, adjusted one-versus-rest logistic regression was performed separately by collection day using a target-union carrier indicator and biological sample, pct_counts_mt, log1p(total_counts), log1p(guides_per_cell), log1p(guide_umi_counts), and log1p(n_genes_by_counts) as covariates. Dot color shows the median target-level adjusted log odds ratio within each class. Dot size shows the fraction of evaluable targets with FDR < {fdr_threshold:g} in the direction of the displayed median. Classes with fewer than {int(cfg.get('minimum_targets_per_class',4))} evaluable targets were omitted. State boundaries were taken from the existing author-defined Leiden mapping and were not re-estimated from target-carrier cells.\n'''
    (outdir / 'draft_panel_legend.md').write_text(legend)
    methods = f'''# Methods note\n\nStates were fixed before occupancy testing using the existing author-defined mapping of `leiden_res_0.5`. Each biological target was represented by a union indicator equal to one when at least one guide targeting that intended_target_name was assigned. Cells with multiple targeting genes contributed to each corresponding target indicator, as expected under the high-MOI design. Cells carrying both control and targeting guides were treated as target carriers in the primary all-cell marginal model and were flagged in the exported metadata. Pure controls were defined as cells with at least one safe/non-targeting guide and no real-targeting guides and were used for descriptive and strict-control sensitivity analyses. Models were fit separately by day and state using binomial logistic regression. Benjamini-Hochberg adjustment was applied within day across the target-by-state testing family.\n'''
    (outdir / 'methods_note.md').write_text(methods)

    # run audit
    audit = {
        'n_cells': int(len(cell)), 'n_targets_with_assignments': int(len(target_cells)),
        'n_tests': int(len(effects)), 'n_classes_displayed': int(class_summary['regulator_class'].nunique()) if len(class_summary) else 0,
        'soft_sensitivity_run': False, 'state_source': 'existing author-defined Leiden mapping',
    }
    (outdir / 'analysis_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
