#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse


def clean_day(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip().str.lower()
    s = s.str.replace('day', 'd', regex=False).str.replace(' ', '', regex=False)
    mapping = {
        '0': 'D0', '0.0': 'D0', 'd0': 'D0',
        '1': 'D1', '1.0': 'D1', 'd1': 'D1',
        '2': 'D2', '2.0': 'D2', 'd2': 'D2',
        '3': 'D3', '3.0': 'D3', 'd3': 'D3',
    }
    return s.map(lambda x: mapping.get(x, x.upper()))


def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {'true', '1', 'yes', 'y', 't'}


def canonical_guide(x: str) -> str:
    return str(x).strip().replace(' ', '')


def load_config(path: Path) -> tuple[dict, Path]:
    cfg = yaml.safe_load(path.read_text())
    base = path.parent.parent.resolve()
    for key in ['guide_metadata_tsv', 'class_membership_tsv', 'curated_program_states_tsv']:
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str((base / p).resolve())
    if cfg.get('program_usage_file'):
        p = Path(cfg['program_usage_file'])
        if not p.is_absolute():
            cfg['program_usage_file'] = str((base / p).resolve())
    for key in ['export_dir', 'output_dir']:
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str((base / p).resolve())
    return cfg, base


def get_guide_assignment(adata, key: str):
    ga = adata.obsm[key]
    if hasattr(ga, 'to_memory'):
        ga = ga.to_memory()
    if sparse.issparse(ga):
        return ga.tocsr().astype(np.int8)
    return sparse.csr_matrix(np.asarray(ga).astype(np.int8))


def get_program_usage(adata, cfg: dict):
    """Return cell x program usage, names, source description, and row-availability mask.

    Priority:
      1) Program 0..49 columns already stored in adata.obs;
      2) adata.obsm['cnmf_usage'];
      3) external consensus-usage table configured by program_usage_file.

    The original integrated H5AD was loaded before cNMF usages were added in the
    analysis notebook, so option 3 is required when those in-memory additions were
    never written back to the H5AD.
    """
    n_programs = int(cfg.get('n_programs', 50))
    prefix = str(cfg.get('program_obs_prefix', 'Program '))
    obs_cols = [f'{prefix}{i}' for i in range(n_programs)]
    source = str(cfg.get('program_usage_source', 'auto'))

    if source in {'auto', 'obs'} and all(c in adata.obs.columns for c in obs_cols):
        usage = adata.obs[obs_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        available = np.isfinite(usage).any(axis=1)
        usage = np.nan_to_num(usage, nan=0.0, posinf=0.0, neginf=0.0)
        return usage, obs_cols, 'adata.obs program columns', available

    key = str(cfg.get('program_obsm_key', 'cnmf_usage'))
    if source in {'auto', 'obsm'} and key in adata.obsm:
        usage = adata.obsm[key]
        if hasattr(usage, 'to_memory'):
            usage = usage.to_memory()
        if isinstance(usage, pd.DataFrame):
            usage_df = usage.copy()
            usage_df.index = usage_df.index.astype(str)
            usage_df = usage_df.reindex(adata.obs_names.astype(str))
            usage = usage_df.to_numpy(dtype=np.float32)
        else:
            if sparse.issparse(usage):
                usage = usage.toarray()
            usage = np.asarray(usage, dtype=np.float32)
        if usage.shape[1] < n_programs:
            raise ValueError(f'obsm[{key!r}] has {usage.shape[1]} columns; expected at least {n_programs}.')
        usage = usage[:, :n_programs]
        available = np.isfinite(usage).any(axis=1)
        usage = np.nan_to_num(usage, nan=0.0, posinf=0.0, neginf=0.0)
        return usage, [f'Program {i}' for i in range(n_programs)], f'adata.obsm[{key!r}]', available

    usage_file = cfg.get('program_usage_file')
    if source in {'auto', 'file'} and usage_file:
        usage_path = Path(usage_file)
        if not usage_path.exists():
            raise FileNotFoundError(
                'Program usages are absent from the H5AD and the configured external '
                f'usage table does not exist: {usage_path}'
            )
        sep = str(cfg.get('program_usage_file_sep', '\\t'))
        if sep == '\\t':
            sep = '\t'
        index_col = int(cfg.get('program_usage_file_index_col', 0))
        print(f'Reading external cNMF usage table: {usage_path}')
        usage_df = pd.read_csv(usage_path, sep=sep, index_col=index_col)
        usage_df.index = usage_df.index.astype(str)
        if usage_df.index.duplicated().any():
            print('[warning] Duplicate cell IDs in usage table; keeping first occurrence.')
            usage_df = usage_df.loc[~usage_df.index.duplicated(keep='first')]
        if usage_df.shape[1] < n_programs:
            raise ValueError(
                f'External usage table has {usage_df.shape[1]} columns; expected at least {n_programs}.'
            )
        # Program identities are defined by column order, matching the original notebook.
        usage_df = usage_df.iloc[:, :n_programs].apply(pd.to_numeric, errors='coerce')
        usage_df.columns = [f'Program {i}' for i in range(n_programs)]
        query_ids = pd.Index(adata.obs_names.astype(str))
        overlap = query_ids.intersection(usage_df.index)
        overlap_fraction = len(overlap) / max(1, len(query_ids))
        print(
            f'cNMF usage overlap: {len(overlap):,}/{len(query_ids):,} cells '
            f'({overlap_fraction:.1%})'
        )
        min_overlap = float(cfg.get('minimum_program_usage_overlap_fraction', 0.75))
        if overlap_fraction < min_overlap:
            raise ValueError(
                f'Only {overlap_fraction:.1%} of H5AD cells occur in the external cNMF usage table; '
                f'minimum configured overlap is {min_overlap:.1%}. Check that the usage file belongs '
                'to this exact integrated H5AD and that cell barcodes use the same format.'
            )
        aligned = usage_df.reindex(query_ids)
        available = aligned.notna().any(axis=1).to_numpy()
        if not bool(cfg.get('allow_missing_program_usage_cells', True)) and not available.all():
            raise ValueError(
                f'{(~available).sum():,} cells lack cNMF usages and allow_missing_program_usage_cells is false.'
            )
        usage = aligned.fillna(0.0).to_numpy(dtype=np.float32)
        return usage, list(aligned.columns), f'external file: {usage_path}', available

    raise ValueError(
        'Could not find program usages. The integrated H5AD may not contain the '
        'in-memory cNMF additions made in ipsc-ec.Hippo.figure.ipynb. Expected '
        f'all obs columns {obs_cols[:3]}... or obsm[{key!r}], or set program_usage_file '
        'to the consensus usage table used in that notebook.'
    )


def write_tsv_gz(df: pd.DataFrame, path: Path):
    df.to_csv(path, sep='\t', index=False, compression='gzip')


def main():
    ap = argparse.ArgumentParser(description='Export real-data inputs for program-defined perturbation-class occupancy analysis.')
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg, package_dir = load_config(cfg_path)
    outdir = Path(cfg['export_dir'])
    outdir.mkdir(parents=True, exist_ok=True)

    h5ad_path = Path(cfg['query_h5ad'])
    if not h5ad_path.exists():
        raise FileNotFoundError(f'Query H5AD not found: {h5ad_path}')

    try:
        import anndata as ad
    except ImportError as e:
        raise RuntimeError('Install anndata/scanpy before running this exporter.') from e

    print(f'Reading H5AD in backed mode: {h5ad_path}')
    adata = ad.read_h5ad(h5ad_path, backed='r')

    required_obs = [
        cfg['day_col'], cfg['replicate_col'],
        'pct_counts_mt', 'total_counts', 'guides_per_cell',
        'guide_umi_counts', 'n_genes_by_counts'
    ]
    missing_obs = [c for c in required_obs if c not in adata.obs.columns]
    if missing_obs:
        raise ValueError(f'Missing required obs columns: {missing_obs}. Available: {adata.obs.columns.tolist()}')
    if cfg['guide_assignment_key'] not in adata.obsm:
        raise ValueError(f"Missing obsm[{cfg['guide_assignment_key']!r}]. Available: {list(adata.obsm.keys())}")
    if cfg['guide_names_key'] not in adata.uns:
        raise ValueError(f"Missing uns[{cfg['guide_names_key']!r}]. Available: {list(adata.uns.keys())}")

    obs = adata.obs.copy()
    obs['cell_id'] = adata.obs_names.astype(str)
    obs['day'] = clean_day(obs[cfg['day_col']])
    obs['biological_replicate'] = obs[cfg['replicate_col']].astype(str)

    # ------------------------------------------------------------------
    # Program-defined states: global argmax over all 50 programs.
    # A cell is never forced into the curated set. The primary analysis
    # retains only cells whose global argmax is a primary curated program
    # expected for the observed day.
    # ------------------------------------------------------------------
    program_table = pd.read_csv(cfg['curated_program_states_tsv'], sep='\t')
    program_table['program_number'] = program_table['program_number'].astype(int)
    program_table['use_in_primary'] = program_table['use_in_primary'].map(parse_bool)
    program_table['use_in_expanded_sensitivity'] = program_table['use_in_expanded_sensitivity'].map(parse_bool)
    program_table = program_table.sort_values('plot_order')

    usage, usage_names, usage_source, has_program_usage = get_program_usage(adata, cfg)
    if usage.shape[0] != adata.n_obs:
        raise ValueError(f'Program usage matrix has {usage.shape[0]} rows; expected {adata.n_obs}.')
    usage = np.nan_to_num(usage, nan=-np.inf)
    dominant_number = np.argmax(usage, axis=1).astype(int)
    top1 = usage[np.arange(usage.shape[0]), dominant_number]
    # second-largest usage for dominance margin
    top2 = np.partition(usage, -2, axis=1)[:, -2]
    finite_usage = np.where(np.isfinite(usage), usage, 0.0)
    total_usage = finite_usage.sum(axis=1)
    dominant_fraction = np.divide(top1, total_usage, out=np.zeros_like(top1, dtype=float), where=total_usage > 0)
    dominance_margin = top1 - top2

    program_meta = program_table.set_index('program_number').to_dict('index')
    obs['has_program_usage'] = has_program_usage
    obs['global_dominant_program_number'] = dominant_number
    obs['global_dominant_program_id'] = [f'P{i}' for i in dominant_number]
    obs['global_dominant_usage'] = top1
    obs['dominant_usage_fraction'] = dominant_fraction
    obs['dominance_margin'] = dominance_margin
    obs['global_dominant_program_name'] = [program_meta.get(i, {}).get('program_name', 'Non-curated program') for i in dominant_number]
    obs['dominant_program_expected_day'] = [program_meta.get(i, {}).get('expected_day', np.nan) for i in dominant_number]

    primary_numbers = set(program_table.loc[program_table['use_in_primary'], 'program_number'])
    expanded_numbers = set(program_table.loc[program_table['use_in_expanded_sensitivity'], 'program_number'])
    min_fraction = float(cfg.get('minimum_dominant_fraction', 0.0))
    min_margin = float(cfg.get('minimum_dominance_margin', 0.0))
    dominance_ok = (obs['dominant_usage_fraction'] >= min_fraction) & (obs['dominance_margin'] >= min_margin)
    expected_day_match = obs['day'].astype(str).to_numpy() == obs['dominant_program_expected_day'].astype(str).to_numpy()

    obs['is_primary_curated_dominant'] = obs['has_program_usage'] & obs['global_dominant_program_number'].isin(primary_numbers) & dominance_ok
    obs['is_expanded_curated_dominant'] = obs['has_program_usage'] & obs['global_dominant_program_number'].isin(expanded_numbers) & dominance_ok
    obs['expected_day_match'] = expected_day_match
    obs['is_primary_program_state'] = obs['is_primary_curated_dominant'] & obs['expected_day_match']
    obs['is_expanded_program_state'] = obs['is_expanded_curated_dominant'] & obs['expected_day_match']
    obs['assigned_state'] = np.where(obs['is_primary_program_state'], obs['global_dominant_program_id'], pd.NA)
    obs['assigned_program_name'] = np.where(obs['is_primary_program_state'], obs['global_dominant_program_name'], pd.NA)

    # ------------------------------------------------------------------
    # Guide and target-union assignments.
    # ------------------------------------------------------------------
    guide_names = list(map(str, adata.uns[cfg['guide_names_key']]))
    guide_to_idx = {g: i for i, g in enumerate(guide_names)}
    canonical_to_idx = {}
    duplicates = set()
    for g, i in guide_to_idx.items():
        cg = canonical_guide(g)
        if cg in canonical_to_idx:
            duplicates.add(cg)
        else:
            canonical_to_idx[cg] = i
    for cg in duplicates:
        canonical_to_idx.pop(cg, None)

    meta = pd.read_csv(cfg['guide_metadata_tsv'], sep='\t')
    needed = {'guide_id', 'type', 'intended_target_name'}
    if not needed.issubset(meta.columns):
        raise ValueError(f'Guide metadata missing columns {sorted(needed - set(meta.columns))}')
    meta = meta.copy()
    meta['guide_id'] = meta['guide_id'].astype(str)
    meta['guide_index'] = meta['guide_id'].map(guide_to_idx)
    meta['match_method'] = np.where(meta['guide_index'].notna(), 'exact', '')
    for idx in meta.index[meta['guide_index'].isna()]:
        cg = canonical_guide(meta.at[idx, 'guide_id'])
        if cg in canonical_to_idx:
            meta.at[idx, 'guide_index'] = canonical_to_idx[cg]
            meta.at[idx, 'match_method'] = 'canonical'
    mapped = meta[meta['guide_index'].notna()].copy()
    mapped['guide_index'] = mapped['guide_index'].astype(int)
    mapped['guide_name_in_h5ad'] = mapped['guide_index'].map(lambda i: guide_names[i])
    mapped['is_targeting'] = mapped['type'].astype(str).eq('targeting')
    mapped['is_control'] = mapped['type'].astype(str).isin(['safe-targeting', 'non-targeting'])

    guide_audit = pd.DataFrame({'guide_index': np.arange(len(guide_names)), 'guide_name_in_h5ad': guide_names})
    guide_audit = guide_audit.merge(
        mapped[['guide_index', 'guide_id', 'type', 'intended_target_name', 'match_method']],
        on='guide_index', how='left'
    )
    guide_audit['mapped'] = guide_audit['guide_id'].notna()
    guide_audit.to_csv(outdir / 'guide_to_target_mapping_audit.tsv', sep='\t', index=False)

    ga = get_guide_assignment(adata, cfg['guide_assignment_key'])
    if ga.shape != (adata.n_obs, len(guide_names)):
        raise ValueError(f'guide_assignment shape {ga.shape} does not match cells/guides {(adata.n_obs, len(guide_names))}')

    targeting_guide_idx = sorted(mapped.loc[mapped['is_targeting'], 'guide_index'].unique())
    control_guide_idx = sorted(mapped.loc[mapped['is_control'], 'guide_index'].unique())
    n_targeting_guides = np.asarray(ga[:, targeting_guide_idx].sum(axis=1)).ravel().astype(int)
    n_control_guides = np.asarray(ga[:, control_guide_idx].sum(axis=1)).ravel().astype(int)
    obs['n_targeting_guides_assigned'] = n_targeting_guides
    obs['n_control_guides_assigned'] = n_control_guides
    obs['has_targeting_guide'] = n_targeting_guides > 0
    obs['has_control_guide'] = n_control_guides > 0
    obs['is_pure_control'] = (n_control_guides > 0) & (n_targeting_guides == 0)
    obs['is_mixed_control_target'] = (n_control_guides > 0) & (n_targeting_guides > 0)

    targeting_map = mapped[mapped['is_targeting'] & mapped['intended_target_name'].notna()].copy()
    targeting_map['target_name'] = targeting_map['intended_target_name'].astype(str)
    target_names = sorted(targeting_map['target_name'].unique())
    target_to_idx = {t: i for i, t in enumerate(target_names)}
    rr = targeting_map['guide_index'].to_numpy(int)
    cc = targeting_map['target_name'].map(target_to_idx).to_numpy(int)
    guide_target_matrix = sparse.csr_matrix((np.ones(len(targeting_map), dtype=np.int8), (rr, cc)), shape=(len(guide_names), len(target_names)))
    target_counts = (ga @ guide_target_matrix).tocsr()
    target_counts.eliminate_zeros()

    coo = target_counts.tocoo()
    assignments = pd.DataFrame({
        'cell_index': coo.row.astype(int),
        'target_index': coo.col.astype(int),
        'n_guides_for_target': coo.data.astype(int),
    })
    assignments['cell_id'] = obs['cell_id'].to_numpy()[assignments['cell_index'].to_numpy()]
    assignments['target_name'] = np.asarray(target_names, dtype=object)[assignments['target_index'].to_numpy()]
    assignments = assignments[['cell_id', 'target_name', 'n_guides_for_target']]
    obs['n_unique_target_genes'] = np.diff(target_counts.indptr)

    cell_cols = [
        'cell_id', 'day', 'biological_replicate',
        'has_program_usage', 'global_dominant_program_number', 'global_dominant_program_id', 'global_dominant_program_name',
        'global_dominant_usage', 'dominant_usage_fraction', 'dominance_margin',
        'dominant_program_expected_day', 'expected_day_match',
        'is_primary_curated_dominant', 'is_expanded_curated_dominant',
        'is_primary_program_state', 'is_expanded_program_state',
        'assigned_state', 'assigned_program_name',
        'pct_counts_mt', 'total_counts', 'guides_per_cell', 'guide_umi_counts', 'n_genes_by_counts',
        'n_targeting_guides_assigned', 'n_control_guides_assigned', 'n_unique_target_genes',
        'has_targeting_guide', 'has_control_guide', 'is_pure_control', 'is_mixed_control_target'
    ]
    cell_meta = obs[cell_cols].copy()
    write_tsv_gz(cell_meta, outdir / 'occupancy_cell_metadata.tsv.gz')
    write_tsv_gz(assignments, outdir / 'occupancy_cell_target_assignments.tsv.gz')

    # Program-state audits.
    dominant_counts = (
        cell_meta.groupby(['day', 'global_dominant_program_id', 'global_dominant_program_name'], dropna=False, as_index=False)
        .agg(n_cells=('cell_id', 'size'), median_dominant_fraction=('dominant_usage_fraction', 'median'))
    )
    dominant_counts.to_csv(outdir / 'dominant_program_counts_by_day.tsv', sep='\t', index=False)

    program_state_counts = (
        cell_meta[cell_meta['is_primary_program_state']]
        .groupby(['day', 'biological_replicate', 'assigned_state', 'assigned_program_name'], as_index=False)
        .agg(n_cells=('cell_id', 'size'), median_dominant_fraction=('dominant_usage_fraction', 'median'))
    )
    program_state_counts.to_csv(outdir / 'program_state_counts_by_day_and_replicate.tsv', sep='\t', index=False)

    day_qc = (
        cell_meta.groupby('day', as_index=False)
        .agg(
            n_all_cells=('cell_id', 'size'),
            n_has_program_usage=('has_program_usage', 'sum'),
            n_primary_curated_dominant=('is_primary_curated_dominant', 'sum'),
            n_primary_program_state=('is_primary_program_state', 'sum'),
            n_expanded_program_state=('is_expanded_program_state', 'sum'),
            n_pure_controls=('is_pure_control', 'sum'),
            n_mixed_control_target=('is_mixed_control_target', 'sum'),
        )
    )
    day_qc['fraction_with_program_usage'] = day_qc['n_has_program_usage'] / day_qc['n_all_cells']
    day_qc['fraction_in_primary_program_state'] = day_qc['n_primary_program_state'] / day_qc['n_all_cells']
    day_qc.to_csv(outdir / 'program_state_assignment_qc.tsv', sep='\t', index=False)

    carrier_counts = (
        assignments.merge(cell_meta[['cell_id', 'day', 'is_primary_program_state']], on='cell_id', how='left')
        .groupby(['target_name', 'day'], as_index=False)
        .agg(
            n_carrier_cells=('cell_id', 'nunique'),
            n_carrier_cells_in_primary_program_states=('is_primary_program_state', 'sum'),
            median_guides_for_target=('n_guides_for_target', 'median')
        )
    )
    carrier_counts.to_csv(outdir / 'target_carrier_counts_by_day.tsv', sep='\t', index=False)

    control_qc = (
        cell_meta.groupby(['day', 'biological_replicate'], as_index=False)
        .agg(
            n_cells=('cell_id', 'size'),
            n_pure_controls=('is_pure_control', 'sum'),
            n_mixed_control_target=('is_mixed_control_target', 'sum'),
            n_any_targeting=('has_targeting_guide', 'sum'),
        )
    )
    control_qc.to_csv(outdir / 'control_definition_qc.tsv', sep='\t', index=False)

    # Copy reviewed tables used by downstream analysis.
    pd.read_csv(cfg['class_membership_tsv'], sep='\t').to_csv(outdir / 'class_membership.tsv', sep='\t', index=False)
    program_table.to_csv(outdir / 'program_state_definitions.tsv', sep='\t', index=False)

    manifest = {
        'query_h5ad': str(h5ad_path),
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'program_usage_source': usage_source,
        'n_cells_with_program_usage': int(has_program_usage.sum()),
        'fraction_cells_with_program_usage': float(has_program_usage.mean()),
        'state_assignment_rule': cfg.get('state_assignment_rule'),
        'n_primary_curated_programs': int(program_table['use_in_primary'].sum()),
        'n_expanded_curated_programs': int(program_table['use_in_expanded_sensitivity'].sum()),
        'n_primary_program_state_cells': int(cell_meta['is_primary_program_state'].sum()),
        'n_target_names': int(len(target_names)),
        'control_definition': 'at least one safe/non-targeting guide and zero targeting guides',
    }
    (outdir / 'export_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
