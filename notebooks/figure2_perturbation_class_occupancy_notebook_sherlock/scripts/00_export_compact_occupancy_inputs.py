#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import zipfile
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


def canonical_guide(x: str) -> str:
    return str(x).strip().replace(' ', '')


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    base = path.parent.parent.resolve()
    for key in ['guide_metadata_tsv', 'legacy_state_map_tsv', 'class_membership_tsv', 'state_definitions_tsv']:
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str((base / p).resolve())
    out = Path(cfg['output_dir'])
    if not out.is_absolute():
        cfg['output_dir'] = str((base / out).resolve())
    return cfg


def get_guide_assignment(adata, key: str):
    ga = adata.obsm[key]
    if hasattr(ga, 'to_memory'):
        ga = ga.to_memory()
    if sparse.issparse(ga):
        return ga.tocsr().astype(np.int8)
    return sparse.csr_matrix(np.asarray(ga).astype(np.int8))


def write_tsv_gz(df: pd.DataFrame, path: Path):
    df.to_csv(path, sep='\t', index=False, compression='gzip')


def main():
    ap = argparse.ArgumentParser(description='Export compact real-data inputs for coarse-state occupancy analysis.')
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    outdir = Path(cfg['output_dir'])
    outdir.mkdir(parents=True, exist_ok=True)

    h5ad_path = Path(cfg['query_h5ad'])
    if not h5ad_path.exists():
        raise FileNotFoundError(f'Query H5AD not found: {h5ad_path}')

    try:
        import anndata as ad
    except ImportError as e:
        raise RuntimeError('Install anndata/scanpy before running this exporter.') from e

    print(f'Reading H5AD metadata/backed object: {h5ad_path}')
    adata = ad.read_h5ad(h5ad_path, backed='r')

    required_obs = [
        cfg['cluster_col'], cfg['day_col'], cfg['replicate_col'],
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
    obs['day_clean'] = clean_day(obs[cfg['day_col']])
    obs['biological_replicate'] = obs[cfg['replicate_col']].astype(str)

    # Apply author-defined legacy state map recovered from the prior notebook.
    state_map = pd.read_csv(cfg['legacy_state_map_tsv'], sep='\t', dtype={'source_cluster': str})
    cluster_to_state = dict(zip(state_map['source_cluster'].astype(str), state_map['assigned_state'].astype(str)))
    obs['source_cluster'] = obs[cfg['cluster_col']].astype(str)
    obs['assigned_state'] = obs['source_cluster'].map(cluster_to_state)
    if obs['assigned_state'].isna().any():
        missing_clusters = sorted(obs.loc[obs['assigned_state'].isna(), 'source_cluster'].unique())
        raise ValueError(f'Unmapped clusters in {cfg["cluster_col"]}: {missing_clusters}')

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

    # Audit all H5AD guide columns.
    mapped_indices = set(mapped['guide_index'])
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

    # Build guide-to-target sparse map and target-union assignments.
    targeting_map = mapped[mapped['is_targeting'] & mapped['intended_target_name'].notna()].copy()
    targeting_map['target_name'] = targeting_map['intended_target_name'].astype(str)
    target_names = sorted(targeting_map['target_name'].unique())
    target_to_idx = {t: i for i, t in enumerate(target_names)}
    rr = targeting_map['guide_index'].to_numpy(int)
    cc = targeting_map['target_name'].map(target_to_idx).to_numpy(int)
    data = np.ones(len(targeting_map), dtype=np.int8)
    guide_target_matrix = sparse.csr_matrix((data, (rr, cc)), shape=(len(guide_names), len(target_names)))
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

    unique_target_counts = np.diff(target_counts.indptr)
    obs['n_unique_target_genes'] = unique_target_counts

    cell_cols = [
        'cell_id', 'day_clean', 'biological_replicate', 'source_cluster', 'assigned_state',
        'pct_counts_mt', 'total_counts', 'guides_per_cell', 'guide_umi_counts', 'n_genes_by_counts',
        'n_targeting_guides_assigned', 'n_control_guides_assigned', 'n_unique_target_genes',
        'has_targeting_guide', 'has_control_guide', 'is_pure_control', 'is_mixed_control_target'
    ]
    cell_meta = obs[cell_cols].copy()
    cell_meta = cell_meta.rename(columns={'day_clean': 'day'})

    write_tsv_gz(cell_meta, outdir / 'occupancy_cell_metadata.tsv.gz')
    write_tsv_gz(assignments, outdir / 'occupancy_cell_target_assignments.tsv.gz')

    carrier_counts = (
        assignments.merge(cell_meta[['cell_id', 'day']], on='cell_id', how='left')
        .groupby(['target_name', 'day'], as_index=False)
        .agg(n_carrier_cells=('cell_id', 'nunique'), median_guides_for_target=('n_guides_for_target', 'median'))
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

    state_counts = (
        cell_meta.groupby(['day', 'biological_replicate', 'assigned_state'], as_index=False)
        .agg(n_cells=('cell_id', 'size'))
    )
    state_counts.to_csv(outdir / 'state_counts_by_day_and_replicate.tsv', sep='\t', index=False)

    # Copy reviewed tables into the export bundle.
    for src_key, out_name in [
        ('class_membership_tsv', 'class_membership.tsv'),
        ('state_definitions_tsv', 'state_definitions.tsv'),
    ]:
        shutil.copyfile(cfg[src_key], outdir / out_name)

    audit = {
        'query_h5ad': str(h5ad_path),
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'n_guides_h5ad': len(guide_names),
        'n_guides_mapped': int(guide_audit['mapped'].sum()),
        'n_targeting_guides_mapped': len(targeting_guide_idx),
        'n_control_guides_mapped': len(control_guide_idx),
        'n_unique_targets_exported': len(target_names),
        'n_cell_target_assignments': len(assignments),
        'n_pure_control_cells': int(cell_meta['is_pure_control'].sum()),
        'n_mixed_control_target_cells': int(cell_meta['is_mixed_control_target'].sum()),
        'state_source': f"existing {cfg['cluster_col']} mapped with prior author-defined labels",
        'soft_state_probabilities_available': False,
    }
    (outdir / 'export_audit.json').write_text(json.dumps(audit, indent=2))

    notes = f'''Compact occupancy export created from the real processed CC-Perturb-seq H5AD.\n\nState definition\n----------------\nStates are the author-defined mapping of existing {cfg['cluster_col']} clusters. No new clustering or target-to-program effects were used.\n\nTarget carrier definition\n-------------------------\nA cell is a carrier for a biological target when it contains at least one guide whose metadata intended_target_name equals that target. Multiple guides for the same target collapse to one target-union indicator; n_guides_for_target is retained. Cells with multiple real targets appear once for each target.\n\nControls\n--------\nPure control = at least one safe-targeting or non-targeting guide and zero targeting guides. Mixed control+target cells are retained as target carriers in the primary high-MOI marginal model and flagged explicitly.\n\nCovariates exported\n-------------------\npct_counts_mt, biological_sample, total_counts, guides_per_cell, guide_umi_counts, n_genes_by_counts. Count-like covariates should be log1p transformed in the occupancy model.\n'''
    (outdir / 'export_method_notes.md').write_text(notes)

    zip_path = outdir.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in outdir.rglob('*'):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(outdir))
    print(json.dumps(audit, indent=2))
    print(f'Wrote compact export: {zip_path}')


if __name__ == '__main__':
    main()
