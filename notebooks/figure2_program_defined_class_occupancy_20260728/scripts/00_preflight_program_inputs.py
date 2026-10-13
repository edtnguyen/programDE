#!/usr/bin/env python
from pathlib import Path
import argparse, yaml, pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--config', required=True)
a=ap.parse_args()
cp=Path(a.config).resolve(); cfg=yaml.safe_load(cp.read_text()); base=cp.parent.parent
q=Path(cfg['query_h5ad']); u=Path(cfg.get('program_usage_file',''))
print('query_h5ad:', q, 'exists=', q.exists())
print('program_usage_file:', u, 'exists=', u.exists())
for key in ['guide_metadata_tsv','class_membership_tsv','curated_program_states_tsv']:
    p=Path(cfg[key]); p=p if p.is_absolute() else base/p
    print(f'{key}:', p, 'exists=', p.exists())
if not q.exists(): raise SystemExit(1)
import anndata as ad
x=ad.read_h5ad(q, backed='r')
print('H5AD shape:', x.shape)
print('Program obs columns:', len([c for c in x.obs.columns if str(c).startswith('Program ')]))
print('obsm keys:', list(x.obsm.keys()))
print('guide_assignment shape:', x.obsm[cfg['guide_assignment_key']].shape if cfg['guide_assignment_key'] in x.obsm else 'MISSING')
print('guide_names length:', len(x.uns[cfg['guide_names_key']]) if cfg['guide_names_key'] in x.uns else 'MISSING')
print('required obs missing:', [c for c in [cfg['day_col'],cfg['replicate_col'],'pct_counts_mt','total_counts','guides_per_cell','guide_umi_counts','n_genes_by_counts'] if c not in x.obs])
if u.exists():
    sep=cfg.get('program_usage_file_sep','\\t'); sep='\t' if sep=='\\t' else sep
    ud=pd.read_csv(u,sep=sep,index_col=int(cfg.get('program_usage_file_index_col',0)))
    ov=pd.Index(x.obs_names.astype(str)).intersection(ud.index.astype(str))
    print('usage table shape:',ud.shape)
    print('cell overlap:',len(ov),'/',x.n_obs,f'({len(ov)/x.n_obs:.1%})')
x.file.close()
