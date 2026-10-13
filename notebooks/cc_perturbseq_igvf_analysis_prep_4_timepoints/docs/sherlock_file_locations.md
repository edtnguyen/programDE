# Sherlock file locations extracted from the attached notebooks

## Known direct source candidates

### Combined annotated cell-by-gene/guide h5ad

```text
/oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/3.integrated_days/3.all_days_concat_pc30_all_genes_sceptre_guides.h5ad
```

Planned use: validate the day field, then create D0-D3 staged subsets. The source is never overwritten.

### Combined PerturbNMF/cNMF h5mu

```text
/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/adata/cNMF_50_0_2.h5mu
```

Planned use: inspect only until DACC resolves the mismatch between one combined source and four workbook MatrixFile rows.

## Supporting sources

### cNMF usage matrix

```text
/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50.usages.k_50.dt_0_2.consensus.txt
```

### Program metadata and annotations

```text
/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/per_day_umaps/data/program_activity_long_v4.csv
```

## Required paths not found in the attachments

- absolute path to `300genes_guide_metadata_v43.tsv`;
- 16 differential-output paths: local/global x guide/element x D0-D3;
- any original source files for the already uploaded PerturbNMF tables, if direct schema comparison is needed rather than Portal inspection.

The staging target from the workbook is:

```text
/oak/stanford/groups/engreitz/Projects/IGVF/CC-Perturb-seq/300gene/analysis_submission
```
