iPSC-EC full-covariate QQ export bundle

Primary notebook:
  iPSC_EC_QQ_export_full_covariates.ipynb

What changed:
  - Backend attach_covariates() now uses the requested full set:
      pct_counts_mt
      biological_sample
      total_counts
      guides_per_cell
      guide_umi_counts
      n_genes_by_counts
  - biological_sample is one-hot encoded with one reference level dropped.
  - total_counts, guides_per_cell, guide_umi_counts, and n_genes_by_counts are log1p transformed.
  - The run cell uses --require-covariates and --overwrite.
  - Backend matplotlib plots are disabled by default; plotnine cells save PDF/SVG/EPS/PNG outputs with the legend above the panel.

Output root used by the notebook:
  results/ipsc_ec_qq_exports_full_covariates/k_50.dt_0_2/

Recommended Illustrator files after running:
  results/ipsc_ec_qq_exports_full_covariates/k_50.dt_0_2/illustrator_editable_full_covariates/*.pdf
  results/ipsc_ec_qq_exports_full_covariates/k_50.dt_0_2/illustrator_editable_full_covariates/*.svg
