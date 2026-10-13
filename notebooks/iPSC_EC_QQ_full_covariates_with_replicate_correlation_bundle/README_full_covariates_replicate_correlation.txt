iPSC-EC QQ export with full covariates + biological replicate effect correlation

Main notebook:
  iPSC_EC_QQ_full_covariates_with_replicate_correlation.ipynb

The notebook writes and uses:
  iPSC_EC_QQ_export_all_days.py

Full-day CRT covariates:
  pct_counts_mt
  biological_sample, one-hot encoded with one reference level dropped
  log1p_total_counts
  log1p_guides_per_cell
  log1p_guide_umi_counts
  log1p_n_genes_by_counts

Notebook sections:
  1. Write backend script
  2. Optional dry run
  3. Run full d0-d3 QQ export with full covariates
  4. Validate covariates and export plotnine QQ plots as PDF/SVG/EPS/PNG
  5. Preview biological sample cell counts
  6. Rerun CRT within each biological_sample and save betas_df per replicate
  7. Plot biological replicate correlation of DE program effects

Replicate-correlation outputs are written under:
  results/ipsc_ec_qq_exports_full_covariates/k_50.dt_0_2/biological_replicate_effects/

Key replicate-correlation files:
  replicate_beta_manifest.csv
  betas_by_replicate/<day>/<day>__<sample>__target_betas_wide.tsv
  plots/biological_replicate_DE_program_effect_correlation_summary.csv
  plots/biological_replicate_DE_program_effect_pairs.csv
  plots/*biological_replicate_DE_program_effect_correlation*.pdf/svg/eps/png

For Illustrator, use PDF or SVG first.
