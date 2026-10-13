iPSC-EC CRT QQ point export bundle
===================================

Purpose
-------
This bundle adapts the IP/GV simple-QQ export package to the iPSC-EC
differentiation data. It exports p-values and QQ-plot coordinates for each day
separately: d0, d1, d2, and d3.

The primary per-day QQ-point outputs are:

  <out-dir>/k_<K>.dt_<DT>/<day>/<day>_observed_target_QQ_points.csv
  <out-dir>/k_<K>.dt_<DT>/<day>/<day>_nontargeting_pseudotarget_QQ_points.csv
  <out-dir>/k_<K>.dt_<DT>/<day>/<day>_QQ_points.csv

The combined all-day output is:

  <out-dir>/k_<K>.dt_<DT>/all_days_QQ_points.csv

Each QQ-point CSV contains the original p-value plus:

  rank
  n_points
  expected_pvalue
  expected_neglog10_p
  observed_neglog10_p
  day
  pvalue_type
  series
  series_type
  target_name / program_name for observed target p-values
  ensemble / group_id / program_name for grouped NTC pseudo-target p-values

Source data paths inferred from the notebooks
---------------------------------------------
From qq_sweep_test.ipynb:

  H5MU data directory:
    /oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/2.normalized_data/normalized_data

  H5MU files:
    d0_normalized_gex_and_guide.h5mu
    d1_normalized_gex_and_guide.h5mu
    d2_normalized_gex_and_guide.h5mu
    d3_normalized_gex_and_guide.h5mu

  cNMF usage directory used for the all-day sweep:
    /oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/012726_100k_cells_20iter_allHVG_torch_halsvar_batch_e7/012726_100k_cells_20iter_allHVG_torch_halsvar_batch_e7_all

  guide long CSV template:
    /oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/{day}_sceptre_transfer/{day}_grna_assignments_long.csv

From ipsc_ec_qq.ipynb:

  single-d2 example data file:
    d2_normalized_gex_and_guide.h5mu

  single-d2 cNMF usage example:
    /oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50.usages.k_50.dt_0_2.consensus.txt

  guide assignment example:
    /oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/d2_sceptre_transfer/d2_grna_assignments_long.csv

Default run command
-------------------
Run this from the programDE notebooks directory or another environment where the
/oak paths are mounted and the programDE Python environment is active.

  python iPSC_EC_QQ_export_all_days.py \
    --programde-root /oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE \
    --data-dir /oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/2.normalized_data/normalized_data \
    --cnmf-dir /oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/012726_100k_cells_20iter_allHVG_torch_halsvar_batch_e7/012726_100k_cells_20iter_allHVG_torch_halsvar_batch_e7_all \
    --guide-long-csv-template '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/{day}_sceptre_transfer/{day}_grna_assignments_long.csv' \
    --out-dir results/ipsc_ec_qq_exports \
    --days d0 d1 d2 d3 \
    --k 50 \
    --density-threshold 0_2 \
    --B 10000 \
    --n-jobs ${SLURM_CPUS_PER_TASK:-4} \
    --ntc-label SAFE \
    --ntc-label non-targeting \
    --ntc-label NTC \
    --skip-if-exists

For a faster test run matching the sweep notebook's B=1023 setting, use:

  --B 1023

If the final K=50 dt=0_2 usage file should come from the 030526 directory used
in ipsc_ec_qq.ipynb, either set --cnmf-dir to that directory or pass the exact
file with --cnmf-file. The script discovers usage files by matching:

  *.usages.k_<K>.dt_<DT>.consensus.txt

and then chooses the candidate with the best overlap to the cells in each day's
H5MU file.

What is exported
----------------
For each day, the script writes:

  <day>_target_pvalues_empirical_wide.tsv
      Raw empirical target x program p-values from out['pvals_raw_df'].
      This table retains any control-label rows such as non-targeting.

  <day>_target_pvalues_empirical_long.tsv
      Long version of the empirical target p-values.

  <day>_target_pvalues_skew_wide.tsv
  <day>_target_pvalues_skew_long.tsv
      Skew-calibrated target p-values, when skew calibration is enabled.

  <day>_target_betas_wide.tsv
      Target x program CRT beta matrix.

  <day>_ntc_pseudotarget_pvalues_empirical_wide.tsv
      Grouped non-targeting pseudo-target empirical p-values from
      crt_pvals_for_ntc_groups_ensemble.

  <day>_ntc_pseudotarget_pvalues_skew_wide.tsv
      Grouped non-targeting pseudo-target skew p-values, when enabled.

  <day>_observed_target_QQ_points.csv
      QQ coordinates for real observed targets. NTC labels are excluded from
      this series using --ntc-label.

  <day>_nontargeting_pseudotarget_QQ_points.csv
      QQ coordinates for grouped non-targeting pseudo-targets. This is the NTC
      series used for the point-only calibration QQ export.

  <day>_observed_ntc_label_row_QQ_points.csv
      Only written if out['pvals_raw_df'] contains rows named SAFE,
      non-targeting, or NTC. This preserves the raw observed control-label row
      separately from the grouped NTC pseudo-target series.

  <day>_QQ_points.csv
      Combined day-specific QQ-point table.

  <day>_QQ_empirical_simple.pdf/png
  <day>_QQ_skew_simple.pdf/png
      Point-only replotting checks. Use --no-plot to skip.

  <day>_run_config.json
      Source paths, cNMF file selected, B, NTC grouping parameters, and output
      file names.

Notes on matching the notebooks
-------------------------------
The script follows ipsc_ec_qq.ipynb for:

  GEX modality: mdata['GEX']
  guide modality: mdata['guide']
  cNMF usage key: adata.obsm['cnmf_usage']
  guide assignment key: adata.obsm['guide_assignment']
  guide-to-gene default: guide_name.split('_', 1)[0]
  covariates:
    log1p(guides_per_cell)  -> grna_n_nonzero
    log1p(total_counts)     -> response_n_umis
    pct_counts_mt           -> response_p_mito
  NTC labels:
    SAFE, non-targeting, NTC
  NTC group size:
    6
  NTC ensemble count:
    10
  NTC bins:
    10
  NTC group seed:
    7
  NTC CRT seed:
    23

The default B=10000 matches ipsc_ec_qq.ipynb. The sweep notebook used B=1023.

Dry run
-------
Use --dry-run to check the resolved H5MU, guide, and candidate cNMF paths without
running CRT:

  python iPSC_EC_QQ_export_all_days.py --dry-run

Validation status
-----------------
The Python file was syntax checked in this environment. It was not executed
against the Sherlock /oak data because those source files are referenced by the
notebooks but are not included in this upload.
