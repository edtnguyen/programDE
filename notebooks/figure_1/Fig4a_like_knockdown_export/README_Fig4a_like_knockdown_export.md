# Fig. 4A-like on-target knockdown export

This package exports plot-input tables for a panel similar to Fig. 4A in the uploaded CD4+ T-cell Perturb-seq paper: target-gene expression per guide, comparing cells carrying each guide with non-targeting-control cells.

## Run on Sherlock

```bash
unzip Fig4a_like_knockdown_export.zip
cd Fig4a_like_knockdown_export
bash run_fig4a_like_knockdown_export_on_sherlock.sh
```

The output bundle will be written to:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure4a_like_knockdown_export/fig4a_like_knockdown_plot_input_bundle.zip
```

Upload that ZIP back to ChatGPT to generate the plot.

## Inputs currently configured

Dataset #2:

```text
teloHAEC circular capture:
  GEX2.h5ad
  GUIDE1.h5ad

teloHAEC CROP-seq:
  CROP_GEX1.h5ad
  CROP_GUIDE1.h5ad
```

Dataset #3:

```text
hiPSC 10-gene pilot Parse GEX matrix:
  WT_combined/MOI8/DGE_filtered/count_matrix.mtx

hiPSC 10-gene pilot Parse guide matrix:
  crispr_combined/MOI8/guide_RNAs_filtered/count_matrix.mtx

Guide metadata:
  eTN3_gRNA_group_table.csv
  eTN3_gRNA_group_pairs.csv
```

## Output tables

```text
knockdown_per_guide.tsv.gz
knockdown_per_target.tsv.gz
guide_carrier_counts.tsv.gz
processing_qc.tsv
quality_checks/*.guide_target_mapping.tsv
quality_checks/*.guide_carrier_counts.tsv
data_dictionary.tsv
manifest.json
```

## Method summary

For each guide with a resolvable target gene:

1. Identify guide-carrier cells using `guide UMI >= guide_threshold`.
2. Compute log-normalized target-gene expression: `log1p(count / total_counts * 10000)`.
3. Compare guide-carrier cells with non-targeting-control cells when available.
4. Fall back to non-carrier cells for the same target if no adequate NTC control cells are found.
5. Run a one-sided Mann-Whitney test for reduced expression in guide-carrier cells.
6. Apply Benjamini-Hochberg correction within each sample.

The key columns for plotting are:

```text
target_rank_by_control_expression
control_mean_log_normalized_expr
control_sd_log_normalized_expr
carrier_mean_log_normalized_expr
significant_knockdown_q10
```

## Notes

- The teloHAEC guide-to-target mapping is inferred from guide names by taking the part before the first underscore, e.g. `RHOQ_-_...` maps to `RHOQ`.
- NTC guides are detected by name using a configurable regex.
- Dataset #3 is small; the per-guide statistics may be underpowered, but the table will still be exported.
