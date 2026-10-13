# Fig. 4A-like per-gene knockdown export

This package exports plot-input tables for a Fig. 4A-like on-target knockdown panel, but collapsed at the target-gene level rather than the guide level.

## What is different from the per-guide export

For each target gene, the script pools all cells carrying any guide assigned to that gene:

```text
carrier cells for target gene = cells with any guide for that target gene at guide UMI >= 5
```

It then compares target-gene expression in those pooled carrier cells to non-targeting-control cells.

The per-guide table is still exported as support, so the final plot can show or filter by guide consistency if needed.

## Inputs

The default config includes:

```text
Dataset #2 teloHAEC circular capture:
  GEX2.h5ad
  GUIDE1.h5ad

Dataset #2 teloHAEC CROP-seq:
  CROP_GEX1.h5ad
  CROP_GUIDE1.h5ad

Dataset #3 hiPSC 10-gene pilot:
  Parse-style GEX and guide matrices
  eTN3 guide mapping tables
```

## Run on Sherlock

```bash
unzip Fig4a_like_knockdown_per_gene_export.zip
cd Fig4a_like_knockdown_per_gene_export
bash run_fig4a_like_knockdown_per_gene_export_on_sherlock.sh
```

The output bundle will be written to:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure4a_like_knockdown_per_gene_export/fig4a_like_knockdown_per_gene_plot_input_bundle.zip
```

Upload that ZIP back to ChatGPT for plotting.

## Main output files

```text
knockdown_per_gene.tsv.gz
knockdown_per_guide_support.tsv.gz
guide_carrier_counts.tsv.gz
processing_qc.tsv
quality_checks/*.tsv
manifest.json
data_dictionary.tsv
```

## Suggested plot

Use `knockdown_per_gene.tsv.gz`.

Recommended axes:

```text
x-axis = target gene rank by control expression
y-axis = carrier mean log-normalized expression of target gene
gray reference = control mean ± SD or SEM
point color = significant knockdown q <= 0.10
```

Useful columns:

```text
gene_rank_by_control_expression
target_gene
control_mean_log_normalized_expr
control_sd_log_normalized_expr
carrier_mean_log_normalized_expr
carrier_vs_control_ratio_logexpr
knockdown_efficiency_1_minus_ratio
qvalue_bh
significant_knockdown_q10
n_guides_for_target
n_carrier_cells_any_target_guide
n_guides_significant_q10
median_per_guide_knockdown_efficiency
```
