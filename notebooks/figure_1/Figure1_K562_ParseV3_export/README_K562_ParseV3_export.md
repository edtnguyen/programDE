# K562/iPSC Parse v3 figure-panel export

This package reprocesses the analysis pattern in `250904.Parse.v3.iPSC.K562.copy.ipynb` and writes plot-input tables for Figure 1 and supplementary panels.

## Run on Sherlock

```bash
unzip Figure1_K562_ParseV3_export.zip
cd Figure1_K562_ParseV3_export
bash run_k562_parsev3_export_on_sherlock.sh
```

The output bundle will be written to:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/250730_eTN10_Parsev3_mixediPSC_K562/figure_panel_export/k562_parsev3_figure_panel_plot_input_bundle.zip
```

Upload that ZIP back to ChatGPT for plotting.

## Notebook-derived inputs

The config uses the same source paths as the attached notebook:

```text
GEX DGE_filtered:
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/250730_eTN10_Parsev3_mixediPSC_K562/splitpipe_out/WT/sub1/all-sample/DGE_filtered/

Guide long table:
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/250730_eTN10_Parsev3_mixediPSC_K562/splitpipe_out/CRISPR_combined/grnaxcell_count.tsv

Guide feature list:
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/250730_eTN10_Parsev3_mixediPSC_K562/splitpipe_out/CRISPR/sub1/all-sample/guide_RNAs_unfiltered/all_guides.csv
```

It also reads transcriptome saturation files from:

```text
WT/sub1/CS1/report/tscp_counts_subsampled.csv
WT/sub1/DC1/report/tscp_counts_subsampled.csv
WT/sub1/Dual/report/tscp_counts_subsampled.csv
WT/sub1/no_guide/report/tscp_counts_subsampled.csv
```

## Main output files

```text
cell_metrics.tsv.gz
main_panel_K562_guide_capture.tsv.gz
primer_guide_capture_cell_metrics.tsv.gz
species_specificity_cell_metrics.tsv.gz
guide_detection_summary.tsv
cell_metric_summary.tsv
guide_metrics_by_sample_celltype.tsv.gz
guide_target_balance.tsv.gz
guide_target_skew_summary.tsv
pairwise_primer_tests.tsv
transcriptome_saturation.tsv
transcriptome_qc_by_sample_celltype.tsv
cell_type_assignment.tsv.gz
umap_embedding_cell_type.tsv.gz   # written if Scanpy clustering succeeds
processing_qc.tsv
manifest.json
data_dictionary.tsv
```

## Intended plots

### Main Figure 1e: K562 additional cell type
Use:

```text
main_panel_K562_guide_capture.tsv.gz
```

Recommended metrics to plot:

```text
expected_species_total_gRNA
K562_total
top_guide_umis
guide_umis_per_detected_guide
```

The default table includes K562 cells from `CS1`, `DC1`, and `Dual` samples.

### Supplement: primer comparison
Use:

```text
primer_guide_capture_cell_metrics.tsv.gz
pairwise_primer_tests.tsv
```

This supports CS1 vs DC1 vs Dual vs no_guide guide-UMI distributions, optionally stratified by inferred cell type.

### Supplement: iPSC vs K562 guide specificity / mixing
Use:

```text
species_specificity_cell_metrics.tsv.gz
```

Recommended plot: log-scale 2D histogram or scatter of:

```text
iPSC_total vs K562_total
```

### Supplement: guide detection metrics
Use:

```text
guide_detection_summary.tsv
```

This contains threshold curves for total guide UMIs, top guide UMIs, expected-species guide UMIs, and off-species guide UMIs.

### Supplement: transcriptome saturation / QC
Use:

```text
transcriptome_saturation.tsv
transcriptome_qc_by_sample_celltype.tsv
```

### Supplement: guide target balance / skew
Use:

```text
guide_target_balance.tsv.gz
guide_target_skew_summary.tsv
```

## Cell type assignment

By default, the script attempts a notebook-like Scanpy workflow:

```text
normalize_total -> log1p -> HVG -> regress total_counts -> PCA -> neighbors -> UMAP -> Leiden
```

It labels clusters using marker scores:

```text
K562 markers: GATA1, KLF1, HBB, HBA1, HBA2
hiPSC markers: SOX2, NANOG, POU5F1, LIN28A
```

If Scanpy or Leiden fails, it falls back to direct marker-score assignment.
