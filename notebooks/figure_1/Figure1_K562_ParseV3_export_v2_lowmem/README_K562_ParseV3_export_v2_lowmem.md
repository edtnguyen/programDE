# K562/iPSC Parse v3 figure-panel export, v2 low-memory

This package is a low-memory replacement for the first K562/iPSC Parse v3 export script.

## Why the first script was killed

The shell message `Killed` usually means the Linux scheduler/OOM killer terminated the Python process because it exceeded available memory. In the first script, the memory-heavy steps were:

1. Scanpy PCA/neighbors/Leiden on the full GEX matrix.
2. Materializing the filtered `grnaxcell_count.tsv` cell-guide table as one large dataframe.
3. Running several groupbys on that large long table and writing it to disk.

The run got past Scanpy and was killed during guide-table aggregation/export.

## What v2 changes

- Uses marker-score cell-type assignment only; no Scanpy/Leiden by default.
- Streams `grnaxcell_count.tsv` in chunks.
- Does not write `filtered_cell_guide_long.tsv.gz`.
- Computes per-cell guide metrics incrementally.
- Computes guide and target summaries incrementally.

The output bundle still contains the main plot-input tables needed for K562 main Fig. 1e and supplementary panels.

## Run on Sherlock

```bash
unzip Figure1_K562_ParseV3_export_v2_lowmem.zip
cd Figure1_K562_ParseV3_export_v2_lowmem
bash run_k562_parsev3_export_v2_lowmem_on_sherlock.sh
```

Output bundle:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/250730_eTN10_Parsev3_mixediPSC_K562/figure_panel_export_v2_lowmem/k562_parsev3_figure_panel_plot_input_bundle.zip
```

Upload that ZIP back to ChatGPT for plotting.

## Key outputs

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
processing_qc.tsv
```

## Tradeoff

The only intentional tradeoff is that v2 does not export the full filtered cell-guide long table and does not compute per-guide median UMI among nonzero cells. Those are not needed for the requested figure panels.
