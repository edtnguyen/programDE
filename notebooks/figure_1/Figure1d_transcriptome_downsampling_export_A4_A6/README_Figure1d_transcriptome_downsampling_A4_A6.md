# Figure 1d transcriptome downsampling export — Circular capture A4:A6

This package is the same post-mapping transcriptome downsampling workflow as v4.1, but changes the circular-capture sample to `A4:A6`.

## Samples used

```text
Circular capture:
  tscp_assignment:
    /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/230820_eTN2_iPSC_mixed/panmg/process/tscp_assignment.csv
  barcode-1 wells:
    A4:A6
  called-cell report:
    /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/230820_eTN2_iPSC_mixed/panmg/FA_polyA/report/cluster_assignment.csv

CROP-seq:
  tscp_assignment:
    /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/230621_Tri_guide_iPSCs/iPSC/process/tscp_assignment.csv
  barcode-1 wells:
    B1:B12
```

The script keeps the v4.1 assumptions:

```text
filter_mode = bc_wells
bc1_bc_wells_position = 1
bc1_cell_barcode_position = 1
cell_identity_column = bc_wells
cell_identity_transform = identity_numeric
```

It excludes guide-like features from transcriptome downsampling:

```text
gene_name beginning with sg_
gene_name equal to b08
gene IDs beginning with gene_id_
gene_id equal to id_b08
```

## Run on Sherlock

```bash
unzip Figure1d_transcriptome_downsampling_export_A4_A6.zip
cd Figure1d_transcriptome_downsampling_export_A4_A6
bash run_fig1d_downsampling_export_A4_A6_on_sherlock.sh
```

Expected output:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_A4_A6/fig1d_transcriptome_downsampling_plot_input_bundle.zip
```

Upload that ZIP back here.

## QC checks before upload

Inspect:

```bash
cat /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_A4_A6/processing_qc.tsv
```

Key columns:

```text
sample_id
requested_wells
called_cells_path_used
n_called_cells_loaded
rows_after_well_filter
rows_after_called_cell_filter
n_cells_retained
assigned_reads_retained
```

For the circular-capture sample, confirm that:

```text
requested_wells = A4,A5,A6
called_cells_path_used contains /panmg/FA_polyA/report/
n_cells_retained is plausible for the FA_polyA report
```

If the A4:A6 report folder is named something other than `FA_polyA`, edit only `called_cells_path` and `called_cells_candidates` in the config.
