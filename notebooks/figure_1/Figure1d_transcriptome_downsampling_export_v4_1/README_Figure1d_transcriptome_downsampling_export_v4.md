# Figure 1d transcriptome downsampling export v4

This version fixes the issue seen in the v3 output bundle:

- v3 filtered `bc_wells` component 3 and then used `perm312`.
- The BS3 report shows called cells like `07_02_52`, and `cell_counts_by_rnd1_well.csv` confirms BS3 cells are in round-1 wells 7 and 9.
- Therefore the correct filtering for these outputs is `bc_wells` component 1, with no permutation of the cell ID.

v4 settings:

```text
bc1_bc_wells_position = 1
bc1_cell_barcode_position = 1
cell_identity_column = bc_wells
cell_identity_transform = identity_numeric
```

It also excludes the CROP-seq guide feature:

```text
gene_name: b08
gene_id: id_b08
```

along with circular-capture guide features:

```text
gene_name beginning with sg_
gene_id beginning with gene_id_
```

## Run on Sherlock

```bash
unzip Figure1d_transcriptome_downsampling_export_v4.zip
cd Figure1d_transcriptome_downsampling_export_v4
bash run_fig1d_downsampling_export_on_sherlock_v4.sh
```

Expected output:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_v4/fig1d_transcriptome_downsampling_plot_input_bundle.zip
```

Upload that ZIP back to ChatGPT.

## QC checks

After running, inspect:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_v4/processing_qc.tsv
```

You should expect approximately:

```text
Circular capture n_cells_retained ~ 217
CROP-seq n_cells_retained close to the called-cell count used by that sample
```

The exact CROP-seq retained cell count depends on which called-cell metadata exists on Sherlock.
