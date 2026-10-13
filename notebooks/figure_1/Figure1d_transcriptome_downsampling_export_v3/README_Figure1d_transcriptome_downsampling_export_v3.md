# Figure 1d transcriptome downsampling export v3

This version fixes the warnings seen in v2 for the BS3 circular-capture sample.

## What changed from v2

1. Forces `cell_identity_column = bc_wells` and `cell_identity_transform = perm312`.
   In the Parse `tscp_assignment.csv`, `bc_wells` is in the order `round2_round3_round1`, while the report/DGE called-cell IDs are `round1_round2_round3`. Therefore `02_52_07` becomes `07_02_52`.
2. Filters by barcode-1 using the third component of `bc_wells`.
   - Circular capture BS3: A7:A9 -> numeric well indices 7,8,9.
   - CROP-seq: B1:B12 -> numeric well indices 13-24.
3. Uses sample report `cluster_assignment.csv` as the first called-cell list when available.
4. Suppresses repeated chunk-level called-cell warnings. Some chunks contain selected well rows but no called cells because `tscp_assignment.csv` is sorted by raw barcode combination. v3 only warns if global overlap is zero after scanning the full file.

## Run on Sherlock

```bash
unzip Figure1d_transcriptome_downsampling_export_v3.zip
cd Figure1d_transcriptome_downsampling_export_v3
bash run_fig1d_downsampling_export_on_sherlock_v3.sh
```

Output bundle:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_v3/fig1d_transcriptome_downsampling_plot_input_bundle.zip
```

Upload that ZIP back to ChatGPT.

## Sanity checks

After running, inspect:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_v3/processing_qc.tsv
```

Expected key fields:

- `cell_identity_source_selected` should be `bc_wells`.
- `cell_identity_transform_selected` should be `perm312`.
- `rows_after_called_cell_filter` should be >0 for both samples.
- `n_cells_retained` should be close to the called-cell counts for the selected sample/wells.

For BS3, the report indicates A7/A9 have called cells and A8 has none, for 217 total cells.
