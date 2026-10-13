# Diagnostic of uploaded v3 bundle

The uploaded bundle at `/mnt/data/fig1d_transcriptome_downsampling_plot_input_bundle.zip` should not be used for the final Figure 1d plot.

Observed in `processing_qc.tsv`:

Circular capture:
- n_called_cells_loaded = 217
- n_cells_retained = 30
- rows_after_well_filter = 534111
- rows_after_called_cell_filter = 486

CROP-seq:
- n_called_cells_loaded = 784
- n_cells_retained = 83
- rows_after_well_filter = 2219899
- rows_after_called_cell_filter = 2228

Additional issue:
- `quality_checks/hiPSC_CROP_seq.top_gene_summary.tsv` shows `b08` as the top retained feature, so the CROP-seq guide was not excluded from the transcriptome feature set in v3.
