# Figure 1 CC-Perturb-seq capture benchmarking package

This package reprocesses the Figure 1 capture-comparison datasets on Sherlock and writes an uploadable plot-input ZIP. Upload that ZIP back to ChatGPT, and I can make the final Figure 1 plotnine panels locally as editable SVGs and PNG previews.

The ZIP contains derived plotting tables only: per-cell guide/transcriptome QC metrics, per-guide detection summaries, sample-level summaries, saturation rows if available, paper-inspired supplemental QC tables, panel-specific subsets, and diagnostics. It does not include raw MatrixMarket, h5ad, h5mu, FASTQ, or sequencing files.

## Path updates retained in this version

The following paths are prefilled in `figure1_config_template.json`:

```text
# teloHAEC parent directory
/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/tony_results

# files expected inside that directory
GEX2.h5ad
GUIDE1.h5ad
CROP_GEX1.h5ad
CROP_GUIDE1.h5ad
GUIDE_S1.{reads}_kb_kite/counts_unfiltered_modified/adata.h5ad
CROP_GUIDE1_S4.{reads}_kb_kite/counts_unfiltered_modified/adata.h5ad

# iPSC pilot guide annotation files
/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/figure_1/eTN3_gRNA_group_table.csv
/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/figure_1/eTN3_gRNA_group_pairs.csv
```

The supplied `CROP_GEX1.h5ad` path is interpreted as one file inside the teloHAEC folder. The config uses the parent directory as `root_dir`. The helper script also tolerates an h5ad file path in `root_dir` by converting it to its parent directory.

## New paper-inspired panels added

Based on PerturbSci-Kinetics-style method figures, the script now also prepares and, when possible, plots:

- guide-assignment class fractions: no guide / singlet / ambiguous-multiplet;
- per-guide recovery curves, useful for showing how uniformly guides are detected;
- cell throughput / cells passing QC by sample;
- transcriptome QC summary: median UMIs versus median genes;
- gene-detection saturation, if transcriptome downsampling rows include detected-feature counts;
- guide RT primer optimization, if an optional table is supplied;
- guide-read source fractions, such as oligo-dT versus guide RT primer, if an optional table is supplied;
- guide-count replicate correlations, if optional replicate count tables are supplied;
- bulk guide count versus single-cell guide recovery correlation, if optional bulk count tables are supplied;
- target-gene knockdown QC, if enabled with a guide-to-target table.

The target-knockdown optional block is present but disabled by default in `figure1_config_template.json`, because it is a quick QC summary rather than a replacement for SCEPTRE. Set `optional_panels.target_knockdown[0].enabled` to `true` to run it for the iPSC pilot.

## Package files

- `CC_Figure1_reprocess_and_plot.ipynb` — Sherlock notebook workflow.
- `CC_Figure1_reprocess_and_export_for_ChatGPT.ipynb` — same workflow, emphasizing upload-bundle creation.
- `figure1_cc_perturbseq_pipeline.py` — helper module and command-line runner.
- `figure1_config_template.json` — editable config with updated paths and optional-panel placeholders.
- `run_figure1_plot_input_bundle_on_sherlock.sh` — optional command-line wrapper.

## Fast command-line workflow on Sherlock

From the folder containing these package files:

```bash
bash run_figure1_plot_input_bundle_on_sherlock.sh figure1_config_template.json
```

Equivalent direct commands:

```bash
python figure1_cc_perturbseq_pipeline.py \
  --config-json figure1_config_template.json \
  --check-paths

python figure1_cc_perturbseq_pipeline.py \
  --config-json figure1_config_template.json \
  --skip-plots
```

The expected upload file is:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_cc_capture/figure1_plot_input_bundle.zip
```

Upload that ZIP back here.

## Main output tables inside the upload ZIP

```text
cell_metrics_for_plotting.tsv.gz
guide_metrics_for_plotting.tsv.gz
sample_summary_for_plotting.tsv
saturation_for_plotting.tsv
assignment_classes_for_plotting.tsv
guide_detection_curve_for_plotting.tsv.gz
primer_optimization_for_plotting.tsv
read_origin_for_plotting.tsv
replicate_correlations_for_plotting.tsv
bulk_singlecell_correlations_for_plotting.tsv.gz
target_knockdown_for_plotting.tsv
panel_specific/fig1b_hiPSC_guide_umis_full.tsv.gz
panel_specific/fig1c_assignment_metrics.tsv
panel_specific/fig1d_transcriptome_saturation.tsv
panel_specific/fig1d_transcriptome_qc_full.tsv.gz
panel_specific/fig1e_other_celltypes_guide_umis_full.tsv.gz
panel_specific/figS_assignment_classes_hiPSC.tsv
panel_specific/figS_guide_detection_curve_hiPSC.tsv.gz
quality_checks/path_check.tsv
quality_checks/processing_errors.tsv
quality_checks/missing_saturation_paths.tsv
manifest.json
data_dictionary.tsv
```

## Figure panel support

- **Fig. 1b:** hiPSC guide UMIs/cell for circular capture vs CROP-seq.
- **Fig. 1c:** cells assigned a guide and guides detected.
- **Fig. 1d:** true transcriptome saturation if supplied; otherwise transcriptome UMI/QC fallback.
- **Fig. 1e:** guide UMIs/cell across additional cell types, including teloHAEC if the Sherlock paths exist.
- **Supplemental candidates:** assignment-class fractions, guide recovery curves, cell throughput, transcriptome QC summary, guide saturation, primer optimization, guide-read source fractions, replicate correlations, bulk-vs-single-cell guide recovery, and target-knockdown QC.

## Optional table formats

The optional table readers are flexible and infer common column names. These are the preferred formats:

### Primer optimization

```text
primer_concentration  primer  guide_umis_per_cell  n_cells_recovered  sample_id  method
1uM                   RT1     42                   1000               pilot      Circular capture
```

### Read-origin fractions

```text
sample_id  method            read_source       n_reads
pilot      Circular capture  guide RT primer   100000
pilot      Circular capture  oligo-dT          12000
```

A `fraction` or `percent` column can be supplied instead of `n_reads`.

### Replicate guide-count correlations

```text
guide_id  replicate_1  replicate_2  replicate_3
sgA       100          120          113
```

### Bulk-vs-single-cell guide recovery

```text
guide_id  n_reads
sgA       10000
```

Point this table to `optional_panels.bulk_guide_count_tables` and set `single_cell_sample_id` to the sample in `guide_metrics` to compare against.

## Remaining optional inputs

1. True hiPSC transcriptome/GEX downsampling files if panel 1d must be a saturation curve instead of a QC fallback.
2. K562 or other additional-cell-type paths if panel 1e should include more than teloHAEC and the iPSC pilot.
3. Raw or processed library-QC tables for guide-read source fractions, barcode-valid guide reads, uniquely mapped guide reads, and replicate correlations.
