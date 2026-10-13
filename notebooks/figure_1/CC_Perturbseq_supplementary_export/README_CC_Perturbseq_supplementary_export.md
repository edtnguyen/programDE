# CC-Perturb-seq supplementary-panel export

This package creates plot-input tables and draft SVG/PNG panels for the CC-Perturb-seq Figure 1 supplementary material from the compact plot-input bundles generated earlier in this project.

## Run on Sherlock

```bash
unzip CC_Perturbseq_supplementary_export.zip
cd CC_Perturbseq_supplementary_export
bash run_cc_perturbseq_supplementary_panels_on_sherlock.sh
```

The default output bundle will be:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/cc_perturbseq_supplementary_panels/cc_perturbseq_supplementary_panels_bundle.zip
```

Upload that ZIP back for final figure assembly.

## Inputs expected by the default config

```text
figure1_bundle:
  /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_cc_capture/figure1_plot_input_bundle.zip

b08_bundle:
  /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_cc_capture_b08_only/figure1_b08_plot_input_bundle.zip

fig1d_downsampling_bundle:
  /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_A4_A6/fig1d_transcriptome_downsampling_plot_input_bundle.zip

k562_parsev3_bundle:
  /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/250730_eTN10_Parsev3_mixediPSC_K562/figure_panel_export_v2_lowmem/k562_parsev3_figure_panel_plot_input_bundle.zip

knockdown_per_gene_bundle:
  /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure4a_like_knockdown_per_gene_export_v2/fig4a_like_knockdown_per_gene_plot_input_bundle.zip

knockdown_per_guide_bundle:
  /oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure4a_like_knockdown_export_v2/fig4a_like_knockdown_plot_input_bundle.zip
```

Edit `cc_perturbseq_supplementary_config.json` if any of these paths differ.

## Panels/tables generated when inputs are present

### S1: hiPSC circular-capture vs CROP-seq method QC

- `S1_hiPSC_guide_detection.tsv` plus `S1_hiPSC_guide_detection.svg/png`
- `S1_transcriptome_downsampling_summary.tsv.gz` plus saturation plots
- `S1_knockdown_per_gene.tsv.gz` and/or `S1_knockdown_per_guide.tsv.gz` plus knockdown plot

### S2: Parse v3 K562/iPSC and RT-primer robustness

- K562 guide-capture violin panel
- hiPSC guide-capture violin panel
- faceted K562/hiPSC/no-guide negative-control panel
- guide-detection threshold curves
- expected-species guide specificity
- transcriptome QC table and plot if the table format is recognized
- pairwise primer-test table

### S3: additional cell types

- teloHAEC guide-capture panel
- teloHAEC guide-detection threshold curve
- teloHAEC transcriptome QC summary

## Missing data report

The script always writes:

```text
missing_data_report.tsv
```

This lists panels that still require additional inputs, including:

- guide-read mapping QC: `% barcode-valid guide reads`, `% uniquely mapped reads`
- read-origin panel: `% guideRNA reads from oligo-dT vs guide RT primer`
- replicate correlations
- primary mouse hepatocyte data
- schematic panels

## Notes

This script does not reprocess raw matrices. It consumes the compact bundles already produced by the earlier export scripts. For final manuscript plots, upload the resulting output bundle and I can generate refined panels from the included tables.
