# Figure 1 b08-only update

This package fixes the hiPSC single-guide comparison by plotting the expected b08 guide only:

- circular capture: `sg_b08`
- CROP-seq: `b08`

The previous draft plotted the top guide across all guide-like features, which is not appropriate for this single-guide experiment.

## Run on Sherlock

```bash
unzip Figure1_b08_only_update.zip
cd Figure1_b08_only_update
bash run_b08_only_on_sherlock.sh
```

The script writes:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_cc_capture_b08_only/figure1_b08_plot_input_bundle.zip
```

Upload that ZIP back to ChatGPT if you want final formatting here.

## Main outputs

```text
b08_cell_metrics_for_plotting.tsv.gz
b08_summary_for_plotting.tsv
b08_detection_thresholds_for_plotting.tsv
quality_checks/b08_feature_check.tsv
plots/fig1b_b08_guide_umis.svg
plots/fig1c_b08_detection_bar.svg
plots/fig1c_b08_detection_threshold_curve.svg
```

## Interpretation of updated panel c

The updated panel c is no longer singlet/multiplet assignment. For a single-guide experiment, guide-identity-based singlet/multiplet assignment is not meaningful. The panel now shows the percent of transcriptome-QC-passing cells with the expected b08 guide detected, using `b08 >= 5 UMIs` by default.

A threshold-curve version is also generated for `>=1`, `>=3`, `>=5`, `>=10`, and `>=20` UMIs.
