# Figure 2 perturbation-class coarse-state occupancy notebook

This package generates the requested publication panel and supplementary heatmap directly on Sherlock from the real processed CC-Perturb-seq H5AD.

## Verified input object

```text
/oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/3.integrated_days/3.all_days_concat_pc30_all_genes_sceptre_guides.h5ad
```

## Main notebook

```text
notebooks/01_generate_perturbation_class_occupancy.ipynb
```

The notebook performs both steps:

1. export compact cell metadata and target-union assignments from the H5AD;
2. fit adjusted occupancy models, summarize reviewed regulator classes, and generate figures.

## Run interactively

```bash
unzip figure2_perturbation_class_occupancy_notebook_sherlock_20260727.zip
cd figure2_perturbation_class_occupancy_notebook_sherlock_20260727

# activate an environment satisfying requirements.txt
jupyter lab notebooks/01_generate_perturbation_class_occupancy.ipynb
```

For an interactive job, request approximately 8 CPUs, 128 GB RAM, and up to 24 hours.

## Run as one Slurm job

```bash
sbatch scripts/run_full_occupancy_pipeline.sbatch
```

## Execute the notebook headlessly

```bash
sbatch scripts/run_notebook_headless.sbatch
```

## State source

The notebook uses the existing author-defined mapping of `leiden_res_0.5` from `ipsc-ec.Hippo.figure.ipynb` and does not recluster target carriers. Therefore the panel should be called an author-defined coarse-state occupancy analysis.

## Primary outputs

```text
occupancy_analysis_results/Fig2_perturbation_class_state_occupancy.svg
occupancy_analysis_results/Fig2_perturbation_class_state_occupancy.pdf
occupancy_analysis_results/Fig2_perturbation_class_state_occupancy.png
occupancy_analysis_results/Fig2_target_state_occupancy_heatmap_supplementary.svg
occupancy_analysis_results/Fig2_target_state_occupancy_heatmap_supplementary.pdf
occupancy_analysis_results/Fig2_target_state_occupancy_heatmap_supplementary.png
occupancy_analysis_results/target_state_occupancy_effects.tsv
occupancy_analysis_results/target_state_occupancy_by_replicate.tsv
occupancy_analysis_results/class_state_occupancy_summary.tsv
occupancy_analysis_results/class_membership.tsv
occupancy_analysis_results/state_definitions.tsv
occupancy_analysis_results/coverage_and_filtering.tsv
occupancy_analysis_results/draft_panel_legend.md
occupancy_analysis_results/methods_note.md
```

## Model

Separate day-specific one-vs-rest logistic models are fit for every target and state using:

```text
target union indicator
+ biological sample
+ pct_counts_mt
+ log1p(total_counts)
+ log1p(guides_per_cell)
+ log1p(guide_umi_counts)
+ log1p(n_genes_by_counts)
```

FDR correction is performed within day across the target×state testing family. Class-level dots summarize target-level results; no class-level pooled exposure is fit.
