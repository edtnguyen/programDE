# Legacy author-defined coarse-state atlas (no review required)

This package bypasses the cluster-review error by using the exact coarse-state mapping already defined in `ipsc-ec.Hippo.figure.ipynb`:

| leiden_res_0.5 | state |
|---|---|
| 0 | artEC |
| 1 | iPSC |
| 2 | Neuro-iPSC |
| 3 | PrimStreak |
| 4 | Lat-Meso |
| 5 | Fibro-Mesen |
| 6 | Pre-EC |
| 7 | Neu-Mesen |
| 8 | Others |

It does **not** recluster and does **not** require `approved_cluster_state_map.tsv` review. It uses the existing `X_umap` and `leiden_res_0.5` values from the processed CC-Perturb-seq H5AD.

## Important interpretation

This solves the execution error, but the result should be described as an **author-defined legacy coarse-state atlas**, not a strictly control-defined atlas. The original Leiden boundaries were learned using all CC-Perturb-seq cells. No target-to-program effects are used here, but perturbed cells contributed to the original clustering.

## Run on Sherlock

```bash
unzip figure2_legacy_author_defined_state_atlas_UPDATED_NO_REVIEW_20260727.zip
cd figure2_legacy_author_defined_state_atlas_UPDATED_NO_REVIEW_20260727

python scripts/01_run_legacy_author_state_atlas.py \
  --config config/legacy_author_state_atlas_config.local.yaml
```

Or Slurm:

```bash
sbatch \
  --export=ALL,PACKAGE_DIR=$PWD,CONFIG=$PWD/config/legacy_author_state_atlas_config.local.yaml \
  scripts/run_legacy_author_state_atlas.sbatch
```

## Expected outputs

```text
legacy_author_defined_state_atlas_results/
├── Fig2_control_defined_state_atlas.svg/pdf/png
├── Fig2_control_defined_state_atlas_candidateA.svg/pdf/png
├── Fig2_control_defined_state_atlas_candidateB.svg/pdf/png
├── control_state_assignments.tsv
├── control_state_annotation_review.tsv
├── state_marker_summary.tsv
├── state_counts_by_day_and_replicate.tsv
├── state_color_map.tsv
├── control_cell_definition_qc.tsv
├── legacy_cluster_state_map_used.tsv
├── state_atlas_qc.tsv
├── draft_panel_legend.md
└── methods_note.md
```

`control_state_assignments.tsv` includes one-hot `probability__...` columns so it can be used directly by the downstream perturbation-class occupancy workflow.
