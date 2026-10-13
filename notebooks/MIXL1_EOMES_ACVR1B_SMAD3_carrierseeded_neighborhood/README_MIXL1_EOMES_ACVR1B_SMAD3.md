# MIXL1 / EOMES / ACVR1B / SMAD3 carrier-seeded neighborhood notebook

This is a fresh, focused copy of the carrier-seeded neighborhood UMAP workflow.

## Targets
- MIXL1
- EOMES
- ACVR1B
- SMAD3

## Method
- Target-carrier cells are seeds.
- KNN neighborhoods are calculated in z-scored cNMF program-usage space.
- K = 300.
- Neighbors and NT baselines are restricted to the same day.
- D0–D3 are shown on a shared integrated UMAP.
- The default `simple_color` plotnine style matches the attached Fig. 3 notebook.

## Sherlock paths already configured
- AnnData: `/oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/3.integrated_days/3.all_days_concat_pc30_all_genes_sceptre_guides.h5ad`
- cNMF usage: `/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50.usages.k_50.dt_0_2.consensus.txt`
- Guide metadata: packaged as `300genes_guide_metadata_v43.tsv`

## Point size settings
In configuration Cell 2:
```python
FACET_BACKGROUND_POINT_SIZE = 0.005
FACET_FOREGROUND_POINT_SIZE = 0.02
```
These match the attached Fig. 3 plotnine notebook. Increase them if the maps look too faint.

## Alternative size/FDR style
Set:
```python
PLOT_STYLE = "size_fdr"
```
Then point size maps to neighborhood carrier fraction and outline indicates FDR significance.

## Outputs
The notebook exports PNG, PDF, SVG, source CSVs, per-target score tables, summaries, and an output ZIP in:
`MIXL1_EOMES_ACVR1B_SMAD3_carrierseeded_neighborhood_outputs/`
