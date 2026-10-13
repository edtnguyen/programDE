# Fig. 3 carrier-seeded neighborhood UMAP maps

This package updates panel f/j to match the `UMAP neighborhood panel` section in `ipsc-ec.Hippo.figure.ipynb`.

## Extracted paths from the notebook

```text
ADATA_H5AD = /oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/3.integrated_days/3.all_days_concat_pc30_all_genes_sceptre_guides.h5ad

CNMF_USAGE_TXT = /oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50.usages.k_50.dt_0_2.consensus.txt

GUIDE_METADATA_TSV = 300genes_guide_metadata_v43.tsv
```

The notebook used the guide metadata as a relative path. If `300genes_guide_metadata_v43.tsv` is not in the run directory, edit `fig3_carrierseeded_config.py` and provide the absolute Sherlock path.

## Method

For each target:

1. Use each target-carrier cell as a seed.
2. Find `K=300` nearest neighbors in z-scored cNMF program-usage space.
3. Restrict neighbors to the same day as the seed.
4. Compare local target-carrier counts to local non-targeting-control counts.
5. Compute same-day local enrichment:

```text
log2((local_target + 0.5)/(local_NT + 0.5) / (background_target + 0.5)/(background_NT + 0.5))
```

6. Compute Fisher exact p-value and BH FDR across seed cells.

This is different from the earlier simplified panel f/j script, which computed KNN maps over all cells and could use UMAP space. This script matches the notebook's carrier-seeded cNMF-neighborhood logic.

## Run

```bash
python make_fig3_carrierseeded_neighborhood_panels.py
```

Outputs go to:

```text
fig3_carrierseeded_neighborhood_outputs/
```

Main outputs:

```text
panel_f_carrierseeded_neighborhood_grid.png/pdf/svg
panel_j_carrierseeded_neighborhood_grid.png/pdf/svg
carrier_seeded_scores_<target>_all_carriers.csv
combined_carrier_seeded_scores_all_carriers.csv
all_days_background_cells_for_plotting.csv
summary_by_target_day_all_carriers.csv
```
