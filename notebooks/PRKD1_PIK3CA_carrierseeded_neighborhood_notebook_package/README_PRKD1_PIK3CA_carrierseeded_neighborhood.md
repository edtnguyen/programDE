# PRKD1 / PIK3CA carrier-seeded neighborhood UMAP notebook

This is a focused notebook for generating carrier-seeded neighborhood UMAP panels for **PRKD1** and **PIK3CA** only.

It was adapted from the plotnine-style Fig. 3 carrier-seeded neighborhood notebook and keeps the same plotting style:

- grey all-cell background on the shared D0-D3 UMAP
- target carrier cells as seed cells
- KNN neighborhoods computed in z-scored cNMF usage space
- K = 300
- neighbors restricted to the same day
- same-day non-targeting guide carrier baseline
- local enrichment shown as log2OR
- PDF/SVG export with editable text

## Main notebook

`PRKD1_PIK3CA_carrierseeded_neighborhood_plotnine_style.ipynb`

## Paths currently set in the notebook

AnnData:
`/oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/3.integrated_days/3.all_days_concat_pc30_all_genes_sceptre_guides.h5ad`

cNMF usage:
`/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50.usages.k_50.dt_0_2.consensus.txt`

Guide metadata:
`300genes_guide_metadata_v43.tsv`

The guide metadata TSV is included in this package. If you run the notebook from a different directory, update `cfg.GUIDE_METADATA_TSV` in the first config cell.

## Output directory

`prkd1_pik3ca_carrierseeded_neighborhood_outputs/`

Expected outputs include PNG/PDF/SVG figures and source CSVs for plotting/reproducibility.
