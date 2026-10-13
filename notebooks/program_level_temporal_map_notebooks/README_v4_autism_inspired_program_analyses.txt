Program-level temporal map notebooks v4

These notebooks implement the program-first analyses inspired by asynchronous-development papers, without using Leiden/UMAP clusters as the main interpretive unit.

Notebooks:
1. 01_program_effect_temporal_map.ipynb
   - Reads cNMF target-program effects from the Excel spreadsheet.
   - Filters to curated expert-checked programs.
   - Makes curated effect heatmaps, program correlation heatmap, selected scatter grids, and Hippo-focused heatmaps.

2. 02_program_temporal_phenotypes_and_archetypes.ipynb
   - Computes developmental-age / heterochrony scores, asynchrony scores, directional axes, first affected day, convergence counts, program coupling classes, and target modules.
   - Assigns rule-based program-defined phenotype tags.

3. 03_program_high_cell_DE_optional_annotation.ipynb
   - Optional AnnData-based annotation notebook.
   - Defines high-P14, high-P17, high-P15, high-P34, EC-high cells and runs DE between program-defined cell groups.
   - This is for biological annotation only, not the primary perturbation-effect analysis.

Main inputs needed:
- cNMF target-program effect spreadsheet, default path:
  /oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/per_day_umaps/data/cNMF_50_0_2_032526_6co.xlsx
- Curated program annotation CSV included here.
- Optional for notebook 03: an AnnData object named adata_int with Program columns in adata_int.obs.

Run notebooks in order 01 -> 02. Notebook 03 is optional.
