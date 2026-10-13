# Initial audit findings

This package preserves the uploaded metadata workbook as the governing draft, but it does not silently resolve the following inconsistencies.

## 1. Four timepoint AnalysisSets are authoritative

The source prompt and metadata workbook explicitly define one existing AnalysisSet per timepoint:

| Day | Active AnalysisSet |
|---|---|
| D0 | `IGVFDS5257JGQO` |
| D1 | `IGVFDS3530HHEU` |
| D2 | `IGVFDS7380DUFB` |
| D3 | `IGVFDS6826YKCZ` |

The supplied Portal links `IGVFDS0463XTUH`, `IGVFDS0991DBQZ`, `IGVFDS4402ZGWJ`, `IGVFDS2307REVJ`, and `IGVFDS9440URJB` are read-only examples containing files already uploaded. They are not replacements for the workbook AnalysisSets and are not used as `file_set` values for new files.

The notebook enforces exactly four unique active AnalysisSets and maps every new file strictly from its D0-D3 day value.

## 2. Combined PerturbNMF h5mu versus four planned h5mu files

The notebooks point to one combined D0-D3 file:

`/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/adata/cNMF_50_0_2.h5mu`

The workbook plans one h5mu per day. The source prompt explicitly prohibits creating four artificial copies when the real output is combined. The notebook therefore inspects this file but blocks h5mu staging until DACC approves a combined-object design.

## 3. Already uploaded PerturbNMF tables

The user states that these file types are already uploaded and should not be recreated:

- gene programs;
- gene universe;
- gene program regulators;
- regulator-regulator correlation;
- program-program correlation.

Rows for these content types remain in the 52-row workbook inventory, but the generated crosswalk marks them `SKIP_ALREADY_UPLOADED_RECONCILE_PORTAL`. The read-only portal audit is used to capture their accessions, warnings, and schemas.

## 4. Three-step workbook versus four-step analysis-plan narrative

The detailed plan describes four conceptual steps, with reporting as Step 4. The current workbook and explicit prompt use three portal AnalysisStep records and fold reporting into PerturbNMF Step 3. This package follows the current workbook and does not add a fourth AnalysisStep.

## 5. Local/global versus cis/trans labels

The workbook maps local results to cis content types and global results to trans content types. The source prompt and analysis plan both require confirmation that the actual statistical outputs meet those definitions. The notebook preserves the current draft labels but marks them as a DACC decision and does not finalize live metadata automatically.

## 6. Known Sherlock candidates

The attached notebooks identify these likely source files:

- combined annotated h5ad: `/oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/3.integrated_days/3.all_days_concat_pc30_all_genes_sceptre_guides.h5ad`
- combined PerturbNMF h5mu: `/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/adata/cNMF_50_0_2.h5mu`
- cNMF usage matrix: `/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50.usages.k_50.dt_0_2.consensus.txt`
- program metadata: `/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/per_day_umaps/data/program_activity_long_v4.csv`

Only the first two are direct candidates for planned MatrixFiles. The notebook checks whether each exists before using it.
