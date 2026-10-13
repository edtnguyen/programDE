# Consolidated information still needed

The four timepoint AnalysisSets are fixed and no longer require a decision:

| Day | Active AnalysisSet |
|---|---|
| D0 | `IGVFDS5257JGQO` |
| D1 | `IGVFDS3530HHEU` |
| D2 | `IGVFDS7380DUFB` |
| D3 | `IGVFDS6826YKCZ` |

The notebook can perform the read-only audit immediately. These remaining items are required before a submission-ready manifest can be finalized.

1. **Combined h5mu handling.** Confirm whether `cNMF_50_0_2.h5mu` should be submitted once, omitted because flattened tables are sufficient, or represented another way approved by DACC. The package will not create four artificial copies.
2. **Differential-result source paths.** Fill the 16 blank paths in `config/source_path_overrides.tsv` for local/global, per-guide/per-element outputs for D0-D3.
3. **Cis/trans definitions.** Provide the actual definitions/model output documentation showing whether “local” is formally cis and “global” is formally trans. Otherwise generic differential content types should be requested from DACC.
4. **Guide metadata path.** Provide the absolute Sherlock path to the already uploaded `300genes_guide_metadata_v43.tsv` if it should be used for identifier cross-checking.
5. **Software versions.** Provide the exact Git tag or commit SHA used for CC-Perturb-seq and PerturbNMF.
6. **Portal warning detail.** Run the notebook’s authenticated Portal-audit cell, or provide a Portal TSV/JSON export, so the exact WARNING/NOT_COMPLIANT messages for the supplied examples can be reviewed.
7. **h5ad day field.** Confirm the intended `obs` column and raw values for D0-D3 if the notebook does not identify them unambiguously.
8. **Existing uploaded file accessions.** The Portal audit should recover these. If it cannot, provide the individual MatrixFile/TabularFile accessions or a downloaded TSV from each prior-upload AnalysisSet.
9. **Reference-file scope.** Confirm that `/reference-files/IGVFFI7217ZMJZ/` is appropriate for all matrix and tabular file types, not only the gene-coordinate-dependent outputs.
