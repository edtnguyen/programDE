# Notebook outputs

The notebook writes only local audit/staging artifacts. It contains no live POST, PATCH, or upload operation.

Expected outputs under `work/`:

- `portal_analysis_sets.tsv`
- `portal_files.tsv`
- `portal_audits.tsv`
- `h5ad_inspection.json`
- `h5mu_inspection.json`
- `source_to_submission_crosswalk.tsv`
- `staging_report.tsv`
- `validation_report.tsv`
- `column_dictionary.tsv`
- `metadata/matrix_file.tsv`
- `metadata/tabular_file.tsv`
- `metadata/eligible_upload_manifest.tsv`
- `dry_run_commands.sh`

The metadata TSVs are created only when:

1. the fixed four-timepoint AnalysisSet mapping validates;
2. source files exist;
3. staged files pass local validation;
4. rows are not marked already uploaded or blocked by a DACC decision.
