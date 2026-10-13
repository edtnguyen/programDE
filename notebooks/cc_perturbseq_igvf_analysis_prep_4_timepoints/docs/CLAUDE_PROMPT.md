# Prompt for the IGVF submission agent

From the `igvf-data-submission-stanford-fcc` repository, give Claude Code this prompt after copying this package under the 300-gene submission directory:

```text
Use the cc_perturbseq_igvf_analysis_prep notebook package as a read-only/local-staging workflow.

Read:
- README.md in the package
- docs/audit_findings.md
- docs/questions_needed.md
- config/expected_files.tsv
- config/config.example.yaml
- the current submission START_HERE.md, STATUS.md, and CHECKLIST.md

Do not POST, PATCH, upload, modify existing Portal objects, or change the shared metadata workbook.
First run the local tests and inspect the configuration. Then help me fill source_path_overrides.tsv, run the read-only portal audit, and execute the notebook with write_staged_files=false. Report all conflicts and wait for approval before enabling staging writes.
```
