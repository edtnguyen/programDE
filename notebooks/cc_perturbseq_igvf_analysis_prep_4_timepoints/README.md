# CC-Perturb-seq IGVF analysis-file preparation notebook

This package audits and stages analysis files for the WTC11 iPSC-to-EC 300-gene CC-Perturb-seq dataset. It follows the uploaded metadata workbook and source prompt, but it deliberately stops at the dry-run preparation boundary.

**It contains no live Portal POST, PATCH, or upload code.**

## What is already handled

- The metadata workbook is parsed into a 52-row expected-file inventory.
- Existing raw SequenceFile accessions from the prior 300-gene upload are included and grouped by day.
- Known Sherlock paths found in the attached notebooks are pre-populated.
- The five PerturbNMF table types that the user says are already uploaded are marked as Portal examples and excluded from new upload preparation by default.
- The notebook can query the four active AnalysisSets plus five prior-upload example AnalysisSets read-only and export exact audit warnings.

## AnalysisSet design

The active submission uses exactly four existing AnalysisSets, one per timepoint:

| Day | AnalysisSet |
|---|---|
| D0 | `IGVFDS5257JGQO` |
| D1 | `IGVFDS3530HHEU` |
| D2 | `IGVFDS7380DUFB` |
| D3 | `IGVFDS6826YKCZ` |

Every new file is assigned by its day. The separately supplied Portal sets are retained only for read-only inspection of already-uploaded files and warnings; they are not alternative target sets.

## Remaining blockers recorded by the package

1. The real PerturbNMF h5mu path appears to be one combined D0-D3 file, while the workbook plans four day-specific h5mu files.
2. Local/global differential outputs cannot be labeled cis/trans without confirming the actual definitions.
3. Exact local paths to the 16 differential-result tables are still needed.

Read `docs/audit_findings.md` and `docs/questions_needed.md` before enabling staging writes.

## Package layout

```text
notebooks/00_prepare_igvf_analysis_files.ipynb
src/igvf_analysis_prep.py
config/config.example.yaml
config/expected_files.tsv
config/source_path_overrides.tsv
config/sherlock_candidate_paths.tsv
config/portal_examples.tsv
config/analysis_set_crosswalk.tsv
config/existing_uploaded_reconciliation_template.tsv
config/sequence_file_accessions.csv
inputs/metadata_workbook_original.xlsx
docs/
scripts/
tests/
```

## 1. Copy to Sherlock

```bash
cd /oak/stanford/groups/engreitz/Users/tri
unzip cc_perturbseq_igvf_analysis_prep_4_timepoints.zip
cd cc_perturbseq_igvf_analysis_prep_4_timepoints
```

The Git clone is preferable for long-term work. You can place the package under:

```text
/oak/stanford/groups/engreitz/Users/tri/repos/
igvf-data-submission-stanford-fcc/
submissions/cc-perturb-seq-300gene/analysis_prep/
```

## 2. Create the analysis environment

Run on a compute node:

```bash
mamba env create \
  --prefix /oak/stanford/groups/engreitz/Users/tri/envs/igvf-analysis-prep \
  --file environment.yml

mamba activate /oak/stanford/groups/engreitz/Users/tri/envs/igvf-analysis-prep
python -m ipykernel install --user \
  --name igvf-analysis-prep \
  --display-name "IGVF analysis prep"
```

Run tests:

```bash
bash scripts/run_tests.sh
```

## 3. Make a working config

```bash
cp config/config.example.yaml config/config.yaml
nano config/config.yaml
nano config/source_path_overrides.tsv
```

The configuration already fixes the active mapping to the four workbook AnalysisSets:

```yaml
active_mapping: workbook_current
write_staged_files: false
perturbnmf_matrix_strategy: BLOCK_UNTIL_DACC_DECISION
```

Keep staging writes disabled until the remaining file-level and DACC decisions are reviewed.

## 4. Export Portal credentials for read-only warning inspection

Unreleased records may require credentials. Enter them in the same shell that launches Jupyter; do not save them in files.

```bash
read -r -s -p "IGVF_API_KEY: " IGVF_API_KEY; echo
export IGVF_API_KEY
read -r -s -p "IGVF_SECRET_KEY: " IGVF_SECRET_KEY; echo
export IGVF_SECRET_KEY
```

## 5. Quick read-only Portal warning audit

After exporting credentials, you can audit the four active AnalysisSets plus five prior-upload example AnalysisSets without opening Jupyter:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/audit_portal_examples.py --out work/portal_audit
```

Review:

```bash
column -s $'\t' -t < work/portal_audit/portal_audits.tsv | less -S
```

## 6. Run interactively

From a Sherlock compute node:

```bash
jupyter lab --no-browser --ip=0.0.0.0
```

Open:

```text
notebooks/00_prepare_igvf_analysis_files.ipynb
```

Use the `IGVF analysis prep` kernel.

## 7. Run as a Slurm batch audit

The default batch run is read-only/local-audit because the config has staging writes disabled.

```bash
sbatch scripts/run_notebook.sbatch
```

The executed notebook and logs are written beneath `runs/<job_id>/`.

## 8. First-run outputs

The first run should generate:

- read-only Portal audit tables for the four active AnalysisSets plus five prior-upload example AnalysisSets;
- source-path existence checks;
- h5ad and h5mu structural reports;
- a 52-row source-to-submission crosswalk;
- a list of missing and blocked files.

It should **not** create live Portal objects or upload files.

## 9. Enable local staging only after reviewing the audit

After resolving the combined-h5mu and remaining file-level decisions:

1. keep `active_mapping: workbook_current`;
2. fill `source_path_overrides.tsv`;
3. set `write_staged_files: true`;
4. rerun on a compute node with sufficient memory.

The notebook then:

- splits the combined h5ad into validated day-specific copies;
- derives candidate guide-quantification and per-cell QC tables;
- converts existing differential tables to deterministic `.tsv.gz` files;
- validates gzip structure, rows, columns, h5ad/h5mu structure, file sizes, and MD5 checksums;
- creates draft MatrixFile/TabularFile POST TSVs and dry-run command templates.

The notebook still does not execute the dry-run or live submission commands automatically.

## Known source candidates

```text
Combined h5ad:
/oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/
2.preprocessing_tonys_pipeline/3.integrated_days/
3.all_days_concat_pc30_all_genes_sceptre_guides.h5ad

Combined PerturbNMF h5mu:
/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/
030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/
adata/cNMF_50_0_2.h5mu

cNMF usage:
/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/
030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/
030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50.usages.k_50.dt_0_2.consensus.txt

Program metadata:
/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/
notebooks/per_day_umaps/data/program_activity_long_v4.csv
```

The notebook checks these paths on Sherlock instead of assuming they still exist.
