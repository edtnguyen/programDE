Corrected MAGMA gene-property resume
=====================================

Cause of the error
------------------
MAGMA v1.10 expects:

    --gene-covar <COVARIATE_FILE> include=P1,...,P30

The previous R script passed:

    --gene-covar file=<COVARIATE_FILE> include=P1,...,P30

MAGMA interpreted the complete literal text "file=/path/..." as the filename
and therefore reported that the file did not exist.

Run
---
This bundle resumes at stage 3. It does not rerun:

- GWAS download and formatting;
- SNP-to-gene annotation;
- MAGMA gene analysis;
- program mapping;
- 20,000 matched gene permutations.

Submit:

    chmod +x *.sh *.sbatch *.R
    bash submit_resume_pah_gwas_program_tests.sh

Output
------
    .../PAH_GWAS_program_burden/GCST007228/summary/
        PAH_GWAS_K30_program_burden_review.xlsx
        PAH_GWAS_program_burden_review.pdf
        PAH_GWAS_program_burden_results.tsv

Logs
----
    .../PAH_GWAS_program_burden/GCST007228/logs/
        program_tests_resume_<jobid>.out
        program_tests_resume_<jobid>.err
