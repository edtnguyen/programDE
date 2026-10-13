MAGMA v1.10 program-test resume v2
=================================

Cause
-----
The previous resume script used:

    --gene-covar FILE include=P1,...,P30

MAGMA v1.10 rejected include as an unknown --gene-covar modifier.

The v1.09+ manual documents:

    filter-read
    filter-skip

for selecting which columns are loaded from a gene-covariate file.

Correct commands
----------------
Unconditioned continuous tests:

    --gene-covar FILE filter-read=P1,...,P30
    --model direction-covar=greater

Human-PAEC-expression-conditioned continuous tests:

    --gene-covar FILE filter-read=mean_logCPM,P1,...,P30
    --model direction-covar=greater condition-hide=mean_logCPM

The script performs P1-only smoke tests for both forms before running all
programs.

Run
---
    chmod +x *.sh *.sbatch *.R
    bash submit_resume_pah_gwas_program_tests_v2.sh

This resumes from the existing genes.raw and program-input files. It does not
rerun the GWAS download, MAGMA gene analysis, program mapping, or matched
permutations.
