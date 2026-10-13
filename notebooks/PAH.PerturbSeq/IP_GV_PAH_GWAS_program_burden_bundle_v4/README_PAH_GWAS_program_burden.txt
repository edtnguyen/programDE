IP/GV K30 program enrichment in PAH common-variant genetics
================================================================

Purpose
-------
This bundle performs the first Ota-inspired human-genetics analysis:

    PAH common-variant GWAS
              ↓
    MAGMA gene associations
              ↓
    loading-weighted K30 cNMF programs

The primary GWAS is GCST007228, the clinically ascertained international PAH
meta-analysis with 2,085 cases and 9,659 controls of European ancestry.

This is an unsigned common-variant enrichment analysis. It is not a direct
replication of Ota et al.'s directional loss-of-function program burden.

Analysis methods
----------------
Primary:
1. MAGMA continuous gene-property analysis using rank-decay program weights.
2. The same MAGMA analysis conditioned on mean expression in human PAECs.
3. Loading-weighted mean MAGMA gene Z with 20,000 matched permutations.

Sensitivity:
4. MAGMA competitive top-50 program gene sets.
5. MAGMA competitive top-300 program gene sets.
6. Repetition with the extended MHC included.

Circularity and locus control
-----------------------------
The primary program weights exclude:

- all experimentally targeted genes, including disease genes and control
  targets;
- human genes in the extended MHC interval, chr6:25–34 Mb.

This is important because GCST007228 has a strong HLA-DPA1/DPB1 locus.

Mouse program genes are transferred through MGI one-to-one protein-coding
human orthologs. The existing GSE185479 PAEC pseudobulk logCPM is used as a
gene-expression covariate and in matched null sampling.

One-time setup
--------------
    bash setup_pah_gwas_program_burden_mamba.sh

Download and format the primary GWAS on a Sherlock login node
--------------------------------------------------------------
    bash 01_download_pah_gwas.sh

This dynamically discovers the current GCST007228 summary-statistics file on
the GWAS Catalog FTP server and records its exact source URL and SHA256.

Submit the analysis
-------------------
    bash submit_pah_gwas_program_burden.sh

Primary outputs
---------------
Root:

    .../results/IP_GV_skcnmf_K30_dt2_0_n100/
        Interpretation/PAH_GWAS_program_burden/GCST007228/

Review files:

    summary/PAH_GWAS_K30_program_burden_review.xlsx
    summary/PAH_GWAS_program_burden_review.pdf
    summary/PAH_GWAS_program_burden_results.tsv
    summary/PAH_GWAS_program_driver_genes.tsv
    summary/HOW_TO_INTERPRET.txt

Logs:

    logs/program_burden_<jobid>.out
    logs/program_burden_<jobid>.err

Optional MVP sensitivity GWAS
-----------------------------
The public MVP European "primary pulmonary hypertension" study can be run as a
less phenotype-specific sensitivity analysis:

    export GWAS_ACCESSION=GCST90477880
    export GWAS_N=448274
    export GWAS_LABEL=MVP_EUR_primary_pulmonary_hypertension

    bash 01_download_pah_gwas.sh
    bash submit_pah_gwas_program_burden.sh

Interpret MVP cautiously: its EHR phecode phenotype is broader and less
clinically curated than GCST007228.

Interpretation
--------------
Strongest evidence is a program that is supported by:

- expression-conditioned MAGMA linear analysis;
- loading-weighted matched permutations;
- MHC-excluded and MHC-included sensitivities;
- independent human PAEC program validation.

A significant result means that genes highly loaded in the program are more
associated with PAH common-variant risk than comparable genes. It does not
provide a direction of program effect on PAH.

MAGMA reference downloads
-------------------------
The first batch run may download the European 1000 Genomes LD reference and
the appropriate GRCh37/GRCh38 gene-location resources through
MAGMA.Celltyping. These are cached under the GWAS output directory.


MAGMA.Celltyping dependency repair
----------------------------------
The original setup attempted remotes::install_local(..., dependencies=TRUE).
That can fail because EWCE and orthogene are Bioconductor packages, while lme4
is a compiled CRAN dependency.

The corrected setup installs these packages with mamba first:

    bioconductor-ewce
    bioconductor-ewcedata
    bioconductor-orthogene
    r-lme4

It then verifies every package in MAGMA.Celltyping DESCRIPTION/Imports and
installs MAGMA.Celltyping with dependencies=FALSE.

When the original setup already created the environment, use:

    bash repair_pah_gwas_magma_celltyping_setup.sh

After repair succeeds, continue with:

    bash 01_download_pah_gwas.sh
    bash submit_pah_gwas_program_burden.sh


MAGMA executable discovery repair
---------------------------------
MAGMA.Celltyping 2.0.16 can download MAGMA into a custom dest_dir and then
fail its final version check because the check searches the package's default
install/cache directory rather than the supplied custom directory.

When MAGMA.Celltyping itself is already installed, run:

    bash repair_pah_gwas_magma_executable_registration.sh

The helper reuses the previously downloaded binary, falls back to the official
static Linux build only when necessary, and registers a copy under both:

    MAGMA.Celltyping:::find_install_dir()
    <conda-environment>/bin/magma

It validates magma_executable(), magma_installed_version(), and
magma_check_version_match() before reporting success.


PAT-free MAGMA gene-location repair
-----------------------------------
MAGMA.Celltyping 2.0.16 downloads NCBI37.3.gene.loc or NCBI38.gene.loc through
piggyback and requires a GitHub PAT. This is unnecessary because the package
documents an immutable public raw-GitHub permalink for the same files.

Bundle v4 replaces run_magma_gene_analysis.R with a PAT-free implementation
that:

1. downloads the build-specific public gene-location file directly;
2. reuses the previously downloaded g1000_eur reference;
3. runs the two official MAGMA commands directly;
4. creates .genes.annot, .genes.out, and .genes.raw;
5. writes the same MAGMA_GENE_ANALYSIS.json consumed downstream.

After the prior job failed, do not repeat setup or GWAS download. Extract v4
and resubmit:

    tar -xzf IP_GV_PAH_GWAS_program_burden_bundle_v4.tar.gz
    chmod +x *.sh *.sbatch *.py *.R
    FORCE_MAGMA_GENE_ANALYSIS=true \
        bash submit_pah_gwas_program_burden.sh

The 1000 Genomes reference already downloaded into magma_gene/cache is reused.
