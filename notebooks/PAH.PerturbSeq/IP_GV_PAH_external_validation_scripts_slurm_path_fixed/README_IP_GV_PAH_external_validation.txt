IP/GV mouse-to-human PAH external validation
================================================

Purpose
-------
This workflow performs the PAH-specific external validation described in the
single-time-point mouse analysis framework.

It transfers mouse cNMF program genes to one-to-one human orthologs and tests
whether those programs align with an independent human PAH endothelial
transcriptomic dataset.

Default human cohort
--------------------
GSE185479 contains cultured pulmonary arterial endothelial cells from:

    Healthy: AH021, AH006, A677
    PAH:     CC027, CC014, CC015

The workflow uses donors, not cells, as the independent units for differential
expression and program-score comparisons.

Circularity control
-------------------
All experimentally targeted genes are read from:

    prepared_h5ad.uns["guide_targets"]

Every non-control target is removed from:

1. the mouse cNMF top-gene list;
2. the mapped human program gene set;
3. the human PAH differential-expression ranking;
4. the hypergeometric overlap universe.

For this experiment, this should remove the 16 targeting genes while retaining
the non-targeting control label.

Ortholog policy
---------------
The primary transfer uses the MGI HOM_ProteinCoding.rpt report, which contains
mouse protein-coding genes with one-to-one human orthology. This avoids
arbitrary one-to-many expansion.

Per-model analyses
------------------
For each of the 16 pilot models:

    K = 5, 10, 15, 20, 25, 30, 40, 50
    density threshold = 0.2, 2.0

the workflow calculates:

- one-to-one ortholog coverage;
- donor-level weighted program scores;
- PAH-minus-healthy Hedges g;
- exact donor-label permutation p-value;
- edgeR pseudobulk PAH-versus-healthy differential expression;
- preranked GSEA NES/FDR;
- overlap with PAH-up and PAH-down differential genes;
- concordance between donor-score and GSEA direction.

One-time setup
--------------
From a Sherlock login shell:

    bash setup_pah_external_validation_env_mamba.sh

Download public data and the MGI report:

    bash 00_download_GSE185479_and_MGI.sh

The GSE archive is approximately 322 MB.


Slurm support-file staging
--------------------------
The submit wrapper copies the Python, R, and sample-metadata files into:

    .../external_validation_human_PAH/code/

and exports that path to every batch job. This is necessary because Slurm
executes a copied SBATCH script under /var/spool/slurmd rather than from the
directory where the bundle was extracted.

Submit
------
After setup and download:

    chmod +x *.sh *.sbatch
    bash submit_pah_external_validation.sh

The dependency chain is:

    prepare GSE185479 + edgeR
          ↓
    16-model validation array
          ↓
    cross-K summary

Change array concurrency:

    MAX_CONCURRENT=2 bash submit_pah_external_validation.sh

Outputs
-------
Root:

    /oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/
    250507_ePAH1_NovaSeq/CRISPR_count/
    IP_GV_unified_sceptre_cnmf_no_celltype/PerturbNMF/
    pilot_array_n20/external_validation_human_PAH/

Per-model outputs:

    models/K20_dt0_2/
        external_validation_summary.tsv
        donor_program_scores.pdf
        human_PAH_effects.pdf
        donor_score_vs_GSEA.pdf
        ortholog_coverage.pdf
        program_PAH_preranked_GSEA.tsv
        program_PAH_DE_overlap.tsv
        program_mouse_to_human_mapping.tsv.gz
        experimentally_targeted_genes_removed.tsv

Cross-K outputs:

    summary_before_K_selection/
        human_PAH_external_validation_review.pdf
        human_PAH_external_validation_review.xlsx
        external_validation_across_models.tsv
        external_validation_all_programs.tsv.gz
        top_external_programs_per_model.tsv
        HOW_TO_INTERPRET.txt

Logs
----
All logs are written to:

    .../external_validation_human_PAH/logs/

Names:

    prepare_<jobid>.out/.err
    model_<array_jobid>_<taskid>.out/.err
    summary_<jobid>.out/.err

Interpretation
--------------
GSE185479 has only six independent donors. The exact score test has 20 possible
3-versus-3 label assignments, so it has coarse p-value resolution. Use effect
size, direction, donor separation, and GSEA jointly.

GSE185479 uses cultured PAECs. A mouse program may fail validation because it
comes from another cell type, an in-vivo vascular niche, or a state lost in
culture. Failure to validate is not evidence that the program is biologically
irrelevant.

Do not select K using human validation alone. Combine it with:

- cNMF stability and reconstruction error;
- mouse-to-mouse reproducibility;
- null-guide calibration;
- GO/Reactome interpretation;
- guide-level consistency.

A second in-vivo human lung PAH dataset can be added later. Required information
would be an H5AD path plus donor, disease, and endothelial-subtype metadata
columns.
