Final K30/dt2.0 validation in human PAH PAECs
================================================

Purpose
-------
This re-runs the existing cross-species PAH validation on the selected final
100-replicate model:

    K = 30
    density threshold = 2.0

Unlike the previous K sweep, the spectra come from:

    .../results/IP_GV_skcnmf_K30_dt2_0_n100/
        Inference/Inference.gene_spectra_score.k_30.dt_2_0.txt

Human dataset
-------------
GSE185479: cultured PAECs from 3 healthy and 3 PAH donors.

The workflow reuses the already downloaded and processed cohort under:

    .../pilot_array_n20/external_validation_human_PAH/

It does not redownload or reprocess the human data.

Circularity control
-------------------
The existing validation code removes all experimentally targeted genes from:

1. mouse program gene lists;
2. mapped human program sets;
3. the human PAH DE ranking;
4. the hypergeometric overlap universe.

Run
---
Place all files in one Sherlock directory:

    chmod +x *.sh *.sbatch *.py
    bash submit_final_K30_dt2_human_validation.sh

Outputs
-------
Root:

    .../results/IP_GV_skcnmf_K30_dt2_0_n100/
        Interpretation/Human_validation_GSE185479/

Primary review:

    summary/K30_dt2_human_validation_review.xlsx
    summary/K30_dt2_human_validation_review.pdf

Focused mouse-to-human table:

    summary/IP_q10_target_program_human_support.tsv

The focused table recalculates Storey q over the IP_GFP non-control CRT tests,
selects q<0.10 target-program pairs, and attaches human validation metrics for
the corresponding programs.

Human-support fields
--------------------
GSEA_FDR_lt_0.25
    Exploratory program enrichment in the human PAH DE ranking.

GSEA_FDR_lt_0.05
    Stricter enrichment support.

large_donor_effect_abs_g_ge_1
    Absolute donor-score Hedges g of at least 1.

complete_separation
    The 3 PAH donor scores are completely above or below all 3 healthy scores.

human_program_support
    GSEA FDR<0.25 OR (|Hedges g|>=1 AND complete separation).

This validates the program's relevance to human PAH endothelial data. It does
not prove that the corresponding mouse perturbation target has the same causal
effect in humans.

Limitations
-----------
With 3 versus 3 donors, the exact label-permutation test has only 20 possible
assignments. Effect size, donor separation, GSEA, ortholog coverage, and
direction consistency should be interpreted jointly.

Cultured PAECs may not represent immune, platelet, epithelial, stromal, or
in-vivo vascular-niche programs.

Logs
----
Logs are saved under:

    .../Interpretation/Human_validation_GSE185479/logs/
