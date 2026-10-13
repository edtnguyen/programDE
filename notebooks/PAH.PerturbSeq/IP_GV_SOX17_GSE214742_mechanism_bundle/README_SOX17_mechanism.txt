SOX17 variant → SOX17 → cNMF program → human PAH analysis
================================================================

Purpose
-------
This workflow tests a directional mechanistic chain using:

1. the final K30, density-threshold=2.0, 100-replicate cNMF model;
2. mouse IP_GFP Sox17 CRT program effects;
3. human HPAEC SOX17 perturbation RNA-seq, GSE214742;
4. independent human PAH PAEC program validation, GSE185479.

GSE214742 contrasts
-------------------
Primary SOX17-loss contrasts:

    siSOX17 versus negative siRNA                     n=4 matched pairs
    promoter CRISPRi versus BFP CRISPRi control      n=6 matched pairs
    signal-1 enhancer repression versus BFP control  n=3 matched pairs

Orthogonal reversal:

    promoter CRISPRa versus BFP CRISPRa control      n=6 matched pairs

Technical control:

    siGAPDH versus negative siRNA                     n=4 matched pairs

For promoter CRISPRa, effects are multiplied by -1 when represented as a
SOX17-loss-aligned effect.

Circularity control
-------------------
All experimentally targeted genes, including SOX17, are removed from the
transferred cNMF program gene sets. Therefore, agreement is not driven by
direct measurement of the perturbed gene.

One-time setup
--------------
    bash setup_sox17_mechanism_env_mamba.sh

Download data
-------------
    bash 00_download_GSE214742_and_MGI.sh

The download step obtains:

    GSE214742_RAW.tar
    MGI one-to-one protein-coding ortholog report
    GENCODE v28 gene annotation

Run
---
    chmod +x *.sh *.sbatch *.py *.R
    bash submit_sox17_mechanism.sh

The dependency chain is:

    parse processed RNA-seq + paired edgeR
                      ↓
    program scoring + GSEA + directional-chain summary

Primary outputs
---------------
Root:

    .../results/IP_GV_skcnmf_K30_dt2_0_n100/
        Interpretation/SOX17_mechanism_GSE214742/

Review:

    analysis/SOX17_variant_to_program_to_PAH_review.xlsx
    analysis/SOX17_variant_to_program_to_PAH_review.pdf
    analysis/SOX17_directional_program_chain.tsv
    analysis/SOX17_expression_QC.tsv
    analysis/SOX17_coherent_program_genes.tsv

Logs:

    logs/prepare_<jobid>.out/.err
    logs/analysis_<jobid>.out/.err

What constitutes a strong program
---------------------------------
A high-priority program should show:

1. the same direction after mouse Sox17 perturbation and at least two human
   SOX17-loss perturbations;
2. an opposite response after human promoter CRISPRa;
3. the same direction in independent human PAH PAECs;
4. coherent program genes changing in the same direction.

Statistical notes
-----------------
- edgeR uses paired designs for every contrast.
- Program-score significance uses exact paired sign-flip tests.
- GSEA uses 5,000 phenotype-independent gene-set permutations.
- Exact sign-flip p-values are coarse with only 3–6 pairs.
- beta_clr is compositional program usage, not conventional expression log2FC.
- The full-directional-chain flag is a structured evidence summary, not a new
  formal mediation test.

Failure modes
-------------
If SOX17 is absent after parsing, inspect:

    GSE214742/processed/GSE214742_parser_report.tsv

The parser supports common Salmon/tximport count and TPM column names and uses
GENCODE v28 when the CSVs contain Ensembl gene IDs.
