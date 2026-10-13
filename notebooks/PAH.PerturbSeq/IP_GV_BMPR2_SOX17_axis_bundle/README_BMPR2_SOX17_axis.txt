BMPR2 → SOX17 → cNMF program → human PAH workflow
=====================================================

Why this is the next step
-------------------------
The completed SOX17 analysis establishes one variant-to-gene-to-program chain.
The next high-impact test is whether BMPR2 loss converges on the same programs,
because BMPR2 is the dominant heritable PAH gene and a published
BMPR2–SOX17 signaling relationship exists.

Human BMPR2 perturbation datasets
---------------------------------
GSE70456
    Primary HPAEC siBMPR2 versus control.
    Four paired biological experiments at baseline and after PMA stimulation.
    Affymetrix PrimeView microarray.

GSE157231
    HPAEC siBMPR2 versus control under:
        normoxia
        6 h hypoxia
        24 h hypoxia
    Two biological patients and two technical replicates per condition.
    Technical replicates are summed before edgeR; patients are the independent
    units.

Directional-chain criterion
---------------------------
A strong program should show:

1. concordance with mouse IP_GFP Bmpr2 CRT beta;
2. concordance in at least three human BMPR2-loss contrasts;
3. agreement between program scores and preranked GSEA;
4. the same direction after human SOX17 loss;
5. the same direction in independent human PAH PAECs.

All experimentally targeted genes are removed from transferred program sets.

Run
---
    tar -xzf IP_GV_BMPR2_SOX17_axis_bundle.tar.gz
    chmod +x *.sh *.sbatch *.py *.R

    bash setup_bmpr2_sox17_axis_mamba.sh
    bash 00_download_BMPR2_GEO.sh
    bash submit_bmpr2_sox17_axis.sh

Outputs
-------
Root:

    .../results/IP_GV_skcnmf_K30_dt2_0_n100/
        Interpretation/BMPR2_SOX17_axis/

Primary review:

    analysis/BMPR2_SOX17_PAH_axis_review.xlsx
    analysis/BMPR2_SOX17_PAH_axis_review.pdf
    analysis/BMPR2_SOX17_PAH_directional_axis.tsv
    analysis/BMPR2_program_score_effects.tsv
    analysis/BMPR2_program_GSEA.tsv

Logs:

    logs/prepare_<jobid>.out/.err
    logs/analysis_<jobid>.out/.err

Limitations
-----------
GSE70456 uses one HPAEC donor across four experimental runs. GSE157231 has two
biological donors, so exact paired p-values are coarse. The value is in
directional replication across independent platforms and contexts, not in any
single nominal p-value.
