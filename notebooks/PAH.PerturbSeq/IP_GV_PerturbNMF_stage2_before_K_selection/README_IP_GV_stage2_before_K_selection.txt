IP/GV PerturbNMF Stage 2A before K selection
================================================

Scope
-----
This bundle evaluates all 16 pilot models:

    K = 5, 10, 15, 20, 25, 30, 40, 50
    density threshold = 0.2, 2.0

It performs the model-evaluation steps that are useful before selecting K:

1. Compile each K × threshold consensus result to H5MU.
2. Mouse/sample association.
3. Per-mouse exploratory perturbation association.
4. Mouse GO Biological Process enrichment.
5. Mouse Reactome enrichment.
6. Explained variance.
7. Cross-K aggregation, GO/Reactome heatmaps, and a review workbook/PDF.

It deliberately does not run:
- human OpenTargets trait enrichment;
- human hg38 motif enrichment;
- CRT calibration across all 16 models.

CRT is computationally expensive and is more appropriate after selecting K
(or for the selected model plus one close runner-up).

One-time setup
--------------
Run this from a Sherlock login shell:

    bash setup_ip_gv_perturbnmf_stage2_env_mamba.sh

The setup creates:

    /oak/stanford/groups/engreitz/Users/tri/envs/
        perturbnmf_stage2_20260620

It patches only the user-owned PerturbNMF checkout. The patches:
- allow dense cNMF scores in perturbation testing;
- remove mandatory torch-cNMF imports for CPU Stage 2;
- make motif dependencies optional;
- fix the upstream reassign_name typo;
- cache mouse GO and Reactome libraries.

Submit the full chain
---------------------
After setup succeeds:

    chmod +x submit_ip_gv_stage2_before_k_selection.sh
    bash submit_ip_gv_stage2_before_k_selection.sh

The dependency chain is:

    preflight
       ↓
    16-model compile array
       ↓
    16-model Stage 2 metrics array
       ↓
    cross-K summary

By default, four array tasks run concurrently. Override this with:

    MAX_COMPILE_CONCURRENT=2 \
    MAX_METRICS_CONCURRENT=2 \
    bash submit_ip_gv_stage2_before_k_selection.sh

Logs
----
All job logs are stored under:

    /oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/
    250507_ePAH1_NovaSeq/CRISPR_count/
    IP_GV_unified_sceptre_cnmf_no_celltype/PerturbNMF/
    pilot_array_n20/stage2_pre_k_selection/logs/

Names:

    preflight_<jobid>.out/.err
    compile_<array_jobid>_<taskid>.out/.err
    metrics_<array_jobid>_<taskid>.out/.err
    summary_<jobid>.out/.err

Evaluation outputs
------------------
Per-model outputs:

    .../pilot_array_n20/merged/Evaluation/<K>_<threshold_tag>/

For example:

    Evaluation/20_0_2/
        20_categorical_association_results.txt
        20_perturbation_association_results_IP_GFP.txt
        20_perturbation_association_results_GV_GFP.txt
        20_geneset_enrichment.txt
        20_GO_term_enrichment.txt
        20_Explained_Variance.txt
        20_Explained_Variance_Summary.txt

Cross-K review outputs:

    .../pilot_array_n20/stage2_pre_k_selection/
        summary_before_k_selection/

Key files:

    pre_K_selection_review.pdf
    model_selection_metrics.tsv
    model_selection_review.xlsx
    GO_terms_across_K_heatmap.pdf
    Reactome_terms_across_K_heatmap.pdf
    perturbation_replication_across_mice.tsv
    top20_genes_all_models.tsv
    HOW_TO_REVIEW_BEFORE_K_SELECTION.txt

Interpretation
--------------
Do not select K from one metric alone. Review:

- stability and prediction error;
- explained variance;
- GO/Reactome interpretability;
- whether programs are dominated by mouse identity;
- whether perturbation effect directions agree across IP_GFP and GV_GFP.

The Stage 2A perturbation test is a per-cell Mann-Whitney test, run separately
within each mouse. Because many cells have multiple guides, treat this result
as an exploratory K-selection diagnostic. Use CRT after selecting K for the
main perturbation-program association analysis.

Recovery
--------
Each task writes a completion marker and the metrics pipeline uses
--skip_existing.

Rerun a single model, for example K=20 and threshold=0.2 (task 6):

    sbatch --array=6 02_stage2_compile_models_array.sbatch
    sbatch --array=6 03_stage2_metrics_array.sbatch

After all model metrics exist:

    sbatch 04_stage2_summarize_before_k_selection.sbatch

Force H5MU recompilation:

    sbatch --export=ALL,RECOMPILE_MODELS=1 \
        --array=6 02_stage2_compile_models_array.sbatch
