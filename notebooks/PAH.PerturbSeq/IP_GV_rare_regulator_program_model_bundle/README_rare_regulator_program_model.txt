Rare-regulator → program → PAH genetics workflow
================================================

Scope
-----
This bundle implements the planned genetics sequence:

1. curate the target-gene rare-variant evidence;
2. attach quantitative directional burden effects (gamma);
3. test IP_GFP beta(g→P) against rare-variant effects;
4. fit a small joint program model with nested leave-one-gene-out prediction;
5. repeat the full inference procedure after gene-label permutations;
6. validate genes in FinnGen/Genebass/AZ/STAAR public rare-variant resources;
7. combine rare-regulator, program, common-variant, and human-PAEC evidence
   into a network-ready graph.

Scientific guardrail
--------------------
The published PAH sequencing papers provide strong gene-level association
evidence and carrier counts, but not a single uniformly comparable gamma and
standard error for all targeted genes.

The primary Ota-style model therefore refuses to run until at least six genes
have finite gamma_log_or values and quantitative_effect_selected=TRUE.

The seed table contains an association_z_proxy for BMPR2, GDF2, KDR, SOX17,
and TBX4. That proxy is reconstructed from published association p-values. It
is not an effect size. Use it only with:

    EFFECT_MODE=association_z_proxy

and label the result exploratory.

Primary exclusions
------------------
EIF2AK4
    Established biallelic/recessive PVOD-PCH mechanism; sensitivity only.

SMAD4
    PAH gene-disease relationship classified as disputed by ClinGen.

NOTCH1, TP53, CDKN1A
    Experimental controls, not PAH genetic outcomes.

One-time setup
--------------
    bash setup_rare_regulator_program_mamba.sh

Prepare review tables
---------------------
    bash submit_rare_regulator_model.sh

This submits only the review job. It creates:

    .../Interpretation/Rare_regulator_program_model/curation/
        rare_variant_curation_review.xlsx
        rare_variant_effects_reviewed.tsv
        portal_results_long.tsv
        IP_GFP_target_by_program_beta_clr.tsv

Portal validation
-----------------
FinnGen:
    Browse I9_HYPTENSPUL publicly. Bulk LoF files require the short FinnGen
    download form. After download, normalize the selected phenotype file with:

        python rare_regulator_program_model.py import-portal \
          --input <FINNGEN_FILE> \
          --curation rare_variant_effects_reviewed.tsv \
          --source FinnGen \
          --release DF13 \
          --endpoint I9_HYPTENSPUL \
          --phenotype-description "Primary pulmonary hypertension" \
          --mask pLoF \
          --test burden \
          --independent-source \
          --output finngen_normalized.tsv

Genebass:
    The browser is public. The full Hail MatrixTable is requester-pays and
    needs a Google Cloud billing project, not UK Biobank individual-level
    permission.

    Optional setup:

        bash setup_genebass_hail_mamba.sh

    Before reading the requester-pays bucket, authenticate Google Cloud and
    set an application-default credential:

        gcloud auth login
        gcloud auth application-default login

    Use a billing-enabled Google Cloud project in --gcp-project.

    Discover phenotype fields/matches:

        conda activate \
          /oak/stanford/groups/engreitz/Users/tri/envs/genebass_hail_20260623

        python genebass_hail_query.py \
          --gcp-project <PROJECT_ID> \
          --output-dir genebass_discovery \
          --tmp-dir /scratch/<USER>/genebass_tmp \
          discover-phenotypes

    Inspect matching_phenotypes.tsv and results_matrix_schema.txt before
    extraction.

AstraZeneca PheWAS and STAAR PheWAS:
    Use the public browser, document the exact phenotype/mask/test, and enter
    values into portal_results_long.tsv. Do not meta-analyze AstraZeneca and
    Genebass because both use UK Biobank.

Selecting quantitative effects
------------------------------
In portal_results_long.tsv:

1. choose the most phenotype-specific and mechanism-compatible row;
2. set selected_for_gene=TRUE;
3. select at most one row per gene unless independent cohorts were
   meta-analyzed separately;
4. record beta, SE, p-value, mask, cases and controls.

Apply selected portal rows:

    python rare_regulator_program_model.py apply-selected-portal \
      --curation rare_variant_effects_reviewed.tsv \
      --portal-results portal_results_long.tsv \
      --output rare_variant_effects_reviewed.tsv

The primary model requires at least six selected quantitative effects.
Eight or more is preferable.

Fit the quantitative model
--------------------------
Copy the reviewed table back to:

    .../Rare_regulator_program_model/curation/
        rare_variant_effects_reviewed.tsv

Then:

    sbatch \
      --export=ALL,RARE_REGULATOR_CODE_DIR=<.../Rare_regulator_program_model/code> \
      02_fit_rare_regulator_model.sbatch

The default final run uses:

    20,000 univariate gene-label permutations
    20,000 full nested-model permutations
    nested leave-one-gene-out program selection
    ridge regression
    up to five selected programs

Exploratory association-strength mode
-------------------------------------
Before quantitative gamma is available:

    EFFECT_MODE=association_z_proxy \
    MINIMUM_GENES=5 \
    MODEL_PERMUTATIONS=5000 \
    sbatch \
      --export=ALL,EFFECT_MODE,MINIMUM_GENES,MODEL_PERMUTATIONS,RARE_REGULATOR_CODE_DIR=<code> \
      02_fit_rare_regulator_model.sbatch

This tests prediction of published association strength, not directional
rare-variant effects.

Outputs
-------
    model/rare_regulator_program_model_review.xlsx
    model/univariate_program_rare_effect_tests.tsv
    model/leave_one_gene_out_predictions.tsv
    model/joint_program_coefficients.tsv
    model/model_summary.json
    model/rare_regulator_program_PAH.graphml
    model/rare_regulator_program_PAH.html
    model/network_nodes.tsv
    model/network_edges.tsv
    model/HOW_TO_INTERPRET.txt

Common-variant integration
--------------------------
If available, the model automatically reads:

    .../PAH_GWAS_program_burden/GCST007228/summary/
        PAH_GWAS_program_burden_results.tsv

and adds common-variant program evidence to the graph.

Interpretation
--------------
This is a small targeted model, not a genome-wide replication of Ota et al.
Ota et al. had thousands of perturbed genes and directional GeneBayes LoF
effects. Here, the number of PAH targets is small and most disease effects are
expected to increase risk, so out-of-sample magnitude prediction is more
informative than sign accuracy.

Program selection is repeated inside every leave-one-gene-out training fold.
The full nested procedure is repeated after gamma-label permutations to avoid
optimistic model-selection p-values.
