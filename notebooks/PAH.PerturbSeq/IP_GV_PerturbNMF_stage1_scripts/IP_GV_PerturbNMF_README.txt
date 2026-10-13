IP/GV PerturbNMF Stage 1 scripts
=================================

Files
-----
1. run_ip_gv_perturbnmf_pilot.sbatch
   - Prepares the combined guide-positive H5AD.
   - Adds obs["sample"], obsm["guide_assignment"], X_pca, and X_umap.
   - Preserves raw RNA counts in X.
   - Validates the object.
   - Runs K = 5, 10, 15, 20, 25, 30, 40, 50 with 20 replicates/K.
   - Runs consensus at density thresholds 0.2 and 2.0.
   - Writes K-selection and diagnostic plots.
   - Does not create redundant H5MU files for all 16 pilot models.

2. run_ip_gv_perturbnmf_final.sbatch
   - Run only after selecting K and density threshold from the pilot.
   - Runs 100 replicates.
   - Compiles the selected result into an H5MU.
   - Preserves sample and guide-assignment metadata for Stage 2.
   - Validates the final H5MU.

Submit
------
Place both scripts in a Sherlock working directory, then:

    sbatch run_ip_gv_perturbnmf_pilot.sbatch

After reviewing the pilot K-selection and clustering plots:

    sbatch run_ip_gv_perturbnmf_final.sbatch <K> <density_threshold>

Example syntax only:

    sbatch run_ip_gv_perturbnmf_final.sbatch 30 0.2

Do not use the example K until the pilot has been reviewed.

Optional MyGene annotation
--------------------------
The final script does not require network annotation. To request it:

    sbatch --export=ALL,RUN_MYGENE_ANNOTATION=1 \
        run_ip_gv_perturbnmf_final.sbatch <K> <density_threshold>

Resume behavior
---------------
Both scripts use --skip_existing. Resubmitting the same run name resumes missing
replicates rather than intentionally recomputing completed replicate files.

Rebuild prepared input
----------------------
The pilot reuses its prepared H5AD. To rebuild it after changing the source:

    sbatch --export=ALL,REBUILD_INPUT=1 \
        run_ip_gv_perturbnmf_pilot.sbatch
