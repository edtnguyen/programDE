IP/GV PerturbNMF array pilot
================================

Files
-----
01_prepare_ip_gv_perturbnmf_array.sbatch
    Prepares and validates the PerturbNMF H5AD, if needed.
    Generates deterministic, K-specific NMF seed files.

02_run_ip_gv_perturbnmf_k_array.sbatch
    SLURM array with tasks 0-7 and a default concurrency cap of 4.
    Each task runs one K:
        0:5, 1:10, 2:15, 3:20, 4:25, 5:30, 6:40, 7:50
    Each K uses 20 cNMF replicates and 3,000 HVGs.
    Completed replicate files are reused on resubmission.

03_merge_ip_gv_perturbnmf_array.sbatch
    Runs only after every K task succeeds.
    Validates and copies all 160 spectra files into one merged run.
    Reconstructs the all-K seed ledger.
    Runs combine, K selection, consensus at 0.2 and 2.0, and diagnostics.

submit_ip_gv_perturbnmf_array_pilot.sh
    Submits the preparation -> array -> merge dependency chain.

04_run_ip_gv_perturbnmf_final.sbatch
    Runs the selected K and threshold with 100 replicates after pilot review.
    Supplies explicit seeds for compatibility with the pinned sk-cNMF fork.

Submit
------
From a Sherlock login shell:

    chmod +x submit_ip_gv_perturbnmf_array_pilot.sh
    bash submit_ip_gv_perturbnmf_array_pilot.sh

The default array specification is:

    0-7%4

To change only the concurrency cap:

    MAX_CONCURRENT=2 bash submit_ip_gv_perturbnmf_array_pilot.sh

Outputs
-------
The array pilot uses a new directory and does not overwrite the original
single-job pilot:

    /oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/
    250507_ePAH1_NovaSeq/CRISPR_count/
    IP_GV_unified_sceptre_cnmf_no_celltype/PerturbNMF/pilot_array_n20/

Primary pilot diagnostics:

    merged/Inference/Inference.k_selection.png
    merged/Inference/Inference.k_selection_stats.tsv
    merged/Inference/diagnosis_plots/
    merged/Inference/Inference.clustering.k_<K>.dt_0_2.png
    merged/Inference/Inference.clustering.k_<K>.dt_2_0.png

After selecting K and threshold
-------------------------------
Use the included final script:

    sbatch 04_run_ip_gv_perturbnmf_final.sbatch <K> <threshold>

Example syntax only:

    sbatch 04_run_ip_gv_perturbnmf_final.sbatch 30 0.2

Recovery
--------
The array task uses --skip_existing. To rerun one failed K task manually after
preparation, submit the corresponding task index, for example K=20:

    sbatch --array=3 02_run_ip_gv_perturbnmf_k_array.sbatch

After all K tasks have completed successfully, submit the merge script:

    sbatch 03_merge_ip_gv_perturbnmf_array.sbatch

To rebuild the prepared H5AD because the source file changed:

    sbatch --export=ALL,REBUILD_INPUT=1 \
        01_prepare_ip_gv_perturbnmf_array.sbatch
