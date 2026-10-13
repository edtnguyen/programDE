IP/GV K30, density-threshold 2.0 summary workbook
======================================================

Files
-----
build_ip_gv_k30_dt2_workbook.py
    Standalone workbook builder adapted to the calibrated CRT output files.

run_ip_gv_k30_dt2_workbook.sbatch
    Sherlock SLURM job.

submit_ip_gv_k30_dt2_workbook.sh
    Submission helper that places logs under the final run directory.

Submit
------
Place all three files in one Sherlock directory, then:

    chmod +x *.sh *.sbatch
    bash submit_ip_gv_k30_dt2_workbook.sh

Output
------
The workbook will be written to:

    /oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/
    250507_ePAH1_NovaSeq/CRISPR_count/
    IP_GV_unified_sceptre_cnmf_no_celltype/PerturbNMF/results/
    IP_GV_skcnmf_K30_dt2_0_n100/Interpretation/Summary/K30_dt2_0/
    IP_GV_K30_dt2_PerturbNMF_summary.xlsx

Sheets
------
Program_summary
Target_summary
Program_loadings_flat
Program_loadings_long
GO
Reactome
CRT_all
CRT_IP_GFP
CRT_GV_GFP
Cross_mouse_CRT
Categorical_assoc
Explained_variance
Explained_var_summary (when available)
Guide_metadata
Sample_summary
Run_metadata
Input_manifest

CRT conventions
---------------
The workbook preserves:

    beta_clr
    p_empirical
    p_skew
    q_empirical_BH
    q_skew_BH
    primary_pvalue
    primary_q_BH

beta_clr is a CLR log-ratio effect. It is not renamed to ordinary expression
log2FC.

Logs
----
Logs are stored under:

    .../Interpretation/Summary/K30_dt2_0/logs/

Sidecar tables
--------------
Program_summary.tsv
Target_summary.tsv
Program_loadings_long.tsv.gz
CRT_all.tsv.gz
workbook_manifest.json


SLURM path handling
-------------------
The submission helper explicitly exports WORKBOOK_SCRIPT_DIR because SLURM
runs a copied batch script from /var/spool/slurmd. Do not derive the companion
Python-file path from BASH_SOURCE[0] inside an SBATCH job.

Direct submission is also supported:

    sbatch       --export=ALL,WORKBOOK_SCRIPT_DIR="$(pwd)"       run_ip_gv_k30_dt2_workbook.sbatch
