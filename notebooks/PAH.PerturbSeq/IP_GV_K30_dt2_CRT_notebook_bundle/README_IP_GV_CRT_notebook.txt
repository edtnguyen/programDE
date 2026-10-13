IP/GV CRT notebook: K=30, density threshold=2.0
====================================================

Files
-----
1. setup_programDE_ipgv_mamba.sh
   Creates a user-owned programDE environment and registers the Jupyter kernel:

       Python (programDE IP-GV)

2. IP_GV_K30_dt2_CRT_calibration_and_discovery.ipynb
   Interactive calibration and final target-program CRT workflow.

Recommended order
-----------------
On a Sherlock login shell:

    bash setup_programDE_ipgv_mamba.sh

Open the notebook in Sherlock Jupyter and select:

    Python (programDE IP-GV)

Notebook workflow
-----------------
1. Locate and validate the final K=30, density-threshold=2.0 H5MU.
2. Compare several scientifically motivated covariate sets using NTC
   pseudo-target calibration.
3. Display raw and skew-calibrated QQ plots inline.
4. Choose the covariate set and primary p-value family after inspection.
5. Optionally run the full 5,000-permutation CRT separately within each mouse.
6. Save empirical and skew p-values, CLR effects, BH q-values, QQ plots, and
   an exploratory cross-mouse signed-Stouffer summary.

Defaults
--------
Calibration:
    1,024 permutations
    20 NTC ensembles
    4 NTC guides per pseudo-target
    7 groups per ensemble
    5 guide-frequency bins

Production:
    5,000 permutations
    BH FDR
    K=30
    density threshold=2.0

The default candidate covariate sets are:
    none
    RNA QC
    RNA QC + guide burden
    RNA QC + guide burden + hemoglobin percentage

The notebook does not automatically choose a model. Prefer the simplest
covariate set that gives acceptable calibration without strong p-value
inflation or excessive deflation.
