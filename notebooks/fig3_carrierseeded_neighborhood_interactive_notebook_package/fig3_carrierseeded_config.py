"""Config for Fig. 3 panel f/j carrier-seeded neighborhood maps.

This version matches the UMAP-neighborhood logic from ipsc-ec.Hippo.figure.ipynb:
- target carrier cells are used as seeds
- KNN is computed in cNMF program-usage space
- neighbors are restricted to the same day
- enrichment compares local target-carrier counts to non-targeting controls from the same day
"""
from pathlib import Path

# Extracted from the uploaded notebook
ADATA_H5AD = "/oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/3.integrated_days/3.all_days_concat_pc30_all_genes_sceptre_guides.h5ad"
CNMF_USAGE_TXT = "/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50.usages.k_50.dt_0_2.consensus.txt"

# The notebook used a relative metadata path. Update this if your working dir differs.
GUIDE_METADATA_TSV = "300genes_guide_metadata_v43.tsv"

OUTDIR = "fig3_carrierseeded_neighborhood_outputs"

# AnnData keys / columns
DAY_COL = "day"
CELLTYPE_COL = "cell_type"
GUIDE_NAMES_KEY = "guide_names"
GUIDE_ASSIGNMENT_KEY = "guide_assignment"
UMAP_KEY = "X_umap"
CNMF_KEY = "cnmf_usage"

# Metadata columns. The notebook used putative_target_genes for this section.
GUIDE_ID_COL = "guide_id"
TARGET_COL_CANDIDATES = ["putative_target_genes", "intended_target_name", "target", "target_name", "gene"]

# Panel target lists
PANEL_F_TARGETS = ["ACVR2A", "BMPR1A", "SMAD1", "SMAD5"]
PANEL_J_TARGETS = ["KDR", "HAND1", "PRKCD", "PLCG1"]

# Controls
CONTROL_TARGET = "non-targeting_control"
CONTROL_PATTERNS = [
    r"non[-_ ]?target",
    r"non[-_ ]?targeting",
    r"\bntc\b",
    r"negative",
    r"neg[-_ ]?ctrl",
    r"scram",
    r"scrambled",
    r"safe",
    r"aavs1",
    r"control",
]

# KNN neighborhood settings from the notebook
PROGRAMS_FOR_KNN = None  # None = all 50 programs
CNMF_TRANSFORM = "zscore"
K = 300
PSEUDOCOUNT = 0.5
RESTRICT_NEIGHBORS_TO_SAME_DAY = True
SAME_DAY_BASELINE = True
MODE = "all_carriers"
DAYS_TO_USE = ["d0", "d1", "d2", "d3"]
LOG2FC_CLIP = 3.0
RANDOM_SEED = 1
EXPORT_NEIGHBOR_MEMBERSHIP = False

# Plot settings
MAX_BACKGROUND_CELLS = 60000
BACKGROUND_POINT_SIZE = 0.08
FOREGROUND_POINT_SIZE = 0.9
CMAP_LOW = "#053061"
CMAP_MID = "#F7F7F7"
CMAP_HIGH = "#67001F"
PLOT_CLIP = 2.0
FIGSIZE_SINGLE = (6.2, 5.4)
FIGSIZE_GRID = (7.5, 6.8)
