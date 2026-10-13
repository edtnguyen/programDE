Compact occupancy export created from the real processed CC-Perturb-seq H5AD.

State definition
----------------
States are the author-defined mapping of existing leiden_res_0.5 clusters. No new clustering or target-to-program effects were used.

Target carrier definition
-------------------------
A cell is a carrier for a biological target when it contains at least one guide whose metadata intended_target_name equals that target. Multiple guides for the same target collapse to one target-union indicator; n_guides_for_target is retained. Cells with multiple real targets appear once for each target.

Controls
--------
Pure control = at least one safe-targeting or non-targeting guide and zero targeting guides. Mixed control+target cells are retained as target carriers in the primary high-MOI marginal model and flagged explicitly.

Covariates exported
-------------------
pct_counts_mt, biological_sample, total_counts, guides_per_cell, guide_umi_counts, n_genes_by_counts. Count-like covariates should be log1p transformed in the occupancy model.
