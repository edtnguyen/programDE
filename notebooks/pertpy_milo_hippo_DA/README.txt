Pertpy Milo Hippo target-carrier DA

Comparison:
  target condition = cells carrying one Hippo target
  control condition = non-targeting carrier cells not carrying that Hippo target

Run structure:
  one Hippo target x one day x one high-MOI mode at a time

Modes:
  all_carriers: target carriers vs NT carriers, excluding cells with target from control
  low_guide_load_leq2: same but n_total_guides <= 2
  single_selected_target: target cells have exactly one targeting gene; controls have NT and zero targeting genes

Milo output interpretation:
  positive logFC = target carriers enriched in that neighborhood vs NT controls
  negative logFC = target carriers depleted in that neighborhood vs NT controls
  SpatialFDR = Milo spatially adjusted neighborhood FDR

High-MOI caveat:
  These are target-carrier enrichment analyses, not pure single-perturbation lineage tracing.
