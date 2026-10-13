Lineage split cell-level export outputs.

Interpretation caveat:
These are high-MOI target-carrier enrichment analyses, not single-perturbation lineage tracing.

Core outputs:
- target_celltype_one_vs_rest_enrichment.csv
- guide_celltype_one_vs_rest_enrichment.csv
- target_binary_state_enrichment.csv
- guide_binary_state_enrichment.csv
- target_axis_distribution_tests.csv
- guide_axis_effects.csv
- guide_concordance_summary.csv
- target_composition_distance_from_control.csv
- target_leadlag_metrics.csv
- pseudobulk_counts_by_sample_target_day_celltype.csv, if sample metadata exists
- neighborhood_enrichment_selected_targets.csv, if export_neighborhoods=True
- highlight_target_cell_score_sample_for_plotting.csv

Axis orientation:
- D1_P13_minus_P30: positive = P13 over P30
- D2_P17_minus_P15: positive = P17 over P15
- D2_YAPLat_minus_PreEC: positive = mean(P17,P34) over mean(P15,P23)
- D3_ALT_minus_EC: positive = mean(P27,P34,P35) over mean(P2,P3,P8,P11)
