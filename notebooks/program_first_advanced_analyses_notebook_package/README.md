# Program-first advanced analyses notebook

This package contains an exploratory notebook for program-first temporal Perturb-seq analyses inspired by asynchronous-development papers, adapted to this EC differentiation system.

The notebook intentionally avoids using Leiden/UMAP clusters as the primary interpretive unit. It uses expert-curated cNMF programs as the analysis unit.

Required input at runtime:
- cNMF spreadsheet with target-program effects, usually at:
  /oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/per_day_umaps/data/cNMF_50_0_2_032526_6co.xlsx
- Sheet: "Perturbation Association 1"
- Optional Sheet: "Summary" for program top loading genes

Included:
- 04_program_first_advanced_analyses.ipynb
- curated_program_annotations.csv
- Programs_annotations.xlsx

Optional inputs for later sections:
- neighborhood enrichment CSV + UMAP coordinate CSV for vulnerability maps
- external gene-set/signature table for program regulon/signature overlay
- genome-wide screen summary table for cross-screen concordance

Outputs are written to outputs_program_first_advanced/.
