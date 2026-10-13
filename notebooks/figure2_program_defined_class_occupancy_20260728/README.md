
## Important: cNMF usages are external to the base H5AD

The original `ipsc-ec.Hippo.figure.ipynb` loaded the integrated H5AD first and then added cNMF usages in memory from:

```text
/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50.usages.k_50.dt_0_2.consensus.txt
```

Those additions may not be persisted in the H5AD. This updated package therefore falls back to the external usage table, aligns by cell barcode, reports overlap, and excludes cells lacking usage from the program-defined state population. The notebook now prints the complete child-process traceback instead of only a generic `CalledProcessError`.

# Program-defined perturbation-class occupancy analysis

This package replaces Leiden-cluster-derived coarse states with cell states defined by the globally dominant cNMF program.

## Core rule

For every cell:

1. Read all 50 cell-level cNMF usages from `obs["Program 0"] ... obs["Program 49"]`; if those columns are absent, use `obsm["cnmf_usage"]`.
2. Identify the global highest-usage program across all 50 programs.
3. Do not force the cell into the curated list.
4. For the primary analysis, retain the cell only when the global dominant program is in the primary curated program table and its expected day matches the observed collection day.

This state assignment is categorical and fixed before target testing. The workflow does not use target-to-program CRT effects or target-to-program differential-effect estimates.

## Actual query object

The configuration is already filled with:

`/oak/stanford/groups/engreitz/Users/opushkar/cc_perturb/2.preprocessing_tonys_pipeline/3.integrated_days/3.all_days_concat_pc30_all_genes_sceptre_guides.h5ad`

## Primary program states

- D0: P7, P22, P29
- D1: P28, P13, P30, P48
- D2: P15, P14, P17
- D3: P33, P34, P2, P3, P8, P11

See `PROGRAM_SELECTION_NOTES.md` and `templates/curated_program_states.tsv`.

## Run in Jupyter

```bash
unzip figure2_program_defined_class_occupancy_sherlock_20260727.zip
cd figure2_program_defined_class_occupancy_sherlock_20260727
jupyter lab notebooks/01_program_defined_perturbation_class_occupancy.ipynb
```

## Run through Slurm

```bash
mkdir -p logs
sbatch scripts/run_full_program_defined_occupancy_pipeline.sbatch
```

## Main model

Within each observed day and for each target/program-state pair:

`program-state membership ~ target-union carrier + biological_sample + technical covariates`

The primary model is restricted to cells whose global dominant program is in the day-matched primary curated set. A sensitivity model compares the same curated program-state outcome against all cells from the observed day.

## Important estimand caveat

The primary analysis estimates redistribution among the curated program-defined state space. A perturbation that moves cells into a non-curated dominant program can reduce the number of cells entering the primary analysis. The package therefore exports the fraction of each day represented by primary curated program states and runs an all-day-cell sensitivity model.

## Target carrier definition

- At least one assigned guide mapped to `intended_target_name`.
- Multiple guides for the same target are collapsed into one target-union indicator.
- Cells carrying multiple targeting genes contribute to each corresponding target model.
- Cells carrying both control and targeting guides remain target carriers and are not pure controls.
- Pure controls are defined as at least one safe/non-targeting guide and zero targeting guides.

## Outputs

See `expected_outputs/README.md`.
