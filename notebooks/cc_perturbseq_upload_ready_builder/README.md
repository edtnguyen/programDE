# CC-Perturb-seq IGVF upload-ready file builder

This package creates the 16 new D0-D3 analysis files that remain in scope:

- 4 annotated h5ad MatrixFiles
- 4 PerturbNMF h5mu MatrixFiles
- 4 guide quantification TabularFiles
- 4 per-cell QC TabularFiles

It validates all files, computes MD5 checksums, and creates staged `iu_register.py` metadata TSVs. It does not contact or write to the IGVF Portal.

## Important day-metadata finding

The attached QQ notebook uses `DAYS = ["d0", "d1", "d2", "d3"]` and loads separate files named `{day}_normalized_gex_and_guide.h5mu`; it does not show a day column inside the combined PerturbNMF h5mu. The builder therefore resolves day from the integrated h5ad, prefers `obs["day"]`, aligns to the h5mu `cNMF` modality by exact cell IDs, checks any h5mu day annotation for agreement, and aborts on mismatch.

## Install an environment

On a Sherlock compute node:

```bash
mamba env create \
  --prefix /oak/stanford/groups/engreitz/Users/tri/envs/igvf-upload-ready \
  --file environment.yml

mamba activate /oak/stanford/groups/engreitz/Users/tri/envs/igvf-upload-ready

python -m ipykernel install --user \
  --name igvf-upload-ready \
  --display-name "IGVF upload-ready builder"
```

An existing environment is sufficient when it can import `anndata`, `mudata` or `muon`, `pandas`, `numpy`, `scipy`, and `h5py`.

## Interactive use

```bash
cd /path/to/cc_perturbseq_upload_ready_builder
jupyter lab --no-browser --ip=0.0.0.0
```

Open `notebooks/01_make_igvf_upload_ready_files.ipynb` and run all cells.

The notebook has `EXECUTE_WRITES = True` and `OVERWRITE_EXISTING = False`. Existing valid outputs are reused. Existing outputs that differ from the deterministic newly derived tables cause a stop rather than silent replacement.

## Batch use

Edit the environment path in `scripts/run_notebook.sbatch` only when needed, then:

```bash
sbatch scripts/run_notebook.sbatch
```

## Expected final marker

A successful run creates:

```text
/oak/stanford/groups/engreitz/Projects/IGVF/CC-Perturb-seq/300gene/analysis_submission/READY_TO_UPLOAD_16_FILES.txt
```

The marker is written only after all 16 files pass validation and the metadata invariants pass.

## Portal submission order

The notebook creates dry-run commands but does not execute them. Submit in this order:

1. four h5ad MatrixFiles;
2. four h5mu MatrixFiles;
3. eight guide/QC TabularFiles.

This order is required because the h5mu and tabular records refer to the h5ad aliases in `derived_from`.
