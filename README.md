# programDE

A pipeline for testing differential effect (DE) of gene targets on gene programs. This pipeline implements a SCEPTRE-like Conditional Randomization Test (CRT) to assess the statistical significance of perturbations.

The core analysis pipeline lives in the `src.sceptre` module.

## Getting Started

### Prerequisites

- Python 3
- `conda` or `virtualenv`

### Installation

1.  **Create a virtual environment:**

    It is recommended to create a dedicated environment. You can use the built-in Makefile command:
    ```bash
    make create_environment
    ```
    This will create a `conda` or `virtualenv` environment named `programDE`. Activate it before proceeding.

2.  **Install dependencies:**

    Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the project source code:**

    To make the `src` directory importable as a package, install it in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

### Data Processing

To process the raw data and generate the final datasets for analysis, run the following command:
```bash
make data
```
This command executes the script `src/data/make_dataset.py`.

### Library Usage

#### Required AnnData inputs

`prepare_crt_inputs` expects the following to be present in `adata`. It searches
common AnnData containers (`.obsm`, `.layers`, `.obsp`, `.uns`, `.obs`) by key.

- **Covariates**: `adata.obsm["covar"]` (shape `N x p`, numeric, catergorical, or object). A `DataFrame`or `ndarray` is fine. Count-based covariates should be already log-transformed. Columns are z-scored and an intercept is added by the pipeline. If `covar` is a `DataFrame` with categorical/object/bool columns, they are one-hot encoded automatically (dropping one level by default to avoid collinearity). Numeric columns with a small number of unique values (<=20 by default) are also treated as categorical and one-hot encoded. If `covar` is a non-numeric array (object/string), it is auto-encoded as well. You can override the numeric threshold via `prepare_crt_inputs(..., numeric_as_category_threshold=...)` or set `numeric_as_category_threshold=None` to disable this heuristics.
- **cNMF usage**: `adata.obsm["cnmf_usage"]` (shape `N x K`, numeric).
  Each row should sum to 1 (the CLR step will floor and renormalize).
- **Guide assignment**: `adata.obsm["guide_assignment"]` (shape `N x G`).
  Can be dense or sparse; nonzero means guide present.
- **Guide → gene mapping**: `adata.uns["guide2gene"]` (dict-like).
  Keys are guide IDs, values are gene names.
- **Guide names** (if guide_assignment is not a DataFrame with columns):
  `adata.uns["guide_names"]` list of guide IDs in column order.
- **Program names** (optional): `adata.uns["program_names"]` list of length `K`.

All matrices must have the same number of rows (`N = number of cells`).


The core functionality is the SCEPTRE-style union CRT. The recommended starting point is skew-normal calibrated p-values:

```python
from src.sceptre import (
    prepare_crt_inputs,
    run_all_genes_union_crt,
    store_results_in_adata,
    limit_threading,
)

# It is recommended to limit BLAS threads for reproducibility
limit_threading()

inputs = prepare_crt_inputs(
    adata,
    usage_key="cnmf_usage",
    covar_key="covar",
    guide_assignment_key="guide_assignment",
    guide2gene_key="guide2gene",
)

out = run_all_genes_union_crt(
    inputs,
    B=1023,      # Number of permutations
    n_jobs=16,   # Number of parallel jobs
    calibrate_skew_normal=True,
    return_skew_normal=True,
)

# Store results back into the AnnData object
store_results_in_adata(
    adata,
    out["pvals_df"],
    out["betas_df"],
    out["treated_df"],
)
```

Here is a minimal usage example without skew-normal calibration:

```python
from src.sceptre import (
    prepare_crt_inputs,
    run_all_genes_union_crt,
    store_results_in_adata,
    limit_threading,
)
import anndata

# It is recommended to limit BLAS threads for reproducibility
limit_threading()

# Assuming 'adata' is an AnnData object containing your single-cell data
# adata = anndata.read_h5ad(...) 

inputs = prepare_crt_inputs(
    adata,
    usage_key="cnmf_usage",
    covar_key="covar",
    guide_assignment_key="guide_assignment",
    guide2gene_key="guide2gene",
)

out = run_all_genes_union_crt(
    inputs,
    B=1023,    # Number of permutations
    n_jobs=16, # Number of parallel jobs
)

# By default, run_all_genes_union_crt returns a dict. For legacy tuple output:
# out = run_all_genes_union_crt(..., return_format="tuple")

# Store results back into the AnnData object
store_results_in_adata(
    adata,
    out["pvals_df"],
    out["betas_df"],
    out["treated_df"],
)
```

#### QQ plot for negative controls

Use this to sanity-check calibration of p-values for negative-control genes.
If you have both raw and skew-calibrated p-values, pass both to compare them.

```python
from src.visualization import qq_plot_ntc_pvals

# Single run that returns both raw and skew-calibrated p-values
out = run_all_genes_union_crt(
    inputs,
    B=1023,
    n_jobs=16,
    calibrate_skew_normal=True,
    return_raw_pvals=True,
    return_skew_normal=True,
)

ax = qq_plot_ntc_pvals(
    out["pvals_raw_df"],
    guide2gene=adata.uns["guide2gene"],
    ntc_genes=["non-targeting", "safe-targeting"],
    pvals_skew_df=out["pvals_df"],
    title="QQ plot: NTC genes (raw vs skew) vs null",
    show_ref_line=True,
    show_conf_band=True,
)

# Display the plot (e.g., in scripts)
# In notebooks, returning `ax` is usually enough; in scripts, call plt.show().
import matplotlib.pyplot as plt
plt.show()
```

#### Skew-normal calibration note

The skew-normal fitting entry point is `fit_skew_normal` (numba-backed).

Manual fitting for a single program:

```python
import numpy as np
from src.sceptre import fit_skew_normal, crt_betas_for_gene

# beta_null is the CRT null distribution for a single gene-program (length B)
# Example: beta_null = beta_null_mat[:, program_index]
# If you already ran the pipeline, you can reuse stored results; otherwise,
# generate beta_null directly:
# beta_obs, beta_null_mat = crt_betas_for_gene(indptr, idx, C, Y, A, CTY, obs_idx, B)
mu = beta_null.mean()
sd = beta_null.std()
z_nulls = (beta_null - mu) / sd

params = fit_skew_normal(z_nulls)  # [xi, omega, alpha, mean, sd]
```

Where do `z_nulls` come from?

`z_nulls` are standardized CRT null statistics. You first generate the null
beta coefficients from CRT resamples, then standardize them by their mean and
standard deviation. The pipeline does this internally when
`calibrate_skew_normal=True`.

### Makefile Commands

This project uses a `Makefile` to streamline common tasks.

- `make requirements`: Install Python dependencies.
- `make data`: Run the data processing pipeline.
- `make lint`: Lint the source code using `flake8`.
- `make clean`: Remove compiled Python files.

### Testing

Run the mock in-memory smoke test:

```bash
python3 scripts/test_mock_crt.py
```

## Project Structure

```
├── Makefile           # Makefile with useful commands
├── README.md          # Project README
├── requirements.txt   # Python package requirements
├── setup.py           # Makes the project pip-installable
├── scripts            # Utility scripts
│   └── test_mock_crt.py
├── src                # Project source code
│   ├── data           # Scripts to download or generate data
│   ├── features       # Scripts to generate features
│   ├── models         # Scripts for model training and prediction
│   ├── sceptre        # Core CRT analysis pipeline
│   │   ├── __init__.py
│   │   ├── adata_utils.py
│   │   ├── crt.py
│   │   ├── pipeline.py
│   │   ├── pipeline_helpers.py
│   │   ├── propensity.py
│   │   ├── shared_low_level_functions.cpp
│   │   └── skew_normal.py
│   └── visualization  # Scripts for plotting and visualization
│   │   ├── __init__.py
│   │   └── qq_plot.py
├── data
│   ├── external       # Data from third-party sources
│   ├── interim        # Intermediate transformed data
│   ├── processed      # Final, canonical datasets
│   └── raw            # Original, immutable data
├── docs               # Project documentation
├── models             # Trained models
├── notebooks          # Jupyter notebooks for exploration
└── reports            # Generated analysis reports and figures
```
---
