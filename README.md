# programDE

A pipeline for testing differential expression (DE) of gene targets on gene programs. This pipeline implements a SCEPTRE-like Conditional Randomization Test (CRT) to assess the statistical significance of perturbations.

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

The core functionality is the SCEPTRE-style union CRT. Here is a typical usage example:

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

pvals_df, betas_df, treated_df, results = run_all_genes_union_crt(
    inputs,
    B=1023,    # Number of permutations
    n_jobs=16, # Number of parallel jobs
)

# Store results back into the AnnData object
store_results_in_adata(adata, pvals_df, betas_df, treated_df)
```

### Makefile Commands

This project uses a `Makefile` to streamline common tasks.

- `make requirements`: Install Python dependencies.
- `make data`: Run the data processing pipeline.
- `make lint`: Lint the source code using `flake8`.
- `make clean`: Remove compiled Python files.

## Project Structure

```
├── Makefile           # Makefile with useful commands
├── README.md          # Project README
├── requirements.txt   # Python package requirements
├── setup.py           # Makes the project pip-installable
├── src                # Project source code
│   ├── data           # Scripts to download or generate data
│   ├── features       # Scripts to generate features
│   ├── models         # Scripts for model training and prediction
│   ├── sceptre        # Core CRT analysis pipeline
│   │   ├── __init__.py
│   │   ├── adata_utils.py
│   │   ├── crt.py
│   │   ├── pipeline.py
│   │   └── propensity.py
│   └── visualization  # Scripts for plotting and visualization
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