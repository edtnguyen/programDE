# programDE

A pipeline for testing differential effect (DE) of gene targets on gene programs. This pipeline implements a Conditional Randomization Test (CRT) to assess the statistical significance of perturbations.

The core analysis pipeline lives in the `src.sceptre` module.

---

## Method overview

This pipeline tests **target gene → program usage** effects using a Conditional Randomization Test (CRT) with optional skew-normal calibration.

### Notation

* Cells: $i = 1,\dots,N$
* Programs: $k = 1,\dots,K$ (here $K \approx 70$)
* Covariates: $C \in \mathbb{R}^{N \times p}$ (includes an intercept column)
* cNMF usage (composition): $u_i \in \Delta^{K-1}$ with $\sum_{k=1}^K u_{ik}=1$ and $u_{ik}\ge 0$

## 1) Linear model for gene effect on program usage

### 1.1 Gene-level perturbation indicator (union)

For each target gene $g$, define a **gene-level union indicator**:

```math
x_i \in \{0,1\},\qquad
x_i = \mathbf{1}\{ \exists\ \mathrm{guide\ targeting}\ g\ \mathrm{in\ cell}\ i \}.
```

### 1.2 CLR transform of program usage

Because program usages are compositional (each row sums to 1), we apply a **centered log-ratio (CLR)** transform. After flooring and renormalization (to avoid $\log 0$), define:

```math
u'_{ik}=\frac{\max(u_{ik},\varepsilon)}{\sum_{j=1}^K \max(u_{ij},\varepsilon)}.
```

Then the CLR-transformed outcome is:

```math
Y_{ik}=\mathrm{CLR}(u'_i)_k
= \log u'_{ik} - \frac{1}{K}\sum_{j=1}^K \log u'_{ij}.
```

Collect outcomes in $Y \in \mathbb{R}^{N \times K}$.

### 1.3 Per-program linear regression

For each program $k$, fit:

```math
Y_{ik} = \beta_k\, x_i + C_i^\top \gamma_k + \varepsilon_{ik}.
```

where $\beta_k$ is the gene effect on program $k$ (on the CLR scale), and $\gamma_k$ are covariate effects.

**Test statistic:** the OLS coefficient $\hat\beta_k$.

## 2) Conditional Randomization Test (CRT)

The CRT tests $H_0: Y \perp x \mid C$ by resampling $x$ from its conditional distribution given covariates.

### 2.1 Propensity model

Fit a regularized logistic regression for the union indicator:

```math
p_i = \mathbb{P}(x_i = 1 \mid C_i)
= \sigma\!\left(C_i^\top \theta\right),
\qquad
\sigma(t)=\frac{1}{1+e^{-t}}.
```

### 2.2 Null resampling of $x$

Generate $B$ synthetic perturbation vectors:

```math
\tilde x_i^{(b)} \sim \mathrm{Bernoulli}(p_i),
\qquad b=1,\dots,B,\ i=1,\dots,N.
```

**Efficient Bernoulli resampling (index sampler).** Instead of drawing $B$ Bernoullis for every cell, we use an equivalent two-stage procedure per cell $i$:

1. Draw how many resamples include cell $i$ as treated:

```math
M_i \sim \mathrm{Binomial}(B, p_i).
```

2. Sample $M_i$ distinct resample indices uniformly without replacement:

```math
S_i \subset \{1,\dots,B\},\quad |S_i|=M_i,\quad S_i\ \mathrm{uniform}.
```

3. Set $\tilde x_i^{(b)}=1$ for $b\in S_i$ and $\tilde x_i^{(b)}=0$ otherwise.

This yields exactly the same distribution as i.i.d. Bernoulli draws across $b$, but is faster when $p_i$ is small (sparse perturbations).

### 2.3 Recompute the test statistic under the null

For each resample $b$, compute OLS coefficients $\hat\beta_k^{(b)}$ for all programs using $\tilde x^{(b)}$:

```math
Y_{ik} = \beta_k^{(b)}\, \tilde x_i^{(b)} + C_i^\top \gamma_k^{(b)} + \varepsilon_{ik}^{(b)}.
```

In the implementation, $\hat\beta_k^{(b)}$ is computed using precomputed OLS summary quantities (no repeated least-squares solves).

### 2.4 Empirical CRT p-values

Let $\hat\beta_k^{(\mathrm{obs})}$ be from observed $x$, and $\hat\beta_k^{(b)}$ from resample $b$. The two-sided CRT p-value is:

```math
p_k
=
\frac{
1 + \sum_{b=1}^B \mathbf{1}\!\left(\left|\hat\beta_k^{(b)}\right| \ge \left|\hat\beta_k^{(\mathrm{obs})}\right|\right)
}{
B+1
}.
```

## 3) Skew-normal calibration (optional)

CRT nulls can be skewed (sparse perturbations, heterogeneous covariates). We optionally smooth tails by fitting a skew-normal distribution to null z-scores.

### 3.1 Null z-scores

Compute mean and standard deviation from the null draws:

```math
\mu_k = \frac{1}{B}\sum_{b=1}^B \hat\beta_k^{(b)},
\qquad
\sigma_k = \sqrt{\frac{1}{B}\sum_{b=1}^B\left(\hat\beta_k^{(b)}-\mu_k\right)^2}.
```

Define null z-scores and the observed z-score:

```math
z_k^{(b)}=\frac{\hat\beta_k^{(b)}-\mu_k}{\sigma_k},
\qquad
z_k^{(\mathrm{obs})}=\frac{\hat\beta_k^{(\mathrm{obs})}-\mu_k}{\sigma_k}.
```

### 3.2 Skew-normal fit

Fit $\mathrm{SN}(\xi,\omega,\alpha)$ to ${z_k^{(b)}}_{b=1}^B$ with density:

```math
f(z;\xi,\omega,\alpha)
=
\frac{2}{\omega}\,
\phi\!\left(\frac{z-\xi}{\omega}\right)\,
\Phi\!\left(\alpha\,\frac{z-\xi}{\omega}\right).
```

where $\phi$ and $\Phi$ are the standard normal PDF and CDF. Let $F(\cdot)$ be the fitted CDF. Calibrated p-values:

* Two-sided:

```math
p = 2 \min\{F(z^{(\mathrm{obs})}),\, 1 - F(z^{(\mathrm{obs})})\}.
```

* Right-tailed:

```math
p = 1 - F(z^{(\mathrm{obs})}).
```

* Left-tailed:

```math
p = F(z^{(\mathrm{obs})}).
```

**Note**: The skew‑normal calibration is two‑sided by default (`skew_normal_side_code=0`). The p‑value is computed as 2 × one tail, where the tail is chosen using the empirical median of the null z‑scores: use the right tail if ($z^{(\mathrm{obs})}$) is above the null median, otherwise the left tail. This is equivalent to $2 \min \\{F(z^{(\mathrm{obs})})\, 1 - F(z^{(\mathrm{obs})}) \\}$ when the fitted skew‑normal median matches the empirical median.



### 3.3 Fallback

If the skew-normal fit is unstable or fails diagnostics, fall back to the empirical CRT p-values.

## Interpretation (CLR scale)

Because $Y_{ik}=\mathrm{CLR}(u'_i)_k$, $\beta_k$ is a **log-ratio shift** of program $k$ relative to the geometric mean across programs:

```math
\beta_k \approx \Delta \log\!\left(\frac{u'_k}{g(u')}\right),
\qquad
g(u')=\left(\prod_{j=1}^K u'_j\right)^{1/K}.
```

If $\beta_k>0$, program $k$ increases **relative to the overall composition**; if $\beta_k<0$, it decreases. Effects are **relative**: increases in some components imply decreases elsewhere.

## Flooring $\varepsilon$ (handling near-zeros)

cNMF usages can be extremely close to zero, making $\log(u_{ik})$ unstable. We apply flooring and renormalization:

```math
u'_{ik}=\frac{\max(u_{ik},\varepsilon)}{\sum_{j=1}^K \max(u_{ij},\varepsilon)}.
```

Practical choices for $\varepsilon$:

* Quantile-based: $\varepsilon=\mathrm{quantile}({u_{ik}}, q)$ with small $q$ (e.g., $10^{-4}$ to $10^{-6}$)
* Fixed: $\varepsilon = 10^{-6}$

Smaller $\varepsilon$ preserves dynamic range but can amplify noise in very small components; larger $\varepsilon$ stabilizes logs but shrinks contrasts involving tiny programs.

---


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

### Library Usage

#### Required AnnData inputs

`prepare_crt_inputs` expects the following to be present in `adata`. It searches
common AnnData containers (`.obsm`, `.layers`, `.obsp`, `.uns`, `.obs`) by key.

- **Covariates**: `adata.obsm["covar"]` (shape `N x p`, numeric, catergorical, or object). A `DataFrame`or `ndarray` is fine. Count-based covariates should be already log-transformed. Columns are z-scored and an intercept is added by the pipeline. If `covar` is a `DataFrame` with categorical/string/bool/mixed columns, they are one-hot encoded automatically (dropping one level by default to avoid collinearity). Numeric columns with a small number of unique values (<=20 by default) are also treated as categorical and one-hot encoded. You can override the numeric threshold via `prepare_crt_inputs(..., numeric_as_category_threshold=...)` or set `numeric_as_category_threshold=None` to disable this heuristics. If you want stratified-permutation CRT with batch stratification, include a `batch` column (or choose another `batch_key`) in this DataFrame.
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
    adata=adata,
    usage_key="cnmf_usage",
    covar_key="covar",
    guide_assignment_key="guide_assignment",
    guide2gene_key="guide2gene",
)

out = run_all_genes_union_crt(
    inputs=inputs,
    B=1023,      # Number of permutations
    n_jobs=16,   # Number of parallel jobs
    calibrate_skew_normal=True,
    return_skew_normal=True,
)

# Store results back into the AnnData object
store_results_in_adata(
    adata=adata,
    pvals_df=out["pvals_df"],
    betas_df=out["betas_df"],
    treated_df=out["treated_df"],
)
```

Optional: switch to the stratified-permutation CRT sampler (S-CRT):

```python
out = run_all_genes_union_crt(
    inputs=inputs,
    B=1023,
    n_jobs=16,
    resampling_method="stratified_perm",
    resampling_kwargs=dict(
        n_bins=20,
        stratify_by_batch=True,
        batch_key="batch",
        min_stratum_size=2,
    ),
    calibrate_skew_normal=False,
    return_raw_pvals=True,
)
```

Rank-based U-test statistic (S-CRT U-test):

```python
out = run_all_genes_union_crt(
    inputs=inputs,
    B=1023,
    n_jobs=16,
    resampling_method="stratified_perm",
    resampling_kwargs=dict(
        n_bins=20,
        stratify_by_batch=True,
        batch_key="batch",
        min_stratum_size=2,
    ),
    test_stat="utest",
    test_stat_kwargs=dict(use="clr", rank_method="average"),
    calibrate_skew_normal=False,
)

# Rank-biserial effect sizes live in out["betas_df"];
# out["stats_df"] is provided as an explicit alias when test_stat="utest".
# To rank raw usage instead of CLR, use test_stat_kwargs={"use": "usage"}.
# This uses the same eps_quantile flooring/renormalization from prepare_crt_inputs.
```

Batch stratification uses the raw covariate DataFrame (before one-hot encoding). If your covariates were provided as an `ndarray`, `stratify_by_batch` is ignored.

#### NTC empirical null (CLR-OLS)

This mode uses NTC pseudo-genes as the null distribution and matches on:
- `n1`: treated count (union)
- `d`: OLS denominator `x^T M_C x` (computed from the same CLR-OLS statistic)

Workflow (don’t skip steps):

1. Ensure `adata.obsm["covar"]` is a **DataFrame** with a `batch` column.
2. Call `prepare_crt_inputs(...)` (captures raw covariates for batch meta‑analysis).
3. Choose NTC labels that exist in `adata.uns["guide2gene"]` values.
4. Run `run_all_genes_union_crt(..., null_method="ntc_empirical")`.
5. Use `out["pvals_df"]` as the empirical‑null p-values; inspect `out["ntc_matching_info"]` for bin sizes/fallbacks.

End‑to‑end example:

```python
from src.sceptre import prepare_crt_inputs, run_all_genes_union_crt

# 1) Ensure covariates are a DataFrame with batch
covar_df = adata.obsm["covar"].copy()
if "batch" not in covar_df.columns:
    raise ValueError("covar_df must contain a 'batch' column for ntc_empirical.")
adata.obsm["covar"] = covar_df

# 2) Prepare inputs
inputs = prepare_crt_inputs(adata=adata, usage_key="cnmf_usage", covar_key="covar")

# 3) Run NTC empirical‑null CRT
out = run_all_genes_union_crt(
    inputs=inputs,
    null_method="ntc_empirical",
    null_kwargs=dict(
        ntc_labels=["SAFE", "non-targeting", "NTC"],
        guides_per_unit=6,
        n_ntc_units=5000,
        batch_mode="meta",          # per-batch pvals + Fisher combine
        combine_method="fisher",
        matching=dict(
            n_n1_bins=10,
            n_d_bins=10,
            min_ntc_per_bin=50,
            use_pbar=True,           # optional: add pbar matching
            n_pbar_bins=8,
        ),
        min_treated=10,
        min_control=10,
    ),
)

pvals_ntc = out["pvals_df"]
betas = out["betas_df"]
matching_info = out.get("ntc_matching_info", {})
```

#### NTC empirical-null QQ plots (cross-fit)

These QQ diagnostics **cross-fit** the NTC null: split NTC pseudo-genes into A/B,
use **A** to define the null, then evaluate calibration using **B vs A**. This avoids
the overly-optimistic “in-sample” QQ that happens if you compare NTC_A to itself.

Recommended usage (gene-level QQ is the primary diagnostic):

```python
from src.sceptre import make_ntc_empirical_qq_plots, run_all_genes_union_crt

out = run_all_genes_union_crt(
    inputs=inputs,
    null_method="ntc_empirical",
    qq_crossfit=True,
    null_kwargs=dict(
        ntc_labels=["SAFE", "non-targeting", "NTC"],
        guides_per_unit=6,
        n_ntc_units=5000,
        batch_mode="meta",
        combine_method="fisher",
        matching=dict(n_n1_bins=10, n_d_bins=10, min_ntc_per_bin=50),
        min_treated=10,
        min_control=10,
        qq_crossfit_seed=11,
        qq_crossfit_frac_A=0.5,
    ),
)

make_ntc_empirical_qq_plots(
    out=out,
    out_dir="results/qq_ntc",
    programs_to_plot="top_var",
    n_programs=6,
    make_per_batch=True,
    make_meta=True,
    envelope_boot=200,
    seed=0,
)
```

Step-by-step workflow (including what to do after plotting):

1. **Prepare inputs**: `adata.obsm["covar"]` must be a DataFrame with a `batch` column, then run `prepare_crt_inputs(...)`.
2. **Run NTC empirical-null with cross-fit**: set `null_method="ntc_empirical"` and `qq_crossfit=True` so `out["ntc_crossfit"]` is produced.
3. **Generate QQ plots**: call `make_ntc_empirical_qq_plots(...)` as shown above.
4. **Inspect outputs on disk** (the function saves PNGs; it does not display them):

```python
from pathlib import Path
print(sorted(Path("results/qq_ntc").glob("**/*.png")))
```

Expected files:
- Meta (combined across batches):
  - `results/qq_ntc/qq_ntc_empirical_genelevel_meta.png`
  - `results/qq_ntc/qq_ntc_empirical_program_<program>_meta.png`
- Per-batch (if `make_per_batch=True`):
  - `results/qq_ntc/per_batch/<batch>/qq_ntc_empirical_genelevel.png`
  - `results/qq_ntc/per_batch/<batch>/qq_ntc_empirical_program_<program>.png`

If no images appear, verify:
- `qq_crossfit=True` and `null_method="ntc_empirical"`
- `out["ntc_crossfit"]` is present (it’s required by the plotting function)
- `programs_to_plot` matches program names in `out["betas_df"].columns`

Notes:
- The holdout calibration curve is **NTC_B vs NTC_A** (with a bootstrap band).
- The “Real genes vs A” curve should be interpreted relative to that holdout band.
- Program-level QQ uses a subset of programs (default: top variance in `betas_df`).
- Enabling `use_pbar` adds a propensity-fit per unit, so it increases runtime.

#### S-CRT workflow (stratified-permutation)

Use this checklist to avoid missing any steps:

1. Ensure `adata.obsm["covar"]` is a **DataFrame** with a `batch` column (or your chosen `batch_key`).
2. (Optional) Add a **burden** column to `covar` (e.g., log1p non-NTC guides per cell).
3. Call `prepare_crt_inputs(...)` (captures raw covariates for stratification).
4. Define `resampling_kwargs` and reuse them everywhere (main CRT, NTC groups, CRT-null pvals).
5. Run `run_all_genes_union_crt(..., resampling_method="stratified_perm")`.
6. Build NTC guide groups and compute grouped NTC p-values with the same sampler options.
7. Compute CRT-null p-values for the **same NTC units** (grouped guides).
8. Plot the QQ curve using the NTC p-values + CRT-null p-values.

End-to-end example:

```python
from src.sceptre import (
    add_burden_covariate,
    build_ntc_group_inputs,
    compute_ntc_group_null_pvals_parallel,
    crt_pvals_for_ntc_groups_ensemble,
    crt_pvals_for_ntc_groups_ensemble_skew,
    make_ntc_groups_ensemble,
    prepare_crt_inputs,
    run_all_genes_union_crt,
)
from src.visualization import qq_plot_ntc_pvals

# 1) Make covariates a DataFrame with batch
covar_df = adata.obsm["covar"].copy()
if "batch" not in covar_df.columns:
    raise ValueError("covar_df must contain a 'batch' column for stratification.")
adata.obsm["covar"] = covar_df

# 2) Optional burden column (exclude NTC guides if desired)
add_burden_covariate(
    adata=adata,
    guide_assignment_key="guide_assignment",
    covar_key="covar",
    guide_names_key="guide_names",
    guide2gene_key="guide2gene",
    burden_key="log1p_non_ntc_guides_per_cell",
    ntc_labels=["non-targeting", "safe-targeting", "NTC"],
    include_ntc=False,
    use_log1p=True,
)

# 3) Prepare CRT inputs
inputs = prepare_crt_inputs(adata=adata, covar_key="covar", batch_key="batch")

# 4) Shared S-CRT settings
resampling_kwargs = dict(
    n_bins=20,
    stratify_by_batch=True,
    batch_key="batch",
    min_stratum_size=2,
    # Optional burden stratification:
    # burden_key="log1p_non_ntc_guides_per_cell",
    # n_burden_bins=8,
    # burden_bin_method="quantile",
)

# 5) Run gene-level CRT with stratified permutation
out = run_all_genes_union_crt(
    inputs=inputs,
    B=1023,
    n_jobs=16,
    resampling_method="stratified_perm",
    resampling_kwargs=resampling_kwargs,
    calibrate_skew_normal=True,
    return_raw_pvals=True,
    return_skew_normal=True,
)

# 6) Build NTC guide groups (6-guide units) + compute NTC p-values
ntc_labels = ["non-targeting", "safe-targeting", "NTC"]
ntc_guides, guide_freq, guide_to_bin, real_sigs = build_ntc_group_inputs(
    inputs=inputs,
    ntc_label=ntc_labels,
    group_size=6,
    n_bins=10,
)
ntc_groups_ens = make_ntc_groups_ensemble(
    ntc_guides=ntc_guides,
    ntc_freq=guide_freq,
    real_gene_bin_sigs=real_sigs,
    guide_to_bin=guide_to_bin,
    n_ensemble=10,
    seed0=7,
    group_size=6,
    max_groups=None,
)
ntc_group_pvals_ens = crt_pvals_for_ntc_groups_ensemble(
    inputs=inputs,
    ntc_groups_ens=ntc_groups_ens,
    B=1023,
    seed0=23,
    resampling_method="stratified_perm",
    resampling_kwargs=resampling_kwargs,
    # If you ran the main CRT with test_stat="utest", pass the same here.
    # test_stat="utest",
    # test_stat_kwargs={"use": "clr", "rank_method": "average"},
)
ntc_group_pvals_skew_ens = crt_pvals_for_ntc_groups_ensemble_skew(
    inputs=inputs,
    ntc_groups_ens=ntc_groups_ens,
    B=1023,
    seed0=23,
    resampling_method="stratified_perm",
    resampling_kwargs=resampling_kwargs,
)

# 7) CRT-null p-values matched to the same NTC units
null_pvals = compute_ntc_group_null_pvals_parallel(
    inputs=inputs,
    ntc_groups_ens=ntc_groups_ens,
    B=1023,
    n_jobs=8,
    backend="threading",
    resampling_method="stratified_perm",
    resampling_kwargs=resampling_kwargs,
    # Keep test_stat/test_stat_kwargs in sync with your main run if needed.
    # test_stat="utest",
    # test_stat_kwargs={"use": "clr", "rank_method": "average"},
)

# 8) QQ plot
ax = qq_plot_ntc_pvals(
    pvals_raw_df=out["pvals_raw_df"],
    guide2gene=adata.uns["guide2gene"],
    ntc_genes=ntc_labels,
    pvals_skew_df=out["pvals_df"],
    null_pvals=null_pvals,
    ntc_group_pvals_ens=ntc_group_pvals_ens,
    ntc_group_pvals_skew_ens=ntc_group_pvals_skew_ens,
    show_ntc_ensemble_band=True,
    show_all_pvals=True,
    title="QQ plot: S-CRT (grouped NTC controls)",
)
```

#### Output meanings (p-values)

`run_all_genes_union_crt` always returns `pvals_df` as the main p-value output.
What it contains depends on `calibrate_skew_normal`:

- `calibrate_skew_normal=False`: `pvals_df` = raw CRT p-values.
- `calibrate_skew_normal=True`: `pvals_df` = skew-normal calibrated p-values.

Optional outputs (only when requested):

- `pvals_raw_df` (if `return_raw_pvals=True`): raw CRT p-values.
- `pvals_skew_df` (if `return_skew_normal=True`): skew-normal p-values.
  When `calibrate_skew_normal=True`, this matches `pvals_df`.


#### QQ plot for negative controls

Use this to sanity-check calibration of p-values for negative-control genes. The dashed null curve is built from CRT-null p-values (leave-one-out), so you must provide `null_pvals` directly or pass `null_stats` and let the helper compute them.

Required inputs:
- `pvals_raw_df`: observed raw CRT p-values (genes × programs).
- `guide2gene`: guide → gene mapping.
- `ntc_genes`: negative-control gene labels (must exist in `guide2gene` values).
- `null_pvals` **or** `null_stats`.

Optional inputs:
- `pvals_skew_df`: skew-calibrated p-values (for comparison).
- `ntc_group_pvals_ens`: grouped NTC control p-values to match real genes which has several guides/gene (recommended; see below).
- `ntc_group_pvals_skew_ens`: grouped skew p-values (required when comparing skew).
- `show_null_skew=True`: add a second null curve drawn from a skew-normal fit to `null_stats`.
- `show_all_pvals=True`: scatter all observed p-values from `pvals_raw_df`.

Terminology:
- `null_stats`: raw CRT null test statistics (e.g., `beta_null` from resamples).
- `null_pvals`: leave-one-out CRT-null p-values computed from `null_stats`.

**Curve notes:**

Important: the null curve should be built from the same unit as the observed NTC curve. If you plot grouped NTC controls (6‑guide units), compute CRT‑null p-values for those same guide groups (not the whole NTC union into one giant gene). When you use grouped NTC controls, concatenate p-values from all groups across all ensemble partitions for both raw and skew curves. One ensemble replicate = one random partition of NTC guides into groups while respecting the guide frequency signature of a real gene. 

If you want all groups generated within each ensemble replicate (no truncation), use `max_groups=None` when building the ensemble and pass the full grouped outputs into `qq_plot_ntc_pvals`.

- NTC raw curve (grouped): QQ of **all grouped raw p-values pooled across all ensembles**.
- NTC skew curve (grouped): QQ of **all grouped skew p-values pooled across all ensembles**.
- NTC ensemble IQR band: variability across ensembles (median + IQR of per-ensemble QQ quantiles), not the curve itself.
- 95% CI band: theoretical uniform QQ band using `m_null` (the number of null p-values), not fit to NTC or skew curves.
- Each curve uses its own `m` (number of p-values) when forming expected quantiles.
- By default, NTC curves include **all programs**; slice DataFrames first if you want a subset.


Example (raw vs skew, grouped NTC controls, CRT-null curve):

```python
from src.sceptre import (
    build_ntc_group_inputs,
    compute_ntc_group_null_pvals_parallel,
    crt_pvals_for_ntc_groups_ensemble,
    crt_pvals_for_ntc_groups_ensemble_skew,
    make_ntc_groups_ensemble,
    run_all_genes_union_crt,
)
from src.visualization import qq_plot_ntc_pvals

out = run_all_genes_union_crt(
    inputs=inputs,
    B=1023,
    n_jobs=16,
    calibrate_skew_normal=True,
    return_raw_pvals=True,
    return_skew_normal=True,
)

ntc_labels = ["non-targeting", "safe-targeting"]
# Identify NTC guides and build guide-frequency bins / real-gene signatures
ntc_guides, guide_freq, guide_to_bin, real_sigs = build_ntc_group_inputs(
    inputs=inputs,
    ntc_label=ntc_labels,
    group_size=6,
    n_bins=10,
)
# Create multiple random partitions (ensembles) of NTC guides into 6-guide groups
ntc_groups_ens = make_ntc_groups_ensemble(
    ntc_guides=ntc_guides,
    ntc_freq=guide_freq,
    real_gene_bin_sigs=real_sigs,
    guide_to_bin=guide_to_bin,
    n_ensemble=10,
    seed0=7,
    group_size=6,
    max_groups=None,
)
# Compute raw CRT p-values for each NTC group in each ensemble
ntc_group_pvals_ens = crt_pvals_for_ntc_groups_ensemble(
    inputs=inputs,
    ntc_groups_ens=ntc_groups_ens,
    B=1023,
    seed0=23,
)
# Compute skew-calibrated CRT p-values for each NTC group in each ensemble
ntc_group_pvals_skew_ens = crt_pvals_for_ntc_groups_ensemble_skew(
    inputs=inputs,
    ntc_groups_ens=ntc_groups_ens,
    B=1023,
    seed0=23,
)

# Build CRT-null p-values matched to NTC group units (recommended)
null_pvals = compute_ntc_group_null_pvals_parallel(
    inputs=inputs,
    ntc_groups_ens=ntc_groups_ens,
    B=1023,
    n_jobs=8,
    backend="threading",
)

ax = qq_plot_ntc_pvals(
    pvals_raw_df=out["pvals_raw_df"],
    guide2gene=adata.uns["guide2gene"],
    ntc_genes=ntc_labels,
    pvals_skew_df=out["pvals_df"],
    null_pvals=null_pvals,
    ntc_group_pvals_ens=ntc_group_pvals_ens,
    ntc_group_pvals_skew_ens=ntc_group_pvals_skew_ens,
    show_ntc_ensemble_band=True,
    show_all_pvals=True,
    title="QQ plot: grouped NTC controls (raw vs skew) vs CRT null",
)

import matplotlib.pyplot as plt
plt.show()
```

Stratified‑permutation variant (S‑CRT) for the QQ plot:

```python
resampling_kwargs = dict(
    n_bins=20,
    stratify_by_batch=True,
    batch_key="batch",
    min_stratum_size=2,
    burden_key="log1p_non_ntc_guides_per_cell",
    n_burden_bins=8,
    burden_bin_method="quantile",
    burden_clip_quantiles=(0.0, 1.0),
)

out = run_all_genes_union_crt(
    inputs=inputs,
    B=1023,
    n_jobs=16,
    resampling_method="stratified_perm",
    resampling_kwargs=resampling_kwargs,
    calibrate_skew_normal=True,
    return_raw_pvals=True,
    return_skew_normal=True,
)

ntc_group_pvals_ens = crt_pvals_for_ntc_groups_ensemble(
    inputs=inputs,
    ntc_groups_ens=ntc_groups_ens,
    B=1023,
    seed0=23,
    resampling_method="stratified_perm",
    resampling_kwargs=resampling_kwargs,
    # If you ran the main CRT with test_stat="utest", pass the same here.
    # test_stat="utest",
    # test_stat_kwargs={"use": "clr", "rank_method": "average"},
)
ntc_group_pvals_skew_ens = crt_pvals_for_ntc_groups_ensemble_skew(
    inputs=inputs,
    ntc_groups_ens=ntc_groups_ens,
    B=1023,
    seed0=23,
    resampling_method="stratified_perm",
    resampling_kwargs=resampling_kwargs,
)
null_pvals = compute_ntc_group_null_pvals_parallel(
    inputs=inputs,
    ntc_groups_ens=ntc_groups_ens,
    B=1023,
    n_jobs=8,
    backend="threading",
    resampling_method="stratified_perm",
    resampling_kwargs=resampling_kwargs,
    # Keep test_stat/test_stat_kwargs in sync with your main run if needed.
    # test_stat="utest",
    # test_stat_kwargs={"use": "clr", "rank_method": "average"},
)
```

Burden helper (compute and store a recommended burden column):

```python
from src.sceptre import add_burden_covariate

add_burden_covariate(
    adata=adata,
    guide_assignment_key="guide_assignment",
    covar_key="covar",
    guide_names_key="guide_names",
    guide2gene_key="guide2gene",
    burden_key="log1p_non_ntc_guides_per_cell",
    ntc_labels=["non-targeting", "safe-targeting", "NTC"],
    include_ntc=False,
    count_nonzero=True,
    use_log1p=True,
)
```

#### Synthetic data

Use the synthetic generator for large‑scale diagnostics:

```python
from tests.synthetic_data import make_sceptre_style_synth
from src.sceptre import prepare_crt_inputs

adata = make_sceptre_style_synth(
    N=10000,
    K=20,
    n_target_genes=200,
    guides_per_gene=6,
    ntc_frac_guides=0.15,
    moi_mean=5.5,
    frac_causal_genes=0.10,
    n_effect_programs=3,
    effect_size=0.6,
    confound_strength=0.0,
    seed=0,
)

# The generator stores usage in adata.obsm["usage"].
# Pass usage_key="usage" when preparing inputs.
inputs = prepare_crt_inputs(adata=adata, usage_key="usage")
```

Note on scale: this generator is designed for large synthetic cohorts (e.g., tens of
thousands of cells). For quick smoke tests, reduce `N`, `K`, and `n_target_genes`
to keep runtime and memory low.

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

Run the full test suite (excluding performance):

```bash
python3 -m pytest -q -m "not performance"
```

Run performance tests explicitly:

```bash
python3 -m pytest -q -m performance
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
