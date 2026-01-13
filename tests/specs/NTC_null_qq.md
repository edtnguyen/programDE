
# Codex spec: QQ plots + calibration diagnostics for **NTC empirical-null (CLR-OLS)** using cross-fitting

We are implementing an NTC empirical-null mode (null_method="ntc_empirical") for CLR-OLS, where each real gene/program test is calibrated against matched NTC pseudo-genes using matching keys like `(n1, denom d = x^T M_C x)` and (by default) **batch meta**.

Now we need a QQ plotting + diagnostic module that is **appropriate for an NTC-derived null**:
- avoids “in-sample” reuse of NTC units that makes QQ look artificially perfect,
- supports both **gene-level** and **program-level** QQ,
- optionally adds uncertainty bands for the NTC holdout QQ (bootstrap envelope),
- integrates seamlessly with existing outputs & plotting code (do not break CRT QQ plots).

Batch labels are stored in `inputs.covar_df_raw["batch"]` (via `prepare_crt_inputs`), consistent with `memory.md`. The repo already has `diagnostics.py` and `global_null_diagnostics.py` which produce QQ plots for CRT; this spec adds **parallel functions** for NTC empirical-null.

Reference context: the pipeline currently supports samplers returning `(indptr, indices)` and stratified CRT, per `memory.md`. No changes should break those paths. :contentReference[oaicite:0]{index=0}

---

## 0) Design goals

1. **Cross-fit NTC**: split NTC pseudo-genes into A/B; use A to define the null; evaluate calibration using B vs A.
2. **Two QQs by default**:
   - **Gene-level aggregated QQ** (primary): one p-value per gene (max-stat across programs).
   - **Program-level QQ** (secondary): p-values per program across genes (optionally a small subset of programs).
3. **Same expected quantile grid** for each curve: `q_i = (i - 0.5) / m` (do not reuse m from other curves).
4. **Bootstrap envelope** for NTC holdout QQ (optional): 5–95% band.
5. Backward compatible: keep existing CRT QQ plotting unchanged; add new functions and a new CLI/config entry (if applicable) without affecting defaults.

---

## 1) New module: `src/sceptre/ntc_qq.py`

Add a module with the following core functions.

### 1.1 Quantile utilities

`expected_quantiles(m: int) -> np.ndarray`
- returns `(np.arange(1, m+1) - 0.5) / m`

`qq_coords(pvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
- filters finite pvals, clips to `(1e-300, 1]`
- sorts ascending
- x = `-log10(expected_quantiles(m))`
- y = `-log10(p_sorted)`

### 1.2 Bootstrap QQ envelope

`bootstrap_qq_envelope(pvals: np.ndarray, n_boot: int = 200, alpha: float = 0.10, seed: int = 0)`
- input: pvals array length m
- for b in 1..n_boot:
  - resample with replacement from pvals, size m
  - compute y_b = `-log10(sort(p_boot))`
- return:
  - x = `-log10(expected_quantiles(m))`
  - y_lo = percentile(alpha/2)
  - y_hi = percentile(1-alpha/2)
- IMPORTANT: x uses the same m as the curve; do not mix m.

Keep it deterministic via `np.random.default_rng(seed)`.

### 1.3 Plotting

`plot_qq_curves(curves: dict[str, np.ndarray], out_png: str, title: str, envelope: tuple[np.ndarray,np.ndarray,np.ndarray] | None = None)`
- `curves` maps label -> pvals array
- plot each curve (x,y) on same axes; also plot y=x reference line
- if envelope provided (x, y_lo, y_hi), fill between for NTC-holdout band
- use matplotlib only (no seaborn), 1 plot per figure (no subplots)
- save to `out_png`

---

## 2) Cross-fit NTC p-values for QQ diagnostics

We need “holdout” NTC p-values computed **exactly the same way as real genes** but with NTC_B treated as “queries” and NTC_A used as “null reference set”.

Add helper(s) in the NTC-null pipeline module (likely `src/sceptre/ntc_null.py`) OR in `ntc_qq.py` if you want it self-contained.

### 2.1 Deterministic A/B split

`split_ntc_units(ntc_unit_ids: np.ndarray, frac_A: float = 0.5, seed: int = 0) -> tuple[np.ndarray,np.ndarray]`
- deterministic shuffle with seed
- return ids_A, ids_B disjoint

### 2.2 Compute cross-fit empirical p-values (program-level)

Implement a function that produces **two p-value matrices**:

- `p_real_vs_A`: (n_genes, K)
- `p_ntcB_vs_A`: (n_ntcB, K)  <-- this is the calibration curve for QQ

Signature:

`compute_ntc_empirical_pvals_crossfit(beta_real: ndarray[G,K], sig_real: DataFrame[G,*], beta_ntcA: ndarray[RA,K], sig_ntcA: DataFrame[RA,*], beta_ntcB: ndarray[RB,K], sig_ntcB: DataFrame[RB,*], matching_cfg: dict, two_sided: bool=True) -> dict`

Where:
- `sig_*` contains at least:
  - `n1` (treated count)
  - `d` (OLS denom x^T M_C x)
  - optionally `batch` if pooling; but for batch-meta we do this per batch, so signatures are per-batch already.

Algorithm (must match how we compute real gene p-values in ntc_empirical mode):
- assign bins based on **real units** (for stability), using:
  - n1 bins (quantile or integer bins)
  - log(d) bins (quantile bins)
- map each unit to a bin id
- for each bin:
  - compare each obs unit in that bin against the null NTC_A units in that bin
  - empirical p-value:
    - `p = (1 + count(|T_ntc| >= |T_obs|)) / (R_bin + 1)`
- implement fallback if bin has too few NTC_A:
  - expand to neighbor bins by Manhattan radius, or fallback to all NTC_A
- return p-value matrices + bin-size diagnostics

**Important:** This function should work for both:
- per-program T = `beta[:,k]`
- aggregated T = `max_k abs(beta)` (see below)

### 2.3 Gene-level aggregated p-values (primary QQ)

For QQ that’s sensitive to sparse signal, compute:
- `T_real_gene[g] = max_k abs(beta_real[g,k])`
- `T_ntcA[r] = max_k abs(beta_ntcA[r,k])`
- `T_ntcB[r] = max_k abs(beta_ntcB[r,k])`

Then compute:
- `p_real_gene_vs_A[g]` using matched comparison of `T_real_gene[g]` vs `T_ntcA` in matched bin
- `p_ntcB_gene_vs_A[r]` using `T_ntcB[r]` vs matched `T_ntcA`

Implement:
`compute_ntc_empirical_pvals_gene_agg_crossfit(...)` OR reuse the same function with `K=1` by passing in `T` as a vector and signatures.

This is the **primary** QQ diagnostic.

---

## 3) Batch handling: per-batch QQ first, then meta QQ

We observed: within batch looks calibrated; pooled breaks. So the QQ tool must support per-batch outputs.

### 3.1 Inputs
In batch-meta mode, the NTC-null pipeline already computes per-batch betas and signatures.

Ensure the NTC empirical-null runner returns a structured object like:

`out["ntc_crossfit"] = {`
- `"batches": list[batch_id]`
- `"p_real_vs_A_by_batch": dict[batch -> DataFrame(G,K)]`
- `"p_ntcB_vs_A_by_batch": dict[batch -> DataFrame(RB,K)]`
- `"p_real_gene_vs_A_by_batch": dict[batch -> Series(G)]`
- `"p_ntcB_gene_vs_A_by_batch": dict[batch -> Series(RB)]`
- `"meta_p_real_vs_A"`: DataFrame(G,K)  (Fisher combined)
- `"meta_p_ntcB_vs_A"`: DataFrame(RB,K) (Fisher combined)
- `"meta_p_real_gene_vs_A"`: Series(G)
- `"meta_p_ntcB_gene_vs_A"`: Series(RB)
- `"matching_diagnostics"`: bin counts / fallback usage
`}`

If the runner does not currently return this, add it only when `null_method="ntc_empirical"` and `qq_crossfit=True` (new flag), keeping default outputs unchanged.

### 3.2 Combine method
If you already meta-combine per-batch p-values (Fisher), reuse the same combine for NTC_B as well (B vs A), so the holdout QQ corresponds exactly to the meta pipeline.

---

## 4) QQ plot entrypoints

Add one main function:

`make_ntc_empirical_qq_plots(out: dict, out_dir: str, programs_to_plot: str|list|None = "top_var", n_programs: int = 6, make_per_batch: bool = True, make_meta: bool = True, envelope_boot: int = 200, seed: int = 0)`

Behavior:
- Requires `out["ntc_crossfit"]` present (from Section 3).
- Always make **gene-level QQ**:
  - curves:
    - `"NTC holdout (B vs A)"` = `meta_p_ntcB_gene_vs_A` (or per batch)
    - `"Real genes (vs A)"` = `meta_p_real_gene_vs_A`
- Make **program-level QQ**:
  - choose programs:
    - if `"top_var"`: compute variance of beta across genes (or variance of CLR usage means) and pick top `n_programs`
    - if list: use provided
  - for each selected program k:
    - curves:
      - `"NTC holdout (B vs A)"` = `meta_p_ntcB_vs_A[:,k]`
      - `"Real genes (vs A)"` = `meta_p_real_vs_A[:,k]`
- For each plot, optionally add envelope:
  - envelope computed from NTC holdout pvals only
  - `bootstrap_qq_envelope(ntc_holdout_pvals, n_boot=envelope_boot, seed=seed+...)`
- Save:
  - `{out_dir}/qq_ntc_empirical_genelevel_meta.png`
  - `{out_dir}/qq_ntc_empirical_program_{k}_meta.png`
  - and per batch versions if enabled:
    - `{out_dir}/per_batch/{batch}/qq_ntc_empirical_genelevel.png`
    - `{out_dir}/per_batch/{batch}/qq_ntc_empirical_program_{k}.png`

---

## 5) Tests (pytest)

Add `tests/test_ntc_empirical_qq.py` with deterministic small settings.

### 5.1 QQ utilities correctness
- `test_expected_quantiles_range_and_monotonic`
- `test_qq_coords_shapes_and_sorting`
- `test_bootstrap_envelope_shapes`

### 5.2 Cross-fit calibration sanity under global null
Using `make_sceptre_style_synth(effect_size=0)` with multiple batches:
- run ntc_empirical pipeline with `qq_crossfit=True` and `n_ntc_units >= 2000` (small for test)
- from returned crossfit p-values:
  - `p_ntcB_gene_vs_A` should be roughly Uniform:
    - check quantiles:
      - q50 in [0.45, 0.55]
      - q10 in [0.07, 0.13]
      - q01 in [0.005, 0.02]
  - also check one program’s `p_ntcB_vs_A[:,k]` similarly (loose)
Use loose bounds to avoid flaky tests.

### 5.3 “In-sample” warning test (optional)
If user mistakenly tries to QQ `NTC_A vs NTC_A`, it should be flagged (not necessarily error):
- implement a guard or warning that “in-sample NTC QQ is not diagnostic”.
- test that this warning is issued when asked.

### 5.4 Backward compatibility
- Ensure calling existing CRT QQ code still works unchanged.
- Ensure `make_ntc_empirical_qq_plots` is only invoked when `null_method="ntc_empirical"`.

---

## 6) Documentation

Update README with a short section:

**NTC empirical-null QQ plots (cross-fit)**
- describe A/B split
- gene-level QQ recommended
- show usage:

```python
out = run_all_genes_union_crt(
    inputs=inputs,
    null_method="ntc_empirical",
    null_kwargs=dict(...),
    qq_crossfit=True,   # new flag
)
make_ntc_empirical_qq_plots(out, out_dir="results/qq_ntc")
````

Mention:

* The calibration curve is `NTC_B vs NTC_A` (holdout), not `NTC_A vs NTC_A`.
* Always interpret “real genes vs NTC_A” relative to the holdout band.

---

## 7) Deliverables

* `src/sceptre/ntc_qq.py` implemented
* minimal additions to NTC-null runner to return `out["ntc_crossfit"]` when enabled
* new pytest file `tests/test_ntc_empirical_qq.py`
* README snippet updated


