```markdown
# Codex instruction: Diagnose conservative NTC QQ under synthetic global null

We have a CRT pipeline for testing gene→program effects with CLR-transformed usage. Under synthetic **global null** (effect_size=0), the **All observed** QQ looks roughly calibrated but **NTC (raw)** is conservative in the tail and **NTC (skew)** is extremely conservative. We need to identify whether this is due to (A) degeneracy/NaNs/p=1 handling, (B) propensity model failure for NTC units, (C) NTC curve accidentally using ungrouped NTC, or (D) skew-normal calibration bug.

## What to implement (no refactor, just diagnostics + 2 reruns)

### 1) Add a “diagnostics dump” mode for a set of units
Create a function (or flag in existing runner) that, for each tested unit (real gene or NTC group), records:

**Per-unit observed-x / OLS diagnostics**
- `unit_id` (gene name or ntc_group_id), `unit_type` in {"real_gene","ntc_group"}
- `x_mean = mean(x)`  
- `n_treated = sum(x)`  
- `n_control = n_cells - n_treated`
- `u = C.T @ x` (vector, but only store summary norms)
  - store `u_norm = ||u||2`
- `den = x^T M_C x` used in beta formula:
  - compute exactly as in production: `den = xTx - u.T @ CtC_inv @ u` where `xTx = sum(x)` for binary x
  - store `den`
- `den_is_bad = (den <= 1e-12)` (same threshold as production)
- `beta_obs_nan = any(isnan(beta_obs))`
- `pvals_raw` (vector length K) OR just: `pmin_raw = min(pvals_raw)`
- `pvals_raw_eq1_frac = mean(pvals_raw == 1.0)`
- If skew-calibration exists:
  - `pvals_skew` and `pmin_skew`, `pvals_skew_eq1_frac`

**Per-unit propensity diagnostics**
When propensity is fit (mode="fit"):
- `p_hat_mean = mean(p_hat)`
- `p_hat_min`, `p_hat_max`
- `p_hat_var = var(p_hat)`
- `propensity_separation_flag`: true if `p_hat_min < 1e-4` or `p_hat_max > 1-1e-4` (near-saturation)
- `propensity_auc` (optional): AUC of p_hat predicting x (sklearn roc_auc_score) — large AUC under null can indicate leakage/overfit but may happen if x depends on C.

**Per-unit CRT-null diagnostics**
- From null betas `beta_null[:,k]`, compute:
  - `null_nan_frac = mean(any isnan across programs or per program)`
  - For at least one program (e.g. k=0), store `beta_null_sd`, `beta_null_mean`

### 2) Produce a compact summary table for NTC vs real genes
After running a representative set (or all units from the QQ plot), print and save a CSV:

Group by `unit_type` and report:
- count units
- median/quantiles of `x_mean` and `n_treated`
- fraction of units with `den_is_bad`
- fraction with `beta_obs_nan`
- fraction of tests (unit×program) with `p==1` for raw and skew separately
- distribution of `p_hat_mean`, and fraction `propensity_separation_flag`

Also compute these *global* summaries:
- overall fraction `p_raw == 1` among NTC tests and among real-gene tests
- overall fraction `beta_obs is nan` among NTC units and among real-gene units

Save outputs to:
- `diagnostics_units.csv`
- `diagnostics_summary.txt` (or JSON)

### 3) Verify NTC curve is truly “grouped 6-guide units”
Add an assertion/check in the plotting pipeline:
- When building NTC curve, confirm each unit has exactly 6 guides.
- Log:
  - number of NTC groups
  - distribution of group sizes (min/median/max)
  - if any group size != 6, raise error.
This prevents accidental fallback to “NTC gene = union of all NTC guides”.

### 4) Rerun NTC grouped pipeline with oracle propensity to isolate propensity failure
Implement a rerun option:
- For the same NTC groups, recompute CRT p-values with `propensity_mode="oracle_const"` where `p_i = mean(x)` for that unit.
- Keep everything else identical: same OLS cache, same B, same random seed base.
- Save p-values: `pvals_ntc_oracle_raw` (and skew if still enabled, but skew can be turned off).

Then:
- Generate QQ curves for:
  - NTC grouped with fit propensity (current)
  - NTC grouped with oracle propensity
If oracle fixes the conservative tail, the culprit is propensity fitting for NTC units.

### 5) Skew-normal calibration sanity checks (minimal)
Add a debug print for a small number of tests (e.g. 5 random NTC tests):
- show `p_raw`, `p_skew`, and `z_obs`
- show null z-score summary: mean, sd, skewness (scipy.stats.skew)
Under global null, skew calibration should not massively inflate p-values relative to raw.

Also add a guard:
- If skew fit fails, or estimated scale <= 0, or produces NaNs, fallback to raw p-values.

### 6) Deterministic seeds
Ensure diagnostics are reproducible:
- Use fixed seeds for:
  - NTC grouping
  - propensity fit (random_state)
  - CRT sampling RNG
Record `seed_base` in the diagnostics output.

## Acceptance criteria (what “good” looks like)
Under effect_size=0:
- For both NTC grouped and real genes:
  - `den_is_bad` should be near 0
  - `beta_obs_nan` near 0
  - fraction `p==1` should be small (only from finite-sample granularity / occasional degenerate x), not massive
- If NTC is conservative with fit propensity but calibrated with oracle propensity:
  - propensity model is the problem (overfit/saturation/misspecification)
- Skew calibration:
  - `p_skew` should be close to `p_raw` under null; if it makes things far more conservative, treat as bug and disable by default.

## Deliverables
- Code changes implementing the diagnostics and oracle rerun
- `diagnostics_units.csv` and `diagnostics_summary.txt` created by the synthetic null example
- A short note in PR/commit message summarizing which failure mode was observed (A/B/C/D) and evidence (key stats)
```
