
# SPEC: NTC guide-group controls (matched-by-frequency + ensemble) for QQ plots

## Goal
Replace the current NTC QQ curve (built from `pvals_raw_df.loc[ntc_genes]`) with an apples-to-apples control where:
- **1 point = 1 NTC guide-group × 1 program** p-value
- each NTC guide-group has **6 guides** (matching real genes’ ~6 guides/gene)
- NTC guide-groups are formed via:
  1) **Matched grouping by guide frequency** (guide prevalence), to match real-gene guide prevalence distribution
  2) **Repeat-grouping ensemble** (multiple random partitions) to reduce variance and prevent cherry-picking

This spec covers both **group construction** and **pipeline/plottable outputs**.

---

## Definitions / Inputs

### Existing inputs (assumed)
- `adata.obsm["guide_assignment"]`: sparse CSR matrix `G` of shape `(N_cells, N_guides)` with binary per-guide assignment per cell.
- `adata.uns["guide2gene"]`: `dict[str, str]` mapping guide name → gene name.
- `adata.obsm["covar"]`: dense covariate matrix `C` shape `(N_cells, p)`.
- `adata.obsm["usage"]`: program usage matrix `U` shape `(N_cells, K)`.
- Pipeline produces `pvals_raw_df`: DataFrame with index = genes, columns = programs.

### New required inputs/config
- `ntc_label`: gene label used for NTC guides in `guide2gene` (default `"NTC"`).
- `group_size`: 6 (fixed).
- `n_ensemble`: number of random groupings (e.g., 10–50; default 20).
- `seed0`: base RNG seed for reproducibility.
- Optional `n_ntc_groups`: cap on number of NTC groups (default: use as many as possible after filtering).

---

## High-level plan

1) Compute per-guide **frequency** (prevalence) across cells for:
   - all NTC guides
   - all real-gene guides (non-NTC)
2) Define a **target frequency distribution** for NTC groups based on real genes’ guide frequencies.
3) For each ensemble replicate `e=1..E`:
   - create an NTC partition into groups of 6 using **matched-by-frequency** sampling
   - compute CRT p-values per group × program (same method as genes)
4) Produce plottable outputs:
   - `pvals_raw_df_ntcgrp_ens[e]`: per-ensemble DataFrame (rows=NTC groups, cols=programs)
   - `ntc_pvals_flat_ens[e]`: flattened p-values for QQ
   - `ntc_pvals_flat_all`: concatenated across ensemble replicates (for stable QQ line)
5) Update QQ plotting:
   - NTC curve uses concatenated NTC-group p-values (or optionally median curve across ensembles)
   - All observed scatter remains unchanged

---

## Detailed requirements

# A) Compute guide frequency (prevalence)

### A1. Implement `guide_frequency(G)`
**File**: `src/yourpkg/ntc_groups.py`

**Signature**
```python
def guide_frequency(G_csr, guide_names: list[str]) -> dict[str, float]:
    """
    Returns prevalence per guide: freq[g] = mean(G[:,g] > 0) across cells.
    """
````

**Implementation notes**

* `G` is CSR (cells × guides).
* Prevalence should be computed fast:

  * If binary matrix: `freq = (G.sum(axis=0).A1 / N)`
  * Ensure dtype is float.
* Return dict name→freq.

### A2. Identify NTC guides and real-gene guides

**NTC guides**:

* `ntc_guides = [g for g in guide_names if guide2gene[g] == ntc_label]`
  **Real guides**:
* `real_guides = [g for g in guide_names if guide2gene[g] != ntc_label]`

---

# B) Define matching target for NTC grouping

We want NTC guide groups to have a similar “difficulty” (treated rate / union prevalence) as real genes.

### B1. Compute per-gene guide-frequency profile (real genes)

For each real gene `gene`:

* obtain its guides `guides_gene` (from `guide2gene`)
* compute their individual frequencies `freq[g]`
* summarize into a gene-level statistic used for matching, e.g.:

  * `gene_freq_summary = sorted(freq[g] for g in guides_gene)[:6]` (use first 6 if >6)
  * and/or `mean`, `median`, `sum`
    Store:
* `real_gene_profiles`: list of vectors length 6 (or summary scalars)

### B2. Choose matching strategy (must implement both; default strategy below)

We match NTC guide groups to real genes by matching **guide frequency distribution**.

**Default matching target**:

* Bin guides by frequency (quantile bins).
* Create NTC groups so that the multiset of bins in each group resembles that of a randomly sampled real gene.

Specifically:

1. Compute quantile bin edges on *real guide frequencies* (e.g., 20 bins).
2. Assign each guide to a bin id.
3. For each real gene, create a “bin signature” of its 6 guides (multiset of 6 bin ids).
4. For each NTC group, sample a bin signature from real genes, then draw one unused NTC guide from each required bin.

This yields matched groups that mimic real genes’ per-guide abundance spectrum.

**Configurable params**

* `n_bins`: default 20
* `drop_remainder`: default True (if not enough guides in bins)

---

# C) Build matched NTC groups (one ensemble replicate)

### C1. Implement `make_ntc_groups_matched_by_freq(...)`

**File**: `src/yourpkg/ntc_groups.py`

**Signature**

```python
def make_ntc_groups_matched_by_freq(
    ntc_guides: list[str],
    ntc_freq: dict[str, float],
    real_gene_bin_sigs: list[list[int]],   # each len=6
    guide_to_bin: dict[str, int],
    group_size: int = 6,
    seed: int = 0,
    max_groups: int | None = None,
    drop_remainder: bool = True,
) -> dict[str, list[str]]:
    """
    Returns dict group_id -> list of 6 NTC guide names (no overlap within a replicate).
    Matched grouping: each group’s bin-multiset is drawn from real genes’ bin signatures.
    """
```

**Algorithm**

1. Partition NTC guides by bin: `bin -> list[guide]`
2. Shuffle each bin list with RNG.
3. Maintain “unused” pools per bin.
4. Repeatedly:

   * sample a real gene bin signature (list of 6 bins, with possible repeats)
   * attempt to draw one unused NTC guide from each requested bin

     * if bin repeats r times, need r distinct unused guides in that bin
   * if successful, create group, mark guides used, add to output
   * if fails:

     * try another signature up to `max_attempts` (e.g., 100)
     * if repeatedly fails and `drop_remainder`, stop
5. Stop when:

   * `max_groups` reached OR insufficient unused guides remain

**Constraints**

* Within one replicate, an NTC guide can appear in **at most one** group.
* Each group has exactly 6 guides.
* Deterministic given seed.

### C2. Emit diagnostics for each replicate

Return also:

* `group_bin_sigs`: dict group_id -> list of bin ids
* `n_groups`, and bin depletion stats (optional)

---

# D) Repeat-grouping ensemble (E replicates)

### D1. Implement ensemble driver

**File**: `src/yourpkg/ntc_groups.py` or `src/yourpkg/diagnostics.py`

**Signature**

```python
def make_ntc_groups_ensemble(
    ...,
    n_ensemble: int,
    seed0: int,
) -> list[dict[str, list[str]]]:
    """
    Returns a list of group dicts, one per ensemble replicate.
    Seeds should be seed0 + e (or hashed).
    """
```

**Requirement**

* Each replicate produces a different partition (different seed).
* Within each replicate, no overlaps; across replicates, overlaps allowed.

---

# E) Compute p-values for NTC groups

We need to reuse existing CRT code that currently runs per gene.

### E1. Refactor gene CRT to accept a generic “unit” defined by a guide set

**Preferred approach (minimal refactor)**:
Create a function:

```python
def crt_pvals_for_guide_set(
    G_csr,
    guide_idx: np.ndarray,     # guides defining this unit
    C: np.ndarray,
    Y: np.ndarray,             # CLR outcomes
    B: int,
    seed: int,
    **crt_kwargs
) -> np.ndarray:
    """
    Returns p-values across programs for one unit: shape (K,)
    Internally:
      x = union(G[:,guide_idx])
      fit propensity p_i = P(x=1|C)
      sample null x~ with fast sampler
      compute beta_obs, beta_null and CRT pvals
    """
```

Then implement wrappers:

* `crt_pvals_for_gene(gene)` calls `crt_pvals_for_guide_set` with gene’s guides
* `crt_pvals_for_ntc_group(group)` calls same function with group’s guides

### E2. NTC ensemble pvals

Implement:

```python
def crt_pvals_for_ntc_groups_ensemble(
    adata,
    ntc_groups_ens: list[dict[str, list[str]]],
    B: int,
    seed0: int,
    **crt_kwargs
) -> dict[int, pd.DataFrame]:
    """
    Returns mapping e -> DataFrame(rows=group_id, cols=programs)
    """
```

Seeds:

* For replicate e and group g:

  * `seed = hash((seed0, e, group_id)) mod 2**31-1`

Output:

* `out["pvals_raw_df_ntcgrp_ens"]` : dict e -> DataFrame
* `out["ntc_groups_ens"]` : list of dict group_id->guides
* `out["ntc_grouping_params"]` : record (group_size=6, n_bins, n_ensemble, etc.)

---

# F) QQ plotting modifications

### F1. NTC curve should be based on NTC groups, not NTC genes

Modify `qq_plot_ntc_pvals`:

* If `out["pvals_raw_df_ntcgrp_ens"]` exists:

  * flatten pvals across all groups × programs × ensemble:

    * `p_ntc = np.concatenate([df.to_numpy().ravel() for df in ens.values()])`
  * compute expected quantiles with `m=len(p_ntc)` (must be per-curve)
  * plot curve labeled e.g. `"NTC (grouped, matched, E=20)"`

### F2. Optional: show ensemble variability band

Compute QQ curve per replicate e:

* for each e:

  * compute sorted -log10 p vs expected grid for that replicate
    Because replicate curves may have different `m` if grouping yields variable #groups:
* Standardize by using a fixed set of quantile positions `q_grid` (e.g., 200 quantiles):

  * for each replicate:

    * compute quantile of p at each q_grid
  * plot median curve and 25–75% band
    This is optional but requested by “ensemble”.

Implement helper:

```python
def qq_curve_quantiles(pvals, q_grid):
    return np.quantile(pvals, q_grid)
```

Then plot:

* median of -log10(quantile p) across replicates
* band of 25–75% (or 5–95%)

### F3. Guardrail: expected axis length checks

In QQ plotting util, enforce:

* expected grid length == sorted pvals length for each curve OR use q_grid-based approach.

---

# G) Tests (pytest)

## G1. Unit test: matched grouping uses correct bins and size

* Each group has exactly 6 guides.
* No guide repeats within a replicate.
* Group bin signatures are drawn from real gene signatures (or at least same bin space).
* Deterministic with seed.

## G2. Integration test: NTC grouped pvals are approximately uniform under null synthetic

On a synthetic null dataset:

* compute NTC grouped pvals for E=5
* concatenate pvals and KS-test vs Uniform with loose threshold
* ensure QQ curve roughly near y=x (optionally via mean ~0.5 and reject rate ~alpha)

## G3. Test: ensemble output stable in size

* number of groups across replicates should not vary wildly; assert within e.g. ±20%

---

# H) Acceptance Criteria

1. `qq_plot_ntc_pvals` no longer uses `pvals_raw_df.loc[ntc_genes]`.
2. NTC control curve is built from **NTC groups of 6 guides** matched by guide frequency.
3. Multiple grouping replicates are supported and can be aggregated into a stable NTC curve.
4. All outputs saved for reproducibility:

   * group definitions per replicate
   * grouping parameters
   * per-replicate pval DataFrames
5. Tests pass on synthetic null data.

---

# Implementation Notes / Pitfalls

* Matching by per-guide frequency does not guarantee matching union prevalence exactly, but should bring it closer than “NTC mega-gene”.
* If NTC guide pool is small or frequency bins are sparse, grouping can fail; implement graceful stopping and report how many groups formed.
* Keep runtime manageable: for plotting, you can compute CRT pvals for NTC groups on a subset of programs if needed, but default should match gene pipeline (all programs).

