```python
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def make_sceptre_style_synth(
    N: int = 50_000,              # cells
    K: int = 70,                  # programs
    n_target_genes: int = 300,    # real target genes (each has 6 guides)
    guides_per_gene: int = 6,     # fixed at 6 per target gene
    ntc_frac_guides: float = 0.15,# NTC guides = 15% of total guides
    moi_mean: float = 5.5,        # average guides/cell
    frac_causal_genes: float = 0.10,  # fraction of target genes with true effects
    n_effect_programs: int = 3,        # sparse effects per causal gene
    effect_size: float = 0.6,          # additive shift in logits
    n_cov_extra: int = 6,              # number of nuisance covariates (besides batch, log_depth)
    confound_strength: float = 0.0,    # 0 = no confounding; >0 makes guide frequencies depend on batch
    eps_near_zero: float = 1e-6,       # sprinkle near-zeros in usage to mimic cNMF
    sprinkle_near_zero_frac: float = 0.15,
    seed: int = 0,
):
    """
    Synthetic generator with:
      - NTC guides have NO effect by construction.
      - NTC guides comprise ~15% of total guides.
      - Target genes each have exactly 6 guides.
      - Some target genes are causal and shift program-usage logits when present (union indicator).
      - High MOI: multiple guides per cell sampled, potentially multiple genes.

    Outputs AnnData with:
      adata.obsm["covar"]            : (N, p) covariates incl intercept
      adata.obsm["usage"]            : (N, K) compositional program usage (rows sum to 1)
      adata.obsm["guide_assignment"] : CSR (N, n_guides) binary
      adata.uns["guide2gene"]        : dict guide_name -> gene_name (NTC gene name = "NTC")
      adata.uns["gene2guides"]       : dict gene_name -> list[guide_name]
      adata.uns["causal_genes"]      : list of causal gene names
      adata.uns["gene_effects"]      : dict gene_name -> effect vector (length K) for causal genes
    """
    rng = np.random.default_rng(seed)

    # -------------------------
    # 1) Define guides: target genes (6 each) + NTC guides at 15% of total
    # -------------------------
    assert guides_per_gene == 6, "This generator assumes 6 guides per target gene."

    n_target_guides = n_target_genes * guides_per_gene
    # Want ntc_guides / (ntc_guides + target_guides) ~= ntc_frac_guides
    n_ntc_guides = int(np.round((ntc_frac_guides / (1.0 - ntc_frac_guides)) * n_target_guides))
    n_guides = n_target_guides + n_ntc_guides

    target_gene_names = [f"GENE{g:05d}" for g in range(n_target_genes)]
    ntc_gene_name = "NTC"

    guide_names = []
    guide2gene = {}
    gene2guides = {}

    # Target guides
    for gi, gene in enumerate(target_gene_names):
        gs = []
        for j in range(guides_per_gene):
            gname = f"{gene}_g{j+1:02d}"
            guide_names.append(gname)
            guide2gene[gname] = gene
            gs.append(gname)
        gene2guides[gene] = gs

    # NTC guides
    ntc_guides = []
    for j in range(n_ntc_guides):
        gname = f"NTC_g{j+1:04d}"
        guide_names.append(gname)
        guide2gene[gname] = ntc_gene_name
        ntc_guides.append(gname)
    gene2guides[ntc_gene_name] = ntc_guides

    # Indices for sampling
    n_target_guides = n_target_genes * guides_per_gene
    ntc_guide_idx = np.arange(n_target_guides, n_guides, dtype=np.int32)

    # Map target gene -> its guide indices (contiguous blocks)
    target_gene_to_guide_idx = {
        gene: np.arange(i * guides_per_gene, (i + 1) * guides_per_gene, dtype=np.int32)
        for i, gene in enumerate(target_gene_names)
    }

    # -------------------------
    # 2) Covariates
    # -------------------------
    batch = rng.integers(0, 2, size=N).astype(np.float64)
    libsize = rng.lognormal(mean=10.0, sigma=0.35, size=N).astype(np.float64)
    log_depth = np.log1p(libsize)

    covs = [batch, log_depth]
    for _ in range(max(0, n_cov_extra)):
        covs.append(rng.normal(0, 1, size=N).astype(np.float64))

    C_no_intercept = np.column_stack(covs)  # (N, 2+n_cov_extra)
    intercept = np.ones((N, 1), dtype=np.float64)
    C = np.column_stack([intercept, C_no_intercept])  # (N, p)
    p = C.shape[1]

    # -------------------------
    # 3) Baseline usage logits from covariates (logistic-normal -> softmax)
    # -------------------------
    W = rng.normal(0, 0.25, size=(p, K))
    logits = C @ W + rng.normal(0, 0.3, size=(N, K))

    # -------------------------
    # 4) Define causal genes and their effects on logits (NTC has zero by construction)
    # -------------------------
    n_causal = max(1, int(np.round(frac_causal_genes * n_target_genes)))
    causal_gene_idx = rng.choice(n_target_genes, size=n_causal, replace=False)
    causal_genes = [target_gene_names[i] for i in causal_gene_idx]

    gene_effects = {}
    for gene in causal_genes:
        eff = np.zeros(K, dtype=np.float64)
        progs = rng.choice(K, size=min(n_effect_programs, K), replace=False)
        signs = rng.choice([-1.0, 1.0], size=progs.size)
        eff[progs] = signs * effect_size
        gene_effects[gene] = eff

    # -------------------------
    # 5) Sample guide assignments per cell (high MOI), optionally confounded by batch
    # -------------------------
    # We sample "events" per cell: for each event choose NTC vs target, then a guide.
    # To introduce confounding, we bias WHICH target genes are chosen by batch.
    gene_base_weights = rng.lognormal(mean=0.0, sigma=0.6, size=n_target_genes).astype(np.float64)
    if confound_strength > 0:
        # batch-specific multiplicative tilts (kept modest)
        tilt = rng.normal(0, confound_strength, size=n_target_genes).astype(np.float64)
        gene_w_batch0 = gene_base_weights * np.exp(-tilt)
        gene_w_batch1 = gene_base_weights * np.exp(+tilt)
    else:
        gene_w_batch0 = gene_base_weights
        gene_w_batch1 = gene_base_weights

    # Normalize to probabilities for sampling genes
    gene_p_batch0 = gene_w_batch0 / gene_w_batch0.sum()
    gene_p_batch1 = gene_w_batch1 / gene_w_batch1.sum()

    # MOI draws
    m = rng.poisson(moi_mean, size=N)
    m = np.clip(m, 0, min(50, n_guides)).astype(np.int32)

    rows = []
    cols = []

    # Also track per-cell union indicators for causal genes only (fast)
    # (We’ll apply effects after sampling; no need to store all gene unions.)
    x_causal = np.zeros((N, len(causal_genes)), dtype=np.int8)
    causal_gene_to_j = {g: j for j, g in enumerate(causal_genes)}

    # Precompute guide indices per target gene for quick guide sampling
    # target gene i has guides [i*6 .. i*6+5]
    for i in range(N):
        mi = int(m[i])
        if mi == 0:
            continue

        # Decide NTC vs target for each event
        is_ntc = rng.random(mi) < ntc_frac_guides
        n_ntc = int(is_ntc.sum())
        n_tgt = mi - n_ntc

        chosen_cols = []

        # Sample NTC guides
        if n_ntc > 0:
            ntc_choices = rng.choice(ntc_guide_idx, size=n_ntc, replace=False if n_ntc <= ntc_guide_idx.size else True)
            chosen_cols.extend(ntc_choices.tolist())

        # Sample target genes, then one guide within gene
        if n_tgt > 0:
            gene_p = gene_p_batch1 if batch[i] >= 0.5 else gene_p_batch0
            gene_idx = rng.choice(n_target_genes, size=n_tgt, replace=True, p=gene_p)

            # For each sampled gene, pick one of its 6 guides
            within = rng.integers(0, guides_per_gene, size=n_tgt)
            guide_idx = gene_idx * guides_per_gene + within
            chosen_cols.extend(guide_idx.tolist())

            # Update causal union indicators
            # (If causal gene appears >=1 time, union=1)
            if len(causal_genes) > 0:
                # check unique sampled genes for speed
                for gidx in np.unique(gene_idx):
                    gene_name = target_gene_names[int(gidx)]
                    j = causal_gene_to_j.get(gene_name, None)
                    if j is not None:
                        x_causal[i, j] = 1

        # Binarize within-cell (drop duplicates)
        chosen_cols = np.unique(np.array(chosen_cols, dtype=np.int32))
        rows.extend([i] * chosen_cols.size)
        cols.extend(chosen_cols.tolist())

    data = np.ones(len(rows), dtype=np.int8)
    G = sp.csr_matrix((data, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
                      shape=(N, n_guides), dtype=np.int8)
    G.eliminate_zeros()

    # -------------------------
    # 6) Apply causal gene effects to logits (additive in logits)
    # -------------------------
    for j, gene in enumerate(causal_genes):
        logits += x_causal[:, [j]].astype(np.float64) * gene_effects[gene][None, :]

    # -------------------------
    # 7) Convert logits -> usage (composition), sprinkle near-zeros
    # -------------------------
    usage = _softmax_rows(logits).astype(np.float64)

    if sprinkle_near_zero_frac > 0:
        mask = rng.random(size=usage.shape) < sprinkle_near_zero_frac
        usage[mask] *= eps_near_zero
        usage /= usage.sum(axis=1, keepdims=True)

    # -------------------------
    # 8) Build AnnData
    # -------------------------
    adata = ad.AnnData(X=sp.csr_matrix((N, 1), dtype=np.float32))
    adata.obs = pd.DataFrame(index=[f"cell{i:07d}" for i in range(N)])

    adata.obsm["covar"] = C
    adata.obsm["usage"] = usage
    adata.obsm["guide_assignment"] = G

    adata.uns["guide2gene"] = guide2gene
    adata.uns["gene2guides"] = gene2guides
    adata.uns["guide_names"] = guide_names
    adata.uns["target_gene_names"] = target_gene_names
    adata.uns["ntc_gene_name"] = ntc_gene_name

    # Ground truth
    adata.uns["causal_genes"] = causal_genes
    adata.uns["gene_effects"] = {g: gene_effects[g].tolist() for g in causal_genes}  # JSON-safe

    # Useful metadata for debugging
    adata.uns["synth_params"] = dict(
        N=N, K=K, n_target_genes=n_target_genes, guides_per_gene=guides_per_gene,
        n_ntc_guides=n_ntc_guides, ntc_frac_guides=ntc_frac_guides, moi_mean=moi_mean,
        frac_causal_genes=frac_causal_genes, n_effect_programs=n_effect_programs, effect_size=effect_size,
        confound_strength=confound_strength, seed=seed
    )

    return adata
```

### What this generator guarantees

* **NTC guides have no effects**: only `causal_genes` get nonzero `gene_effects`; `"NTC"` is never in `causal_genes`.
* **NTC guide count = ~15% of all guides**: by construction using `ntc_frac_guides`.
* **6 guides per target gene**: fixed blocks of 6 per gene.
* **Some real genes have effects**: `frac_causal_genes` controls how many; each affects `n_effect_programs` programs with ±`effect_size` shifts in logits.
