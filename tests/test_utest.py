import numpy as np
from scipy.stats import mannwhitneyu, rankdata

from src.sceptre.pipeline import prepare_crt_inputs, run_all_genes_union_crt
from src.sceptre.pipeline_helpers import _rank_biserial_from_rank_sum, _rank_sums_from_indices
from tests.synthetic_data import make_synthetic_adata


def test_utest_stat_matches_mannwhitneyu():
    rng = np.random.default_rng(0)
    N = 500
    K = 5
    Y = rng.normal(size=(N, K))

    obs_idx = rng.choice(N, size=120, replace=False)
    x_obs = np.zeros(N, dtype=bool)
    x_obs[obs_idx] = True
    n1 = int(x_obs.sum())
    n0 = N - n1

    R = np.column_stack([rankdata(Y[:, k], method="average") for k in range(K)])
    rank_sum = R[obs_idx].sum(axis=0)
    rbc_obs = _rank_biserial_from_rank_sum(rank_sum, n1, N)

    rbc_ref = []
    for k in range(K):
        res = mannwhitneyu(Y[x_obs, k], Y[~x_obs, k], alternative="greater")
        U = res.statistic
        rbc_ref.append(2.0 * U / (n1 * n0) - 1.0)
    rbc_ref = np.array(rbc_ref, dtype=np.float64)

    assert np.allclose(rbc_obs, rbc_ref, rtol=1e-6, atol=1e-6)


def test_rank_sum_sparse_matches_naive():
    rng = np.random.default_rng(1)
    N = 50
    B = 20
    K = 4

    R = rng.normal(size=(N, K))
    indices = []
    indptr = [0]
    for _ in range(B):
        m = int(rng.integers(0, N // 2))
        rows = rng.choice(N, size=m, replace=False)
        indices.extend(rows.tolist())
        indptr.append(len(indices))

    indices = np.array(indices, dtype=np.int32)
    indptr = np.array(indptr, dtype=np.int64)

    rank_sum_null, n1b = _rank_sums_from_indices(indptr, indices, R, N, B)

    rank_sum_naive = np.zeros((B, K), dtype=np.float64)
    for b in range(B):
        rows = indices[indptr[b] : indptr[b + 1]]
        if rows.size:
            rank_sum_naive[b] = R[rows].sum(axis=0)

    assert np.allclose(rank_sum_null, rank_sum_naive)
    assert np.all(n1b == np.diff(indptr))


def test_ols_compatibility_explicit():
    rng = np.random.default_rng(4)
    adata, _ = make_synthetic_adata(
        rng, n_cells=120, n_programs=4, n_genes=3, guides_per_gene=2
    )
    inputs = prepare_crt_inputs(adata)

    out_default = run_all_genes_union_crt(inputs, B=31, n_jobs=1)
    out_ols = run_all_genes_union_crt(inputs, B=31, n_jobs=1, test_stat="ols")

    assert np.allclose(out_default["pvals_df"], out_ols["pvals_df"])
    assert np.allclose(out_default["betas_df"], out_ols["betas_df"])
