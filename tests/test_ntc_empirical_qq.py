import numpy as np
import pandas as pd

from src.sceptre.ntc_qq import bootstrap_qq_envelope, expected_quantiles, qq_coords
from src.sceptre.pipeline import prepare_crt_inputs, run_all_genes_union_crt
from tests.synthetic_data import make_sceptre_style_synth


def _covar_df_from_synth(adata):
    covar_mat = np.asarray(adata.obsm["covar"])
    cols = ["batch", "log_depth"] + [
        f"covar_{i}" for i in range(max(0, covar_mat.shape[1] - 3))
    ]
    return pd.DataFrame(covar_mat[:, 1:], columns=cols, index=adata.obs_names)


def test_expected_quantiles_range_and_monotonic():
    q = expected_quantiles(10)
    assert np.all(q > 0.0)
    assert np.all(q < 1.0)
    assert np.all(np.diff(q) > 0.0)


def test_qq_coords_shapes_and_sorting():
    pvals = np.array([0.2, 0.05, np.nan, 0.9, 0.01])
    x, y = qq_coords(pvals)
    assert x.shape == y.shape
    assert x.size == 4
    assert y[0] >= y[-1]


def test_bootstrap_envelope_shapes():
    rng = np.random.default_rng(0)
    pvals = rng.uniform(size=50)
    x, lo, hi = bootstrap_qq_envelope(pvals, n_boot=10, alpha=0.10, seed=0)
    assert x.shape == lo.shape == hi.shape
    assert np.all(lo <= hi)
    assert np.all(np.diff(x) < 0.0)


def test_crossfit_ntc_holdout_uniform_under_global_null():
    adata = make_sceptre_style_synth(
        N=3000,
        K=8,
        n_target_genes=20,
        guides_per_gene=6,
        ntc_frac_guides=0.2,
        frac_causal_genes=0.10,
        n_effect_programs=2,
        effect_size=0.0,
        confound_strength=0.0,
        seed=7,
    )
    adata.obsm["covar"] = _covar_df_from_synth(adata)
    inputs = prepare_crt_inputs(adata=adata, usage_key="usage")

    out = run_all_genes_union_crt(
        inputs=inputs,
        null_method="ntc_empirical",
        qq_crossfit=True,
        null_kwargs=dict(
            ntc_labels=["NTC"],
            guides_per_unit=6,
            n_ntc_units=2000,
            batch_mode="meta",
            combine_method="fisher",
            matching=dict(n_n1_bins=8, n_d_bins=8, min_ntc_per_bin=30),
            min_treated=3,
            min_control=3,
            qq_crossfit_seed=11,
        ),
        n_jobs=1,
        base_seed=11,
    )

    crossfit = out.get("ntc_crossfit")
    assert crossfit is not None

    p_gene = np.asarray(crossfit["meta_p_ntcB_gene_vs_A"])
    p_gene = p_gene[np.isfinite(p_gene)]
    assert p_gene.size > 100
    q01, q10, q50 = np.quantile(p_gene, [0.01, 0.10, 0.50])
    assert 0.005 <= q01 <= 0.03
    assert 0.06 <= q10 <= 0.14
    assert 0.40 <= q50 <= 0.60

    p_prog = crossfit["meta_p_ntcB_vs_A"].iloc[:, 0].to_numpy()
    p_prog = p_prog[np.isfinite(p_prog)]
    q01, q10, q50 = np.quantile(p_prog, [0.01, 0.10, 0.50])
    assert 0.005 <= q01 <= 0.03
    assert 0.06 <= q10 <= 0.14
    assert 0.40 <= q50 <= 0.60
