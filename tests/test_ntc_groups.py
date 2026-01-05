import numpy as np
from scipy.stats import kstest

from src.sceptre.ntc_groups import (
    build_ntc_group_inputs,
    crt_pvals_for_ntc_groups_ensemble,
    make_ntc_groups_ensemble,
    make_ntc_groups_matched_by_freq,
)
from src.sceptre.pipeline import prepare_crt_inputs
from tests.synthetic_data import make_synthetic_adata


def test_ntc_groups_matched_by_freq_deterministic():
    rng = np.random.default_rng(0)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=300,
        n_programs=4,
        n_genes=4,
        guides_per_gene=6,
        n_covariates=3,
        ntc_genes=("NTC",),
        propensity_mode="constant",
        propensity_range=(0.1, 0.3),
    )
    inputs = prepare_crt_inputs(adata)
    ntc_guides, guide_freq, guide_to_bin, real_sigs = build_ntc_group_inputs(
        inputs,
        ntc_label="NTC",
        group_size=6,
        n_bins=10,
    )

    groups1 = make_ntc_groups_matched_by_freq(
        ntc_guides=ntc_guides,
        ntc_freq=guide_freq,
        real_gene_bin_sigs=real_sigs,
        guide_to_bin=guide_to_bin,
        group_size=6,
        seed=123,
    )
    groups2 = make_ntc_groups_matched_by_freq(
        ntc_guides=ntc_guides,
        ntc_freq=guide_freq,
        real_gene_bin_sigs=real_sigs,
        guide_to_bin=guide_to_bin,
        group_size=6,
        seed=123,
    )
    assert groups1 == groups2

    all_guides = [g for guides in groups1.values() for g in guides]
    assert len(all_guides) == len(set(all_guides))
    assert all(len(guides) == 6 for guides in groups1.values())

    real_sig_set = {tuple(sorted(sig)) for sig in real_sigs}
    for guides in groups1.values():
        sig = tuple(sorted(guide_to_bin[g] for g in guides))
        assert sig in real_sig_set


def test_ntc_groups_ensemble_size_stable():
    rng = np.random.default_rng(1)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=300,
        n_programs=4,
        n_genes=4,
        guides_per_gene=12,
        n_covariates=3,
        ntc_genes=("NTC",),
        propensity_mode="constant",
        propensity_range=(0.1, 0.3),
    )
    inputs = prepare_crt_inputs(adata)
    ntc_guides, guide_freq, guide_to_bin, real_sigs = build_ntc_group_inputs(
        inputs,
        ntc_label="NTC",
        group_size=6,
        n_bins=10,
    )

    groups_ens = make_ntc_groups_ensemble(
        ntc_guides=ntc_guides,
        ntc_freq=guide_freq,
        real_gene_bin_sigs=real_sigs,
        guide_to_bin=guide_to_bin,
        n_ensemble=5,
        seed0=42,
        group_size=6,
    )
    counts = np.array([len(g) for g in groups_ens], dtype=np.float64)
    median = np.median(counts)
    assert np.all(np.abs(counts - median) <= 0.2 * max(1.0, median))


def test_ntc_group_pvals_uniform_under_null():
    rng = np.random.default_rng(2)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=500,
        n_programs=6,
        n_genes=5,
        guides_per_gene=12,
        n_covariates=3,
        ntc_genes=("NTC", "NTC2"),
        propensity_mode="constant",
        propensity_range=(0.1, 0.3),
        effect_gene=None,
        effect_size=0.0,
    )
    inputs = prepare_crt_inputs(adata)
    ntc_guides, guide_freq, guide_to_bin, real_sigs = build_ntc_group_inputs(
        inputs,
        ntc_label=("NTC", "NTC2"),
        group_size=6,
        n_bins=6,
    )
    groups_ens = make_ntc_groups_ensemble(
        ntc_guides=ntc_guides,
        ntc_freq=guide_freq,
        real_gene_bin_sigs=real_sigs,
        guide_to_bin=guide_to_bin,
        n_ensemble=3,
        seed0=7,
        group_size=6,
    )
    pvals_ens = crt_pvals_for_ntc_groups_ensemble(
        inputs=inputs,
        ntc_groups_ens=groups_ens,
        B=79,
        seed0=99,
    )

    pvals = np.concatenate(
        [df.to_numpy().ravel() for df in pvals_ens.values()]
    )
    pvals = pvals[np.isfinite(pvals)]
    assert pvals.size > 0
    stat, pval = kstest(pvals, "uniform")
    assert pval > 1e-4
