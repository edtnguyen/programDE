from pathlib import Path
import sys

import numpy as np


class _DummyInputs:
    def __init__(self, guide_names):
        self.guide_names = guide_names


def _load_module_globals():
    base = Path(__file__).parents[1] if "__file__" in globals() else Path.cwd()
    module_path = base / "src" / "sceptre" / "ntc_parallel.py"
    module_globals = {"__name__": "ntc_parallel_test"}
    module_code = module_path.read_text()
    exec(module_code, module_globals)
    return module_globals


def _build_manual(ntc_groups_ens, inputs, compute_fn, B, base_seed):
    guide_to_col = {g: i for i, g in enumerate(inputs.guide_names)}
    manual = []
    for groups in ntc_groups_ens:
        for guides in groups.values():
            cols = [guide_to_col[g] for g in guides]
            manual.append(compute_fn(cols, inputs, B, base_seed).ravel())
    return np.concatenate(manual)


def test_compute_ntc_group_null_pvals_parallel_matches_manual():
    guide_names = ["ntc0", "ntc1", "ntc2", "ntc3", "gA", "gB"]
    inputs = _DummyInputs(guide_names)

    ntc_groups_ens = [
        {"g0": ["ntc0", "ntc1"], "g1": ["ntc2", "ntc3"]},
        {"g0": ["ntc0", "ntc2"], "g1": ["ntc1", "ntc3"]},
    ]

    def _fake_compute(guide_idx, inputs, B, base_seed, **kwargs):
        val = float(sum(guide_idx) + base_seed)
        return np.full((B, 1), val, dtype=np.float64)

    module_globals = _load_module_globals()
    module_globals["compute_guide_set_null_pvals"] = _fake_compute

    manual = _build_manual(ntc_groups_ens, inputs, _fake_compute, 5, 7)

    sequential = module_globals["compute_ntc_group_null_pvals_parallel"](
        inputs=inputs,
        ntc_groups_ens=ntc_groups_ens,
        B=5,
        base_seed=7,
        n_jobs=1,
        backend="threading",
    )

    assert sequential.shape == manual.shape
    assert np.allclose(sequential, manual)

    class _FakeJoblib:
        @staticmethod
        def delayed(func):
            def _wrap(*args, **kwargs):
                return lambda: func(*args, **kwargs)

            return _wrap

        class Parallel:
            def __init__(self, n_jobs=None, backend=None):
                self.n_jobs = n_jobs
                self.backend = backend

            def __call__(self, iterable):
                out = []
                for item in iterable:
                    out.append(item() if callable(item) else item)
                return out

    sys.modules["joblib"] = _FakeJoblib
    try:
        parallel = module_globals["compute_ntc_group_null_pvals_parallel"](
            inputs=inputs,
            ntc_groups_ens=ntc_groups_ens,
            B=5,
            base_seed=7,
            n_jobs=2,
            backend="threading",
        )
    finally:
        sys.modules.pop("joblib", None)

    assert parallel.shape == manual.shape
    assert np.allclose(parallel, manual)
