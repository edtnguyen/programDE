"""
QQ plot helpers for NTC empirical-null cross-fit diagnostics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def expected_quantiles(m: int) -> np.ndarray:
    """
    Expected QQ quantiles for m p-values.
    """
    if m <= 0:
        raise ValueError("m must be positive.")
    return (np.arange(1, m + 1) - 0.5) / float(m)


def qq_coords(pvals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute QQ plot coordinates (x, y) for a p-value array.
    """
    arr = np.asarray(pvals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("pvals must contain at least one finite value.")
    arr = np.clip(arr, 1e-300, 1.0)
    arr_sorted = np.sort(arr)
    m = arr_sorted.size
    x = -np.log10(expected_quantiles(m))
    y = -np.log10(arr_sorted)
    return x, y


def bootstrap_qq_envelope(
    pvals: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap envelope for QQ plot curves.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")
    arr = np.asarray(pvals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("pvals must contain at least one finite value.")
    arr = np.clip(arr, 1e-300, 1.0)
    m = arr.size
    rng = np.random.default_rng(seed)
    y_boot = np.empty((n_boot, m), dtype=np.float64)
    for b in range(n_boot):
        samp = rng.choice(arr, size=m, replace=True)
        samp = np.sort(samp)
        y_boot[b] = -np.log10(samp)
    lo = np.percentile(y_boot, 100.0 * (alpha / 2.0), axis=0)
    hi = np.percentile(y_boot, 100.0 * (1.0 - alpha / 2.0), axis=0)
    x = -np.log10(expected_quantiles(m))
    return x, lo, hi


def plot_qq_curves(
    curves: Mapping[str, np.ndarray],
    out_png: str,
    title: str,
    envelope: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    """
    Plot QQ curves and save to disk.
    """
    if not curves:
        raise ValueError("curves must contain at least one entry.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    max_x = 0.0
    for label, pvals in curves.items():
        x, y = qq_coords(pvals)
        ax.plot(x, y, lw=2.0, label=label)
        max_x = max(max_x, float(np.max(x)))

    if envelope is not None:
        x_env, y_lo, y_hi = envelope
        ax.fill_between(
            x_env, y_lo, y_hi, color="#d9d9d9", alpha=0.5, label="NTC holdout band"
        )
        max_x = max(max_x, float(np.max(x_env)))

    ax.plot(
        [0.0, max_x],
        [0.0, max_x],
        color="#7f7f7f",
        linestyle="--",
        lw=1.0,
        label="y=x",
    )
    ax.set_xlabel("-log10(expected)")
    ax.set_ylabel("-log10(observed)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _sanitize_label(label: Union[str, int]) -> str:
    text = str(label)
    text = text.replace(os.sep, "_")
    return text.replace(" ", "_")


def _select_programs(
    out: Mapping[str, Any],
    programs_to_plot: Union[str, Sequence[Union[str, int]], None],
    n_programs: int,
) -> List[Any]:
    betas_df = out.get("betas_df")
    if not isinstance(betas_df, pd.DataFrame):
        raise ValueError("out['betas_df'] must be a DataFrame to select programs.")

    if programs_to_plot is None:
        return []
    if isinstance(programs_to_plot, str):
        if programs_to_plot != "top_var":
            return [programs_to_plot]
        var = betas_df.var(axis=0).sort_values(ascending=False)
        return var.index[: int(n_programs)].tolist()

    selected: List[Any] = []
    for item in programs_to_plot:
        if isinstance(item, int):
            if item < 0 or item >= betas_df.shape[1]:
                raise ValueError("program index out of range.")
            selected.append(betas_df.columns[item])
        else:
            selected.append(item)
    return selected


def make_ntc_empirical_qq_plots(
    out: Dict[str, Any],
    out_dir: str,
    programs_to_plot: Union[str, Sequence[Union[str, int]], None] = "top_var",
    n_programs: int = 6,
    make_per_batch: bool = True,
    make_meta: bool = True,
    envelope_boot: int = 200,
    seed: int = 0,
) -> None:
    """
    Generate QQ plots for NTC empirical-null cross-fit diagnostics.
    """
    crossfit = out.get("ntc_crossfit")
    if crossfit is None:
        raise ValueError("out['ntc_crossfit'] is required; set qq_crossfit=True.")

    programs = _select_programs(out, programs_to_plot, n_programs)

    def _plot_genelevel(curves: Mapping[str, np.ndarray], filename: str) -> None:
        envelope = None
        ntc_holdout = curves.get("NTC holdout (B vs A)")
        if envelope_boot > 0 and ntc_holdout is not None:
            envelope = bootstrap_qq_envelope(
                ntc_holdout, n_boot=envelope_boot, seed=seed
            )
        plot_qq_curves(
            curves,
            os.path.join(out_dir, filename),
            "NTC empirical-null QQ (gene-level)",
            envelope=envelope,
        )

    def _plot_program(
        curves: Mapping[str, np.ndarray],
        filename: str,
        program: Any,
        seed_offset: int,
    ) -> None:
        envelope = None
        ntc_holdout = curves.get("NTC holdout (B vs A)")
        if envelope_boot > 0 and ntc_holdout is not None:
            envelope = bootstrap_qq_envelope(
                ntc_holdout, n_boot=envelope_boot, seed=seed + seed_offset
            )
        title = f"NTC empirical-null QQ (program {program})"
        plot_qq_curves(
            curves, os.path.join(out_dir, filename), title, envelope=envelope
        )

    if make_meta:
        meta_real = crossfit.get("meta_p_real_gene_vs_A")
        meta_ntc = crossfit.get("meta_p_ntcB_gene_vs_A")
        if meta_real is None or meta_ntc is None:
            raise ValueError("meta cross-fit outputs are missing.")
        curves = {
            "NTC holdout (B vs A)": np.asarray(meta_ntc),
            "Real genes (vs A)": np.asarray(meta_real),
        }
        _plot_genelevel(curves, "qq_ntc_empirical_genelevel_meta.png")

        if programs:
            meta_real_df = crossfit.get("meta_p_real_vs_A")
            meta_ntc_df = crossfit.get("meta_p_ntcB_vs_A")
            if not isinstance(meta_real_df, pd.DataFrame) or not isinstance(
                meta_ntc_df, pd.DataFrame
            ):
                raise ValueError("meta program-level outputs are missing.")
            for idx, program in enumerate(programs):
                if program not in meta_real_df.columns:
                    raise ValueError(
                        f"Program {program} not found in meta_p_real_vs_A."
                    )
                curves = {
                    "NTC holdout (B vs A)": meta_ntc_df[program].to_numpy(),
                    "Real genes (vs A)": meta_real_df[program].to_numpy(),
                }
                safe = _sanitize_label(program)
                _plot_program(
                    curves,
                    f"qq_ntc_empirical_program_{safe}_meta.png",
                    program,
                    seed_offset=7 + idx,
                )

    if make_per_batch:
        per_batch_real = crossfit.get("p_real_gene_vs_A_by_batch", {})
        per_batch_ntc = crossfit.get("p_ntcB_gene_vs_A_by_batch", {})
        per_batch_real_prog = crossfit.get("p_real_vs_A_by_batch", {})
        per_batch_ntc_prog = crossfit.get("p_ntcB_vs_A_by_batch", {})

        for batch_id in crossfit.get("batches", []):
            if batch_id not in per_batch_real or batch_id not in per_batch_ntc:
                continue
            batch_safe = _sanitize_label(batch_id)
            os.makedirs(os.path.join(out_dir, "per_batch", batch_safe), exist_ok=True)
            curves = {
                "NTC holdout (B vs A)": np.asarray(per_batch_ntc[batch_id]),
                "Real genes (vs A)": np.asarray(per_batch_real[batch_id]),
            }
            _plot_genelevel(
                curves,
                os.path.join("per_batch", batch_safe, "qq_ntc_empirical_genelevel.png"),
            )

            if programs:
                real_df = per_batch_real_prog.get(batch_id)
                ntc_df = per_batch_ntc_prog.get(batch_id)
                if not isinstance(real_df, pd.DataFrame) or not isinstance(
                    ntc_df, pd.DataFrame
                ):
                    continue
                for idx, program in enumerate(programs):
                    if program not in real_df.columns:
                        raise ValueError(
                            f"Program {program} not found in per-batch outputs."
                        )
                    curves = {
                        "NTC holdout (B vs A)": ntc_df[program].to_numpy(),
                        "Real genes (vs A)": real_df[program].to_numpy(),
                    }
                    safe = _sanitize_label(program)
                    _plot_program(
                        curves,
                        os.path.join(
                            "per_batch",
                            batch_safe,
                            f"qq_ntc_empirical_program_{safe}.png",
                        ),
                        program,
                        seed_offset=17 + idx,
                    )
