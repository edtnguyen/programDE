#!/usr/bin/env python
"""Two-stage CLR flooring sensitivity analysis for iPSC-EC programDE CRT.

This backend uses the six-source-covariate full-day model:
  * pct_counts_mt (identity)
  * biological_sample (one-hot encoded, one reference level dropped)
  * total_counts (log1p)
  * guides_per_cell (log1p)
  * guide_umi_counts (log1p)
  * n_genes_by_counts (log1p)

It independently varies:
  1. The initial exact-zero replacement floor applied to raw cNMF usages.
  2. The internal programDE lower-tail quantile floor used before CLR.

The transformation audit always evaluates the full Cartesian grid. The CRT can
run either an axis-isolating primary grid (all internal quantiles at the current
initial floor plus all initial-floor factors at the current internal quantile)
or the complete factorial grid. Only the empirical Bernoulli CRT is used; no
skew-normal or stratified-resampling analysis is performed.
"""

from __future__ import annotations

import os

for _variable in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_variable, "1")

import argparse
import gc
import hashlib
import inspect
import json
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import iPSC_EC_QQ_export_all_days as qqb


DEFAULT_DAYS = ["d0", "d1", "d2", "d3"]
DEFAULT_NTC_LABELS = ["SAFE", "non-targeting", "NTC"]


@dataclass
class RunRecord:
    day: str
    initial_floor_factor: float
    internal_eps_quantile: float
    initial_floor_label: str
    internal_quantile_label: str
    combination_label: str
    is_reference: bool
    status: str
    output_dir: str
    n_cells: int | None = None
    n_programs: int | None = None
    n_guides: int | None = None
    n_covariate_columns: int | None = None
    n_real_target_program_pairs: int | None = None
    n_ntc_pvalues_empirical: int | None = None
    raw_usage_zero_fraction: float | None = None
    base_raw_zero_floor: float | None = None
    applied_raw_zero_floor: float | None = None
    effective_internal_floor: float | None = None
    stage2_fraction_entries_modified: float | None = None
    stage2_fraction_positive_entries_modified: float | None = None
    stage2_fraction_zero_derived_entries_modified: float | None = None
    median_stage2_pseudomass_fraction: float | None = None
    runtime_seconds: float | None = None
    config_fingerprint: str | None = None
    message: str | None = None


def safe_float_token(value: float) -> str:
    text = f"{float(value):.12g}"
    return (
        text.replace("-", "m")
        .replace("+", "p")
        .replace(".", "_")
    )


def initial_floor_label(factor: float) -> str:
    return f"{safe_float_token(factor)}x"


def internal_quantile_label(quantile: float) -> str:
    value = float(quantile)
    if float_equal(value, 0.0):
        return "q0"
    if abs(value) < 0.01:
        text = f"{value:.0e}"
        text = text.replace("-", "m").replace("+", "p")
        return f"q{text}"
    return f"q{safe_float_token(value)}"


def combination_label(initial_factor: float, internal_quantile: float) -> str:
    return (
        f"initial_{initial_floor_label(initial_factor)}"
        f"__internal_{internal_quantile_label(internal_quantile)}"
    )


def float_equal(a: float, b: float, atol: float = 1e-15) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=atol))


def unique_floats(values: Iterable[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        current = float(value)
        if not any(float_equal(current, existing) for existing in result):
            result.append(current)
    return result


def json_normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(json_normalize(v) for v in value)
    if isinstance(value, Mapping):
        return {str(k): json_normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_normalize(v) for v in value]
    return value


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        json_normalize(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_normalize(payload), indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def git_commit(path: str | Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(path).expanduser()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit or None


def bh_qvalues(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float).ravel()
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    pv = np.clip(p[finite], 0.0, 1.0)
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_finite = np.empty_like(pv)
    q_finite[order] = np.clip(ranked, 0.0, 1.0)
    q[finite] = q_finite
    return q


def get_prepare_eps_quantile(programde: SimpleNamespace) -> tuple[bool, float | None]:
    signature = inspect.signature(programde.prepare_crt_inputs)
    parameter = signature.parameters.get("eps_quantile")
    if parameter is None:
        return False, None
    if parameter.default is inspect.Parameter.empty:
        return True, None
    try:
        return True, float(parameter.default)
    except (TypeError, ValueError):
        return True, None


def resolve_reference_quantile(
    requested: float | None,
    local_default: float | None,
) -> float:
    if requested is not None:
        reference = float(requested)
    elif local_default is not None:
        reference = float(local_default)
    else:
        raise RuntimeError(
            "Could not infer the current programDE eps_quantile default. "
            "Supply --reference-internal-eps-quantile explicitly."
        )
    if not np.isfinite(reference) or not 0.0 <= reference <= 1.0:
        raise ValueError(
            f"Reference internal eps quantile must be in [0, 1]; got {reference}."
        )
    return reference


def build_audit_combinations(
    initial_factors: Sequence[float],
    internal_quantiles: Sequence[float],
    reference_initial: float,
    reference_quantile: float,
) -> list[tuple[float, float]]:
    combinations = [
        (float(initial), float(quantile))
        for initial, quantile in product(initial_factors, internal_quantiles)
    ]
    combinations.sort(
        key=lambda pair: (
            0
            if float_equal(pair[0], reference_initial)
            and float_equal(pair[1], reference_quantile)
            else 1,
            pair[0],
            pair[1],
        )
    )
    return combinations


def build_crt_combinations(
    initial_factors: Sequence[float],
    internal_quantiles: Sequence[float],
    reference_initial: float,
    reference_quantile: float,
    mode: str,
) -> list[tuple[float, float]]:
    if mode == "full":
        return build_audit_combinations(
            initial_factors,
            internal_quantiles,
            reference_initial,
            reference_quantile,
        )
    if mode != "primary":
        raise ValueError(f"Unknown CRT grid mode: {mode!r}")

    combinations: list[tuple[float, float]] = []

    # Hold the initial floor fixed and vary the internal quantile.
    for quantile in internal_quantiles:
        combinations.append((float(reference_initial), float(quantile)))

    # Hold the internal quantile fixed and vary the initial floor.
    for initial in initial_factors:
        pair = (float(initial), float(reference_quantile))
        if not any(
            float_equal(pair[0], existing[0])
            and float_equal(pair[1], existing[1])
            for existing in combinations
        ):
            combinations.append(pair)

    combinations.sort(
        key=lambda pair: (
            0
            if float_equal(pair[0], reference_initial)
            and float_equal(pair[1], reference_quantile)
            else 1,
            0 if float_equal(pair[0], reference_initial) else 1,
            pair[0],
            pair[1],
        )
    )
    return combinations


def floor_and_normalize_raw_usage(
    usage_raw: pd.DataFrame,
    factor: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace exact zeros with a scaled day-specific floor, then row-normalize."""
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"Initial floor factor must be positive and finite; got {factor!r}")

    values = usage_raw.to_numpy(dtype=float, copy=True)
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"Invalid usage matrix shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Raw usage matrix contains non-finite values.")
    if np.any(values < 0):
        raise ValueError("Raw usage matrix contains negative values.")

    positive = values[values > 0]
    positive_q01 = float(np.quantile(positive, 0.01)) if positive.size else np.nan
    base_floor = max(1e-8, positive_q01 * 0.1) if positive.size else 1e-8
    applied_floor = base_floor * float(factor)

    zero_mask = values == 0
    zeros_per_cell = zero_mask.sum(axis=1)
    raw_row_sums = values.sum(axis=1)
    if np.any(raw_row_sums <= 0):
        bad = int((raw_row_sums <= 0).sum())
        raise ValueError(f"Raw usage contains {bad:,} all-zero rows.")

    floored_values = np.where(zero_mask, applied_floor, values)
    row_sums = floored_values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0) or not np.isfinite(row_sums).all():
        raise ValueError("Invalid row sums after initial exact-zero replacement.")
    normalized = floored_values / row_sums

    initial_added_mass = zeros_per_cell * applied_floor
    initial_pseudomass_fraction = initial_added_mass / (
        raw_row_sums + initial_added_mass
    )

    zero_derived_values = normalized[zero_mask]

    metadata = {
        "initial_floor_factor": float(factor),
        "initial_floor_label": initial_floor_label(factor),
        "n_cells": int(normalized.shape[0]),
        "n_programs": int(normalized.shape[1]),
        "n_positive_raw_entries": int(positive.size),
        "n_exact_zero_raw_entries": int(zero_mask.sum()),
        "fraction_exact_zero_raw_entries": float(zero_mask.mean()),
        "pct_cells_with_any_raw_zero": float(100.0 * (zeros_per_cell > 0).mean()),
        "median_raw_zeros_per_cell": float(np.median(zeros_per_cell)),
        "p90_raw_zeros_per_cell": float(np.quantile(zeros_per_cell, 0.90)),
        "p95_raw_zeros_per_cell": float(np.quantile(zeros_per_cell, 0.95)),
        "max_raw_zeros_per_cell": int(zeros_per_cell.max()),
        "positive_raw_usage_q01": positive_q01,
        "base_raw_zero_floor": float(base_floor),
        "applied_raw_zero_floor": float(applied_floor),
        "median_initial_pseudomass_fraction": float(
            np.median(initial_pseudomass_fraction)
        ),
        "p95_initial_pseudomass_fraction": float(
            np.quantile(initial_pseudomass_fraction, 0.95)
        ),
        "max_initial_pseudomass_fraction": float(
            np.max(initial_pseudomass_fraction)
        ),
        "median_zero_derived_value_after_stage1": float(
            np.median(zero_derived_values)
        )
        if zero_derived_values.size
        else np.nan,
        "p05_zero_derived_value_after_stage1": float(
            np.quantile(zero_derived_values, 0.05)
        )
        if zero_derived_values.size
        else np.nan,
        "p95_zero_derived_value_after_stage1": float(
            np.quantile(zero_derived_values, 0.95)
        )
        if zero_derived_values.size
        else np.nan,
    }

    return (
        pd.DataFrame(
            normalized,
            index=usage_raw.index,
            columns=usage_raw.columns,
        ),
        metadata,
    )


def stage2_floor_metrics_and_manual_clr(
    stage1_usage: pd.DataFrame,
    usage_raw: pd.DataFrame,
    internal_quantile: float,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    """Reproduce programDE's internal floor and return stage-specific diagnostics."""
    quantile = float(internal_quantile)
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError(f"Internal eps quantile must be in [0, 1]; got {quantile}.")

    values = stage1_usage.to_numpy(dtype=float, copy=False)
    raw_values = usage_raw.to_numpy(dtype=float, copy=False)
    if values.shape != raw_values.shape:
        raise ValueError("Stage-1 and raw usage matrices have different shapes.")

    internal_floor = float(np.quantile(values, quantile))
    if not np.isfinite(internal_floor) or internal_floor <= 0.0:
        internal_floor = float(np.finfo(np.float64).tiny)

    raw_zero_mask = raw_values == 0
    raw_positive_mask = raw_values > 0
    modified_mask = values < internal_floor
    modified_positive_mask = modified_mask & raw_positive_mask
    modified_zero_derived_mask = modified_mask & raw_zero_mask

    added = np.maximum(internal_floor - values, 0.0)
    added_per_cell = added.sum(axis=1)
    stage2_pseudomass_fraction = added_per_cell / (1.0 + added_per_cell)

    floored = np.maximum(values, internal_floor)
    floored = floored / floored.sum(axis=1, keepdims=True)
    log_values = np.log(floored)
    manual_clr = log_values - log_values.mean(axis=1, keepdims=True)

    modified_per_cell = modified_mask.sum(axis=1)
    modified_positive_per_cell = modified_positive_mask.sum(axis=1)
    modified_zero_per_cell = modified_zero_derived_mask.sum(axis=1)

    zero_derived_values = values[raw_zero_mask]
    modified_positive_values = values[modified_positive_mask]
    positive_uplift = (
        internal_floor / modified_positive_values
        if modified_positive_values.size
        else np.empty(0, dtype=float)
    )

    median_zero_value = (
        float(np.median(zero_derived_values))
        if zero_derived_values.size
        else np.nan
    )

    summary = {
        "internal_eps_quantile": quantile,
        "internal_quantile_label": internal_quantile_label(quantile),
        "effective_internal_floor": internal_floor,
        "stage2_n_entries_modified": int(modified_mask.sum()),
        "stage2_fraction_entries_modified": float(modified_mask.mean()),
        "stage2_n_originally_positive_entries_modified": int(
            modified_positive_mask.sum()
        ),
        "stage2_fraction_originally_positive_entries_modified": float(
            modified_positive_mask.sum() / raw_positive_mask.sum()
        )
        if raw_positive_mask.any()
        else np.nan,
        "stage2_n_zero_derived_entries_modified": int(
            modified_zero_derived_mask.sum()
        ),
        "stage2_fraction_zero_derived_entries_modified": float(
            modified_zero_derived_mask.sum() / raw_zero_mask.sum()
        )
        if raw_zero_mask.any()
        else np.nan,
        "stage2_pct_cells_with_any_entry_modified": float(
            100.0 * (modified_per_cell > 0).mean()
        ),
        "stage2_pct_cells_with_any_originally_positive_entry_modified": float(
            100.0 * (modified_positive_per_cell > 0).mean()
        ),
        "stage2_median_entries_modified_per_cell": float(
            np.median(modified_per_cell)
        ),
        "stage2_p95_entries_modified_per_cell": float(
            np.quantile(modified_per_cell, 0.95)
        ),
        "stage2_median_positive_entries_modified_per_cell": float(
            np.median(modified_positive_per_cell)
        ),
        "stage2_p95_positive_entries_modified_per_cell": float(
            np.quantile(modified_positive_per_cell, 0.95)
        ),
        "median_stage2_pseudomass_fraction": float(
            np.median(stage2_pseudomass_fraction)
        ),
        "p95_stage2_pseudomass_fraction": float(
            np.quantile(stage2_pseudomass_fraction, 0.95)
        ),
        "max_stage2_pseudomass_fraction": float(
            np.max(stage2_pseudomass_fraction)
        ),
        "median_zero_derived_value_before_stage2": median_zero_value,
        "internal_floor_over_median_zero_derived_value": float(
            internal_floor / median_zero_value
        )
        if np.isfinite(median_zero_value) and median_zero_value > 0
        else np.nan,
        "median_uplift_factor_for_modified_positive_entries": float(
            np.median(positive_uplift)
        )
        if positive_uplift.size
        else np.nan,
        "p95_uplift_factor_for_modified_positive_entries": float(
            np.quantile(positive_uplift, 0.95)
        )
        if positive_uplift.size
        else np.nan,
    }

    programs = list(map(str, stage1_usage.columns))
    program_rows: list[dict[str, Any]] = []
    for j, program in enumerate(programs):
        raw_pos_j = raw_positive_mask[:, j]
        raw_zero_j = raw_zero_mask[:, j]
        modified_j = modified_mask[:, j]
        modified_pos_j = modified_positive_mask[:, j]
        modified_zero_j = modified_zero_derived_mask[:, j]
        added_j = added[:, j]
        program_rows.append(
            {
                "program_name": program,
                "stage2_fraction_entries_modified": float(modified_j.mean()),
                "stage2_fraction_originally_positive_entries_modified": float(
                    modified_pos_j.sum() / raw_pos_j.sum()
                )
                if raw_pos_j.any()
                else np.nan,
                "stage2_fraction_zero_derived_entries_modified": float(
                    modified_zero_j.sum() / raw_zero_j.sum()
                )
                if raw_zero_j.any()
                else np.nan,
                "stage2_total_added_mass_before_renormalization": float(
                    added_j.sum()
                ),
                "stage2_median_added_mass_per_cell": float(np.median(added_j)),
                "stage2_p95_added_mass_per_cell": float(
                    np.quantile(added_j, 0.95)
                ),
            }
        )

    cell_diagnostics = {
        "raw_zeros_per_cell": raw_zero_mask.sum(axis=1).astype(float),
        "stage2_modified_entries_per_cell": modified_per_cell.astype(float),
        "stage2_modified_positive_entries_per_cell": modified_positive_per_cell.astype(
            float
        ),
        "stage2_modified_zero_derived_entries_per_cell": modified_zero_per_cell.astype(
            float
        ),
        "stage2_pseudomass_fraction_per_cell": stage2_pseudomass_fraction.astype(
            float
        ),
    }

    return manual_clr, summary, pd.DataFrame(program_rows), cell_diagnostics


def align_raw_usage(
    usage_path: Path,
    obs_names: pd.Index,
) -> tuple[pd.DataFrame, int]:
    usage_raw = pd.read_csv(usage_path, sep="\t", index_col=0)
    usage_raw.index = usage_raw.index.astype(str)
    usage_raw.columns = usage_raw.columns.astype(str)
    obs_as_str = pd.Index(obs_names.astype(str))
    matched = int(usage_raw.index.intersection(obs_as_str).size)

    aligned = usage_raw.reindex(obs_as_str)
    fully_missing = aligned.isna().all(axis=1)
    if fully_missing.any():
        examples = list(aligned.index[fully_missing][:5])
        raise ValueError(
            f"{int(fully_missing.sum()):,} H5MU cells are missing entirely from "
            f"{usage_path}; first few: {examples}."
        )
    if aligned.isna().any().any():
        n_missing = int(aligned.isna().sum().sum())
        raise ValueError(f"Usage alignment produced {n_missing:,} partial missing entries.")

    aligned.index = obs_names
    return aligned, matched


def prepare_inputs(
    programde: SimpleNamespace,
    adata: Any,
    eps_quantile: float,
) -> Any:
    signature = inspect.signature(programde.prepare_crt_inputs)
    if "eps_quantile" not in signature.parameters:
        raise TypeError(
            "The local programDE prepare_crt_inputs() has no eps_quantile parameter."
        )
    kwargs: dict[str, Any] = {
        "adata": adata,
        "usage_key": "cnmf_usage",
        "covar_key": "covar",
        "guide_assignment_key": "guide_assignment",
        "guide2gene_key": "guide2gene",
        "eps_quantile": float(eps_quantile),
    }
    if "numeric_as_category_threshold" in signature.parameters:
        kwargs["numeric_as_category_threshold"] = None
    return programde.prepare_crt_inputs(**kwargs)


def iter_object_fields(obj: Any) -> list[tuple[str, Any]]:
    fields: list[tuple[str, Any]] = []
    if isinstance(obj, Mapping):
        fields.extend((str(k), v) for k, v in obj.items())
    if hasattr(obj, "_asdict"):
        try:
            fields.extend((str(k), v) for k, v in obj._asdict().items())
        except Exception:
            pass
    if hasattr(obj, "__dataclass_fields__"):
        for name in getattr(obj, "__dataclass_fields__", {}):
            try:
                fields.append((str(name), getattr(obj, name)))
            except Exception:
                pass
    try:
        fields.extend((str(k), v) for k, v in vars(obj).items())
    except TypeError:
        pass

    seen: set[str] = set()
    unique: list[tuple[str, Any]] = []
    for name, value in fields:
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, value))
    return unique


def extract_response_matrix(
    inputs: Any,
    n_cells: int,
    n_programs: int,
) -> tuple[np.ndarray, str]:
    candidates: list[tuple[int, str, np.ndarray]] = []
    preferred_tokens = ["y", "response", "outcome", "clr", "usage"]

    for name, value in iter_object_fields(inputs):
        try:
            if hasattr(value, "to_numpy"):
                array = value.to_numpy(dtype=float)
            else:
                array = np.asarray(value, dtype=float)
        except Exception:
            continue
        if array.ndim != 2:
            continue
        if array.shape == (n_cells, n_programs):
            normalized = array
        elif array.shape == (n_programs, n_cells):
            normalized = array.T
        else:
            continue
        lower_name = name.lower()
        score = 0
        for index, token in enumerate(preferred_tokens):
            if lower_name == token:
                score += 100 - index
            elif token in lower_name:
                score += 50 - index
        candidates.append((score, name, normalized))

    if not candidates:
        observed_shapes = []
        for name, value in iter_object_fields(inputs):
            try:
                observed_shapes.append((name, np.asarray(value).shape))
            except Exception:
                pass
        raise RuntimeError(
            "Could not locate the response matrix in prepare_crt_inputs output. "
            f"Expected {(n_cells, n_programs)}; observed {observed_shapes}."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, field_name, matrix = candidates[0]
    matrix = np.asarray(matrix, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"Extracted response matrix {field_name!r} contains non-finite values."
        )
    return matrix, field_name


def rowwise_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    result = np.full(x.shape[0], np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def safe_flat_pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return np.nan
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def safe_flat_spearman(
    x: np.ndarray,
    y: np.ndarray,
    max_entries: int = 300_000,
) -> float:
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if len(a) > max_entries:
        rng = np.random.default_rng(1)
        indices = rng.choice(len(a), size=max_entries, replace=False)
        a = a[indices]
        b = b[indices]
    if len(a) < 3:
        return np.nan
    a_rank = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    b_rank = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.std(a_rank) == 0 or np.std(b_rank) == 0:
        return np.nan
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def column_response_metrics(
    current: np.ndarray,
    reference: np.ndarray,
    program_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for j, program in enumerate(program_names):
        current_j = current[:, j]
        reference_j = reference[:, j]
        delta = current_j - reference_j
        rows.append(
            {
                "program_name": str(program),
                "response_pearson_vs_reference": safe_flat_pearson(
                    current_j,
                    reference_j,
                ),
                "response_spearman_vs_reference": safe_flat_spearman(
                    current_j,
                    reference_j,
                    max_entries=len(current_j),
                ),
                "response_rmse_vs_reference": float(
                    np.sqrt(np.mean(delta**2))
                ),
                "response_median_abs_change_vs_reference": float(
                    np.median(np.abs(delta))
                ),
                "response_p95_abs_change_vs_reference": float(
                    np.quantile(np.abs(delta), 0.95)
                ),
            }
        )
    return pd.DataFrame(rows)


def make_target_program_zero_burden(
    adata: Any,
    raw_usage: pd.DataFrame,
    ntc_labels: list[str],
) -> pd.DataFrame:
    guide_assignment = adata.obsm["guide_assignment"]
    guide_names = list(map(str, adata.uns["guide_names"]))
    guide2gene = {str(k): str(v) for k, v in adata.uns["guide2gene"].items()}

    if isinstance(guide_assignment, pd.DataFrame):
        guide_df = guide_assignment.copy()
        guide_df.index = adata.obs_names
        guide_df.columns = guide_df.columns.astype(str)
        guide_df = guide_df.reindex(columns=guide_names, fill_value=0)
        guide_values = guide_df.to_numpy(dtype=bool, copy=False)
    else:
        guide_values = np.asarray(guide_assignment, dtype=bool)
        if guide_values.shape[1] != len(guide_names):
            raise ValueError(
                "Guide assignment matrix column count does not match guide_names."
            )

    target_to_indices: dict[str, list[int]] = {}
    ntc_keys = {str(value).strip().lower() for value in ntc_labels}
    for index, guide in enumerate(guide_names):
        target = guide2gene.get(guide, guide.split("_", 1)[0])
        if str(target).strip().lower() in ntc_keys:
            continue
        target_to_indices.setdefault(str(target), []).append(index)

    zero_mask = raw_usage.to_numpy(dtype=float, copy=False) == 0
    programs = list(map(str, raw_usage.columns))
    rows: list[pd.DataFrame] = []

    for target, indices in target_to_indices.items():
        treated = guide_values[:, indices].any(axis=1)
        n_treated = int(treated.sum())
        n_control = int((~treated).sum())
        if n_treated == 0 or n_control == 0:
            continue
        treated_fraction = zero_mask[treated].mean(axis=0)
        control_fraction = zero_mask[~treated].mean(axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "target_name": target,
                    "program_name": programs,
                    "n_treated_cells": n_treated,
                    "n_control_cells": n_control,
                    "zero_fraction_treated": treated_fraction,
                    "zero_fraction_control": control_fraction,
                    "zero_fraction_treated_minus_control": (
                        treated_fraction - control_fraction
                    ),
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "target_name",
                "program_name",
                "n_treated_cells",
                "n_control_cells",
                "zero_fraction_treated",
                "zero_fraction_control",
                "zero_fraction_treated_minus_control",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def save_wide(frame: pd.DataFrame, path: Path) -> None:
    current = frame.copy()
    current.index = current.index.astype(str)
    current.columns = current.columns.astype(str)
    current.index.name = "target_name"
    path.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(path, sep="\t")


def flatten_ntc_frames(
    frames: Mapping[Any, pd.DataFrame] | None,
) -> np.ndarray:
    if not frames:
        return np.empty(0, dtype=float)
    arrays = [
        frame.to_numpy(dtype=float, copy=False).ravel()
        for frame in frames.values()
    ]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=float)


def call_empirical_real_crt(
    programde: SimpleNamespace,
    inputs: Any,
    B: int,
    n_jobs: int,
    real_crt_seed: int,
) -> tuple[dict[str, Any], str | None]:
    """Run the uncalibrated empirical CRT with the Bernoulli index sampler.

    Current programDE returns the empirical p-value matrix as ``pvals_df`` when
    skew-normal calibration is disabled.  Some older checkouts expose extra
    return flags, so they are set only when present and are kept false to avoid
    requesting skew-normal/raw-calibration outputs.
    """
    kwargs: dict[str, Any] = {
        "inputs": inputs,
        "B": B,
        "n_jobs": n_jobs,
        "calibrate_skew_normal": False,
    }
    signature = inspect.signature(programde.run_all_genes_union_crt)
    if "return_raw_pvals" in signature.parameters:
        kwargs["return_raw_pvals"] = False
    if "return_skew_normal" in signature.parameters:
        kwargs["return_skew_normal"] = False
    if "resampling_method" in signature.parameters:
        kwargs["resampling_method"] = "bernoulli_index"
    seed_parameter = None
    for candidate in ["base_seed", "seed0", "seed", "random_state"]:
        if candidate in signature.parameters:
            kwargs[candidate] = real_crt_seed
            seed_parameter = candidate
            break
    return programde.run_all_genes_union_crt(**kwargs), seed_parameter


def expected_run_files(run_dir: Path, day: str) -> list[Path]:
    return [
        run_dir / f"{day}_target_betas_wide.tsv",
        run_dir / f"{day}_target_pvalues_empirical_wide.tsv",
        run_dir / f"{day}_target_treated_counts.tsv",
        run_dir / f"{day}_ntc_pseudotarget_pvalues_empirical.npy",
        run_dir / f"{day}_run_config.json",
    ]


def matching_complete_run(
    run_dir: Path,
    day: str,
    fingerprint: str,
) -> bool:
    config_path = run_dir / f"{day}_run_config.json"
    if not all(path.exists() for path in expected_run_files(run_dir, day)):
        return False
    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return False
    return config.get("config_fingerprint") == fingerprint


def count_ntc_groups(ntc_groups: Any) -> dict[str, int | None]:
    n_ensembles: int | None = None
    total_groups: int | None = None
    try:
        n_ensembles = len(ntc_groups)
    except Exception:
        pass
    try:
        if isinstance(ntc_groups, Mapping):
            total_groups = sum(len(value) for value in ntc_groups.values())
        else:
            total_groups = sum(len(value) for value in ntc_groups)
    except Exception:
        pass
    return {
        "n_ensembles": n_ensembles,
        "total_groups_across_ensembles": total_groups,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two-stage cNMF CLR flooring sensitivity with the six-covariate "
            "iPSC-EC Bernoulli CRT."
        )
    )
    parser.add_argument("--programde-root", default=qqb.DEFAULT_PROGRAMDE_ROOT)
    parser.add_argument("--data-dir", default=qqb.DEFAULT_DATA_DIR)
    parser.add_argument("--cnmf-dir", default=qqb.DEFAULT_CNMF_DIR)
    parser.add_argument("--cnmf-file", default=None)
    parser.add_argument("--usage-file-template", default=None)
    parser.add_argument(
        "--guide-long-csv-template",
        default=qqb.DEFAULT_GUIDE_LONG_TEMPLATE,
    )
    parser.add_argument(
        "--guide-metadata-tsv",
        default="./300genes_guide_metadata_v43.tsv",
    )
    parser.add_argument("--guide-metadata-guide-col", default=None)
    parser.add_argument("--guide-metadata-target-col", default=None)
    parser.add_argument(
        "--out-dir",
        default="results/ipsc_ec_clr_two_stage_floor_sensitivity_6cov",
    )
    parser.add_argument("--days", nargs="+", default=DEFAULT_DAYS)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--density-threshold", default="0_2")
    parser.add_argument("--gex-modality", default=None)
    parser.add_argument("--guide-modality", default=None)
    parser.add_argument(
        "--initial-floor-factors",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0],
    )
    parser.add_argument(
        "--internal-eps-quantiles",
        nargs="+",
        type=float,
        default=[0.0, 1e-5, 1e-4, 1e-3],
    )
    parser.add_argument(
        "--reference-initial-floor-factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--reference-internal-eps-quantile",
        type=float,
        default=None,
        help=(
            "Current internal programDE flooring quantile. Omit to use the local "
            "prepare_crt_inputs default."
        ),
    )
    parser.add_argument(
        "--crt-grid-mode",
        choices=["primary", "full"],
        default="primary",
        help=(
            "primary: vary each flooring stage separately around the current "
            "reference; full: run the entire Cartesian grid."
        ),
    )
    parser.add_argument("--B", type=int, default=10000)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument("--real-crt-seed", type=int, default=123)
    parser.add_argument("--ntc-label", action="append", default=None)
    parser.add_argument("--ntc-group-size", type=int, default=6)
    parser.add_argument("--ntc-ensembles", type=int, default=10)
    parser.add_argument("--ntc-bins", type=int, default=10)
    parser.add_argument("--ntc-max-groups", type=int, default=None)
    parser.add_argument("--ntc-group-seed", type=int, default=7)
    parser.add_argument("--ntc-crt-seed", type=int, default=23)
    parser.add_argument("--require-covariates", action="store_true")
    parser.add_argument("--reuse-matching", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-crt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.ntc_label is None:
        args.ntc_label = DEFAULT_NTC_LABELS.copy()
    if args.B < 1:
        raise ValueError("--B must be >= 1")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be >= 1")

    args.initial_floor_factors = unique_floats(args.initial_floor_factors)
    args.internal_eps_quantiles = unique_floats(args.internal_eps_quantiles)

    if not all(
        np.isfinite(value) and value > 0
        for value in args.initial_floor_factors
    ):
        raise ValueError("All initial-floor factors must be positive and finite.")
    if not all(
        np.isfinite(value) and 0.0 <= value <= 1.0
        for value in args.internal_eps_quantiles
    ):
        raise ValueError("All internal eps quantiles must lie in [0, 1].")
    if not np.isfinite(args.reference_initial_floor_factor) or (
        args.reference_initial_floor_factor <= 0
    ):
        raise ValueError("Reference initial-floor factor must be positive.")
    if args.reuse_matching and args.overwrite:
        raise ValueError("Choose either --reuse-matching or --overwrite, not both.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    programde = qqb.load_programde(args.programde_root)
    has_eps_quantile, local_default_quantile = get_prepare_eps_quantile(programde)
    if not has_eps_quantile:
        raise RuntimeError(
            "The local programDE prepare_crt_inputs() does not expose eps_quantile."
        )

    reference_quantile = resolve_reference_quantile(
        args.reference_internal_eps_quantile,
        local_default_quantile,
    )
    reference_initial = float(args.reference_initial_floor_factor)

    initial_factors = unique_floats(
        [*args.initial_floor_factors, reference_initial]
    )
    internal_quantiles = unique_floats(
        [*args.internal_eps_quantiles, reference_quantile]
    )
    initial_factors.sort()
    internal_quantiles.sort()

    audit_combinations = build_audit_combinations(
        initial_factors,
        internal_quantiles,
        reference_initial,
        reference_quantile,
    )
    crt_combinations = build_crt_combinations(
        initial_factors,
        internal_quantiles,
        reference_initial,
        reference_quantile,
        args.crt_grid_mode,
    )

    density_token = qqb.normalize_density_token(args.density_threshold)
    model_root = Path(args.out_dir).expanduser() / qqb.model_tag(
        args.k,
        density_token,
    )
    model_root.mkdir(parents=True, exist_ok=True)

    programde_commit = git_commit(args.programde_root)
    programde_prepare_source = inspect.getsourcefile(programde.prepare_crt_inputs)
    programde_prepare_signature = str(
        inspect.signature(programde.prepare_crt_inputs)
    )

    plan_rows = []
    crt_set = {
        (float(initial), float(quantile))
        for initial, quantile in crt_combinations
    }
    for initial, quantile in audit_combinations:
        plan_rows.append(
            {
                "initial_floor_factor": initial,
                "initial_floor_label": initial_floor_label(initial),
                "internal_eps_quantile": quantile,
                "internal_quantile_label": internal_quantile_label(quantile),
                "combination_label": combination_label(initial, quantile),
                "is_reference": bool(
                    float_equal(initial, reference_initial)
                    and float_equal(quantile, reference_quantile)
                ),
                "run_transformation_audit": True,
                "run_empirical_crt": (initial, quantile) in crt_set
                and not args.skip_crt,
            }
        )
    plan = pd.DataFrame(plan_rows)
    plan_path = model_root / "two_stage_flooring_planned_combinations.csv"
    plan.to_csv(plan_path, index=False)

    print("iPSC-EC two-stage CLR floor sensitivity: six-covariate model")
    print("days:", args.days)
    print("initial-floor factors:", initial_factors)
    print("internal eps quantiles:", internal_quantiles)
    print("reference initial factor:", reference_initial)
    print("reference internal quantile:", reference_quantile)
    print("local programDE default quantile:", local_default_quantile)
    print("CRT grid mode:", args.crt_grid_mode)
    print("audit combinations:", len(audit_combinations))
    print("CRT combinations:", 0 if args.skip_crt else len(crt_combinations))
    print("B:", args.B, "n_jobs:", args.n_jobs)
    print("model root:", model_root)
    print("programDE commit:", programde_commit)
    print("planned combinations:", plan_path)

    helper_args = SimpleNamespace(
        cnmf_file=args.cnmf_file,
        usage_file_template=args.usage_file_template,
        cnmf_dir=args.cnmf_dir,
        k=args.k,
    )

    if args.dry_run:
        path_rows = []
        for day in args.days:
            path_rows.append(
                {
                    "day": day,
                    "h5mu": str(
                        qqb.h5mu_path_for_day(Path(args.data_dir), day)
                    ),
                    "guide_csv": str(
                        qqb.guide_long_path_for_day(
                            args.guide_long_csv_template,
                            day,
                        )
                    ),
                    "usage_candidates": ";".join(
                        str(path)
                        for path in qqb.usage_candidates_from_args(
                            helper_args,
                            day,
                            density_token,
                        )[:10]
                    ),
                }
            )
        path_table = pd.DataFrame(path_rows)
        path_table.to_csv(model_root / "dry_run_paths.csv", index=False)
        print(path_table.to_string(index=False))
        print("\nCombination plan:")
        print(plan.to_string(index=False))
        return 0

    import muon as mu

    run_records: list[RunRecord] = []
    audit_rows: list[dict[str, Any]] = []
    program_audit_rows: list[pd.DataFrame] = []
    zero_burden_paths: list[dict[str, str]] = []
    ntc_group_rows: list[dict[str, Any]] = []

    for day in args.days:
        print(f"\n{'=' * 100}\n{day}\n{'=' * 100}")
        day_started = time.time()

        h5mu_path = qqb.h5mu_path_for_day(
            Path(args.data_dir).expanduser(),
            day,
        )
        guide_csv = qqb.guide_long_path_for_day(
            args.guide_long_csv_template,
            day,
        )
        if not h5mu_path.exists():
            raise FileNotFoundError(f"Missing H5MU for {day}: {h5mu_path}")

        mdata = mu.read(h5mu_path)
        gex_modality = qqb.resolve_modality(
            mdata,
            args.gex_modality,
            ["GEX", "gex", "rna", "RNA"],
            "GEX/RNA",
        )
        guide_modality = qqb.resolve_modality(
            mdata,
            args.guide_modality,
            ["guide", "Guide", "gRNA", "grna"],
            "guide",
        )
        adata = mdata[gex_modality].copy()

        usage_candidates = qqb.usage_candidates_from_args(
            helper_args,
            day,
            density_token,
        )
        usage_path, matched_usage_cells = qqb.choose_usage_file(
            usage_candidates,
            pd.Index(adata.obs_names),
            day,
        )
        usage_raw, matched_usage_cells = align_raw_usage(
            usage_path,
            pd.Index(adata.obs_names),
        )

        guide_names, _ = qqb.load_and_attach_guides(
            adata,
            mdata,
            guide_modality,
            guide_csv,
            args.ntc_label,
            args.guide_metadata_tsv,
            args.guide_metadata_guide_col,
            args.guide_metadata_target_col,
        )
        covariates = qqb.attach_covariates(
            adata,
            require_covariates=args.require_covariates,
        )

        print(
            f"cells={adata.n_obs:,}; programs={usage_raw.shape[1]:,}; "
            f"guides={len(guide_names):,}; matched_usage_cells={matched_usage_cells:,}"
        )
        print("covariate columns:", list(covariates.columns))

        zero_burden = make_target_program_zero_burden(
            adata,
            usage_raw,
            args.ntc_label,
        )
        zero_burden.insert(0, "day", day)
        zero_burden_path = (
            model_root
            / "zero_burden"
            / f"{day}_target_program_zero_burden.tsv"
        )
        zero_burden_path.parent.mkdir(parents=True, exist_ok=True)
        zero_burden.to_csv(zero_burden_path, sep="\t", index=False)
        zero_burden_paths.append({"day": day, "path": str(zero_burden_path)})

        def prepare_variant(
            initial_factor: float,
            internal_quantile: float,
        ) -> dict[str, Any]:
            stage1_usage, stage1_meta = floor_and_normalize_raw_usage(
                usage_raw,
                initial_factor,
            )
            stage1_usage.index = adata.obs_names
            manual_clr, stage2_meta, stage2_program, cell_diag = (
                stage2_floor_metrics_and_manual_clr(
                    stage1_usage,
                    usage_raw,
                    internal_quantile,
                )
            )

            adata.obsm["cnmf_usage"] = stage1_usage
            adata.uns["program_names"] = list(map(str, stage1_usage.columns))
            inputs = prepare_inputs(
                programde,
                adata,
                eps_quantile=internal_quantile,
            )
            response, response_field = extract_response_matrix(
                inputs,
                n_cells=adata.n_obs,
                n_programs=stage1_usage.shape[1],
            )
            response = np.asarray(response, dtype=float).copy()

            manual_difference = np.abs(response - manual_clr)
            max_manual_difference = float(np.max(manual_difference))
            rmse_manual_difference = float(
                np.sqrt(np.mean(manual_difference**2))
            )
            if not np.allclose(
                response,
                manual_clr,
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            ):
                raise RuntimeError(
                    "The extracted programDE response does not match the manually "
                    "reconstructed two-stage CLR transform. Maximum absolute "
                    f"difference={max_manual_difference:.6g}. Inspect the local "
                    "clr_from_usage implementation before interpreting this audit."
                )

            combo = combination_label(initial_factor, internal_quantile)
            stage2_program = stage2_program.copy()
            stage2_program.insert(0, "combination_label", combo)
            stage2_program.insert(
                0,
                "internal_quantile_label",
                internal_quantile_label(internal_quantile),
            )
            stage2_program.insert(0, "internal_eps_quantile", internal_quantile)
            stage2_program.insert(
                0,
                "initial_floor_label",
                initial_floor_label(initial_factor),
            )
            stage2_program.insert(0, "initial_floor_factor", initial_factor)
            stage2_program.insert(0, "day", day)

            return {
                "inputs": inputs,
                "response": response,
                "response_field": response_field,
                "stage1_usage": stage1_usage,
                "stage1_meta": stage1_meta,
                "stage2_meta": stage2_meta,
                "stage2_program": stage2_program,
                "cell_diag": cell_diag,
                "manual_vs_programde_max_abs_difference": max_manual_difference,
                "manual_vs_programde_rmse": rmse_manual_difference,
            }

        reference = prepare_variant(reference_initial, reference_quantile)
        reference_response = reference["response"].copy()
        program_names = list(map(str, usage_raw.columns))

        ntc_guides, guide_freq, guide_to_bin, real_signatures = (
            programde.build_ntc_group_inputs(
                inputs=reference["inputs"],
                ntc_label=args.ntc_label,
                group_size=args.ntc_group_size,
                n_bins=args.ntc_bins,
            )
        )
        ntc_groups = programde.make_ntc_groups_ensemble(
            ntc_guides=ntc_guides,
            ntc_freq=guide_freq,
            real_gene_bin_sigs=real_signatures,
            guide_to_bin=guide_to_bin,
            n_ensemble=args.ntc_ensembles,
            seed0=args.ntc_group_seed,
            group_size=args.ntc_group_size,
            max_groups=args.ntc_max_groups,
        )
        ntc_group_summary = count_ntc_groups(ntc_groups)
        ntc_group_rows.append(
            {
                "day": day,
                "n_ntc_guides": int(len(ntc_guides)),
                "ntc_group_size": args.ntc_group_size,
                "ntc_bins": args.ntc_bins,
                "ntc_ensembles_requested": args.ntc_ensembles,
                "ntc_group_seed": args.ntc_group_seed,
                **ntc_group_summary,
            }
        )

        for initial_factor, internal_quantile in audit_combinations:
            is_reference = bool(
                float_equal(initial_factor, reference_initial)
                and float_equal(internal_quantile, reference_quantile)
            )
            current = (
                reference
                if is_reference
                else prepare_variant(initial_factor, internal_quantile)
            )

            response = current["response"]
            delta = response - reference_response
            absolute_delta = np.abs(delta)
            cell_rmse = np.sqrt(np.mean(delta**2, axis=1))
            cell_similarity = rowwise_cosine(response, reference_response)

            cell_diag = current["cell_diag"]
            audit_rows.append(
                {
                    "day": day,
                    "initial_floor_factor": float(initial_factor),
                    "initial_floor_label": initial_floor_label(initial_factor),
                    "internal_eps_quantile": float(internal_quantile),
                    "internal_quantile_label": internal_quantile_label(
                        internal_quantile
                    ),
                    "combination_label": combination_label(
                        initial_factor,
                        internal_quantile,
                    ),
                    "is_reference": is_reference,
                    "reference_initial_floor_factor": reference_initial,
                    "reference_internal_eps_quantile": reference_quantile,
                    **current["stage1_meta"],
                    **current["stage2_meta"],
                    "response_matrix_field": current["response_field"],
                    "manual_vs_programde_max_abs_difference": current[
                        "manual_vs_programde_max_abs_difference"
                    ],
                    "manual_vs_programde_rmse": current[
                        "manual_vs_programde_rmse"
                    ],
                    "response_entrywise_pearson_vs_reference": safe_flat_pearson(
                        response,
                        reference_response,
                    ),
                    "response_entrywise_spearman_vs_reference": safe_flat_spearman(
                        response,
                        reference_response,
                    ),
                    "response_rmse_vs_reference": float(
                        np.sqrt(np.mean(delta**2))
                    ),
                    "response_median_abs_change_vs_reference": float(
                        np.median(absolute_delta)
                    ),
                    "response_p95_abs_change_vs_reference": float(
                        np.quantile(absolute_delta, 0.95)
                    ),
                    "response_p99_abs_change_vs_reference": float(
                        np.quantile(absolute_delta, 0.99)
                    ),
                    "median_cell_response_cosine_vs_reference": float(
                        np.nanmedian(cell_similarity)
                    ),
                    "p05_cell_response_cosine_vs_reference": float(
                        np.nanquantile(cell_similarity, 0.05)
                    ),
                    "spearman_raw_zeros_per_cell_vs_cell_response_rmse": safe_flat_spearman(
                        cell_diag["raw_zeros_per_cell"],
                        cell_rmse,
                        max_entries=len(cell_rmse),
                    ),
                    "spearman_stage2_modified_entries_per_cell_vs_cell_response_rmse": safe_flat_spearman(
                        cell_diag["stage2_modified_entries_per_cell"],
                        cell_rmse,
                        max_entries=len(cell_rmse),
                    ),
                    "spearman_stage2_modified_positive_entries_per_cell_vs_cell_response_rmse": safe_flat_spearman(
                        cell_diag["stage2_modified_positive_entries_per_cell"],
                        cell_rmse,
                        max_entries=len(cell_rmse),
                    ),
                    "spearman_stage2_pseudomass_vs_cell_response_rmse": safe_flat_spearman(
                        cell_diag["stage2_pseudomass_fraction_per_cell"],
                        cell_rmse,
                        max_entries=len(cell_rmse),
                    ),
                    "usage_path": str(usage_path),
                    "h5mu_path": str(h5mu_path),
                }
            )

            program_metrics = column_response_metrics(
                response,
                reference_response,
                program_names,
            )
            program_metrics = current["stage2_program"].merge(
                program_metrics,
                on="program_name",
                how="left",
                validate="one_to_one",
            )
            program_metrics["is_reference"] = is_reference
            program_audit_rows.append(program_metrics)

            if not is_reference:
                del current
                gc.collect()

        if not args.skip_crt:
            for initial_factor, internal_quantile in crt_combinations:
                run_started = time.time()
                is_reference = bool(
                    float_equal(initial_factor, reference_initial)
                    and float_equal(internal_quantile, reference_quantile)
                )
                label = combination_label(initial_factor, internal_quantile)
                run_dir = model_root / "crt_runs" / label / day
                run_dir.mkdir(parents=True, exist_ok=True)

                current = (
                    reference
                    if is_reference
                    else prepare_variant(initial_factor, internal_quantile)
                )

                config_core = {
                    "analysis": "iPSC-EC two-stage CLR floor sensitivity",
                    "covariate_model": "six-source-covariate full-day model",
                    "resampling_method": "bernoulli_index",
                    "skew_normal_calibration": False,
                    "stratified_resampling": False,
                    "day": day,
                    "initial_floor_factor": initial_factor,
                    "initial_floor_label": initial_floor_label(initial_factor),
                    "internal_eps_quantile": internal_quantile,
                    "internal_quantile_label": internal_quantile_label(
                        internal_quantile
                    ),
                    "combination_label": label,
                    "is_reference": is_reference,
                    "reference_initial_floor_factor": reference_initial,
                    "reference_internal_eps_quantile": reference_quantile,
                    "local_programde_default_eps_quantile": local_default_quantile,
                    "B": args.B,
                    "n_jobs": args.n_jobs,
                    "real_crt_seed_requested": args.real_crt_seed,
                    "ntc_labels": args.ntc_label,
                    "ntc_group_size": args.ntc_group_size,
                    "ntc_ensembles": args.ntc_ensembles,
                    "ntc_bins": args.ntc_bins,
                    "ntc_max_groups": args.ntc_max_groups,
                    "ntc_group_seed": args.ntc_group_seed,
                    "ntc_crt_seed": args.ntc_crt_seed,
                    "stage1_floor_metadata": current["stage1_meta"],
                    "stage2_floor_metadata": current["stage2_meta"],
                    "manual_vs_programde_max_abs_difference": current[
                        "manual_vs_programde_max_abs_difference"
                    ],
                    "covariate_columns": list(covariates.columns),
                    "covariate_sources": adata.uns.get("covariate_sources", {}),
                    "h5mu_path": str(h5mu_path),
                    "usage_path": str(usage_path),
                    "guide_long_csv": str(guide_csv),
                    "guide_metadata_tsv": args.guide_metadata_tsv,
                    "n_cells": int(adata.n_obs),
                    "n_programs": int(usage_raw.shape[1]),
                    "n_guides": int(len(guide_names)),
                    "matched_usage_cells": int(matched_usage_cells),
                    "programde_root": str(args.programde_root),
                    "programde_git_commit": programde_commit,
                    "programde_prepare_crt_inputs_source": programde_prepare_source,
                    "programde_prepare_crt_inputs_signature": programde_prepare_signature,
                }
                fingerprint = config_fingerprint(config_core)

                if args.reuse_matching and matching_complete_run(
                    run_dir,
                    day,
                    fingerprint,
                ):
                    print(f"{day} / {label}: reusing config-matched complete run")
                    run_records.append(
                        RunRecord(
                            day=day,
                            initial_floor_factor=initial_factor,
                            internal_eps_quantile=internal_quantile,
                            initial_floor_label=initial_floor_label(initial_factor),
                            internal_quantile_label=internal_quantile_label(
                                internal_quantile
                            ),
                            combination_label=label,
                            is_reference=is_reference,
                            status="reused_matching",
                            output_dir=str(run_dir),
                            n_cells=int(adata.n_obs),
                            n_programs=int(usage_raw.shape[1]),
                            n_guides=int(len(guide_names)),
                            n_covariate_columns=int(covariates.shape[1]),
                            raw_usage_zero_fraction=current["stage1_meta"][
                                "fraction_exact_zero_raw_entries"
                            ],
                            base_raw_zero_floor=current["stage1_meta"][
                                "base_raw_zero_floor"
                            ],
                            applied_raw_zero_floor=current["stage1_meta"][
                                "applied_raw_zero_floor"
                            ],
                            effective_internal_floor=current["stage2_meta"][
                                "effective_internal_floor"
                            ],
                            stage2_fraction_entries_modified=current[
                                "stage2_meta"
                            ]["stage2_fraction_entries_modified"],
                            stage2_fraction_positive_entries_modified=current[
                                "stage2_meta"
                            ][
                                "stage2_fraction_originally_positive_entries_modified"
                            ],
                            stage2_fraction_zero_derived_entries_modified=current[
                                "stage2_meta"
                            ]["stage2_fraction_zero_derived_entries_modified"],
                            median_stage2_pseudomass_fraction=current[
                                "stage2_meta"
                            ]["median_stage2_pseudomass_fraction"],
                            config_fingerprint=fingerprint,
                        )
                    )
                    if not is_reference:
                        del current
                        gc.collect()
                    continue

                if not args.overwrite and any(
                    path.exists() for path in expected_run_files(run_dir, day)
                ):
                    raise FileExistsError(
                        f"Existing outputs found in {run_dir}. Use --overwrite or "
                        "--reuse-matching after checking the saved configuration."
                    )

                print(
                    f"{day} / {label}: initial floor="
                    f"{current['stage1_meta']['applied_raw_zero_floor']:.6g}; "
                    f"internal q={internal_quantile:.6g}; "
                    f"effective internal floor="
                    f"{current['stage2_meta']['effective_internal_floor']:.6g}; "
                    "running empirical Bernoulli CRT"
                )

                out, real_seed_parameter = call_empirical_real_crt(
                    programde,
                    current["inputs"],
                    B=args.B,
                    n_jobs=args.n_jobs,
                    real_crt_seed=args.real_crt_seed,
                )
                p_empirical = out.get("pvals_raw_df")
                if p_empirical is None:
                    p_empirical = out.get("pvals_df")
                if p_empirical is None:
                    raise RuntimeError(
                        f"{day} / {label}: CRT output lacks empirical p-values."
                    )
                betas = out.get("betas_df")
                treated = out.get("treated_df")
                if betas is None or treated is None:
                    raise RuntimeError(
                        f"{day} / {label}: CRT output lacks betas_df or treated_df."
                    )

                ntc_frames = programde.crt_pvals_for_ntc_groups_ensemble(
                    inputs=current["inputs"],
                    ntc_groups_ens=ntc_groups,
                    B=args.B,
                    seed0=args.ntc_crt_seed,
                )
                ntc_empirical = flatten_ntc_frames(ntc_frames)

                save_wide(
                    betas,
                    run_dir / f"{day}_target_betas_wide.tsv",
                )
                save_wide(
                    p_empirical,
                    run_dir / f"{day}_target_pvalues_empirical_wide.tsv",
                )
                treated_frame = pd.DataFrame(
                    {
                        "target_name": treated.index.astype(str),
                        "n_union_positive_cells": treated.to_numpy(),
                    }
                )
                treated_frame.to_csv(
                    run_dir / f"{day}_target_treated_counts.tsv",
                    sep="\t",
                    index=False,
                )
                np.save(
                    run_dir / f"{day}_ntc_pseudotarget_pvalues_empirical.npy",
                    ntc_empirical,
                )

                config = {
                    **config_core,
                    "config_fingerprint": fingerprint,
                    "real_crt_seed_parameter_used": real_seed_parameter,
                    "response_matrix_field": current["response_field"],
                    "runtime_seconds": round(time.time() - run_started, 3),
                    "output_dir": str(run_dir),
                }
                atomic_write_json(
                    run_dir / f"{day}_run_config.json",
                    config,
                )

                run_records.append(
                    RunRecord(
                        day=day,
                        initial_floor_factor=initial_factor,
                        internal_eps_quantile=internal_quantile,
                        initial_floor_label=initial_floor_label(initial_factor),
                        internal_quantile_label=internal_quantile_label(
                            internal_quantile
                        ),
                        combination_label=label,
                        is_reference=is_reference,
                        status="written",
                        output_dir=str(run_dir),
                        n_cells=int(adata.n_obs),
                        n_programs=int(usage_raw.shape[1]),
                        n_guides=int(len(guide_names)),
                        n_covariate_columns=int(covariates.shape[1]),
                        n_real_target_program_pairs=int(np.prod(betas.shape)),
                        n_ntc_pvalues_empirical=int(ntc_empirical.size),
                        raw_usage_zero_fraction=current["stage1_meta"][
                            "fraction_exact_zero_raw_entries"
                        ],
                        base_raw_zero_floor=current["stage1_meta"][
                            "base_raw_zero_floor"
                        ],
                        applied_raw_zero_floor=current["stage1_meta"][
                            "applied_raw_zero_floor"
                        ],
                        effective_internal_floor=current["stage2_meta"][
                            "effective_internal_floor"
                        ],
                        stage2_fraction_entries_modified=current["stage2_meta"][
                            "stage2_fraction_entries_modified"
                        ],
                        stage2_fraction_positive_entries_modified=current[
                            "stage2_meta"
                        ]["stage2_fraction_originally_positive_entries_modified"],
                        stage2_fraction_zero_derived_entries_modified=current[
                            "stage2_meta"
                        ]["stage2_fraction_zero_derived_entries_modified"],
                        median_stage2_pseudomass_fraction=current["stage2_meta"][
                            "median_stage2_pseudomass_fraction"
                        ],
                        runtime_seconds=round(time.time() - run_started, 3),
                        config_fingerprint=fingerprint,
                    )
                )

                if not is_reference:
                    del current
                    gc.collect()

        del reference
        del mdata, adata, usage_raw
        gc.collect()
        print(f"{day} complete in {(time.time() - day_started) / 60:.2f} minutes")

    audit = pd.DataFrame(audit_rows).sort_values(
        ["day", "initial_floor_factor", "internal_eps_quantile"]
    )
    audit_path = model_root / "two_stage_CLR_transformation_audit.csv"
    audit.to_csv(audit_path, index=False)

    program_audit = (
        pd.concat(program_audit_rows, ignore_index=True)
        if program_audit_rows
        else pd.DataFrame()
    )
    program_audit_path = model_root / "two_stage_CLR_program_audit.csv"
    program_audit.to_csv(program_audit_path, index=False)

    manifest = pd.DataFrame(asdict(record) for record in run_records)
    if not manifest.empty:
        manifest = manifest.sort_values(
            ["day", "initial_floor_factor", "internal_eps_quantile"]
        )
    manifest_path = model_root / "two_stage_floor_sensitivity_run_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    pd.DataFrame(zero_burden_paths).to_csv(
        model_root / "target_program_zero_burden_manifest.csv",
        index=False,
    )
    pd.DataFrame(ntc_group_rows).to_csv(
        model_root / "ntc_group_summary.csv",
        index=False,
    )

    overall_config = {
        "analysis": "iPSC-EC two-stage CLR floor sensitivity",
        "covariate_model": "six-source-covariate full-day model",
        "initial_floor_factors": initial_factors,
        "internal_eps_quantiles": internal_quantiles,
        "reference_initial_floor_factor": reference_initial,
        "reference_internal_eps_quantile": reference_quantile,
        "local_programde_default_eps_quantile": local_default_quantile,
        "crt_grid_mode": args.crt_grid_mode,
        "audit_combinations": audit_combinations,
        "crt_combinations": [] if args.skip_crt else crt_combinations,
        "B": args.B,
        "n_jobs": args.n_jobs,
        "days": args.days,
        "programde_root": args.programde_root,
        "programde_git_commit": programde_commit,
        "programde_prepare_crt_inputs_source": programde_prepare_source,
        "programde_prepare_crt_inputs_signature": programde_prepare_signature,
        "skew_normal_calibration": False,
        "stratified_resampling": False,
        "resampling_method": "bernoulli_index",
    }
    atomic_write_json(
        model_root / "two_stage_floor_sensitivity_overall_config.json",
        overall_config,
    )

    print("\nSaved:")
    print(" ", audit_path)
    print(" ", program_audit_path)
    print(" ", manifest_path)
    print(" ", model_root / "target_program_zero_burden_manifest.csv")
    print(" ", model_root / "ntc_group_summary.csv")
    print(" ", model_root / "two_stage_floor_sensitivity_overall_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
