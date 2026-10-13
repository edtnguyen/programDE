"""
Figure 1 reprocessing and plotting utilities for CC-Perturb-seq capture benchmarking.

The functions are written to run on Sherlock against the original count matrices.
They avoid hard-coded local notebook state and write Illustrator-editable SVGs with
matplotlib text preserved as text (`svg.fonttype = 'none'`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import re
import warnings
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp


# -----------------------------
# General helpers
# -----------------------------


def as_path(x: str | os.PathLike[str]) -> Path:
    return Path(x).expanduser().resolve()


def coerce_root_dir(path_like: str | os.PathLike[str]) -> Path:
    """Return a directory path, accepting either a directory or a path to an h5ad file.

    This is useful when a config entry is accidentally set to something like
    `/path/to/tony_results/CROP_GEX1.h5ad`; the h5ad-pair loader expects the
    parent directory containing all paired h5ad files.
    """
    root = Path(path_like).expanduser()
    if root.suffix.lower() in {".h5ad", ".h5mu", ".h5", ".loom"}:
        root = root.parent
    return root.resolve()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path = as_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_1d(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def ensure_csr(x: Any) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def make_unique(names: Sequence[Any]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in names:
        name = str(raw)
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}-{seen[name]}")
    return out


def find_existing(base: Path, candidates: Sequence[str]) -> Path | None:
    for candidate in candidates:
        fp = base / candidate
        if fp.exists():
            return fp
    return None


def infer_column(df: pd.DataFrame, preferred: Sequence[str], fallback_contains: Sequence[str] = ()) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    for token in fallback_contains:
        matches = [c for c in df.columns if token.lower() in c.lower()]
        if matches:
            return matches[0]
    # Fall back to the first non-numeric-like column.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
    return df.columns[0]


# -----------------------------
# Parse Biosciences matrix loading
# -----------------------------


def read_parse_matrix(
    matrix_dir: str | os.PathLike[str],
    matrix_candidates: Sequence[str] = ("count_matrix.mtx", "DGE.mtx", "matrix.mtx"),
    feature_csv_candidates: Sequence[str] = ("all_genes.csv", "all_guides.csv", "features.csv"),
    cell_csv_candidates: Sequence[str] = ("cell_metadata.csv", "barcodes.csv", "cells.csv"),
    feature_name_preferred: Sequence[str] = ("gene_name", "guide_id", "guide_name", "name", "feature_name"),
    cell_name_preferred: Sequence[str] = ("bc_wells", "barcode", "cell_barcode", "cell", "cell_id"),
) -> dict[str, Any]:
    """Read a Parse-style MatrixMarket directory and return a cell x feature sparse matrix.

    The original attached notebooks read `count_matrix.mtx`/`DGE.mtx`, `all_genes.csv`,
    and `cell_metadata.csv` manually. This function handles either matrix orientation
    by comparing matrix dimensions to the feature and cell metadata tables.
    """
    base = as_path(matrix_dir)
    matrix_fp = find_existing(base, matrix_candidates)
    feature_fp = find_existing(base, feature_csv_candidates)
    cell_fp = find_existing(base, cell_csv_candidates)

    missing = []
    if matrix_fp is None:
        missing.append(f"matrix file: one of {matrix_candidates}")
    if feature_fp is None:
        missing.append(f"feature CSV: one of {feature_csv_candidates}")
    if missing:
        raise FileNotFoundError(f"Missing in {base}: " + "; ".join(missing))

    X = ensure_csr(sio.mmread(str(matrix_fp)))
    feature_df = pd.read_csv(feature_fp)
    feature_col = infer_column(feature_df, feature_name_preferred, fallback_contains=("gene", "guide", "name"))
    feature_names_raw = feature_df[feature_col]

    cell_df: pd.DataFrame | None = None
    if cell_fp is not None:
        cell_df = pd.read_csv(cell_fp)
        cell_col = infer_column(cell_df, cell_name_preferred, fallback_contains=("bc", "barcode", "cell"))
        cell_names_raw = cell_df[cell_col]
    else:
        cell_col = None
        cell_names_raw = pd.Series([f"cell_{i}" for i in range(max(X.shape))])

    n_features = len(feature_df)
    n_cells = len(cell_df) if cell_df is not None else None

    # Orient to cells x features.
    if n_cells is not None and X.shape == (n_cells, n_features):
        pass
    elif n_cells is not None and X.shape == (n_features, n_cells):
        X = X.T.tocsr()
    elif X.shape[1] == n_features:
        # Cell metadata absent or mismatched; rows are cells.
        pass
    elif X.shape[0] == n_features:
        X = X.T.tocsr()
    else:
        raise ValueError(
            f"Could not infer matrix orientation for {matrix_fp}. "
            f"Matrix shape={X.shape}, n_features={n_features}, n_cells={n_cells}."
        )

    valid_features = feature_names_raw.notna().to_numpy()
    if len(valid_features) == X.shape[1] and not valid_features.all():
        X = X[:, valid_features].tocsr()
        feature_df = feature_df.loc[valid_features].reset_index(drop=True)
        feature_names_raw = feature_names_raw.loc[valid_features].reset_index(drop=True)

    feature_names = make_unique(feature_names_raw.astype(str).tolist())

    if cell_df is not None and len(cell_df) == X.shape[0]:
        cell_names = make_unique(cell_names_raw.astype(str).tolist())
        obs = cell_df.copy()
    else:
        cell_names = [f"cell_{i}" for i in range(X.shape[0])]
        obs = pd.DataFrame(index=cell_names)
        if cell_df is not None:
            warnings.warn(
                f"Cell metadata row count ({len(cell_df)}) does not match matrix rows ({X.shape[0]}) "
                f"for {base}; using synthetic cell IDs."
            )

    var = feature_df.copy()
    obs.index = cell_names
    var.index = feature_names

    return {
        "X": X,
        "obs": obs,
        "var": var,
        "cell_names": cell_names,
        "feature_names": feature_names,
        "matrix_path": str(matrix_fp),
        "feature_path": str(feature_fp),
        "cell_path": str(cell_fp) if cell_fp is not None else None,
        "feature_name_col": feature_col,
        "cell_name_col": cell_col,
    }


# -----------------------------
# QC and guide metric calculations
# -----------------------------


def qc_from_matrix(X: sp.spmatrix, feature_names: Sequence[str]) -> pd.DataFrame:
    X = ensure_csr(X)
    total_counts = to_1d(X.sum(axis=1))
    n_features = np.diff(X.indptr)
    mt_mask = np.array([str(x).upper().startswith("MT-") for x in feature_names], dtype=bool)
    if mt_mask.any():
        mt_counts = to_1d(X[:, mt_mask].sum(axis=1))
        pct_mt = np.divide(mt_counts, total_counts, out=np.zeros_like(mt_counts, dtype=float), where=total_counts > 0) * 100
    else:
        pct_mt = np.zeros(X.shape[0], dtype=float)
    return pd.DataFrame(
        {
            "total_counts": total_counts,
            "n_genes_by_counts": n_features,
            "pct_counts_mt": pct_mt,
        }
    )


def filter_cells_by_qc(
    qc: pd.DataFrame,
    min_genes: int | None = 300,
    max_genes: int | None = 10000,
    max_counts: int | None = 100000,
    max_pct_mt: float | None = 10,
) -> np.ndarray:
    keep = np.ones(qc.shape[0], dtype=bool)
    if min_genes is not None:
        keep &= qc["n_genes_by_counts"].to_numpy() >= min_genes
    if max_genes is not None:
        keep &= qc["n_genes_by_counts"].to_numpy() < max_genes
    if max_counts is not None:
        keep &= qc["total_counts"].to_numpy() < max_counts
    if max_pct_mt is not None:
        keep &= qc["pct_counts_mt"].to_numpy() < max_pct_mt
    return keep


def feature_regex_mask(feature_names: Sequence[str], regex: str) -> np.ndarray:
    pattern = re.compile(regex)
    return np.array([bool(pattern.search(str(name))) for name in feature_names], dtype=bool)


def top_two_sparse_rows(X: sp.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = ensure_csr(X)
    n = X.shape[0]
    top = np.zeros(n, dtype=float)
    second = np.zeros(n, dtype=float)
    top_idx = np.full(n, -1, dtype=int)
    for i in range(n):
        start, end = X.indptr[i], X.indptr[i + 1]
        data = X.data[start:end]
        idx = X.indices[start:end]
        if data.size == 0:
            continue
        if data.size == 1:
            top[i] = float(data[0])
            top_idx[i] = int(idx[0])
            continue
        # Avoid full sorting of long rows.
        order = np.argpartition(data, -2)[-2:]
        vals = data[order]
        ids = idx[order]
        if vals[0] >= vals[1]:
            top[i], second[i] = float(vals[0]), float(vals[1])
            top_idx[i] = int(ids[0])
        else:
            top[i], second[i] = float(vals[1]), float(vals[0])
            top_idx[i] = int(ids[1])
    return top, second, top_idx


def compute_guide_metrics(
    guide_X: sp.spmatrix,
    guide_names: Sequence[str],
    cell_names: Sequence[str],
    *,
    dataset_id: str,
    sample_id: str,
    method: str,
    cell_type: str,
    guide_threshold: float = 5,
    min_cells_per_guide: int = 10,
    assignment_mode: str = "single_guide",
    top_share_min: float = 0.8,
    top_second_ratio_min: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-cell metrics, per-guide metrics, and one-row sample summary.

    assignment_mode:
        - "single_guide": confident assignment requires one dominant guide.
        - "high_moi": assigned_any is used as the assignment metric.
    """
    guide_X = ensure_csr(guide_X)
    guide_names = list(map(str, guide_names))
    cell_names = list(map(str, cell_names))
    if guide_X.shape != (len(cell_names), len(guide_names)):
        raise ValueError(
            f"guide_X shape {guide_X.shape} does not match "
            f"{len(cell_names)} cells x {len(guide_names)} guides."
        )

    total = to_1d(guide_X.sum(axis=1)).astype(float)
    detected_ge_threshold = to_1d((guide_X >= guide_threshold).sum(axis=1)).astype(int)
    top, second, top_idx = top_two_sparse_rows(guide_X)
    top_guide = np.array([guide_names[i] if i >= 0 else "" for i in top_idx], dtype=object)
    top_share = np.divide(top, total, out=np.zeros_like(top, dtype=float), where=total > 0)
    top_second_ratio = np.divide(top, second + 1.0, out=np.zeros_like(top, dtype=float), where=top > 0)

    assigned_any = top >= guide_threshold
    if assignment_mode == "high_moi":
        assigned_confident = assigned_any
    elif assignment_mode == "single_guide":
        assigned_confident = assigned_any & (top_share >= top_share_min) & (top_second_ratio >= top_second_ratio_min)
    else:
        raise ValueError("assignment_mode must be 'single_guide' or 'high_moi'.")

    if assignment_mode == "single_guide":
        guide_assignment_class = np.where(
            ~assigned_any,
            "No guide",
            np.where(assigned_confident, "Singlet", "Ambiguous/multiplet"),
        )
    else:
        guide_assignment_class = np.where(
            ~assigned_any,
            "No guide",
            np.where(detected_ge_threshold <= 1, "One guide", "Multiple guides"),
        )

    cell_df = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "method": method,
            "cell_type": cell_type,
            "cell_id": cell_names,
            "guide_umis_total": total,
            "guide_umis_top": top,
            "second_guide_umis": second,
            "top_guide": top_guide,
            "top_guide_share": top_share,
            "top_second_ratio": top_second_ratio,
            "n_guides_ge_threshold": detected_ge_threshold,
            "assigned_any_guide": assigned_any,
            "assigned_confident_guide": assigned_confident,
            "guide_assignment_class": guide_assignment_class,
        }
    )

    per_guide_nonzero = to_1d((guide_X > 0).sum(axis=0)).astype(int)
    per_guide_threshold = to_1d((guide_X >= guide_threshold).sum(axis=0)).astype(int)
    per_guide_umis = to_1d(guide_X.sum(axis=0)).astype(float)
    guide_df = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "method": method,
            "cell_type": cell_type,
            "guide_id": guide_names,
            "n_cells_nonzero": per_guide_nonzero,
            "n_cells_ge_threshold": per_guide_threshold,
            "guide_umis_total": per_guide_umis,
            "detected_min_cells": per_guide_threshold >= min_cells_per_guide,
        }
    )

    summary = pd.DataFrame(
        {
            "dataset_id": [dataset_id],
            "sample_id": [sample_id],
            "method": [method],
            "cell_type": [cell_type],
            "n_cells": [guide_X.shape[0]],
            "n_guides": [guide_X.shape[1]],
            "median_guide_umis_top_all_cells": [float(np.median(top))],
            "median_guide_umis_top_assigned": [float(np.median(top[assigned_any])) if assigned_any.any() else np.nan],
            "median_guide_umis_total_all_cells": [float(np.median(total))],
            "median_guide_umis_total_assigned": [float(np.median(total[assigned_any])) if assigned_any.any() else np.nan],
            "pct_cells_any_guide": [float(np.mean(assigned_any))],
            "pct_cells_confident_guide": [float(np.mean(assigned_confident))],
            "pct_guides_detected_min_cells": [float(np.mean(guide_df["detected_min_cells"])) if guide_X.shape[1] else np.nan],
            "guide_threshold": [guide_threshold],
            "min_cells_per_guide": [min_cells_per_guide],
            "assignment_mode": [assignment_mode],
        }
    )
    return cell_df, guide_df, summary


def process_parse_embedded_guides(
    *,
    matrix_dir: str | os.PathLike[str],
    guide_feature_regex: str,
    dataset_id: str,
    sample_id: str,
    method: str,
    cell_type: str,
    guide_threshold: float = 5,
    min_cells_per_guide: int = 10,
    assignment_mode: str = "single_guide",
    qc_filter: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed = read_parse_matrix(matrix_dir)
    feature_names = np.array(parsed["feature_names"], dtype=object)
    guide_mask = feature_regex_mask(feature_names, guide_feature_regex)
    if not guide_mask.any():
        raise ValueError(f"No guide features matched regex {guide_feature_regex!r} in {matrix_dir}.")

    gex_mask = ~guide_mask
    X_gex = parsed["X"][:, gex_mask].tocsr()
    X_guide = parsed["X"][:, guide_mask].tocsr()
    gex_feature_names = feature_names[gex_mask].astype(str).tolist()
    guide_names = feature_names[guide_mask].astype(str).tolist()

    qc = qc_from_matrix(X_gex, gex_feature_names)
    qc.index = parsed["cell_names"]
    qc_filter = dict(qc_filter or {})
    keep = filter_cells_by_qc(qc, **qc_filter)

    cell_df, guide_df, summary = compute_guide_metrics(
        X_guide[keep, :],
        guide_names,
        np.array(parsed["cell_names"], dtype=object)[keep].astype(str).tolist(),
        dataset_id=dataset_id,
        sample_id=sample_id,
        method=method,
        cell_type=cell_type,
        guide_threshold=guide_threshold,
        min_cells_per_guide=min_cells_per_guide,
        assignment_mode=assignment_mode,
    )
    qc_kept = qc.loc[cell_df["cell_id"].to_numpy()].reset_index(drop=True)
    cell_df = pd.concat([cell_df.reset_index(drop=True), qc_kept], axis=1)
    summary["median_gex_umis"] = float(qc_kept["total_counts"].median())
    summary["median_genes"] = float(qc_kept["n_genes_by_counts"].median())
    summary["pct_counts_mt_median"] = float(qc_kept["pct_counts_mt"].median())
    summary["matrix_path"] = parsed["matrix_path"]
    return cell_df, guide_df, summary


def process_parse_separate_gex_and_guides(
    *,
    gex_dir: str | os.PathLike[str],
    guide_dir: str | os.PathLike[str],
    dataset_id: str,
    sample_id: str,
    method: str,
    cell_type: str,
    guide_threshold: float = 5,
    min_cells_per_guide: int = 10,
    assignment_mode: str = "high_moi",
    qc_filter: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gex = read_parse_matrix(as_path(gex_dir) / "DGE_filtered")
    guide = read_parse_matrix(
        as_path(guide_dir) / "guide_RNAs_filtered",
        feature_csv_candidates=("all_guides.csv", "all_genes.csv", "features.csv"),
        feature_name_preferred=("guide_id", "guide_name", "gene_name", "name", "feature_name"),
    )

    qc = qc_from_matrix(gex["X"], gex["feature_names"])
    qc.index = gex["cell_names"]
    qc_filter = dict(qc_filter or {})
    keep_gex = filter_cells_by_qc(qc, **qc_filter)
    gex_cells = np.array(gex["cell_names"], dtype=object)
    guide_cells = np.array(guide["cell_names"], dtype=object)

    if set(gex_cells) & set(guide_cells):
        kept_cells = pd.Index(gex_cells[keep_gex].astype(str)).intersection(pd.Index(guide_cells.astype(str)))
        gex_pos = pd.Index(gex_cells.astype(str)).get_indexer(kept_cells)
        guide_pos = pd.Index(guide_cells.astype(str)).get_indexer(kept_cells)
        qc_kept = qc.iloc[gex_pos].reset_index(drop=True)
        guide_X = guide["X"][guide_pos, :]
        cell_names = kept_cells.astype(str).tolist()
    else:
        # Fall back to row-order alignment after QC if guide cell metadata are absent.
        warnings.warn(
            f"Could not match cell IDs between {gex_dir} and {guide_dir}. "
            "Falling back to row-order alignment after GEX QC. Confirm this before publication."
        )
        n = min(int(keep_gex.sum()), guide["X"].shape[0])
        guide_X = guide["X"][:n, :]
        qc_kept = qc.loc[keep_gex].iloc[:n].reset_index(drop=True)
        cell_names = gex_cells[keep_gex].astype(str).tolist()[:n]

    cell_df, guide_df, summary = compute_guide_metrics(
        guide_X,
        guide["feature_names"],
        cell_names,
        dataset_id=dataset_id,
        sample_id=sample_id,
        method=method,
        cell_type=cell_type,
        guide_threshold=guide_threshold,
        min_cells_per_guide=min_cells_per_guide,
        assignment_mode=assignment_mode,
    )
    cell_df = pd.concat([cell_df.reset_index(drop=True), qc_kept], axis=1)
    summary["median_gex_umis"] = float(qc_kept["total_counts"].median())
    summary["median_genes"] = float(qc_kept["n_genes_by_counts"].median())
    summary["pct_counts_mt_median"] = float(qc_kept["pct_counts_mt"].median())
    summary["gex_matrix_path"] = gex["matrix_path"]
    summary["guide_matrix_path"] = guide["matrix_path"]
    return cell_df, guide_df, summary


# -----------------------------
# h5ad / h5mu loading
# -----------------------------


def _get_anndata_count_matrix(adata: Any, layers_to_sum: Sequence[str] | None = None, preferred_layer: str | None = None) -> sp.csr_matrix:
    if layers_to_sum and all(layer in adata.layers for layer in layers_to_sum):
        X = None
        for layer in layers_to_sum:
            X = adata.layers[layer] if X is None else X + adata.layers[layer]
        return ensure_csr(X)
    if preferred_layer and preferred_layer in adata.layers:
        return ensure_csr(adata.layers[preferred_layer])
    if "raw_counts" in adata.layers:
        return ensure_csr(adata.layers["raw_counts"])
    if "counts" in adata.layers:
        return ensure_csr(adata.layers["counts"])
    return ensure_csr(adata.X)


def process_h5ad_gex_guide_pair(
    *,
    gex_h5ad: str | os.PathLike[str],
    guide_h5ad: str | os.PathLike[str],
    dataset_id: str,
    sample_id: str,
    method: str,
    cell_type: str,
    guide_threshold: float = 5,
    min_cells_per_guide: int = 10,
    assignment_mode: str = "high_moi",
    gex_layers_to_sum: Sequence[str] | None = ("ambiguous", "mature", "nascent"),
    qc_filter: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import scanpy as sc  # Imported lazily so this module can be linted without scanpy installed.

    gex = sc.read_h5ad(str(gex_h5ad))
    guide = sc.read_h5ad(str(guide_h5ad))
    common = pd.Index(gex.obs_names.astype(str)).intersection(pd.Index(guide.obs_names.astype(str)))
    if len(common) == 0:
        raise ValueError(f"No overlapping cells between {gex_h5ad} and {guide_h5ad}.")
    gex = gex[common].copy()
    guide = guide[common].copy()

    X_gex = _get_anndata_count_matrix(gex, layers_to_sum=gex_layers_to_sum)
    X_guide = _get_anndata_count_matrix(guide)
    qc = qc_from_matrix(X_gex, gex.var_names.astype(str).tolist())
    qc.index = common.astype(str)
    qc_filter = dict(qc_filter or {})
    keep = filter_cells_by_qc(qc, **qc_filter)

    cell_df, guide_df, summary = compute_guide_metrics(
        X_guide[keep, :],
        guide.var_names.astype(str).tolist(),
        common[keep].astype(str).tolist(),
        dataset_id=dataset_id,
        sample_id=sample_id,
        method=method,
        cell_type=cell_type,
        guide_threshold=guide_threshold,
        min_cells_per_guide=min_cells_per_guide,
        assignment_mode=assignment_mode,
    )
    qc_kept = qc.iloc[np.where(keep)[0]].reset_index(drop=True)
    cell_df = pd.concat([cell_df.reset_index(drop=True), qc_kept], axis=1)
    summary["median_gex_umis"] = float(qc_kept["total_counts"].median())
    summary["median_genes"] = float(qc_kept["n_genes_by_counts"].median())
    summary["pct_counts_mt_median"] = float(qc_kept["pct_counts_mt"].median())
    summary["gex_h5ad"] = str(gex_h5ad)
    summary["guide_h5ad"] = str(guide_h5ad)
    return cell_df, guide_df, summary


def process_h5mu_gex_guide(
    *,
    h5mu_path: str | os.PathLike[str],
    dataset_id: str,
    sample_id: str,
    method: str,
    cell_type: str,
    guide_threshold: float = 5,
    min_cells_per_guide: int = 10,
    assignment_mode: str = "high_moi",
    gex_mod: str = "GEX",
    guide_mod: str = "guide",
    gex_layer: str = "raw_counts",
    qc_filter: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import muon as mu

    mdata = mu.read(str(h5mu_path))
    gex = mdata[gex_mod]
    guide = mdata[guide_mod]
    common = pd.Index(gex.obs_names.astype(str)).intersection(pd.Index(guide.obs_names.astype(str)))
    if len(common) == 0:
        raise ValueError(f"No overlapping cells between GEX and guide modalities in {h5mu_path}.")
    gex = gex[common].copy()
    guide = guide[common].copy()
    X_gex = _get_anndata_count_matrix(gex, preferred_layer=gex_layer)
    X_guide = _get_anndata_count_matrix(guide)

    qc = qc_from_matrix(X_gex, gex.var_names.astype(str).tolist())
    qc.index = common.astype(str)
    qc_filter = dict(qc_filter or {})
    keep = filter_cells_by_qc(qc, **qc_filter)

    cell_df, guide_df, summary = compute_guide_metrics(
        X_guide[keep, :],
        guide.var_names.astype(str).tolist(),
        common[keep].astype(str).tolist(),
        dataset_id=dataset_id,
        sample_id=sample_id,
        method=method,
        cell_type=cell_type,
        guide_threshold=guide_threshold,
        min_cells_per_guide=min_cells_per_guide,
        assignment_mode=assignment_mode,
    )
    qc_kept = qc.iloc[np.where(keep)[0]].reset_index(drop=True)
    cell_df = pd.concat([cell_df.reset_index(drop=True), qc_kept], axis=1)
    summary["median_gex_umis"] = float(qc_kept["total_counts"].median())
    summary["median_genes"] = float(qc_kept["n_genes_by_counts"].median())
    summary["pct_counts_mt_median"] = float(qc_kept["pct_counts_mt"].median())
    summary["h5mu_path"] = str(h5mu_path)
    return cell_df, guide_df, summary


# -----------------------------
# Saturation curves
# -----------------------------


def collect_h5ad_saturation_curve(
    *,
    pattern: str,
    read_counts: Sequence[int],
    method: str,
    dataset_id: str,
    sample_id: str,
    cell_type: str,
    modality: str,
    cells_to_keep: Iterable[str] | None = None,
    layers_to_sum: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Collect median UMIs/cell from a set of downsampled h5ad files.

    `pattern` should contain `{reads}`. Example:
    `GUIDE_S1.{reads}_kb_kite/counts_unfiltered_modified/adata.h5ad`.
    """
    import scanpy as sc

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    keep_index = pd.Index([str(x) for x in cells_to_keep]) if cells_to_keep is not None else None
    for reads in read_counts:
        fp = Path(pattern.format(reads=reads))
        if not fp.exists():
            missing.append(str(fp))
            continue
        adata = sc.read_h5ad(str(fp))
        if keep_index is not None:
            common = pd.Index(adata.obs_names.astype(str)).intersection(keep_index)
            adata = adata[common].copy()
        X = _get_anndata_count_matrix(adata, layers_to_sum=layers_to_sum)
        umi_per_cell = to_1d(X.sum(axis=1)).astype(float)
        features_per_cell = np.diff(ensure_csr(X).indptr).astype(float)
        n_cells = int(X.shape[0])
        rows.append(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "method": method,
                "cell_type": cell_type,
                "modality": modality,
                "reads_total": int(reads),
                "n_cells": n_cells,
                "reads_per_cell": float(reads / n_cells) if n_cells else np.nan,
                "median_umis_per_cell": float(np.median(umi_per_cell)) if n_cells else np.nan,
                "mean_umis_per_cell": float(np.mean(umi_per_cell)) if n_cells else np.nan,
                "median_features_per_cell": float(np.median(features_per_cell)) if n_cells else np.nan,
                "mean_features_per_cell": float(np.mean(features_per_cell)) if n_cells else np.nan,
                "file": str(fp),
            }
        )
    return pd.DataFrame(rows), missing


# -----------------------------
# Plotnine figure panels
# -----------------------------


CAPTURE_COLORS = {
    "Circular capture": "#D55E00",
    "Direct capture": "#D55E00",
    "CC-Perturb-seq": "#D55E00",
    "CROP-seq": "#0072B2",
    "Pol III / circular capture": "#D55E00",
    "Pol II / CROP-seq": "#0072B2",
}


def _plotnine_setup() -> None:
    import matplotlib as mpl

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"


def fig_theme(base_size: int = 7):
    from plotnine import theme_classic, theme, element_text, element_line, element_blank

    return (
        theme_classic(base_size=base_size, base_family="Arial")
        + theme(
            axis_text=element_text(color="black"),
            axis_ticks=element_line(color="black", size=0.35),
            axis_line=element_line(color="black", size=0.35),
            strip_background=element_blank(),
            strip_text=element_text(color="black", weight="bold"),
            legend_title=element_blank(),
            legend_key=element_blank(),
            panel_spacing=0.08,
        )
    )


def save_plot(plot: Any, filename: str | os.PathLike[str], width: float, height: float, dpi: int = 300) -> Path:
    _plotnine_setup()
    fp = as_path(filename)
    fp.parent.mkdir(parents=True, exist_ok=True)
    plot.save(str(fp), width=width, height=height, units="in", dpi=dpi, verbose=False)
    return fp


def plot_guide_umis_distribution(
    cell_metrics: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    cell_type: str | None = None,
    y_col: str = "guide_umis_top",
    title: str = "",
):
    from plotnine import (
        ggplot,
        aes,
        geom_violin,
        geom_boxplot,
        scale_fill_manual,
        scale_y_log10,
        labs,
        theme,
        element_blank,
    )

    df = cell_metrics.copy()
    if dataset_id is not None:
        df = df[df["dataset_id"] == dataset_id]
    if cell_type is not None:
        df = df[df["cell_type"] == cell_type]
    df = df[np.isfinite(df[y_col]) & (df[y_col] > 0)].copy()
    if df.empty:
        raise ValueError("No positive guide UMI values available for guide UMI distribution plot.")
    method_order = [m for m in ["Circular capture", "Direct capture", "CROP-seq"] if m in df["method"].unique()]
    if method_order:
        df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    return (
        ggplot(df, aes(x="method", y=y_col, fill="method"))
        + geom_violin(width=0.9, scale="width", trim=True, alpha=0.85)
        + geom_boxplot(width=0.16, outlier_alpha=0.12, alpha=0.95)
        + scale_fill_manual(values=CAPTURE_COLORS)
        + scale_y_log10()
        + labs(x="", y="Guide UMIs per cell", title=title)
        + fig_theme()
        + theme(legend_position="none", axis_title_x=element_blank())
    )


def plot_assignment_metrics(summary: pd.DataFrame, *, dataset_id: str | None = None, title: str = ""):
    from plotnine import (
        ggplot,
        aes,
        geom_col,
        geom_text,
        position_dodge,
        scale_fill_manual,
        scale_y_continuous,
        labs,
        theme,
        element_text,
    )
    from mizani.formatters import percent_format

    df = summary.copy()
    if dataset_id is not None:
        df = df[df["dataset_id"] == dataset_id]
    metric_map = {
        "pct_cells_any_guide": "Cells with ≥1 guide",
        "pct_cells_confident_guide": "Confidently assigned cells",
        "pct_guides_detected_min_cells": "Guides detected",
    }
    long = df.melt(
        id_vars=["dataset_id", "sample_id", "method", "cell_type"],
        value_vars=[c for c in metric_map if c in df.columns],
        var_name="metric",
        value_name="fraction",
    )
    long["metric_label"] = long["metric"].map(metric_map)
    long["percent_label"] = (100 * long["fraction"]).map(lambda x: f"{x:.0f}%" if np.isfinite(x) else "NA")
    if long.empty:
        raise ValueError("No assignment metrics available to plot.")
    return (
        ggplot(long, aes(x="metric_label", y="fraction", fill="method"))
        + geom_col(position=position_dodge(width=0.72), width=0.62)
        + geom_text(
            aes(label="percent_label"),
            position=position_dodge(width=0.72),
            va="bottom",
            size=6,
            format_string="{}",
        )
        + scale_fill_manual(values=CAPTURE_COLORS)
        + scale_y_continuous(labels=percent_format(), limits=(0, 1.08))
        + labs(x="", y="Fraction", title=title)
        + fig_theme()
        + theme(axis_text_x=element_text(rotation=25, ha="right"))
    )


def plot_transcriptome_saturation(saturation: pd.DataFrame, *, dataset_id: str | None = None, title: str = ""):
    from plotnine import ggplot, aes, geom_line, geom_point, scale_color_manual, labs

    df = saturation.copy()
    if dataset_id is not None:
        df = df[df["dataset_id"] == dataset_id]
    df = df[df["modality"].isin(["GEX", "gex", "transcriptome", "mRNA"])]
    if df.empty:
        raise ValueError("No transcriptome saturation rows found. Supply downsampled GEX h5ad files or a saturation TSV.")
    return (
        ggplot(df, aes(x="reads_per_cell", y="median_umis_per_cell", color="method", group="method"))
        + geom_line(size=0.7)
        + geom_point(size=1.8)
        + scale_color_manual(values=CAPTURE_COLORS)
        + labs(x="Reads per cell", y="Median transcriptome UMIs/cell", title=title)
        + fig_theme()
    )


def plot_transcriptome_qc_distribution(
    cell_metrics: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    title: str = "",
):
    from plotnine import ggplot, aes, geom_violin, geom_boxplot, scale_fill_manual, scale_y_log10, labs, theme, element_blank

    df = cell_metrics.copy()
    if dataset_id is not None:
        df = df[df["dataset_id"] == dataset_id]
    df = df[np.isfinite(df["total_counts"]) & (df["total_counts"] > 0)].copy()
    if df.empty:
        raise ValueError("No transcriptome UMI data available to plot.")
    return (
        ggplot(df, aes(x="method", y="total_counts", fill="method"))
        + geom_violin(width=0.9, scale="width", trim=True, alpha=0.85)
        + geom_boxplot(width=0.16, outlier_alpha=0.10, alpha=0.95)
        + scale_fill_manual(values=CAPTURE_COLORS)
        + scale_y_log10()
        + labs(x="", y="Transcriptome UMIs per cell", title=title)
        + fig_theme()
        + theme(legend_position="none", axis_title_x=element_blank())
    )


def plot_other_celltype_guide_umis(
    cell_metrics: pd.DataFrame,
    *,
    exclude_dataset_id: str | None = "ipsc_single_guide",
    y_col: str = "guide_umis_top",
    title: str = "",
):
    from plotnine import (
        ggplot,
        aes,
        geom_boxplot,
        geom_jitter,
        facet_wrap,
        scale_fill_manual,
        scale_y_log10,
        labs,
        theme,
        element_text,
    )

    df = cell_metrics.copy()
    if exclude_dataset_id is not None:
        df = df[df["dataset_id"] != exclude_dataset_id]
    df = df[np.isfinite(df[y_col]) & (df[y_col] > 0)].copy()
    if df.empty:
        raise ValueError("No additional-cell-type guide UMI data available to plot.")
    return (
        ggplot(df, aes(x="method", y=y_col, fill="method"))
        + geom_boxplot(width=0.55, outlier_alpha=0.0, alpha=0.85)
        + geom_jitter(width=0.12, height=0, alpha=0.05, size=0.12)
        + facet_wrap("~cell_type", scales="free_x")
        + scale_fill_manual(values=CAPTURE_COLORS)
        + scale_y_log10()
        + labs(x="", y="Guide UMIs per cell", title=title)
        + fig_theme()
        + theme(axis_text_x=element_text(rotation=25, ha="right"), legend_position="none")
    )


def plot_guide_saturation(saturation: pd.DataFrame, *, dataset_id: str | None = None, title: str = ""):
    from plotnine import ggplot, aes, geom_line, geom_point, scale_color_manual, labs

    df = saturation.copy()
    if dataset_id is not None:
        df = df[df["dataset_id"] == dataset_id]
    df = df[df["modality"].isin(["guide", "GUIDE", "gRNA"])]
    if df.empty:
        raise ValueError("No guide saturation rows found.")
    return (
        ggplot(df, aes(x="reads_per_cell", y="median_umis_per_cell", color="method", group="method"))
        + geom_line(size=0.7)
        + geom_point(size=1.8)
        + scale_color_manual(values=CAPTURE_COLORS)
        + labs(x="Guide reads per cell", y="Median guide UMIs/cell", title=title)
        + fig_theme()
    )




# -----------------------------
# Additional paper-inspired QC panels and optional analyses
# -----------------------------


def _find_column_fuzzy(
    df: pd.DataFrame,
    preferred: Sequence[str],
    contains: Sequence[str] = (),
    numeric: bool | None = None,
) -> str | None:
    """Return a column by exact preferred names, then by substring matching."""
    if df is None or df.empty:
        return None
    preferred_lower = {str(x).lower(): str(x) for x in preferred}
    for col in df.columns:
        if str(col).lower() in preferred_lower:
            if numeric is None or pd.api.types.is_numeric_dtype(df[col]) == numeric:
                return col
    for token in contains:
        token_l = str(token).lower()
        for col in df.columns:
            if token_l in str(col).lower():
                if numeric is None or pd.api.types.is_numeric_dtype(df[col]) == numeric:
                    return col
    if numeric is not None:
        candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) == numeric]
        return candidates[0] if candidates else None
    return None


def _read_any_table(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read CSV/TSV/TXT/XLSX table with light format inference."""
    fp = Path(path).expanduser()
    if not fp.exists():
        raise FileNotFoundError(str(fp))
    suffixes = "".join(fp.suffixes).lower()
    if suffixes.endswith(".xlsx") or suffixes.endswith(".xls"):
        return pd.read_excel(fp)
    sep = "\t" if fp.suffix.lower() in {".tsv", ".txt"} or suffixes.endswith(".tsv.gz") else None
    if sep == "\t":
        return pd.read_csv(fp, sep="\t", compression="infer")
    return pd.read_csv(fp, sep=None, engine="python", compression="infer")


def _optional_panel_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("optional_panels", {}) or {}


def _load_external_table_list(
    config: Mapping[str, Any],
    key: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Load zero or more optional external tables from config['optional_panels'][key]."""
    rows: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    entries = _optional_panel_config(config).get(key, [])
    if isinstance(entries, Mapping):
        entries = [entries]
    for i, entry in enumerate(entries or []):
        if not entry or not entry.get("enabled", True):
            continue
        path = entry.get("path") or entry.get("file")
        if not path:
            continue
        try:
            df = _read_any_table(path)
            for meta_col in ["dataset_id", "sample_id", "method", "cell_type", "condition", "note"]:
                if meta_col in entry and meta_col not in df.columns:
                    df[meta_col] = entry[meta_col]
            df["source_path"] = str(path)
            df["source_table_key"] = key
            df["source_table_index"] = i
            rows.append(df)
        except Exception as exc:
            errors.append({"dataset_key": f"optional_panels.{key}.{i}", "error": repr(exc)})
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), errors


def compute_assignment_class_summary(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize no-guide/singlet/multiplet classes by dataset and method."""
    if cell_metrics.empty:
        return pd.DataFrame()
    df = cell_metrics.copy()
    if "guide_assignment_class" not in df.columns:
        assigned_any = df.get("assigned_any_guide", pd.Series(False, index=df.index)).astype(bool)
        assigned_conf = df.get("assigned_confident_guide", pd.Series(False, index=df.index)).astype(bool)
        df["guide_assignment_class"] = np.where(~assigned_any, "No guide", np.where(assigned_conf, "Assigned", "Ambiguous/multiplet"))
    group_cols = [c for c in ["dataset_id", "sample_id", "method", "cell_type", "assignment_mode"] if c in df.columns]
    counts = df.groupby(group_cols + ["guide_assignment_class"], dropna=False, observed=True).size().reset_index(name="n_cells_class")
    totals = df.groupby(group_cols, dropna=False, observed=True).size().reset_index(name="n_cells_total")
    out = counts.merge(totals, on=group_cols, how="left")
    out["fraction"] = out["n_cells_class"] / out["n_cells_total"].replace(0, np.nan)
    out["percent"] = out["fraction"] * 100
    return out


def compute_guide_detection_curve(guide_metrics: pd.DataFrame, sample_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank guides by recovered cells/UMIs to visualize guide dropout or uniformity."""
    if guide_metrics.empty:
        return pd.DataFrame()
    df = guide_metrics.copy()
    merge_cols = [c for c in ["dataset_id", "sample_id", "method", "cell_type"] if c in df.columns and c in sample_summary.columns]
    if merge_cols and not sample_summary.empty and "n_cells" in sample_summary.columns:
        df = df.merge(sample_summary[merge_cols + ["n_cells"]].drop_duplicates(), on=merge_cols, how="left", suffixes=("", "_sample"))
    elif "n_cells" not in df.columns:
        df["n_cells"] = np.nan
    group_cols = [c for c in ["dataset_id", "sample_id", "method", "cell_type"] if c in df.columns]
    pieces: list[pd.DataFrame] = []
    for _, sub in df.groupby(group_cols, dropna=False, observed=True):
        sub = sub.sort_values(["n_cells_ge_threshold", "guide_umis_total"], ascending=False).reset_index(drop=True).copy()
        sub["guide_rank"] = np.arange(1, len(sub) + 1)
        sub["guide_rank_fraction"] = sub["guide_rank"] / max(len(sub), 1)
        if "n_cells" in sub.columns:
            sub["fraction_cells_with_guide"] = sub["n_cells_ge_threshold"] / sub["n_cells"].replace(0, np.nan)
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def standardize_primer_optimization_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize optional primer/concentration optimization tables if supplied."""
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    primer_col = _find_column_fuzzy(df, ["primer", "primer_name", "rt_primer", "guide_rt_primer"], ["primer"], numeric=False)
    concentration_col = _find_column_fuzzy(df, ["concentration", "primer_concentration", "sgRNA RT primer concentration", "rt_primer_concentration"], ["concentration", "conc"], numeric=None)
    guide_umi_col = _find_column_fuzzy(df, ["guide_umis_per_cell", "sgRNA UMI", "sgRNA UMI counts", "guide_umi", "guide_umis", "umis"], ["guide", "sgrna", "umi"], numeric=True)
    n_cells_col = _find_column_fuzzy(df, ["n_cells", "cell_number", "cells_recovered", "recovered_cells", "normalized_cell_number"], ["cell", "recovered"], numeric=True)
    condition_col = _find_column_fuzzy(df, ["condition", "sample", "sample_id"], ["condition", "sample"], numeric=False)
    if primer_col and "primer" not in df.columns:
        df["primer"] = df[primer_col].astype(str)
    elif "primer" not in df.columns:
        df["primer"] = "primer"
    if concentration_col and "primer_concentration" not in df.columns:
        df["primer_concentration"] = df[concentration_col].astype(str)
    elif "primer_concentration" not in df.columns:
        df["primer_concentration"] = "not specified"
    if guide_umi_col and "guide_umis_per_cell" not in df.columns:
        df["guide_umis_per_cell"] = pd.to_numeric(df[guide_umi_col], errors="coerce")
    if n_cells_col and "n_cells_recovered" not in df.columns:
        df["n_cells_recovered"] = pd.to_numeric(df[n_cells_col], errors="coerce")
    if condition_col and "condition" not in df.columns:
        df["condition"] = df[condition_col].astype(str)
    elif "condition" not in df.columns:
        df["condition"] = df["primer"].astype(str) + " " + df["primer_concentration"].astype(str)
    return df


def standardize_read_origin_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize optional guide-read origin tables: oligo-dT vs guide RT primer."""
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    source_col = _find_column_fuzzy(df, ["read_source", "source", "origin", "primer", "library"], ["source", "origin", "primer"], numeric=False)
    reads_col = _find_column_fuzzy(df, ["n_reads", "reads", "read_count", "guide_reads", "count"], ["read", "count"], numeric=True)
    fraction_col = _find_column_fuzzy(df, ["fraction", "frac", "pct", "percent", "percentage"], ["fraction", "percent", "pct"], numeric=True)
    if source_col and "read_source" not in df.columns:
        df["read_source"] = df[source_col].astype(str)
    elif "read_source" not in df.columns:
        df["read_source"] = "unspecified"
    if reads_col and "n_reads" not in df.columns:
        df["n_reads"] = pd.to_numeric(df[reads_col], errors="coerce")
    if fraction_col and "fraction" not in df.columns:
        frac = pd.to_numeric(df[fraction_col], errors="coerce")
        # Accept either 0-1 or 0-100 scale.
        df["fraction"] = np.where(frac > 1.0, frac / 100.0, frac)
    if "fraction" not in df.columns and "n_reads" in df.columns:
        group_cols = [c for c in ["dataset_id", "sample_id", "method", "cell_type"] if c in df.columns]
        totals = df.groupby(group_cols, dropna=False, observed=True)["n_reads"].transform("sum") if group_cols else df["n_reads"].sum()
        df["fraction"] = df["n_reads"] / pd.Series(totals).replace(0, np.nan).to_numpy()
    df["percent"] = df["fraction"] * 100 if "fraction" in df.columns else np.nan
    return df


def compute_replicate_correlations_from_config(config: Mapping[str, Any]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Compute correlation matrices from optional guide-count tables with replicate columns."""
    opt = _optional_panel_config(config)
    entries = opt.get("replicate_count_tables", [])
    if isinstance(entries, Mapping):
        entries = [entries]
    all_long: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for i, entry in enumerate(entries or []):
        if not entry or not entry.get("enabled", True):
            continue
        path = entry.get("path") or entry.get("file")
        if not path:
            continue
        try:
            df = _read_any_table(path)
            guide_col = _find_column_fuzzy(df, ["guide_id", "sgRNA", "sgRNA_id", "grna_id", "guide"], ["guide", "sgrna", "grna"], numeric=False)
            if guide_col is None:
                guide_col = df.columns[0]
            candidate_cols = entry.get("replicate_columns")
            if candidate_cols:
                count_cols = [c for c in candidate_cols if c in df.columns]
            else:
                count_cols = [c for c in df.columns if c != guide_col and pd.api.types.is_numeric_dtype(df[c])]
            if len(count_cols) < 2:
                raise ValueError("Need at least two numeric replicate columns.")
            mat = np.log1p(df[count_cols].astype(float))
            corr_method = entry.get("correlation", "pearson")
            corr = mat.corr(method=corr_method)
            long = corr.stack().reset_index()
            long.columns = ["replicate_1", "replicate_2", "correlation"]
            long["dataset_id"] = entry.get("dataset_id", "optional_replicates")
            long["sample_id"] = entry.get("sample_id", Path(path).stem)
            long["method"] = entry.get("method", "")
            long["cell_type"] = entry.get("cell_type", "")
            long["correlation_method"] = corr_method
            long["source_path"] = str(path)
            all_long.append(long)
        except Exception as exc:
            errors.append({"dataset_key": f"optional_panels.replicate_count_tables.{i}", "error": repr(exc)})
    return (pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()), errors


def compute_bulk_singlecell_correlation_from_config(
    config: Mapping[str, Any],
    guide_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Correlate bulk guide read counts against single-cell guide recovery if optional bulk tables are supplied."""
    if guide_metrics.empty:
        return pd.DataFrame(), []
    opt = _optional_panel_config(config)
    entries = opt.get("bulk_guide_count_tables", [])
    if isinstance(entries, Mapping):
        entries = [entries]
    rows: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for i, entry in enumerate(entries or []):
        if not entry or not entry.get("enabled", True):
            continue
        path = entry.get("path") or entry.get("file")
        if not path:
            continue
        try:
            bulk = _read_any_table(path)
            guide_col = _find_column_fuzzy(bulk, ["guide_id", "sgRNA", "sgRNA_id", "grna_id", "guide"], ["guide", "sgrna", "grna"], numeric=False)
            count_col = entry.get("count_col") or _find_column_fuzzy(bulk, ["n_reads", "reads", "read_count", "count", "bulk_count"], ["read", "count"], numeric=True)
            if guide_col is None or count_col is None:
                raise ValueError("Could not infer guide/count columns in bulk guide count table.")
            sample_id = entry.get("single_cell_sample_id") or entry.get("sample_id")
            gm = guide_metrics.copy()
            if sample_id and "sample_id" in gm.columns:
                gm = gm[gm["sample_id"].astype(str) == str(sample_id)]
            if gm.empty:
                raise ValueError(f"No guide_metrics rows matched sample_id={sample_id!r}.")
            sc_metric = entry.get("single_cell_metric", "n_cells_ge_threshold")
            if sc_metric not in gm.columns:
                raise ValueError(f"single_cell_metric {sc_metric!r} is not in guide_metrics.")
            merged = gm[["guide_id", sc_metric, "dataset_id", "sample_id", "method", "cell_type"]].merge(
                bulk[[guide_col, count_col]].rename(columns={guide_col: "guide_id", count_col: "bulk_count"}),
                on="guide_id",
                how="inner",
            )
            merged["single_cell_count"] = pd.to_numeric(merged[sc_metric], errors="coerce")
            merged["bulk_count"] = pd.to_numeric(merged["bulk_count"], errors="coerce")
            merged["log1p_single_cell_count"] = np.log1p(merged["single_cell_count"])
            merged["log1p_bulk_count"] = np.log1p(merged["bulk_count"])
            merged["source_path"] = str(path)
            merged["comparison_label"] = entry.get("comparison_label", Path(path).stem)
            rows.append(merged)
        except Exception as exc:
            errors.append({"dataset_key": f"optional_panels.bulk_guide_count_tables.{i}", "error": repr(exc)})
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), errors


def _prepare_parse_gex_guide_alignment(
    *,
    gex_dir: str | os.PathLike[str],
    guide_dir: str | os.PathLike[str],
    qc_filter: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and align Parse GEX/guide matrices for optional target-knockdown calculations."""
    gex = read_parse_matrix(as_path(gex_dir) / "DGE_filtered")
    guide = read_parse_matrix(
        as_path(guide_dir) / "guide_RNAs_filtered",
        feature_csv_candidates=("all_guides.csv", "all_genes.csv", "features.csv"),
        feature_name_preferred=("guide_id", "guide_name", "gene_name", "name", "feature_name"),
    )
    qc = qc_from_matrix(gex["X"], gex["feature_names"])
    qc.index = gex["cell_names"]
    keep = filter_cells_by_qc(qc, **dict(qc_filter or {}))
    gex_cells = pd.Index(np.array(gex["cell_names"], dtype=object).astype(str))
    guide_cells = pd.Index(np.array(guide["cell_names"], dtype=object).astype(str))
    if len(gex_cells.intersection(guide_cells)) > 0:
        kept_cells = gex_cells[keep].intersection(guide_cells)
        gex_pos = gex_cells.get_indexer(kept_cells)
        guide_pos = guide_cells.get_indexer(kept_cells)
    else:
        warnings.warn("Could not match GEX and guide cell IDs; using row-order alignment for optional knockdown calculation.")
        n = min(int(keep.sum()), guide["X"].shape[0])
        gex_pos = np.where(keep)[0][:n]
        guide_pos = np.arange(n)
        kept_cells = pd.Index([str(x) for x in np.array(gex["cell_names"], dtype=object)[gex_pos]])
    return {
        "gex": gex,
        "guide": guide,
        "qc": qc.iloc[gex_pos].reset_index(drop=True),
        "gex_pos": np.asarray(gex_pos),
        "guide_pos": np.asarray(guide_pos),
        "cell_names": kept_cells.astype(str).tolist(),
    }


def compute_target_knockdown_from_parse(
    *,
    gex_dir: str | os.PathLike[str],
    guide_dir: str | os.PathLike[str],
    guide_target_table: str | os.PathLike[str],
    dataset_id: str,
    sample_id: str,
    method: str,
    cell_type: str,
    guide_threshold: float = 5,
    assignment_mode: str = "single_guide",
    guide_id_col: str = "grna_id",
    target_col: str = "grna_target",
    control_target_regex: str = r"(?i)(^ntc|non[-_ ]?target|safe|control|scramble)",
    min_cells_per_guide: int = 10,
    qc_filter: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Estimate target-gene knockdown using top-guide assignments and pseudobulk CPM.

    This is intended for small pilot or validation datasets with a guide-to-target table.
    It is not a replacement for SCEPTRE; it gives a compact QC panel similar to the
    target-expression validation panels in PerturbSci-style method papers.
    """
    aligned = _prepare_parse_gex_guide_alignment(gex_dir=gex_dir, guide_dir=guide_dir, qc_filter=qc_filter)
    gex = aligned["gex"]
    guide = aligned["guide"]
    qc = aligned["qc"]
    gex_X = ensure_csr(gex["X"])[aligned["gex_pos"], :]
    guide_X = ensure_csr(guide["X"])[aligned["guide_pos"], :]
    guide_names = list(map(str, guide["feature_names"]))
    cell_metrics, _, _ = compute_guide_metrics(
        guide_X,
        guide_names,
        aligned["cell_names"],
        dataset_id=dataset_id,
        sample_id=sample_id,
        method=method,
        cell_type=cell_type,
        guide_threshold=guide_threshold,
        min_cells_per_guide=min_cells_per_guide,
        assignment_mode=assignment_mode,
    )
    meta = _read_any_table(guide_target_table)
    src_guide_col = guide_id_col if guide_id_col in meta.columns else infer_column(meta, (guide_id_col, "guide_id", "guide_name", "grna_id"), ("guide", "grna"))
    src_target_col = target_col if target_col in meta.columns else infer_column(meta, (target_col, "intended_target_name", "target", "gene", "grna_target"), ("target", "gene"))
    guide_map = meta[[src_guide_col, src_target_col]].rename(columns={src_guide_col: "top_guide", src_target_col: "target_gene"}).copy()
    guide_map["top_guide"] = guide_map["top_guide"].astype(str)
    guide_map["target_gene"] = guide_map["target_gene"].astype(str)
    cm = cell_metrics.merge(guide_map, on="top_guide", how="left")
    cm["target_gene"] = cm["target_gene"].fillna("")
    control_mask = cm["assigned_any_guide"].astype(bool) & cm["target_gene"].str.contains(control_target_regex, regex=True, na=False)
    if control_mask.sum() < max(10, min_cells_per_guide):
        # Fall back to cells with no guide for a rough baseline if explicit controls are absent.
        control_mask = ~cm["assigned_any_guide"].astype(bool)
    if control_mask.sum() == 0:
        return pd.DataFrame()

    gene_names = list(map(str, gex["feature_names"]))
    gene_index = {name.upper(): i for i, name in enumerate(gene_names)}
    total_counts = qc["total_counts"].to_numpy(dtype=float)
    total_counts = np.where(total_counts <= 0, np.nan, total_counts)
    rows: list[dict[str, Any]] = []
    control_idx = np.where(control_mask.to_numpy())[0]
    for guide_id, sub in cm[cm["assigned_any_guide"].astype(bool)].groupby("top_guide", dropna=False, observed=True):
        target = str(sub["target_gene"].iloc[0]) if "target_gene" in sub.columns else ""
        if not target or re.search(control_target_regex, target):
            continue
        gene_pos = gene_index.get(target.upper())
        if gene_pos is None:
            continue
        test_idx = sub.index.to_numpy(dtype=int)
        if test_idx.size < min_cells_per_guide:
            continue
        test_expr = to_1d(gex_X[test_idx, gene_pos].toarray()) / total_counts[test_idx] * 1e6
        ctrl_expr = to_1d(gex_X[control_idx, gene_pos].toarray()) / total_counts[control_idx] * 1e6
        test_expr = test_expr[np.isfinite(test_expr)]
        ctrl_expr = ctrl_expr[np.isfinite(ctrl_expr)]
        if test_expr.size == 0 or ctrl_expr.size == 0:
            continue
        mean_test = float(np.mean(test_expr))
        mean_ctrl = float(np.mean(ctrl_expr))
        median_test = float(np.median(test_expr))
        median_ctrl = float(np.median(ctrl_expr))
        log2fc_mean = float(np.log2((mean_test + 1.0) / (mean_ctrl + 1.0)))
        log2fc_median = float(np.log2((median_test + 1.0) / (median_ctrl + 1.0)))
        rows.append(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "method": method,
                "cell_type": cell_type,
                "guide_id": str(guide_id),
                "target_gene": target,
                "n_cells_guide": int(test_idx.size),
                "n_cells_control": int(control_idx.size),
                "mean_cpm_target_cells": mean_test,
                "mean_cpm_control_cells": mean_ctrl,
                "median_cpm_target_cells": median_test,
                "median_cpm_control_cells": median_ctrl,
                "log2fc_mean_vs_control": log2fc_mean,
                "log2fc_median_vs_control": log2fc_median,
                "pct_knockdown_mean": float((1.0 - 2.0 ** log2fc_mean) * 100.0),
                "pct_knockdown_median": float((1.0 - 2.0 ** log2fc_median) * 100.0),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        target_summary = (
            out.groupby(["dataset_id", "sample_id", "method", "cell_type", "target_gene"], dropna=False, observed=True)
            .agg(
                n_guides=("guide_id", "nunique"),
                median_pct_knockdown=("pct_knockdown_mean", "median"),
                median_log2fc=("log2fc_mean_vs_control", "median"),
                total_cells=("n_cells_guide", "sum"),
            )
            .reset_index()
        )
        out = out.merge(target_summary, on=["dataset_id", "sample_id", "method", "cell_type", "target_gene"], how="left")
    return out


def compute_target_knockdown_from_config(config: Mapping[str, Any]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    opt = _optional_panel_config(config)
    entries = opt.get("target_knockdown", [])
    if isinstance(entries, Mapping):
        entries = [entries]
    all_rows: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for i, entry in enumerate(entries or []):
        if not entry or not entry.get("enabled", True):
            continue
        try:
            table = compute_target_knockdown_from_parse(
                gex_dir=entry["gex_dir"],
                guide_dir=entry["guide_dir"],
                guide_target_table=entry["guide_target_table"],
                dataset_id=entry.get("dataset_id", "target_knockdown"),
                sample_id=entry.get("sample_id", Path(str(entry.get("gex_dir", "sample"))).name),
                method=entry.get("method", "Circular capture"),
                cell_type=entry.get("cell_type", ""),
                guide_threshold=entry.get("guide_threshold", 5),
                assignment_mode=entry.get("assignment_mode", "single_guide"),
                guide_id_col=entry.get("guide_id_col", "grna_id"),
                target_col=entry.get("target_col", "grna_target"),
                control_target_regex=entry.get("control_target_regex", r"(?i)(^ntc|non[-_ ]?target|safe|control|scramble)"),
                min_cells_per_guide=entry.get("min_cells_per_guide", 10),
                qc_filter=config.get("qc_filter_parse", {}),
            )
            if not table.empty:
                all_rows.append(table)
        except Exception as exc:
            errors.append({"dataset_key": f"optional_panels.target_knockdown.{i}", "error": repr(exc)})
    return (pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()), errors


def run_optional_analyses(config: Mapping[str, Any], outputs: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add derived tables for main/supplementary QC panels."""
    outs = dict(outputs)
    errors = _bundle_safe_table(outs, "errors")
    error_rows: list[dict[str, str]] = []
    cell_metrics = _bundle_safe_table(outs, "cell_metrics")
    guide_metrics = _bundle_safe_table(outs, "guide_metrics")
    sample_summary = _bundle_safe_table(outs, "sample_summary")

    try:
        outs["assignment_classes"] = compute_assignment_class_summary(cell_metrics)
    except Exception as exc:
        outs["assignment_classes"] = pd.DataFrame()
        error_rows.append({"dataset_key": "derived.assignment_classes", "error": repr(exc)})
    try:
        outs["guide_detection_curve"] = compute_guide_detection_curve(guide_metrics, sample_summary)
    except Exception as exc:
        outs["guide_detection_curve"] = pd.DataFrame()
        error_rows.append({"dataset_key": "derived.guide_detection_curve", "error": repr(exc)})

    primer_raw, errs = _load_external_table_list(config, "primer_optimization_tables")
    error_rows.extend(errs)
    try:
        outs["primer_optimization"] = standardize_primer_optimization_table(primer_raw)
    except Exception as exc:
        outs["primer_optimization"] = pd.DataFrame()
        error_rows.append({"dataset_key": "optional.primer_optimization", "error": repr(exc)})

    read_origin_raw, errs = _load_external_table_list(config, "read_origin_tables")
    error_rows.extend(errs)
    try:
        outs["read_origin"] = standardize_read_origin_table(read_origin_raw)
    except Exception as exc:
        outs["read_origin"] = pd.DataFrame()
        error_rows.append({"dataset_key": "optional.read_origin", "error": repr(exc)})

    rep_corr, errs = compute_replicate_correlations_from_config(config)
    outs["replicate_correlations"] = rep_corr
    error_rows.extend(errs)

    bulk_corr, errs = compute_bulk_singlecell_correlation_from_config(config, guide_metrics)
    outs["bulk_singlecell_correlations"] = bulk_corr
    error_rows.extend(errs)

    kd, errs = compute_target_knockdown_from_config(config)
    outs["target_knockdown"] = kd
    error_rows.extend(errs)

    if error_rows:
        add = pd.DataFrame(error_rows)
        outs["errors"] = pd.concat([errors, add], ignore_index=True) if not errors.empty else add
    else:
        outs["errors"] = errors
    return outs


# Additional plot functions for main/supplementary panels


def plot_assignment_classes_stacked(assignment_classes: pd.DataFrame, *, dataset_id: str | None = None, title: str = ""):
    from plotnine import ggplot, aes, geom_col, scale_y_continuous, labs, theme, element_text
    from mizani.formatters import percent_format

    df = assignment_classes.copy()
    if dataset_id is not None and "dataset_id" in df.columns:
        df = df[df["dataset_id"] == dataset_id]
    if df.empty:
        raise ValueError("No assignment class summary available.")
    if "guide_assignment_class" in df.columns:
        order = ["No guide", "Ambiguous/multiplet", "One guide", "Multiple guides", "Singlet", "Assigned"]
        present = [x for x in order if x in set(df["guide_assignment_class"].astype(str))]
        df["guide_assignment_class"] = pd.Categorical(df["guide_assignment_class"], categories=present, ordered=True)
    return (
        ggplot(df, aes(x="method", y="fraction", fill="guide_assignment_class"))
        + geom_col(width=0.62)
        + scale_y_continuous(labels=percent_format(), limits=(0, 1))
        + labs(x="", y="Cells", fill="Assignment", title=title)
        + fig_theme()
        + theme(axis_text_x=element_text(rotation=25, ha="right"))
    )


def plot_guide_detection_curve(guide_detection_curve: pd.DataFrame, *, dataset_id: str | None = None, title: str = ""):
    from plotnine import ggplot, aes, geom_line, geom_point, scale_color_manual, labs

    df = guide_detection_curve.copy()
    if dataset_id is not None and "dataset_id" in df.columns:
        df = df[df["dataset_id"] == dataset_id]
    if df.empty:
        raise ValueError("No guide detection curve available.")
    return (
        ggplot(df, aes(x="guide_rank_fraction", y="n_cells_ge_threshold", color="method", group="sample_id"))
        + geom_line(size=0.65)
        + geom_point(size=0.9, alpha=0.75)
        + scale_color_manual(values=CAPTURE_COLORS)
        + labs(x="Guides ranked by recovery", y="Cells per guide", title=title)
        + fig_theme()
    )


def plot_cell_throughput(sample_summary: pd.DataFrame, *, title: str = ""):
    from plotnine import ggplot, aes, geom_col, scale_fill_manual, labs, theme, element_text

    df = sample_summary.copy()
    if df.empty or "n_cells" not in df.columns:
        raise ValueError("No sample-level cell-count summary available.")
    df["sample_label"] = df.get("cell_type", "").astype(str) + "\n" + df.get("method", "").astype(str)
    return (
        ggplot(df, aes(x="sample_label", y="n_cells", fill="method"))
        + geom_col(width=0.68)
        + scale_fill_manual(values=CAPTURE_COLORS)
        + labs(x="", y="Cells passing QC", title=title)
        + fig_theme()
        + theme(axis_text_x=element_text(rotation=30, ha="right"), legend_position="none")
    )


def plot_transcriptome_qc_summary(sample_summary: pd.DataFrame, *, dataset_id: str | None = None, title: str = ""):
    from plotnine import ggplot, aes, geom_point, scale_color_manual, labs

    df = sample_summary.copy()
    if dataset_id is not None and "dataset_id" in df.columns:
        df = df[df["dataset_id"] == dataset_id]
    needed = {"median_gex_umis", "median_genes"}
    if df.empty or not needed.issubset(df.columns):
        raise ValueError("No sample-level transcriptome QC summary available.")
    return (
        ggplot(df, aes(x="median_gex_umis", y="median_genes", color="method"))
        + geom_point(size=2.5)
        + scale_color_manual(values=CAPTURE_COLORS)
        + labs(x="Median transcriptome UMIs/cell", y="Median genes/cell", title=title)
        + fig_theme()
    )


def plot_transcriptome_gene_saturation(saturation: pd.DataFrame, *, dataset_id: str | None = None, title: str = ""):
    from plotnine import ggplot, aes, geom_line, geom_point, scale_color_manual, labs

    df = saturation.copy()
    if dataset_id is not None and "dataset_id" in df.columns:
        df = df[df["dataset_id"] == dataset_id]
    df = df[df["modality"].astype(str).str.lower().isin(["gex", "transcriptome", "mrna"])] if "modality" in df.columns else df
    if df.empty or "median_features_per_cell" not in df.columns:
        raise ValueError("No gene-count saturation rows found.")
    return (
        ggplot(df, aes(x="reads_per_cell", y="median_features_per_cell", color="method", group="method"))
        + geom_line(size=0.7)
        + geom_point(size=1.8)
        + scale_color_manual(values=CAPTURE_COLORS)
        + labs(x="Reads per cell", y="Median detected genes/cell", title=title)
        + fig_theme()
    )


def plot_primer_optimization(primer_optimization: pd.DataFrame, *, title: str = ""):
    from plotnine import ggplot, aes, geom_boxplot, geom_point, labs, theme, element_text

    df = primer_optimization.copy()
    if df.empty or "guide_umis_per_cell" not in df.columns:
        raise ValueError("No primer optimization guide-UMI table available.")
    df = df[np.isfinite(df["guide_umis_per_cell"])]
    if df.empty:
        raise ValueError("Primer optimization table has no finite guide_umis_per_cell values.")
    return (
        ggplot(df, aes(x="primer_concentration", y="guide_umis_per_cell"))
        + geom_boxplot(width=0.55, outlier_alpha=0.25)
        + geom_point(alpha=0.25, size=0.7)
        + labs(x="Guide RT primer concentration", y="Guide UMIs/cell", title=title)
        + fig_theme()
        + theme(axis_text_x=element_text(rotation=30, ha="right"))
    )


def plot_read_origin_fraction(read_origin: pd.DataFrame, *, title: str = ""):
    from plotnine import ggplot, aes, geom_col, scale_y_continuous, labs, theme, element_text
    from mizani.formatters import percent_format

    df = read_origin.copy()
    if df.empty or "fraction" not in df.columns:
        raise ValueError("No read-origin fraction table available.")
    df["sample_label"] = df.get("sample_id", "sample").astype(str)
    return (
        ggplot(df, aes(x="sample_label", y="fraction", fill="read_source"))
        + geom_col(width=0.65)
        + scale_y_continuous(labels=percent_format(), limits=(0, 1))
        + labs(x="", y="Guide-read fraction", fill="Read source", title=title)
        + fig_theme()
        + theme(axis_text_x=element_text(rotation=30, ha="right"))
    )


def plot_replicate_correlation_heatmap(replicate_correlations: pd.DataFrame, *, title: str = ""):
    from plotnine import ggplot, aes, geom_tile, geom_text, labs, theme, element_text, coord_fixed

    df = replicate_correlations.copy()
    if df.empty:
        raise ValueError("No replicate correlation matrix available.")
    df["label"] = df["correlation"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "")
    return (
        ggplot(df, aes(x="replicate_1", y="replicate_2", fill="correlation"))
        + geom_tile()
        + geom_text(aes(label="label"), size=6)
        + coord_fixed()
        + labs(x="", y="", fill="r", title=title)
        + fig_theme()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
    )


def plot_bulk_singlecell_correlation(bulk_singlecell_correlations: pd.DataFrame, *, title: str = ""):
    from plotnine import ggplot, aes, geom_point, geom_smooth, labs

    df = bulk_singlecell_correlations.copy()
    if df.empty:
        raise ValueError("No bulk-vs-single-cell correlation table available.")
    return (
        ggplot(df, aes(x="log1p_bulk_count", y="log1p_single_cell_count"))
        + geom_point(alpha=0.45, size=0.8)
        + geom_smooth(method="lm", se=False, size=0.5)
        + labs(x="log1p bulk guide reads", y="log1p single-cell guide recovery", title=title)
        + fig_theme()
    )


def plot_target_knockdown(target_knockdown: pd.DataFrame, *, title: str = ""):
    from plotnine import ggplot, aes, geom_hline, geom_point, labs, theme, element_text

    df = target_knockdown.copy()
    if df.empty or "log2fc_mean_vs_control" not in df.columns:
        raise ValueError("No target-knockdown table available.")
    df = df[np.isfinite(df["log2fc_mean_vs_control"])]
    if df.empty:
        raise ValueError("Target-knockdown table has no finite log2FC values.")
    return (
        ggplot(df, aes(x="target_gene", y="log2fc_mean_vs_control"))
        + geom_hline(yintercept=0, linetype="dashed", size=0.35)
        + geom_point(alpha=0.75, size=1.4)
        + labs(x="Target gene", y="Target expression log2FC vs control", title=title)
        + fig_theme()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
    )


# -----------------------------
# Pipeline orchestration
# -----------------------------


def default_config() -> dict[str, Any]:
    return {'out_dir': '/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_cc_capture',
     'qc_filter_parse': {'min_genes': 300, 'max_genes': 10000, 'max_counts': 100000, 'max_pct_mt': 10},
     'qc_filter_h5ad': {'min_genes': 100, 'max_genes': None, 'max_counts': None, 'max_pct_mt': 25},
     'datasets': {'ipsc_single_cc': {'enabled': True,
                                     'type': 'parse_embedded_guides',
                                     'matrix_dir': '/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/230820_eTN2_iPSC_mixed/panmg/BS3/DGE_filtered',
                                     'guide_feature_regex': 'sg_',
                                     'dataset_id': 'ipsc_single_guide',
                                     'sample_id': 'hiPSC_circular_capture',
                                     'method': 'Circular capture',
                                     'cell_type': 'hiPSC',
                                     'guide_threshold': 5,
                                     'assignment_mode': 'single_guide'},
                  'ipsc_single_crop': {'enabled': True,
                                       'type': 'parse_embedded_guides',
                                       'matrix_dir': '/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/230621_Tri_guide_iPSCs/iPSC/iPSC_guide/DGE_filtered',
                                       'guide_feature_regex': 'b08|sg_',
                                       'dataset_id': 'ipsc_single_guide',
                                       'sample_id': 'hiPSC_CROP_seq',
                                       'method': 'CROP-seq',
                                       'cell_type': 'hiPSC',
                                       'guide_threshold': 5,
                                       'assignment_mode': 'single_guide'},
                  'telohaec_cc': {'enabled': True,
                                  'type': 'h5ad_pair',
                                  'root_dir': '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/tony_results',
                                  'gex_h5ad': 'GEX2.h5ad',
                                  'guide_h5ad': 'GUIDE1.h5ad',
                                  'dataset_id': 'telohaec_capture_comparison',
                                  'sample_id': 'teloHAEC_circular_capture',
                                  'method': 'Circular capture',
                                  'cell_type': 'teloHAEC',
                                  'guide_threshold': 5,
                                  'assignment_mode': 'high_moi',
                                  'gex_layers_to_sum': ['ambiguous', 'mature', 'nascent']},
                  'telohaec_crop': {'enabled': True,
                                    'type': 'h5ad_pair',
                                    'root_dir': '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/tony_results',
                                    'gex_h5ad': 'CROP_GEX1.h5ad',
                                    'guide_h5ad': 'CROP_GUIDE1.h5ad',
                                    'dataset_id': 'telohaec_capture_comparison',
                                    'sample_id': 'teloHAEC_CROP_seq',
                                    'method': 'CROP-seq',
                                    'cell_type': 'teloHAEC',
                                    'guide_threshold': 3,
                                    'assignment_mode': 'high_moi',
                                    'gex_layers_to_sum': ['ambiguous', 'mature', 'nascent']},
                  'ipsc_pilot_cc': {'enabled': True,
                                    'type': 'parse_separate_gex_guides',
                                    'gex_dir': '/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/231210_eTN3_iPSC_Pilot/WT_combined/MOI8',
                                    'guide_dir': '/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/231210_eTN3_iPSC_Pilot/crispr_combined/MOI8',
                                    'dataset_id': 'ipsc_10gene_pilot',
                                    'sample_id': 'hiPSC_10gene_pilot',
                                    'method': 'Circular capture',
                                    'cell_type': 'hiPSC pilot',
                                    'guide_threshold': 5,
                                    'assignment_mode': 'high_moi',
                                    'guide_target_table': '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/figure_1/eTN3_gRNA_group_table.csv',
                                    'guide_group_pairs': '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/figure_1/eTN3_gRNA_group_pairs.csv',
                                    'guide_id_col': 'grna_id',
                                    'target_col': 'grna_target'}},
     'saturation': {'enabled': True,
                    'read_counts': [1000000, 3000000, 10000000, 30000000],
                    'curves': [{'dataset_id': 'telohaec_capture_comparison',
                                'sample_id': 'teloHAEC_circular_capture_guide_saturation',
                                'method': 'Circular capture',
                                'cell_type': 'teloHAEC',
                                'modality': 'guide',
                                'pattern': '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/tony_results/GUIDE_S1.{reads}_kb_kite/counts_unfiltered_modified/adata.h5ad',
                                'cells_from_sample_id': 'teloHAEC_circular_capture'},
                               {'dataset_id': 'telohaec_capture_comparison',
                                'sample_id': 'teloHAEC_CROP_seq_guide_saturation',
                                'method': 'CROP-seq',
                                'cell_type': 'teloHAEC',
                                'modality': 'guide',
                                'pattern': '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/tony_results/CROP_GUIDE1_S4.{reads}_kb_kite/counts_unfiltered_modified/adata.h5ad',
                                'cells_from_sample_id': 'teloHAEC_CROP_seq'}],
                    'transcriptome_saturation_note': 'True hiPSC GEX/transcriptome saturation inputs are still needed for '
                                                     'manuscript panel 1d saturation. The pipeline exports a transcriptome '
                                                     'UMI/QC fallback until those paths are added.'},
     'upload_bundle': {'max_cells_per_sample': 50000, 'include_full_cell_metrics': False, 'histogram_bins': 80},
     'guide_annotations': {'ipsc_pilot': {'guide_target_table': '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/figure_1/eTN3_gRNA_group_table.csv',
                                          'guide_group_pairs': '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/figure_1/eTN3_gRNA_group_pairs.csv'}},
     'handoff': {'enabled': True,
                 'out_dir': None,
                 'max_cells_per_group': 20000,
                 'random_seed': 1729,
                 'include_full_cell_metrics': False},
     'optional_panels': {'primer_optimization_tables': [],
                         'read_origin_tables': [],
                         'replicate_count_tables': [],
                         'bulk_guide_count_tables': [],
                         'target_knockdown': [{'enabled': False,
                                              'gex_dir': '/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/231210_eTN3_iPSC_Pilot/WT_combined/MOI8',
                                              'guide_dir': '/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/231210_eTN3_iPSC_Pilot/crispr_combined/MOI8',
                                              'guide_target_table': '/oak/stanford/groups/engreitz/Users/tri/CodeBase/programDE/notebooks/figure_1/eTN3_gRNA_group_table.csv',
                                              'dataset_id': 'ipsc_10gene_pilot',
                                              'sample_id': 'hiPSC_10gene_pilot',
                                              'method': 'Circular capture',
                                              'cell_type': 'hiPSC pilot',
                                              'guide_threshold': 5,
                                              'assignment_mode': 'high_moi',
                                              'guide_id_col': 'grna_id',
                                              'target_col': 'grna_target'}]},
     'notes': {'telohaec_root_dir_interpretation': 'The user supplied /oak/.../tony_results/CROP_GEX1.h5ad. The config '
                                                   'uses the parent directory /oak/.../tony_results as root_dir for all '
                                                   'four h5ad files and kb_kite folders.'}}

def check_config_paths(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, ds in config.get("datasets", {}).items():
        if not ds.get("enabled", False):
            rows.append({"section": "datasets", "key": key, "path_type": "dataset", "path": "disabled", "exists": False})
            continue
        typ = ds.get("type")
        if typ == "parse_embedded_guides":
            base = Path(ds["matrix_dir"])
            rows.append({"section": "datasets", "key": key, "path_type": "matrix_dir", "path": str(base), "exists": base.exists()})
        elif typ == "parse_separate_gex_guides":
            for path_type in ["gex_dir", "guide_dir"]:
                base = Path(ds[path_type])
                rows.append({"section": "datasets", "key": key, "path_type": path_type, "path": str(base), "exists": base.exists()})
            for optional_key in ["guide_target_table", "guide_group_pairs"]:
                if optional_key in ds and ds.get(optional_key):
                    fp = Path(ds[optional_key])
                    rows.append({"section": "datasets", "key": key, "path_type": optional_key, "path": str(fp), "exists": fp.exists()})
        elif typ == "h5ad_pair":
            root = coerce_root_dir(ds.get("root_dir", "."))
            rows.append({"section": "datasets", "key": key, "path_type": "root_dir", "path": str(root), "exists": root.exists()})
            for path_type in ["gex_h5ad", "guide_h5ad"]:
                fp = root / ds[path_type]
                rows.append({"section": "datasets", "key": key, "path_type": path_type, "path": str(fp), "exists": fp.exists()})
        elif typ == "h5mu":
            fp = Path(ds["h5mu_path"])
            rows.append({"section": "datasets", "key": key, "path_type": "h5mu_path", "path": str(fp), "exists": fp.exists()})
        else:
            rows.append({"section": "datasets", "key": key, "path_type": "unknown_type", "path": str(typ), "exists": False})

    if config.get("saturation", {}).get("enabled", False):
        read_counts = config["saturation"].get("read_counts", [])
        for i, curve in enumerate(config["saturation"].get("curves", [])):
            for reads in read_counts:
                fp = Path(curve["pattern"].format(reads=reads))
                rows.append({"section": "saturation", "key": f"curve_{i}", "path_type": str(reads), "path": str(fp), "exists": fp.exists()})

    opt = config.get("optional_panels", {}) or {}
    for opt_key in ["primer_optimization_tables", "read_origin_tables", "replicate_count_tables", "bulk_guide_count_tables"]:
        entries = opt.get(opt_key, [])
        if isinstance(entries, Mapping):
            entries = [entries]
        for i, entry in enumerate(entries or []):
            if not entry or not entry.get("enabled", True):
                continue
            path = entry.get("path") or entry.get("file")
            if path:
                fp = Path(path)
                rows.append({"section": "optional_panels", "key": f"{opt_key}_{i}", "path_type": "path", "path": str(fp), "exists": fp.exists()})
    entries = opt.get("target_knockdown", [])
    if isinstance(entries, Mapping):
        entries = [entries]
    for i, entry in enumerate(entries or []):
        if not entry or not entry.get("enabled", True):
            continue
        for path_type in ["gex_dir", "guide_dir", "guide_target_table"]:
            path = entry.get(path_type)
            if path:
                fp = Path(path)
                rows.append({"section": "optional_panels", "key": f"target_knockdown_{i}", "path_type": path_type, "path": str(fp), "exists": fp.exists()})
    return pd.DataFrame(rows)


def run_reprocessing(config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    out_dir = ensure_dir(config.get("out_dir", "figure1_outputs"))
    all_cell: list[pd.DataFrame] = []
    all_guide: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for key, ds in config.get("datasets", {}).items():
        if not ds.get("enabled", False):
            continue
        try:
            typ = ds.get("type")
            if typ == "parse_embedded_guides":
                cell, guide, summary = process_parse_embedded_guides(
                    matrix_dir=ds["matrix_dir"],
                    guide_feature_regex=ds["guide_feature_regex"],
                    dataset_id=ds["dataset_id"],
                    sample_id=ds["sample_id"],
                    method=ds["method"],
                    cell_type=ds["cell_type"],
                    guide_threshold=ds.get("guide_threshold", 5),
                    min_cells_per_guide=ds.get("min_cells_per_guide", 10),
                    assignment_mode=ds.get("assignment_mode", "single_guide"),
                    qc_filter=config.get("qc_filter_parse", {}),
                )
            elif typ == "parse_separate_gex_guides":
                cell, guide, summary = process_parse_separate_gex_and_guides(
                    gex_dir=ds["gex_dir"],
                    guide_dir=ds["guide_dir"],
                    dataset_id=ds["dataset_id"],
                    sample_id=ds["sample_id"],
                    method=ds["method"],
                    cell_type=ds["cell_type"],
                    guide_threshold=ds.get("guide_threshold", 5),
                    min_cells_per_guide=ds.get("min_cells_per_guide", 10),
                    assignment_mode=ds.get("assignment_mode", "high_moi"),
                    qc_filter=config.get("qc_filter_parse", {}),
                )
            elif typ == "h5ad_pair":
                root = coerce_root_dir(ds.get("root_dir", "."))
                cell, guide, summary = process_h5ad_gex_guide_pair(
                    gex_h5ad=root / ds["gex_h5ad"],
                    guide_h5ad=root / ds["guide_h5ad"],
                    dataset_id=ds["dataset_id"],
                    sample_id=ds["sample_id"],
                    method=ds["method"],
                    cell_type=ds["cell_type"],
                    guide_threshold=ds.get("guide_threshold", 5),
                    min_cells_per_guide=ds.get("min_cells_per_guide", 10),
                    assignment_mode=ds.get("assignment_mode", "high_moi"),
                    qc_filter=config.get("qc_filter_h5ad", {}),
                )
            elif typ == "h5mu":
                cell, guide, summary = process_h5mu_gex_guide(
                    h5mu_path=ds["h5mu_path"],
                    dataset_id=ds["dataset_id"],
                    sample_id=ds["sample_id"],
                    method=ds["method"],
                    cell_type=ds["cell_type"],
                    guide_threshold=ds.get("guide_threshold", 5),
                    min_cells_per_guide=ds.get("min_cells_per_guide", 10),
                    assignment_mode=ds.get("assignment_mode", "high_moi"),
                    qc_filter=config.get("qc_filter_h5ad", {}),
                )
            else:
                raise ValueError(f"Unknown dataset type: {typ}")
            all_cell.append(cell)
            all_guide.append(guide)
            all_summary.append(summary)
        except Exception as exc:  # keep notebook running and report missing paths cleanly
            errors.append({"dataset_key": key, "error": repr(exc)})
            warnings.warn(f"Skipping {key}: {exc!r}")

    cell_metrics = pd.concat(all_cell, ignore_index=True) if all_cell else pd.DataFrame()
    guide_metrics = pd.concat(all_guide, ignore_index=True) if all_guide else pd.DataFrame()
    sample_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()

    saturation_rows: list[pd.DataFrame] = []
    saturation_missing: list[str] = []
    if config.get("saturation", {}).get("enabled", False):
        for curve in config["saturation"].get("curves", []):
            try:
                cells_to_keep = None
                cells_from_sample_id = curve.get("cells_from_sample_id")
                if cells_from_sample_id and not cell_metrics.empty and {"sample_id", "cell_id"}.issubset(cell_metrics.columns):
                    candidate_cells = cell_metrics.loc[
                        cell_metrics["sample_id"].astype(str) == str(cells_from_sample_id),
                        "cell_id",
                    ].astype(str)
                    if len(candidate_cells) > 0:
                        cells_to_keep = candidate_cells.tolist()
                    else:
                        warnings.warn(
                            f"No cells found for cells_from_sample_id={cells_from_sample_id!r}; "
                            "using all cells in saturation h5ad files."
                        )
                sat, missing = collect_h5ad_saturation_curve(
                    pattern=curve["pattern"],
                    read_counts=config["saturation"].get("read_counts", []),
                    method=curve["method"],
                    dataset_id=curve["dataset_id"],
                    sample_id=curve["sample_id"],
                    cell_type=curve["cell_type"],
                    modality=curve["modality"],
                    cells_to_keep=cells_to_keep,
                    layers_to_sum=curve.get("layers_to_sum"),
                )
                saturation_rows.append(sat)
                saturation_missing.extend(missing)
            except Exception as exc:
                errors.append({"dataset_key": curve.get("sample_id", "saturation"), "error": repr(exc)})
                warnings.warn(f"Skipping saturation curve {curve.get('sample_id')}: {exc!r}")

    saturation = pd.concat(saturation_rows, ignore_index=True) if saturation_rows else pd.DataFrame()
    errors_df = pd.DataFrame(errors)
    missing_df = pd.DataFrame({"missing_saturation_path": saturation_missing})

    outputs = {
        "cell_metrics": cell_metrics,
        "guide_metrics": guide_metrics,
        "sample_summary": sample_summary,
        "saturation": saturation,
        "errors": errors_df,
        "missing_saturation": missing_df,
    }
    outputs = run_optional_analyses(config, outputs)
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)
    with open(out_dir / "config_used.json", "w") as handle:
        json.dump(config, handle, indent=2)
    return outputs


def generate_figure1_panels(outputs: Mapping[str, pd.DataFrame], out_dir: str | os.PathLike[str]) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    rows: list[dict[str, str | float]] = []
    cell_metrics = outputs.get("cell_metrics", pd.DataFrame())
    summary = outputs.get("sample_summary", pd.DataFrame())
    saturation = outputs.get("saturation", pd.DataFrame())
    assignment_classes = outputs.get("assignment_classes", pd.DataFrame())
    guide_detection_curve = outputs.get("guide_detection_curve", pd.DataFrame())
    primer_optimization = outputs.get("primer_optimization", pd.DataFrame())
    read_origin = outputs.get("read_origin", pd.DataFrame())
    replicate_correlations = outputs.get("replicate_correlations", pd.DataFrame())
    bulk_singlecell_correlations = outputs.get("bulk_singlecell_correlations", pd.DataFrame())
    target_knockdown = outputs.get("target_knockdown", pd.DataFrame())

    if not cell_metrics.empty:
        try:
            p = plot_guide_umis_distribution(
                cell_metrics,
                dataset_id="ipsc_single_guide",
                cell_type="hiPSC",
                y_col="guide_umis_top",
                title="hiPSC guide capture",
            )
            fp = save_plot(p, out_dir / "fig1b_guide_umis_hiPSC.svg", width=2.25, height=2.0)
            rows.append({"panel": "1b", "description": "Guide UMIs/cell in hiPSCs", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "1b", "description": f"not generated: {exc}", "file": ""})

        try:
            p = plot_transcriptome_qc_distribution(
                cell_metrics,
                dataset_id="ipsc_single_guide",
                title="Transcriptome complexity",
            )
            fp = save_plot(p, out_dir / "fig1d_transcriptome_umis_hiPSC.svg", width=2.25, height=2.0)
            rows.append({"panel": "1d_fallback", "description": "Transcriptome UMIs/cell distribution; use if GEX saturation is unavailable", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "1d_fallback", "description": f"not generated: {exc}", "file": ""})

        try:
            p = plot_other_celltype_guide_umis(
                cell_metrics,
                exclude_dataset_id="ipsc_single_guide",
                y_col="guide_umis_top",
                title="Guide capture across cell types",
            )
            fp = save_plot(p, out_dir / "fig1e_guide_umis_other_celltypes.svg", width=3.0, height=2.2)
            rows.append({"panel": "1e", "description": "Guide UMIs/cell in additional cell types", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "1e", "description": f"not generated: {exc}", "file": ""})

    if not summary.empty:
        try:
            p = plot_assignment_metrics(summary, dataset_id="ipsc_single_guide", title="Guide assignment")
            fp = save_plot(p, out_dir / "fig1c_assignment_metrics.svg", width=3.0, height=2.2)
            rows.append({"panel": "1c", "description": "Cells assigned a guide and guides detected", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "1c", "description": f"not generated: {exc}", "file": ""})

    if not saturation.empty:
        try:
            p = plot_transcriptome_saturation(saturation, dataset_id="ipsc_single_guide", title="Transcriptome saturation")
            fp = save_plot(p, out_dir / "fig1d_transcriptome_saturation_hiPSC.svg", width=2.5, height=2.0)
            rows.append({"panel": "1d", "description": "Transcriptome saturation", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "1d", "description": f"not generated: {exc}", "file": ""})

        try:
            p = plot_guide_saturation(saturation, dataset_id="telohaec_capture_comparison", title="Guide saturation")
            fp = save_plot(p, out_dir / "figS1_guide_saturation_teloHAEC.svg", width=2.5, height=2.0)
            rows.append({"panel": "supp", "description": "Guide saturation in teloHAEC", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

        try:
            p = plot_transcriptome_gene_saturation(saturation, dataset_id="ipsc_single_guide", title="Gene detection saturation")
            fp = save_plot(p, out_dir / "figS1_transcriptome_gene_saturation_hiPSC.svg", width=2.5, height=2.0)
            rows.append({"panel": "supp", "description": "Detected genes/cell transcriptome saturation", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

    if not assignment_classes.empty:
        try:
            p = plot_assignment_classes_stacked(assignment_classes, dataset_id="ipsc_single_guide", title="Guide assignment classes")
            fp = save_plot(p, out_dir / "figS1_assignment_classes_hiPSC.svg", width=2.35, height=2.0)
            rows.append({"panel": "supp", "description": "Stacked no-guide/singlet/multiplet assignment classes", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

    if not guide_detection_curve.empty:
        try:
            p = plot_guide_detection_curve(guide_detection_curve, dataset_id="ipsc_single_guide", title="Guide recovery by guide")
            fp = save_plot(p, out_dir / "figS1_guide_detection_curve_hiPSC.svg", width=2.6, height=2.0)
            rows.append({"panel": "supp", "description": "Per-guide recovery curve", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

    if not summary.empty:
        try:
            p = plot_cell_throughput(summary, title="Cell recovery")
            fp = save_plot(p, out_dir / "figS1_cell_throughput.svg", width=3.3, height=2.2)
            rows.append({"panel": "supp", "description": "Cells passing QC by sample", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})
        try:
            p = plot_transcriptome_qc_summary(summary, dataset_id="ipsc_single_guide", title="Transcriptome QC summary")
            fp = save_plot(p, out_dir / "figS1_transcriptome_qc_summary_hiPSC.svg", width=2.4, height=2.0)
            rows.append({"panel": "supp", "description": "Median UMIs vs median genes per cell", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

    if not primer_optimization.empty:
        try:
            p = plot_primer_optimization(primer_optimization, title="Guide RT primer optimization")
            fp = save_plot(p, out_dir / "figS1_primer_optimization.svg", width=3.0, height=2.1)
            rows.append({"panel": "supp", "description": "Guide RT primer / concentration optimization", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

    if not read_origin.empty:
        try:
            p = plot_read_origin_fraction(read_origin, title="Guide reads by RT source")
            fp = save_plot(p, out_dir / "figS1_read_origin_fraction.svg", width=2.6, height=2.0)
            rows.append({"panel": "supp", "description": "Fraction of guide reads from oligo-dT versus guide RT", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

    if not replicate_correlations.empty:
        try:
            p = plot_replicate_correlation_heatmap(replicate_correlations, title="Guide count replicate correlations")
            fp = save_plot(p, out_dir / "figS1_replicate_correlation_heatmap.svg", width=2.4, height=2.2)
            rows.append({"panel": "supp", "description": "Replicate guide-count correlation heatmap", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

    if not bulk_singlecell_correlations.empty:
        try:
            p = plot_bulk_singlecell_correlation(bulk_singlecell_correlations, title="Bulk vs single-cell guide recovery")
            fp = save_plot(p, out_dir / "figS1_bulk_singlecell_correlation.svg", width=2.3, height=2.0)
            rows.append({"panel": "supp", "description": "Bulk guide counts versus single-cell guide recovery", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

    if not target_knockdown.empty:
        try:
            p = plot_target_knockdown(target_knockdown, title="Target knockdown QC")
            fp = save_plot(p, out_dir / "figS1_target_knockdown.svg", width=3.0, height=2.1)
            rows.append({"panel": "supp", "description": "Target-gene expression log2FC for assigned guides", "file": str(fp)})
        except Exception as exc:
            rows.append({"panel": "supp", "description": f"not generated: {exc}", "file": ""})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "figure1_panel_manifest.tsv", sep="\t", index=False)
    return manifest

# -----------------------------
# Optional SCEPTRE handoff helpers
# -----------------------------


def _write_vector_csv(values: Sequence[str], fp: str | os.PathLike[str]) -> None:
    pd.Series(list(map(str, values))).to_csv(fp, index=False, header=False)


def write_sceptre_inputs_from_parse(
    *,
    gex_dir: str | os.PathLike[str],
    guide_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    guide_target_table: str | os.PathLike[str] | None = None,
    guide_id_col: str = "grna_id",
    target_col: str = "grna_target",
    qc_filter: Mapping[str, Any] | None = None,
) -> Path:
    """Write SCEPTRE input files from Parse GEX and guide matrices.

    This mirrors the handoff pattern in the attached R/Jupyter analyses but keeps the
    conversion in Python. SCEPTRE expects response_matrix and grna_matrix as
    features x cells, plus guide and cell metadata.
    """
    out = ensure_dir(out_dir)
    gex = read_parse_matrix(as_path(gex_dir) / "DGE_filtered")
    guide = read_parse_matrix(
        as_path(guide_dir) / "guide_RNAs_filtered",
        feature_csv_candidates=("all_guides.csv", "all_genes.csv", "features.csv"),
        feature_name_preferred=("guide_id", "guide_name", "gene_name", "name", "feature_name"),
    )

    qc = qc_from_matrix(gex["X"], gex["feature_names"])
    qc.index = gex["cell_names"]
    keep = filter_cells_by_qc(qc, **dict(qc_filter or {}))
    gex_cells = pd.Index(np.array(gex["cell_names"], dtype=object).astype(str))
    guide_cells = pd.Index(np.array(guide["cell_names"], dtype=object).astype(str))

    if len(gex_cells.intersection(guide_cells)) > 0:
        kept_cells = gex_cells[keep].intersection(guide_cells)
        gex_pos = gex_cells.get_indexer(kept_cells)
        guide_pos = guide_cells.get_indexer(kept_cells)
    else:
        warnings.warn("Could not match GEX and guide cell IDs; using row-order alignment for SCEPTRE export.")
        n = min(int(keep.sum()), guide["X"].shape[0])
        gex_pos = np.where(keep)[0][:n]
        guide_pos = np.arange(n)
        kept_cells = pd.Index([str(x) for x in np.array(gex["cell_names"], dtype=object)[gex_pos]])

    response = gex["X"][gex_pos, :].T.tocsr()
    grna = guide["X"][guide_pos, :].T.tocsr()
    sio.mmwrite(out / "response_matrix.mtx", response)
    sio.mmwrite(out / "grna_matrix.mtx", grna)
    _write_vector_csv(gex["feature_names"], out / "gene_names.csv")
    _write_vector_csv(guide["feature_names"], out / "guide_names.csv")
    _write_vector_csv(kept_cells.astype(str).tolist(), out / "cell_names.csv")

    covariates = qc.iloc[gex_pos].copy()
    covariates.index = kept_cells.astype(str)
    covariates = covariates.rename(
        columns={
            "total_counts": "response_n_umis",
            "n_genes_by_counts": "response_n_nonzero",
            "pct_counts_mt": "mt_percent",
        }
    )
    covariates.to_csv(out / "covariates.csv")

    guides = pd.DataFrame({"grna_id": list(map(str, guide["feature_names"]))})
    if guide_target_table is not None and Path(guide_target_table).exists():
        meta = pd.read_csv(guide_target_table)
        src_guide_col = guide_id_col if guide_id_col in meta.columns else infer_column(meta, (guide_id_col, "guide_id", "guide_name", "grna_id"), ("guide", "grna"))
        src_target_col = target_col if target_col in meta.columns else infer_column(meta, (target_col, "intended_target_name", "target", "gene"), ("target", "gene"))
        guide_mapping = guides.merge(
            meta[[src_guide_col, src_target_col]].rename(columns={src_guide_col: "grna_id", src_target_col: "grna_target"}),
            on="grna_id",
            how="left",
        )
    else:
        guide_mapping = guides.copy()
        guide_mapping["grna_target"] = guide_mapping["grna_id"].str.replace(r"([_+\-].*)$", "", regex=True)
        warnings.warn(
            "No guide_target_table was supplied. Guide targets were guessed from guide IDs; "
            "replace guide_mapping.csv before publication/SCEPTRE analysis if this is not exact."
        )
    guide_mapping.to_csv(out / "guide_mapping.csv", index=False)

    gene_map = pd.DataFrame({"response_id": list(map(str, gex["feature_names"])), "grna_target": list(map(str, gex["feature_names"]))})
    gene_map.to_csv(out / "gene_symbol_map.csv", index=False)

    return out


SCEPTRE_R_TEMPLATE = r'''
library(sceptre)
library(Matrix)

data_dir <- "{data_dir}"
response_mat <- readMM(file.path(data_dir, "response_matrix.mtx"))
grna_mat <- readMM(file.path(data_dir, "grna_matrix.mtx"))
gene_names <- read.csv(file.path(data_dir, "gene_names.csv"), header=FALSE)$V1
guide_names <- read.csv(file.path(data_dir, "guide_names.csv"), header=FALSE)$V1
cell_names <- read.csv(file.path(data_dir, "cell_names.csv"), header=FALSE)$V1
rownames(response_mat) <- gene_names
colnames(response_mat) <- cell_names
rownames(grna_mat) <- guide_names
colnames(grna_mat) <- cell_names

covariates <- read.csv(file.path(data_dir, "covariates.csv"), row.names=1)
guide_mapping <- read.csv(file.path(data_dir, "guide_mapping.csv"))

reserved_cols <- c("response_n_nonzero", "response_n_umis", "grna_n_nonzero", "grna_n_umis", "response_p_mito")
extra_covariates <- covariates[, !(colnames(covariates) %in% reserved_cols), drop=FALSE]

sceptre_object <- import_data(
  response_matrix = response_mat,
  grna_matrix = grna_mat,
  grna_target_data_frame = guide_mapping,
  extra_covariates = extra_covariates,
  moi = "high"
)

gene_map <- read.csv(file.path(data_dir, "gene_symbol_map.csv"))
discovery_pairs <- merge(guide_mapping, gene_map, by="grna_target")
discovery_pairs <- unique(discovery_pairs[, c("grna_target", "response_id")])

sceptre_object <- set_analysis_parameters(
  sceptre_object = sceptre_object,
  discovery_pairs = discovery_pairs,
  side = "both",
  control_group = "complement"
)

png(file.path(data_dir, "grna_count_distributions.png"), width=3000, height=2000, res=300)
print(plot_grna_count_distributions(sceptre_object = sceptre_object, n_grnas_to_plot = 9))
dev.off()

sceptre_object <- assign_grnas(sceptre_object = sceptre_object, method = "mixture", parallel = TRUE)
assignments <- get_grna_assignments(sceptre_object)
writeMM(assignments, file.path(data_dir, "grna_assignments.mtx"))

sceptre_object <- run_qc(sceptre_object = sceptre_object, p_mito_threshold = {p_mito_threshold})
sceptre_object <- run_calibration_check(sceptre_object = sceptre_object, parallel = FALSE)
write.csv(get_result(sceptre_object, analysis="run_calibration_check"), file.path(data_dir, "calibration_result.csv"), row.names=FALSE)

sceptre_object <- run_discovery_analysis(sceptre_object = sceptre_object, parallel = FALSE)
write.csv(get_result(sceptre_object, analysis="run_discovery_analysis"), file.path(data_dir, "TargetKD.csv"), row.names=FALSE)
'''


# -----------------------------
# Shareable plot-input bundle
# -----------------------------


def write_plot_bundle_table(df: pd.DataFrame, fp: str | os.PathLike[str], *, index: bool = False) -> Path:
    """Write a TSV or TSV.GZ table, creating parent directories."""
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(fp).endswith(".gz") else None
    df.to_csv(fp, sep="\t", index=index, compression=compression)
    return fp


def load_outputs_from_dir(out_dir: str | os.PathLike[str]) -> dict[str, pd.DataFrame]:
    """Load standard pipeline TSV outputs from an existing Figure 1 output directory."""
    base = as_path(out_dir)
    outputs: dict[str, pd.DataFrame] = {}
    for name in [
        "cell_metrics",
        "guide_metrics",
        "sample_summary",
        "saturation",
        "errors",
        "missing_saturation",
        "path_check",
        "assignment_classes",
        "guide_detection_curve",
        "primer_optimization",
        "read_origin",
        "replicate_correlations",
        "bulk_singlecell_correlations",
        "target_knockdown",
    ]:
        candidates = [base / f"{name}.tsv", base / f"{name}.tsv.gz"]
        fp = next((x for x in candidates if x.exists()), None)
        if fp is None:
            outputs[name] = pd.DataFrame()
        else:
            try:
                outputs[name] = pd.read_csv(fp, sep="\t", compression="infer")
            except pd.errors.EmptyDataError:
                outputs[name] = pd.DataFrame()
    return outputs


def _bundle_safe_table(outputs: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    obj = outputs.get(name, pd.DataFrame())
    return obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)


def _panel_cell_columns(df: pd.DataFrame) -> list[str]:
    wanted = [
        "dataset_id",
        "sample_id",
        "method",
        "cell_type",
        "cell_id",
        "cell_id_hash",
        "guide_umis_top",
        "guide_umis_total",
        "second_guide_umis",
        "top_guide",
        "top_guide_share",
        "top_second_ratio",
        "n_guides_ge_threshold",
        "assigned_any_guide",
        "assigned_confident_guide",
        "guide_assignment_class",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
    ]
    return [c for c in wanted if c in df.columns]


def _bundle_summary_stats(df: pd.DataFrame, group_cols: Sequence[str], value_col: str) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    available_groups = [c for c in group_cols if c in df.columns]
    use = df[available_groups + [value_col]].copy()
    use = use[np.isfinite(use[value_col])]
    if use.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    groupby_obj = use.groupby(available_groups, dropna=False, observed=True) if available_groups else [((), use)]
    for keys, sub in groupby_obj:
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = sub[value_col].to_numpy(dtype=float)
        row = {col: key for col, key in zip(available_groups, keys)}
        row.update(
            {
                "metric": value_col,
                "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "q1": float(np.quantile(vals, 0.25)),
                "q3": float(np.quantile(vals, 0.75)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _bundle_stratified_sample(
    df: pd.DataFrame,
    *,
    strata_cols: Sequence[str],
    max_rows_per_stratum: int | None = 50000,
    random_state: int = 1,
) -> pd.DataFrame:
    """Deterministic downsample for fast preview plotting while retaining full tables separately."""
    if df.empty or max_rows_per_stratum is None or max_rows_per_stratum <= 0:
        return df.copy()
    pieces: list[pd.DataFrame] = []
    available = [c for c in strata_cols if c in df.columns]
    if not available:
        return df.sample(n=min(len(df), max_rows_per_stratum), random_state=random_state).copy()
    for _, sub in df.groupby(available, dropna=False, observed=True):
        if len(sub) > max_rows_per_stratum:
            pieces.append(sub.sample(n=max_rows_per_stratum, random_state=random_state))
        else:
            pieces.append(sub)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=df.columns)


def _bundle_assignment_long_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    metric_map = {
        "pct_cells_any_guide": "Cells with >=1 guide",
        "pct_cells_confident_guide": "Confidently assigned cells",
        "pct_guides_detected_min_cells": "Guides detected above threshold",
    }
    value_vars = [c for c in metric_map if c in summary.columns]
    if not value_vars:
        return pd.DataFrame()
    id_vars = [c for c in ["dataset_id", "sample_id", "method", "cell_type", "n_cells", "n_guides", "guide_threshold", "min_cells_per_guide", "assignment_mode"] if c in summary.columns]
    long = summary.melt(id_vars=id_vars, value_vars=value_vars, var_name="metric", value_name="fraction")
    long["metric_label"] = long["metric"].map(metric_map)
    long["percent"] = long["fraction"] * 100
    return long


def _bundle_data_dictionary() -> pd.DataFrame:
    rows = [
        ("cell_metrics_for_plotting.tsv.gz", "dataset_id", "Logical dataset identifier used to facet/filter panels."),
        ("cell_metrics_for_plotting.tsv.gz", "sample_id", "Sample-level identifier."),
        ("cell_metrics_for_plotting.tsv.gz", "method", "Capture method, e.g. Circular capture or CROP-seq."),
        ("cell_metrics_for_plotting.tsv.gz", "cell_type", "Cell type or sample class."),
        ("cell_metrics_for_plotting.tsv.gz", "cell_id", "Cell barcode / cell identifier from the processed matrix."),
        ("cell_metrics_for_plotting.tsv.gz", "guide_umis_top", "UMIs from the most abundant guide in that cell; main y-axis for panel 1b."),
        ("cell_metrics_for_plotting.tsv.gz", "guide_umis_total", "Total guide UMIs observed in that cell."),
        ("cell_metrics_for_plotting.tsv.gz", "assigned_any_guide", "Whether any guide passed the guide UMI threshold."),
        ("cell_metrics_for_plotting.tsv.gz", "assigned_confident_guide", "Whether the cell passed the method-specific confident guide assignment rule."),
        ("cell_metrics_for_plotting.tsv.gz", "guide_assignment_class", "No-guide / singlet / ambiguous class for stacked guide-assignment panels."),
        ("cell_metrics_for_plotting.tsv.gz", "total_counts", "Transcriptome UMIs per cell after excluding guide features where applicable."),
        ("cell_metrics_for_plotting.tsv.gz", "n_genes_by_counts", "Detected transcriptome features per cell."),
        ("cell_metrics_for_plotting.tsv.gz", "pct_counts_mt", "Percent mitochondrial transcriptome UMIs per cell."),
        ("guide_metrics_for_plotting.tsv.gz", "guide_id", "Guide identifier."),
        ("guide_metrics_for_plotting.tsv.gz", "n_cells_ge_threshold", "Number of cells with guide UMIs at or above the threshold."),
        ("guide_metrics_for_plotting.tsv.gz", "detected_min_cells", "Whether the guide was detected in at least min_cells_per_guide cells."),
        ("sample_summary_for_plotting.tsv", "pct_cells_any_guide", "Fraction of cells with at least one guide above threshold."),
        ("sample_summary_for_plotting.tsv", "pct_cells_confident_guide", "Fraction of cells confidently assigned under the selected assignment rule."),
        ("sample_summary_for_plotting.tsv", "pct_guides_detected_min_cells", "Fraction of guides detected in at least min_cells_per_guide cells."),
        ("saturation_for_plotting.tsv", "reads_per_cell", "Downsampled reads per cell for saturation curves, if available."),
        ("saturation_for_plotting.tsv", "median_umis_per_cell", "Median UMIs per cell at the downsampled depth."),
        ("saturation_for_plotting.tsv", "median_features_per_cell", "Median detected features/genes per cell at the downsampled depth, if available."),
        ("assignment_classes_for_plotting.tsv", "guide_assignment_class", "Guide class used for no-guide/singlet/multiplet stacked bars."),
        ("assignment_classes_for_plotting.tsv", "fraction", "Fraction of cells in this guide-assignment class."),
        ("guide_detection_curve_for_plotting.tsv.gz", "guide_rank_fraction", "Rank of each guide after sorting by cells recovered, scaled from 0 to 1."),
        ("guide_detection_curve_for_plotting.tsv.gz", "n_cells_ge_threshold", "Number of cells in which each guide passed the guide UMI threshold."),
        ("primer_optimization_for_plotting.tsv", "guide_umis_per_cell", "Optional per-cell or summarized guide UMIs for primer/concentration tests."),
        ("read_origin_for_plotting.tsv", "fraction", "Optional fraction of guide reads by source, e.g. oligo-dT versus guide RT primer."),
        ("replicate_correlations_for_plotting.tsv", "correlation", "Optional guide-count replicate correlation values."),
        ("bulk_singlecell_correlations_for_plotting.tsv.gz", "single_cell_count", "Optional single-cell guide recovery metric merged with bulk guide counts."),
        ("target_knockdown_for_plotting.tsv", "log2fc_mean_vs_control", "Optional target-gene expression log2FC in guide-assigned cells versus controls."),
        ("panel_specific/*.tsv.gz", "", "Panel-specific subsets used for faster plotting and review."),
    ]
    return pd.DataFrame(rows, columns=["file", "column", "description"])


def export_plot_input_bundle(
    outputs: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any] | None = None,
    *,
    bundle_dir: str | os.PathLike[str] | None = None,
    zip_name: str = "figure1_plot_input_bundle.zip",
    max_rows_per_panel_stratum: int | None = 50000,
    random_state: int = 1,
) -> Path:
    """Export derived plot-input tables that can be uploaded back for Figure 1 plotting.

    The bundle contains standardized metrics, panel-specific subsets, summary tables,
    and diagnostics. It does not contain raw count matrices or FASTQs.
    """
    import datetime as _dt
    import platform as _platform
    import zipfile as _zipfile

    cfg = dict(config or {})
    root = Path(bundle_dir) if bundle_dir is not None else Path(cfg.get("out_dir", "figure1_outputs")) / "plot_input_bundle"
    root = ensure_dir(root)
    panel_dir = ensure_dir(root / "panel_specific")
    qc_dir = ensure_dir(root / "quality_checks")

    cell_metrics = _bundle_safe_table(outputs, "cell_metrics")
    guide_metrics = _bundle_safe_table(outputs, "guide_metrics")
    sample_summary = _bundle_safe_table(outputs, "sample_summary")
    saturation = _bundle_safe_table(outputs, "saturation")
    errors = _bundle_safe_table(outputs, "errors")
    missing_saturation = _bundle_safe_table(outputs, "missing_saturation")
    assignment_classes = _bundle_safe_table(outputs, "assignment_classes")
    guide_detection_curve = _bundle_safe_table(outputs, "guide_detection_curve")
    primer_optimization = _bundle_safe_table(outputs, "primer_optimization")
    read_origin = _bundle_safe_table(outputs, "read_origin")
    replicate_correlations = _bundle_safe_table(outputs, "replicate_correlations")
    bulk_singlecell_correlations = _bundle_safe_table(outputs, "bulk_singlecell_correlations")
    target_knockdown = _bundle_safe_table(outputs, "target_knockdown")

    written: dict[str, str] = {}
    written["cell_metrics"] = str(write_plot_bundle_table(cell_metrics, root / "cell_metrics_for_plotting.tsv.gz"))
    written["guide_metrics"] = str(write_plot_bundle_table(guide_metrics, root / "guide_metrics_for_plotting.tsv.gz"))
    written["sample_summary"] = str(write_plot_bundle_table(sample_summary, root / "sample_summary_for_plotting.tsv"))
    written["saturation"] = str(write_plot_bundle_table(saturation, root / "saturation_for_plotting.tsv"))
    written["assignment_classes"] = str(write_plot_bundle_table(assignment_classes, root / "assignment_classes_for_plotting.tsv"))
    written["guide_detection_curve"] = str(write_plot_bundle_table(guide_detection_curve, root / "guide_detection_curve_for_plotting.tsv.gz"))
    written["primer_optimization"] = str(write_plot_bundle_table(primer_optimization, root / "primer_optimization_for_plotting.tsv"))
    written["read_origin"] = str(write_plot_bundle_table(read_origin, root / "read_origin_for_plotting.tsv"))
    written["replicate_correlations"] = str(write_plot_bundle_table(replicate_correlations, root / "replicate_correlations_for_plotting.tsv"))
    written["bulk_singlecell_correlations"] = str(write_plot_bundle_table(bulk_singlecell_correlations, root / "bulk_singlecell_correlations_for_plotting.tsv.gz"))
    written["target_knockdown"] = str(write_plot_bundle_table(target_knockdown, root / "target_knockdown_for_plotting.tsv"))

    cell_cols = _panel_cell_columns(cell_metrics)
    if cell_metrics.empty:
        fig1b = pd.DataFrame(columns=cell_cols)
        fig1d_qc = pd.DataFrame(columns=cell_cols)
        fig1e = pd.DataFrame(columns=cell_cols)
    else:
        dataset_series = cell_metrics["dataset_id"] if "dataset_id" in cell_metrics.columns else pd.Series("", index=cell_metrics.index)
        cell_type_series = cell_metrics["cell_type"] if "cell_type" in cell_metrics.columns else pd.Series("", index=cell_metrics.index)
        fig1b = cell_metrics.loc[(dataset_series == "ipsc_single_guide") & (cell_type_series == "hiPSC"), cell_cols].copy()
        fig1d_qc = fig1b.copy()
        fig1e = cell_metrics.loc[dataset_series != "ipsc_single_guide", cell_cols].copy()

    fig1b_sample = _bundle_stratified_sample(fig1b, strata_cols=["sample_id", "method"], max_rows_per_stratum=max_rows_per_panel_stratum, random_state=random_state)
    fig1d_qc_sample = _bundle_stratified_sample(fig1d_qc, strata_cols=["sample_id", "method"], max_rows_per_stratum=max_rows_per_panel_stratum, random_state=random_state)
    fig1e_sample = _bundle_stratified_sample(fig1e, strata_cols=["dataset_id", "sample_id", "method", "cell_type"], max_rows_per_stratum=max_rows_per_panel_stratum, random_state=random_state)

    written["fig1b_full"] = str(write_plot_bundle_table(fig1b, panel_dir / "fig1b_hiPSC_guide_umis_full.tsv.gz"))
    written["fig1b_plot_sample"] = str(write_plot_bundle_table(fig1b_sample, panel_dir / "fig1b_hiPSC_guide_umis_plot_sample.tsv.gz"))
    written["fig1b_summary_stats"] = str(write_plot_bundle_table(_bundle_summary_stats(fig1b, ["dataset_id", "sample_id", "method", "cell_type"], "guide_umis_top"), panel_dir / "fig1b_hiPSC_guide_umis_summary_stats.tsv"))

    assign_long = _bundle_assignment_long_table(sample_summary)
    fig1c = assign_long[assign_long["dataset_id"] == "ipsc_single_guide"].copy() if (not assign_long.empty and "dataset_id" in assign_long.columns) else assign_long
    written["fig1c_assignment_metrics"] = str(write_plot_bundle_table(fig1c, panel_dir / "fig1c_assignment_metrics.tsv"))

    if saturation.empty or "modality" not in saturation.columns or "dataset_id" not in saturation.columns:
        fig1d_sat = pd.DataFrame()
    else:
        fig1d_sat = saturation[
            (saturation["dataset_id"] == "ipsc_single_guide")
            & saturation["modality"].astype(str).str.lower().isin(["gex", "transcriptome", "mrna"])
        ].copy()
    written["fig1d_transcriptome_saturation"] = str(write_plot_bundle_table(fig1d_sat, panel_dir / "fig1d_transcriptome_saturation.tsv"))
    written["fig1d_transcriptome_qc_full"] = str(write_plot_bundle_table(fig1d_qc, panel_dir / "fig1d_transcriptome_qc_full.tsv.gz"))
    written["fig1d_transcriptome_qc_plot_sample"] = str(write_plot_bundle_table(fig1d_qc_sample, panel_dir / "fig1d_transcriptome_qc_plot_sample.tsv.gz"))
    written["fig1d_transcriptome_qc_summary_stats"] = str(write_plot_bundle_table(_bundle_summary_stats(fig1d_qc, ["dataset_id", "sample_id", "method", "cell_type"], "total_counts"), panel_dir / "fig1d_transcriptome_qc_summary_stats.tsv"))

    written["fig1e_full"] = str(write_plot_bundle_table(fig1e, panel_dir / "fig1e_other_celltypes_guide_umis_full.tsv.gz"))
    written["fig1e_plot_sample"] = str(write_plot_bundle_table(fig1e_sample, panel_dir / "fig1e_other_celltypes_guide_umis_plot_sample.tsv.gz"))
    written["fig1e_summary_stats"] = str(write_plot_bundle_table(_bundle_summary_stats(fig1e, ["dataset_id", "sample_id", "method", "cell_type"], "guide_umis_top"), panel_dir / "fig1e_other_celltypes_guide_umis_summary_stats.tsv"))

    if not assignment_classes.empty and "dataset_id" in assignment_classes.columns:
        assignment_hiPSC = assignment_classes[assignment_classes["dataset_id"] == "ipsc_single_guide"].copy()
    else:
        assignment_hiPSC = assignment_classes.copy()
    if not guide_detection_curve.empty and "dataset_id" in guide_detection_curve.columns:
        guide_detection_hiPSC = guide_detection_curve[guide_detection_curve["dataset_id"] == "ipsc_single_guide"].copy()
    else:
        guide_detection_hiPSC = guide_detection_curve.copy()
    written["figS_assignment_classes_hiPSC"] = str(write_plot_bundle_table(assignment_hiPSC, panel_dir / "figS_assignment_classes_hiPSC.tsv"))
    written["figS_guide_detection_curve_hiPSC"] = str(write_plot_bundle_table(guide_detection_hiPSC, panel_dir / "figS_guide_detection_curve_hiPSC.tsv.gz"))
    written["figS_primer_optimization"] = str(write_plot_bundle_table(primer_optimization, panel_dir / "figS_primer_optimization.tsv"))
    written["figS_read_origin"] = str(write_plot_bundle_table(read_origin, panel_dir / "figS_read_origin.tsv"))
    written["figS_replicate_correlations"] = str(write_plot_bundle_table(replicate_correlations, panel_dir / "figS_replicate_correlations.tsv"))
    written["figS_bulk_singlecell_correlations"] = str(write_plot_bundle_table(bulk_singlecell_correlations, panel_dir / "figS_bulk_singlecell_correlations.tsv.gz"))
    written["figS_target_knockdown"] = str(write_plot_bundle_table(target_knockdown, panel_dir / "figS_target_knockdown.tsv"))

    written["errors"] = str(write_plot_bundle_table(errors if not errors.empty else pd.DataFrame(columns=["dataset_key", "error"]), qc_dir / "processing_errors.tsv"))
    written["missing_saturation"] = str(write_plot_bundle_table(missing_saturation if not missing_saturation.empty else pd.DataFrame(columns=["missing_saturation_path"]), qc_dir / "missing_saturation_paths.tsv"))
    try:
        path_check = check_config_paths(cfg) if cfg else pd.DataFrame()
    except Exception as exc:
        path_check = pd.DataFrame({"error": [repr(exc)]})
    written["path_check"] = str(write_plot_bundle_table(path_check, qc_dir / "path_check.tsv"))

    written["data_dictionary"] = str(write_plot_bundle_table(_bundle_data_dictionary(), root / "data_dictionary.tsv"))

    table_shapes = {
        "cell_metrics": list(cell_metrics.shape),
        "guide_metrics": list(guide_metrics.shape),
        "sample_summary": list(sample_summary.shape),
        "saturation": list(saturation.shape),
        "assignment_classes": list(assignment_classes.shape),
        "guide_detection_curve": list(guide_detection_curve.shape),
        "primer_optimization": list(primer_optimization.shape),
        "read_origin": list(read_origin.shape),
        "replicate_correlations": list(replicate_correlations.shape),
        "bulk_singlecell_correlations": list(bulk_singlecell_correlations.shape),
        "target_knockdown": list(target_knockdown.shape),
        "fig1b_full": list(fig1b.shape),
        "fig1c_assignment_metrics": list(fig1c.shape),
        "fig1d_transcriptome_saturation": list(fig1d_sat.shape),
        "fig1d_transcriptome_qc_full": list(fig1d_qc.shape),
        "fig1e_full": list(fig1e.shape),
    }
    manifest = {
        "bundle_version": "figure1_plot_input_bundle_v2_paper_qc_panels",
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "purpose": "Derived plot-input tables for CC-Perturb-seq Figure 1 panels b-e plus paper-inspired QC/supplementary panels.",
        "contains_raw_count_matrices": False,
        "python": _platform.python_version(),
        "table_shapes": table_shapes,
        "files": {k: str(Path(v).relative_to(root)) for k, v in written.items()},
        "recommended_upload_to_chatgpt": "Upload the ZIP file named figure1_plot_input_bundle.zip. Raw MatrixMarket, h5ad, h5mu, or FASTQ files are not needed for plotting.",
        "panel_inputs": {
            "fig1b": ["panel_specific/fig1b_hiPSC_guide_umis_full.tsv.gz", "panel_specific/fig1b_hiPSC_guide_umis_summary_stats.tsv"],
            "fig1c": ["panel_specific/fig1c_assignment_metrics.tsv", "sample_summary_for_plotting.tsv"],
            "fig1d": ["panel_specific/fig1d_transcriptome_saturation.tsv", "panel_specific/fig1d_transcriptome_qc_full.tsv.gz"],
            "fig1e": ["panel_specific/fig1e_other_celltypes_guide_umis_full.tsv.gz", "panel_specific/fig1e_other_celltypes_guide_umis_summary_stats.tsv"],
            "supp_assignment_classes": ["panel_specific/figS_assignment_classes_hiPSC.tsv"],
            "supp_guide_detection_curve": ["panel_specific/figS_guide_detection_curve_hiPSC.tsv.gz"],
            "supp_primer_optimization": ["panel_specific/figS_primer_optimization.tsv"],
            "supp_read_origin": ["panel_specific/figS_read_origin.tsv"],
            "supp_replicate_correlations": ["panel_specific/figS_replicate_correlations.tsv"],
            "supp_bulk_singlecell_correlation": ["panel_specific/figS_bulk_singlecell_correlations.tsv.gz"],
            "supp_target_knockdown": ["panel_specific/figS_target_knockdown.tsv"],
        },
    }
    with open(root / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    readme = f"""# Figure 1 plot-input bundle

Upload `{zip_name}` back to ChatGPT when you want the Figure 1 plots made or revised.

This bundle contains derived plotting tables only, not raw count matrices. It includes:

- `cell_metrics_for_plotting.tsv.gz`: standardized per-cell guide and transcriptome QC metrics.
- `guide_metrics_for_plotting.tsv.gz`: per-guide detection metrics.
- `sample_summary_for_plotting.tsv`: per-sample medians and assignment metrics.
- `saturation_for_plotting.tsv`: saturation-curve data, if the config supplied downsampled inputs.
- `assignment_classes_for_plotting.tsv`: no-guide/singlet/multiplet class fractions.
- `guide_detection_curve_for_plotting.tsv.gz`: per-guide recovery curves.
- Optional input-derived tables when supplied: `primer_optimization_for_plotting.tsv`, `read_origin_for_plotting.tsv`, `replicate_correlations_for_plotting.tsv`, `bulk_singlecell_correlations_for_plotting.tsv.gz`, and `target_knockdown_for_plotting.tsv`.
- `panel_specific/`: smaller panel-specific subsets and summary-statistics tables.
- `quality_checks/`: path checks, processing errors, and missing saturation paths.
- `manifest.json`: table sizes and file map.

For exact violin/box plots, use the `*_full.tsv.gz` files. For quick drafts, use the `*_plot_sample.tsv.gz` files. The full tables are usually still small enough to upload because they contain only derived metrics.

Created at UTC: {manifest['created_at_utc']}
"""
    (root / "README_upload_this_to_chatgpt.md").write_text(readme)

    zip_path = root.parent / zip_name
    if zip_path.exists():
        zip_path.unlink()
    with _zipfile.ZipFile(zip_path, mode="w", compression=_zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fp in sorted(root.rglob("*")):
            if fp.is_file():
                zf.write(fp, arcname=str(fp.relative_to(root)))
    return zip_path



def make_chatgpt_plot_bundle(
    outputs: Mapping[str, pd.DataFrame],
    out_dir: str | os.PathLike[str],
    *,
    config: Mapping[str, Any] | None = None,
    path_check: pd.DataFrame | None = None,
    max_cells_per_group: int | None = 25000,
    include_full_cell_metrics: bool = False,
    hash_cell_ids: bool = True,
    random_state: int = 1,
) -> dict[str, Any]:
    """Create the ZIP to upload back for ChatGPT-side Figure 1 plotting.

    The ZIP is written to:
    `<out_dir>/plot_data_for_chatgpt/figure1_data_for_chatgpt.zip`.
    """
    outs = dict(outputs)
    if path_check is not None:
        outs["path_check"] = path_check

    # Remove raw cell barcodes from derived plotting tables by default.
    cell_metrics = _bundle_safe_table(outs, "cell_metrics")
    if hash_cell_ids and not cell_metrics.empty and "cell_id" in cell_metrics.columns:
        cm = cell_metrics.copy()
        cm["cell_id_hash"] = cm["cell_id"].astype(str).map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest()[:16])
        cm = cm.drop(columns=["cell_id"])
        outs["cell_metrics"] = cm

    bundle_root = Path(out_dir) / "plot_data_for_chatgpt" / "figure1_data_for_chatgpt"
    zip_path = export_plot_input_bundle(
        outs,
        config,
        bundle_dir=bundle_root,
        zip_name="figure1_data_for_chatgpt.zip",
        max_rows_per_panel_stratum=max_cells_per_group,
        random_state=random_state,
    )
    return {
        "bundle_zip": str(zip_path),
        "bundle_dir": str(bundle_root),
        "bundle_zip_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 3) if zip_path.exists() else None,
        "max_cells_per_group": max_cells_per_group,
        "include_full_cell_metrics": include_full_cell_metrics,
        "hash_cell_ids": hash_cell_ids,
    }

def load_plot_input_bundle(bundle_path: str | os.PathLike[str]) -> dict[str, pd.DataFrame]:
    """Load a bundle created by export_plot_input_bundle()."""
    import tempfile as _tempfile
    import zipfile as _zipfile

    path = Path(bundle_path)
    if path.is_file() and path.suffix.lower() == ".zip":
        root = Path(_tempfile.mkdtemp(prefix="figure1_plot_bundle_"))
        with _zipfile.ZipFile(path, "r") as zf:
            zf.extractall(root)
    else:
        root = path

    def read(fp: Path) -> pd.DataFrame:
        if not fp.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(fp, sep="\t", compression="infer")
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return {
        "cell_metrics": read(root / "cell_metrics_for_plotting.tsv.gz"),
        "guide_metrics": read(root / "guide_metrics_for_plotting.tsv.gz"),
        "sample_summary": read(root / "sample_summary_for_plotting.tsv"),
        "saturation": read(root / "saturation_for_plotting.tsv"),
        "assignment_classes": read(root / "assignment_classes_for_plotting.tsv"),
        "guide_detection_curve": read(root / "guide_detection_curve_for_plotting.tsv.gz"),
        "primer_optimization": read(root / "primer_optimization_for_plotting.tsv"),
        "read_origin": read(root / "read_origin_for_plotting.tsv"),
        "replicate_correlations": read(root / "replicate_correlations_for_plotting.tsv"),
        "bulk_singlecell_correlations": read(root / "bulk_singlecell_correlations_for_plotting.tsv.gz"),
        "target_knockdown": read(root / "target_knockdown_for_plotting.tsv"),
    }


def generate_figure1_panels_from_bundle(bundle_path: str | os.PathLike[str], out_dir: str | os.PathLike[str]) -> pd.DataFrame:
    """Regenerate Figure 1 SVGs from an exported plot-input bundle."""
    outputs = load_plot_input_bundle(bundle_path)
    return generate_figure1_panels(outputs, out_dir)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run Figure 1 CC-Perturb-seq reprocessing and plot/bundle generation.")
    parser.add_argument("--config-json", default=None, help="Path to JSON config. If omitted, uses default_config().")
    parser.add_argument("--check-paths", action="store_true", help="Only print path existence table; do not process data.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip SVG panel generation on Sherlock.")
    parser.add_argument("--no-plot-bundle", action="store_true", help="Do not write the uploadable figure1_plot_input_bundle.zip.")
    parser.add_argument("--bundle-dir", default=None, help="Optional directory for the extracted plot-input bundle.")
    parser.add_argument("--from-output-dir", default=None, help="Build plots/bundle from an existing output directory containing cell_metrics.tsv etc.")
    parser.add_argument("--from-existing-output-dir", default=None, help="Alias for --from-output-dir.")
    parser.add_argument("--from-bundle", default=None, help="Generate SVGs from an existing figure1_plot_input_bundle.zip or extracted bundle directory.")
    parser.add_argument("--max-rows-per-panel-stratum", type=int, default=50000, help="Rows per sample/method stratum in panel preview tables. Full tables are still exported.")
    parser.add_argument("--max-cells-per-group", type=int, default=None, help="Alias for --max-rows-per-panel-stratum.")
    parser.add_argument("--no-cell-subsample", action="store_true", help="Do not downsample panel preview tables.")
    parser.add_argument("--export-bundle", action="store_true", help="Write plot_data_for_chatgpt/figure1_data_for_chatgpt.zip.")
    parser.add_argument("--include-full-cell-metrics", action="store_true", help="Accepted for compatibility; full derived cell metrics are already exported.")
    args = parser.parse_args(argv)
    if args.from_existing_output_dir and not args.from_output_dir:
        args.from_output_dir = args.from_existing_output_dir
    if args.max_cells_per_group is not None:
        args.max_rows_per_panel_stratum = args.max_cells_per_group
    if args.no_cell_subsample:
        args.max_rows_per_panel_stratum = None

    if args.config_json:
        with open(args.config_json) as handle:
            cfg = json.load(handle)
    else:
        cfg = default_config()

    if args.from_bundle:
        svg_out = Path(cfg.get("out_dir", "figure1_outputs")) / "svg_from_bundle"
        manifest = generate_figure1_panels_from_bundle(args.from_bundle, svg_out)
        print(manifest.to_string(index=False))
        return 0

    output_root = Path(args.from_output_dir) if args.from_output_dir else Path(cfg.get("out_dir", "figure1_outputs"))
    if args.from_output_dir:
        outs = load_outputs_from_dir(args.from_output_dir)
        path_table = outs.get("path_check", pd.DataFrame())
    else:
        out_dir = ensure_dir(output_root)
        path_table = check_config_paths(cfg)
        print(path_table.to_string(index=False))
        path_table.to_csv(out_dir / "path_check.tsv", sep="\t", index=False)
        if args.check_paths:
            return 0
        outs = run_reprocessing(cfg)
        outs["path_check"] = path_table

    if not args.no_plot_bundle:
        if args.export_bundle:
            bundle_dir = Path(args.bundle_dir) if args.bundle_dir else output_root / "plot_data_for_chatgpt" / "figure1_data_for_chatgpt"
            zip_name = "figure1_data_for_chatgpt.zip"
        else:
            bundle_dir = args.bundle_dir
            zip_name = "figure1_plot_input_bundle.zip"
        bundle_zip = export_plot_input_bundle(
            outs,
            cfg,
            bundle_dir=bundle_dir,
            zip_name=zip_name,
            max_rows_per_panel_stratum=args.max_rows_per_panel_stratum,
        )
        print(f"Plot-input bundle written: {bundle_zip}")
        print("Upload this ZIP here and I can make the final Figure 1 plots from it.")

    if not args.skip_plots:
        svg_dir = Path(cfg.get("out_dir", "figure1_outputs")) / "svg"
        manifest = generate_figure1_panels(outs, svg_dir)
        print(manifest.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
