#!/usr/bin/env python3
"""
Export plot-input tables for a Fig. 4A-like on-target knockdown panel.

This script computes, for each guide RNA with a resolvable target gene:
  - target-gene expression in cells carrying that guide
  - target-gene expression in non-targeting-control cells
  - an approximate one-sided Mann-Whitney test for knockdown
  - per-guide and per-target summary tables

Designed for the CC-Perturb-seq manuscript datasets:
  dataset #2: teloHAEC Pol II/CROP vs Pol III/circular capture h5ad files
  dataset #3: hiPSC 10-gene pilot Parse-style matrices

The output bundle is intended to be uploaded back to ChatGPT for plotting.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None


###############################################################################
# Helpers
###############################################################################


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    path = str(path)
    sep = kwargs.pop("sep", None)
    if sep is None:
        if path.endswith(".tsv") or path.endswith(".tsv.gz") or path.endswith(".txt") or path.endswith(".txt.gz"):
            sep = "\t"
        else:
            sep = ","
    return pd.read_csv(path, sep=sep, **kwargs)


def first_existing(paths: Sequence[str | Path | None]) -> Optional[Path]:
    for p in paths:
        if p is None:
            continue
        pp = Path(p)
        if pp.exists():
            return pp
    return None


def choose_first_existing_column(df: pd.DataFrame, candidates: Sequence[str], fallback_first: bool = True) -> str:
    lower_to_col = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_to_col:
            return lower_to_col[c.lower()]
    if fallback_first:
        return df.columns[0]
    raise KeyError(f"None of candidate columns found: {candidates}; available={list(df.columns)}")


def vector_to_1d(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def sparse_sum_axis1(x: sparse.spmatrix | np.ndarray) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.sum(axis=1)).ravel()
    return np.asarray(x.sum(axis=1)).ravel()


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    q[finite] = out
    return q


def normalize_cell_ids(ids: Sequence[str], mode: str) -> List[str]:
    ids = [str(x) for x in ids]
    if mode == "exact":
        return ids
    if mode == "strip_suffix_double_underscore":
        return [x.split("__")[0] for x in ids]
    if mode == "strip_suffix_dash":
        return [x.split("-")[0] for x in ids]
    if mode == "auto_basic":
        # keep exact here; matching function tries alternate modes
        return ids
    raise ValueError(f"Unknown cell_id_normalization mode: {mode}")


def compute_intersection_indices(a_ids: Sequence[str], b_ids: Sequence[str], mode: str = "auto_basic") -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Return indices into a and b for overlapping IDs, preserving a order."""
    modes = [mode] if mode != "auto_basic" else ["exact", "strip_suffix_double_underscore", "strip_suffix_dash"]
    best = None
    for m in modes:
        a_norm = normalize_cell_ids(a_ids, m)
        b_norm = normalize_cell_ids(b_ids, m)
        b_map = {}
        for j, x in enumerate(b_norm):
            if x not in b_map:
                b_map[x] = j
        ai, bi, ids = [], [], []
        for i, x in enumerate(a_norm):
            if x in b_map:
                ai.append(i)
                bi.append(b_map[x])
                ids.append(x)
        cand = (np.array(ai, dtype=int), np.array(bi, dtype=int), ids, m)
        if best is None or len(ids) > len(best[2]):
            best = cand
    assert best is not None
    return best


###############################################################################
# Matrix readers
###############################################################################


@dataclass
class MatrixData:
    X: sparse.spmatrix | np.ndarray
    cell_ids: List[str]
    feature_ids: List[str]
    var: pd.DataFrame = field(default_factory=pd.DataFrame)
    obs: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_path: str = ""


def infer_parse_companion_paths(matrix_path: str | Path) -> Tuple[Optional[Path], Optional[Path]]:
    mtx = Path(matrix_path)
    d = mtx.parent
    feature_candidates = [
        d / "all_genes.csv",
        d / "genes.csv",
        d / "features.csv",
        d / "all_guides.csv",
        d / "guides.csv",
        d / "guide_RNAs.csv",
    ]
    cell_candidates = [
        d / "cell_metadata.csv",
        d / "cells.csv",
        d / "barcodes.csv",
        d / "cell_barcodes.csv",
    ]
    return first_existing(feature_candidates), first_existing(cell_candidates)


def load_parse_matrix(matrix_path: str | Path, feature_path: str | Path | None = None, cell_path: str | Path | None = None) -> MatrixData:
    matrix_path = Path(matrix_path)
    if feature_path is None or cell_path is None:
        inferred_feature, inferred_cell = infer_parse_companion_paths(matrix_path)
        feature_path = Path(feature_path) if feature_path is not None else inferred_feature
        cell_path = Path(cell_path) if cell_path is not None else inferred_cell
    if feature_path is None or cell_path is None:
        raise FileNotFoundError(f"Could not infer feature/cell metadata paths for {matrix_path}")

    var = read_table_auto(feature_path)
    obs = read_table_auto(cell_path)
    feature_col = choose_first_existing_column(var, ["gene_name", "gene", "name", "feature_name", "id", "gene_id", "guide", "guide_id"])
    cell_col = choose_first_existing_column(obs, ["cell_barcode", "barcode", "cell_id", "bc_wells", "cell", "id"])
    feature_ids = var[feature_col].astype(str).tolist()
    cell_ids = obs[cell_col].astype(str).tolist()

    X = mmread(str(matrix_path)).tocsr()
    # Parse outputs can be feature x cell or cell x feature.
    if X.shape == (len(feature_ids), len(cell_ids)):
        X = X.T.tocsr()
    elif X.shape == (len(cell_ids), len(feature_ids)):
        X = X.tocsr()
    else:
        raise ValueError(
            f"Matrix shape {X.shape} does not match features ({len(feature_ids)}) and cells ({len(cell_ids)}) for {matrix_path}"
        )
    return MatrixData(X=X, cell_ids=cell_ids, feature_ids=feature_ids, var=var, obs=obs, source_path=str(matrix_path))


def load_h5ad_matrix(path: str | Path, transpose_if_needed: bool = False) -> MatrixData:
    if ad is None:
        raise ImportError("anndata is required to read h5ad files. Install with: pip install anndata")
    a = ad.read_h5ad(str(path))
    if transpose_if_needed:
        a = a.T
    X = a.X
    if sparse.issparse(X):
        X = X.tocsr()
    return MatrixData(
        X=X,
        cell_ids=[str(x) for x in a.obs_names.tolist()],
        feature_ids=[str(x) for x in a.var_names.tolist()],
        var=a.var.copy(),
        obs=a.obs.copy(),
        source_path=str(path),
    )


def load_matrix_from_config(cfg: Dict[str, Any], prefix: str) -> MatrixData:
    fmt = cfg.get(f"{prefix}_format")
    if fmt is None:
        # Infer from path.
        p = cfg.get(f"{prefix}_h5ad") or cfg.get(f"{prefix}_matrix_path")
        if p and str(p).endswith(".h5ad"):
            fmt = "h5ad"
        else:
            fmt = "parse_mtx"
    if fmt == "h5ad":
        return load_h5ad_matrix(cfg[f"{prefix}_h5ad"])
    if fmt == "parse_mtx":
        return load_parse_matrix(cfg[f"{prefix}_matrix_path"], cfg.get(f"{prefix}_feature_path"), cfg.get(f"{prefix}_cell_path"))
    raise ValueError(f"Unsupported {prefix}_format: {fmt}")


###############################################################################
# Guide mapping and feature matching
###############################################################################


NTC_REGEX_DEFAULT = r"(?i)(^|[_\-\.])(ntc|non.?target|non.?targeting|neg|negative|safe|scramble|control)([_\-\.]|$)"


def infer_target_from_guide_id(guide_id: str) -> str:
    g = str(guide_id)
    # Explicit NTC/safe names.
    if re.search(NTC_REGEX_DEFAULT, g):
        return "NTC"
    # Common guide names in these datasets: GENE_-_pos, GENE_single_..., GENE_g-...
    for sep in ["_", ":", "|"]:
        if sep in g:
            first = g.split(sep)[0]
            if first:
                return first
    return g


def read_guide_mapping_tables(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p:
            continue
        pp = Path(p)
        if not pp.exists():
            continue
        try:
            df = read_table_auto(pp)
            df["__source_mapping_file"] = str(pp)
            frames.append(df)
        except Exception as e:
            print(f"WARNING: failed to read guide mapping table {pp}: {e}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def standardize_guide_mapping(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["guide_id", "target_gene", "guide_category", "source_mapping_file"])
    guide_col = None
    for c in ["guide_id", "guide", "gRNA", "grna", "sgRNA", "sgRNA_id", "feature_id", "name"]:
        if c in raw.columns:
            guide_col = c
            break
    if guide_col is None:
        # Use first object-like column.
        guide_col = raw.columns[0]
    target_col = None
    for c in ["target_gene", "gene", "target", "target_name", "gene_name", "group", "guide_target", "perturbation"]:
        if c in raw.columns and c != guide_col:
            target_col = c
            break
    if target_col is None:
        # Try a column containing target/gene.
        for c in raw.columns:
            lc = str(c).lower()
            if c != guide_col and ("target" in lc or "gene" in lc):
                target_col = c
                break
    out = pd.DataFrame()
    out["guide_id"] = raw[guide_col].astype(str)
    if target_col is not None:
        out["target_gene"] = raw[target_col].astype(str)
    else:
        out["target_gene"] = out["guide_id"].map(infer_target_from_guide_id)
    cat_col = None
    for c in ["guide_category", "category", "type", "control_type"]:
        if c in raw.columns:
            cat_col = c
            break
    if cat_col is not None:
        out["guide_category"] = raw[cat_col].astype(str)
    else:
        out["guide_category"] = np.where(out["target_gene"].str.contains(NTC_REGEX_DEFAULT, regex=True, na=False), "NTC", "targeting")
    out["source_mapping_file"] = raw.get("__source_mapping_file", "").astype(str) if "__source_mapping_file" in raw.columns else ""
    out = out.drop_duplicates(subset=["guide_id"], keep="first")
    return out


def resolve_guide_mapping(guide_ids: Sequence[str], mapping_tables: Sequence[str | Path], ntc_regex: str) -> pd.DataFrame:
    gids = pd.DataFrame({"guide_id": [str(x) for x in guide_ids]})
    raw = read_guide_mapping_tables(mapping_tables)
    std = standardize_guide_mapping(raw)
    out = gids.merge(std, on="guide_id", how="left")
    missing = out["target_gene"].isna()
    out.loc[missing, "target_gene"] = out.loc[missing, "guide_id"].map(infer_target_from_guide_id)
    out["target_gene"] = out["target_gene"].astype(str)
    # Clean common unmapped strings.
    out.loc[out["target_gene"].str.lower().isin(["nan", "none", "", "na"]), "target_gene"] = out["guide_id"].map(infer_target_from_guide_id)
    out["is_ntc"] = out["guide_id"].str.contains(ntc_regex, regex=True, na=False) | out["target_gene"].str.contains(ntc_regex, regex=True, na=False)
    out["guide_category"] = np.where(out["is_ntc"], "NTC", out.get("guide_category", "targeting"))
    return out


def build_gene_lookup(feature_ids: Sequence[str], var: pd.DataFrame) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    def add(name: Any, idx: int) -> None:
        s = str(name)
        if not s or s.lower() in {"nan", "none"}:
            return
        for key in [s, s.upper(), s.lower()]:
            if key not in lookup:
                lookup[key] = idx
    for i, f in enumerate(feature_ids):
        add(f, i)
    for col in ["gene_name", "gene", "symbol", "gene_symbol", "feature_name", "name", "id", "gene_id"]:
        if col in var.columns:
            for i, val in enumerate(var[col].tolist()):
                add(val, i)
    return lookup


def match_target_gene(target: str, lookup: Dict[str, int]) -> Tuple[Optional[int], Optional[str]]:
    t = str(target)
    for key in [t, t.upper(), t.lower()]:
        if key in lookup:
            return lookup[key], key
    # Strip version or common aliases.
    stripped = re.sub(r"\.\d+$", "", t)
    for key in [stripped, stripped.upper(), stripped.lower()]:
        if key in lookup:
            return lookup[key], key
    return None, None


###############################################################################
# Analysis
###############################################################################


def log_normalized_gene_expression(X: sparse.spmatrix | np.ndarray, gene_idx: int, total_counts: np.ndarray, scale_factor: float) -> np.ndarray:
    counts = vector_to_1d(X[:, gene_idx])
    denom = np.maximum(total_counts, 1)
    return np.log1p(counts / denom * scale_factor)


def summarize_sample(sample_cfg: Dict[str, Any], outdir: Path, global_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_id = sample_cfg["sample_id"]
    method = sample_cfg.get("method", "")
    cell_type = sample_cfg.get("cell_type", "")
    guide_threshold = float(sample_cfg.get("guide_threshold", global_cfg.get("guide_threshold", 5)))
    min_carrier_cells = int(sample_cfg.get("min_carrier_cells", global_cfg.get("min_carrier_cells", 10)))
    min_control_cells = int(sample_cfg.get("min_control_cells", global_cfg.get("min_control_cells", 20)))
    scale_factor = float(global_cfg.get("scale_factor", 10000))
    ntc_regex = sample_cfg.get("ntc_regex", global_cfg.get("ntc_regex", NTC_REGEX_DEFAULT))
    cell_norm_mode = sample_cfg.get("cell_id_normalization", global_cfg.get("cell_id_normalization", "auto_basic"))
    strict_ntc_no_target_guides = bool(sample_cfg.get("strict_ntc_no_target_guides", global_cfg.get("strict_ntc_no_target_guides", False)))

    log(f"Loading sample {sample_id}: GEX")
    gex = load_matrix_from_config(sample_cfg, "gex")
    log(f"Loading sample {sample_id}: guide")
    guide = load_matrix_from_config(sample_cfg, "guide")

    # Align cells.
    gex_idx, guide_idx, common_ids, chosen_norm = compute_intersection_indices(gex.cell_ids, guide.cell_ids, mode=cell_norm_mode)
    if len(common_ids) == 0:
        raise ValueError(f"No overlapping cell IDs between GEX and guide matrices for {sample_id}")
    if len(common_ids) < min(len(gex.cell_ids), len(guide.cell_ids)) * 0.5:
        print(f"WARNING: only {len(common_ids)} overlapping cells for {sample_id}; chosen normalization={chosen_norm}", file=sys.stderr)
    Xgex = gex.X[gex_idx, :]
    Xguide = guide.X[guide_idx, :].tocsc() if sparse.issparse(guide.X) else guide.X[guide_idx, :]
    cell_ids = [common_ids[i] for i in range(len(common_ids))]
    total_counts = sparse_sum_axis1(Xgex)

    mapping_paths = sample_cfg.get("guide_mapping_tables", []) or []
    guide_map = resolve_guide_mapping(guide.feature_ids, mapping_paths, ntc_regex=ntc_regex)
    guide_map["sample_id"] = sample_id
    guide_map["method"] = method
    guide_map["cell_type"] = cell_type

    gene_lookup = build_gene_lookup(gex.feature_ids, gex.var)
    gene_match = []
    for tg in guide_map["target_gene"].tolist():
        idx, matched = (None, None) if str(tg) == "NTC" else match_target_gene(tg, gene_lookup)
        gene_match.append((idx, matched))
    guide_map["target_gene_feature_index"] = [x[0] if x[0] is not None else np.nan for x in gene_match]
    guide_map["target_gene_matched_feature"] = [x[1] if x[1] is not None else "" for x in gene_match]
    guide_map["target_gene_found"] = guide_map["target_gene_feature_index"].notna()

    # Guide carrier masks and counts.
    n_cells = Xguide.shape[0]
    n_guides = Xguide.shape[1]
    guide_carrier_counts = []
    # Binary carrier matrix, sparse bool/int.
    if sparse.issparse(Xguide):
        carrier = Xguide >= guide_threshold
        carrier = carrier.astype(np.int8).tocsc()
        guides_per_cell = np.asarray(carrier.sum(axis=1)).ravel()
    else:
        carrier = (Xguide >= guide_threshold).astype(np.int8)
        guides_per_cell = carrier.sum(axis=1)
    guide_map["n_carrier_cells"] = [int(carrier[:, j].sum()) if sparse.issparse(carrier) else int(carrier[:, j].sum()) for j in range(n_guides)]
    guide_map["guide_threshold"] = guide_threshold

    ntc_guide_idx = np.where(guide_map["is_ntc"].values)[0]
    target_guide_idx = np.where(~guide_map["is_ntc"].values)[0]
    if len(ntc_guide_idx) > 0:
        if sparse.issparse(carrier):
            ntc_mask = np.asarray(carrier[:, ntc_guide_idx].sum(axis=1)).ravel() > 0
        else:
            ntc_mask = carrier[:, ntc_guide_idx].sum(axis=1) > 0
        control_type_base = "NTC_carrier"
        if strict_ntc_no_target_guides and len(target_guide_idx) > 0:
            if sparse.issparse(carrier):
                target_any = np.asarray(carrier[:, target_guide_idx].sum(axis=1)).ravel() > 0
            else:
                target_any = carrier[:, target_guide_idx].sum(axis=1) > 0
            ntc_mask = ntc_mask & (~target_any)
            control_type_base = "NTC_carrier_no_target_guides"
    else:
        ntc_mask = np.zeros(n_cells, dtype=bool)
        control_type_base = "fallback_non_carriers_for_same_target"

    per_guide_rows = []
    control_rows = []
    expr_cache: Dict[int, np.ndarray] = {}

    for j, row in guide_map.iterrows():
        if bool(row["is_ntc"]):
            continue
        if not bool(row["target_gene_found"]):
            per_guide_rows.append({
                "sample_id": sample_id, "method": method, "cell_type": cell_type,
                "guide_id": row["guide_id"], "target_gene": row["target_gene"],
                "target_gene_found": False, "skip_reason": "target_gene_not_found_in_GEX",
            })
            continue
        gene_idx = int(row["target_gene_feature_index"])
        if gene_idx not in expr_cache:
            expr_cache[gene_idx] = log_normalized_gene_expression(Xgex, gene_idx, total_counts, scale_factor)
        expr = expr_cache[gene_idx]
        if sparse.issparse(carrier):
            carrier_mask = vector_to_1d(carrier[:, j]).astype(bool)
        else:
            carrier_mask = carrier[:, j].astype(bool)
        n_carrier = int(carrier_mask.sum())

        # Control cells: prefer NTC carrier cells. If unavailable/too few, use cells not carrying any guide for same target.
        control_mask = ntc_mask.copy()
        control_type = control_type_base
        if int(control_mask.sum()) < min_control_cells:
            same_target_idx = guide_map.index[(guide_map["target_gene"].astype(str) == str(row["target_gene"])) & (~guide_map["is_ntc"])]
            if len(same_target_idx) > 0:
                if sparse.issparse(carrier):
                    same_target_carrier = np.asarray(carrier[:, same_target_idx.tolist()].sum(axis=1)).ravel() > 0
                else:
                    same_target_carrier = carrier[:, same_target_idx.tolist()].sum(axis=1) > 0
                control_mask = ~same_target_carrier
                control_type = "fallback_non_carriers_for_same_target"
            else:
                control_mask = ~carrier_mask
                control_type = "fallback_non_carriers_for_guide"
        n_control = int(control_mask.sum())

        pval = np.nan
        if n_carrier >= min_carrier_cells and n_control >= min_control_cells and mannwhitneyu is not None:
            try:
                # One-sided test for knockdown: carrier expression < control expression.
                pval = float(mannwhitneyu(expr[carrier_mask], expr[control_mask], alternative="less").pvalue)
            except Exception:
                pval = np.nan

        ctrl_mean = float(np.mean(expr[control_mask])) if n_control > 0 else np.nan
        ctrl_sd = float(np.std(expr[control_mask], ddof=1)) if n_control > 1 else np.nan
        carrier_mean = float(np.mean(expr[carrier_mask])) if n_carrier > 0 else np.nan
        carrier_sd = float(np.std(expr[carrier_mask], ddof=1)) if n_carrier > 1 else np.nan
        ratio = carrier_mean / ctrl_mean if ctrl_mean and np.isfinite(ctrl_mean) and ctrl_mean > 0 else np.nan
        log2_ratio = math.log2(ratio) if ratio and np.isfinite(ratio) and ratio > 0 else np.nan
        kd_eff = 1 - ratio if np.isfinite(ratio) else np.nan

        per_guide_rows.append({
            "sample_id": sample_id,
            "method": method,
            "cell_type": cell_type,
            "guide_id": row["guide_id"],
            "target_gene": row["target_gene"],
            "target_gene_matched_feature": row["target_gene_matched_feature"],
            "target_gene_found": True,
            "guide_threshold": guide_threshold,
            "control_type": control_type,
            "n_cells_total": n_cells,
            "n_carrier_cells": n_carrier,
            "n_control_cells": n_control,
            "control_mean_log_normalized_expr": ctrl_mean,
            "control_sd_log_normalized_expr": ctrl_sd,
            "carrier_mean_log_normalized_expr": carrier_mean,
            "carrier_sd_log_normalized_expr": carrier_sd,
            "carrier_vs_control_ratio_logexpr": ratio,
            "carrier_vs_control_log2_ratio_logexpr": log2_ratio,
            "knockdown_efficiency_1_minus_ratio": kd_eff,
            "pvalue_mannwhitney_less": pval,
            "skip_reason": "" if n_carrier >= min_carrier_cells else "too_few_carrier_cells",
        })

    per_guide = pd.DataFrame(per_guide_rows)
    if "pvalue_mannwhitney_less" in per_guide.columns:
        per_guide["qvalue_bh"] = bh_adjust(per_guide["pvalue_mannwhitney_less"].values)
        per_guide["significant_knockdown_q10"] = (
            (per_guide["qvalue_bh"] <= 0.10)
            & (per_guide["carrier_mean_log_normalized_expr"] < per_guide["control_mean_log_normalized_expr"])
            & (per_guide["n_carrier_cells"] >= min_carrier_cells)
        )
    else:
        per_guide["qvalue_bh"] = np.nan
        per_guide["significant_knockdown_q10"] = False

    # Rank target genes by control expression, for Fig. 4A-like x-axis.
    if not per_guide.empty and "control_mean_log_normalized_expr" in per_guide.columns:
        rank_df = (
            per_guide[per_guide["target_gene_found"] == True]
            .groupby("target_gene", as_index=False)["control_mean_log_normalized_expr"]
            .mean()
            .sort_values("control_mean_log_normalized_expr", ascending=True)
            .reset_index(drop=True)
        )
        rank_df["target_rank_by_control_expression"] = np.arange(1, len(rank_df) + 1)
        per_guide = per_guide.merge(rank_df[["target_gene", "target_rank_by_control_expression"]], on="target_gene", how="left")
    else:
        per_guide["target_rank_by_control_expression"] = np.nan

    # Per-target aggregation: combine all cells carrying any guide for that target.
    per_target_rows = []
    for target, sub in guide_map[(~guide_map["is_ntc"]) & (guide_map["target_gene_found"])].groupby("target_gene"):
        idxs = sub.index.tolist()
        gene_idx = int(sub["target_gene_feature_index"].iloc[0])
        if gene_idx not in expr_cache:
            expr_cache[gene_idx] = log_normalized_gene_expression(Xgex, gene_idx, total_counts, scale_factor)
        expr = expr_cache[gene_idx]
        if sparse.issparse(carrier):
            target_mask = np.asarray(carrier[:, idxs].sum(axis=1)).ravel() > 0
        else:
            target_mask = carrier[:, idxs].sum(axis=1) > 0
        control_mask = ntc_mask.copy()
        control_type = control_type_base
        if int(control_mask.sum()) < min_control_cells:
            control_mask = ~target_mask
            control_type = "fallback_non_carriers_for_same_target"
        n_carrier = int(target_mask.sum())
        n_control = int(control_mask.sum())
        pval = np.nan
        if n_carrier >= min_carrier_cells and n_control >= min_control_cells and mannwhitneyu is not None:
            try:
                pval = float(mannwhitneyu(expr[target_mask], expr[control_mask], alternative="less").pvalue)
            except Exception:
                pval = np.nan
        ctrl_mean = float(np.mean(expr[control_mask])) if n_control > 0 else np.nan
        carrier_mean = float(np.mean(expr[target_mask])) if n_carrier > 0 else np.nan
        ratio = carrier_mean / ctrl_mean if ctrl_mean and np.isfinite(ctrl_mean) and ctrl_mean > 0 else np.nan
        per_target_rows.append({
            "sample_id": sample_id,
            "method": method,
            "cell_type": cell_type,
            "target_gene": target,
            "n_guides_for_target": len(idxs),
            "n_carrier_cells_any_target_guide": n_carrier,
            "n_control_cells": n_control,
            "control_type": control_type,
            "control_mean_log_normalized_expr": ctrl_mean,
            "carrier_mean_log_normalized_expr": carrier_mean,
            "carrier_vs_control_ratio_logexpr": ratio,
            "carrier_vs_control_log2_ratio_logexpr": math.log2(ratio) if ratio and np.isfinite(ratio) and ratio > 0 else np.nan,
            "knockdown_efficiency_1_minus_ratio": 1 - ratio if np.isfinite(ratio) else np.nan,
            "pvalue_mannwhitney_less": pval,
        })
    per_target = pd.DataFrame(per_target_rows)
    if not per_target.empty:
        per_target["qvalue_bh"] = bh_adjust(per_target["pvalue_mannwhitney_less"].values)
        per_target["significant_knockdown_q10"] = (
            (per_target["qvalue_bh"] <= 0.10)
            & (per_target["carrier_mean_log_normalized_expr"] < per_target["control_mean_log_normalized_expr"])
        )

    # QC rows.
    qc = pd.DataFrame([{
        "sample_id": sample_id,
        "method": method,
        "cell_type": cell_type,
        "gex_source_path": gex.source_path,
        "guide_source_path": guide.source_path,
        "n_gex_cells": len(gex.cell_ids),
        "n_guide_cells": len(guide.cell_ids),
        "n_overlapping_cells": len(cell_ids),
        "cell_id_normalization_selected": chosen_norm,
        "n_gex_features": len(gex.feature_ids),
        "n_guide_features": len(guide.feature_ids),
        "guide_threshold": guide_threshold,
        "n_ntc_guides_detected_by_name": int(guide_map["is_ntc"].sum()),
        "n_ntc_control_cells": int(ntc_mask.sum()),
        "n_targeting_guides": int((~guide_map["is_ntc"]).sum()),
        "n_guides_with_target_gene_found": int(((~guide_map["is_ntc"]) & guide_map["target_gene_found"]).sum()),
        "median_gex_umis_per_cell": float(np.median(total_counts)),
        "median_guides_per_cell_ge_threshold": float(np.median(guides_per_cell)),
    }])

    guide_counts = guide_map[[
        "sample_id", "method", "cell_type", "guide_id", "target_gene", "guide_category", "is_ntc",
        "target_gene_found", "target_gene_matched_feature", "n_carrier_cells", "guide_threshold"
    ]].copy()

    # Write per-sample intermediate files.
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    guide_map.to_csv(outdir / f"{safe}.guide_target_mapping.tsv", sep="\t", index=False)
    guide_counts.to_csv(outdir / f"{safe}.guide_carrier_counts.tsv", sep="\t", index=False)
    return per_guide, per_target, qc, guide_counts


def write_data_dictionary(outdir: Path) -> None:
    rows = [
        ("knockdown_per_guide.tsv.gz", "Main plot-input table with one row per guide RNA and target-gene expression in carrier versus control cells."),
        ("knockdown_per_target.tsv.gz", "Target-level aggregation combining cells carrying any guide for a target."),
        ("guide_carrier_counts.tsv.gz", "Guide-to-target mapping, detected NTC status, and carrier cell counts."),
        ("processing_qc.tsv", "Input paths, cell overlap, guide counts, NTC counts, and matching QC for each sample."),
        ("manifest.json", "Run metadata and file list."),
        ("quality_checks/*.tsv", "Per-sample guide mapping and guide carrier QC."),
    ]
    pd.DataFrame(rows, columns=["file", "description"]).to_csv(outdir / "data_dictionary.tsv", sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Fig. 4A-like knockdown plot input tables")
    ap.add_argument("--config", required=True, help="JSON config file")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    outdir = ensure_dir(cfg.get("output_dir", "fig4a_like_knockdown_export"))
    qc_dir = ensure_dir(outdir / "quality_checks")
    samples = cfg["samples"]

    all_per_guide = []
    all_per_target = []
    all_qc = []
    all_guide_counts = []

    for sample_cfg in samples:
        per_guide, per_target, qc, guide_counts = summarize_sample(sample_cfg, qc_dir, cfg)
        all_per_guide.append(per_guide)
        all_per_target.append(per_target)
        all_qc.append(qc)
        all_guide_counts.append(guide_counts)

    per_guide_all = pd.concat(all_per_guide, ignore_index=True, sort=False) if all_per_guide else pd.DataFrame()
    per_target_all = pd.concat(all_per_target, ignore_index=True, sort=False) if all_per_target else pd.DataFrame()
    qc_all = pd.concat(all_qc, ignore_index=True, sort=False) if all_qc else pd.DataFrame()
    guide_counts_all = pd.concat(all_guide_counts, ignore_index=True, sort=False) if all_guide_counts else pd.DataFrame()

    per_guide_path = outdir / "knockdown_per_guide.tsv.gz"
    per_target_path = outdir / "knockdown_per_target.tsv.gz"
    guide_counts_path = outdir / "guide_carrier_counts.tsv.gz"
    qc_path = outdir / "processing_qc.tsv"
    per_guide_all.to_csv(per_guide_path, sep="\t", index=False, compression="gzip")
    per_target_all.to_csv(per_target_path, sep="\t", index=False, compression="gzip")
    guide_counts_all.to_csv(guide_counts_path, sep="\t", index=False, compression="gzip")
    qc_all.to_csv(qc_path, sep="\t", index=False)
    write_data_dictionary(outdir)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": Path(__file__).name,
        "config": os.path.abspath(args.config),
        "figure_style_reference": "Fig. 4A-like on-target knockdown efficiency per gRNA",
        "description": "Target-gene expression in guide-carrier cells compared with non-targeting-control cells.",
        "files": sorted([str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()]),
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    bundle_path = Path(cfg.get("bundle_path", outdir / "fig4a_like_knockdown_plot_input_bundle.zip"))
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in outdir.rglob("*"):
            if p.is_file() and p.resolve() != bundle_path.resolve():
                zf.write(p, arcname=str(p.relative_to(outdir)))
    log(f"Wrote {bundle_path}")


if __name__ == "__main__":
    main()
