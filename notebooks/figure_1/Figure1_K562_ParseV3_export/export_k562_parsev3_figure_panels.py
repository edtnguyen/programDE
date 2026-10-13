#!/usr/bin/env python3
"""
Export plot-input tables for K562/iPSC Parse v3 circular-capture panels.

This script reprocesses the analysis pattern in:
    250904.Parse.v3.iPSC.K562.copy.ipynb

It is designed to run on Sherlock and write a zip bundle of derived tables that
can be uploaded for publication-quality plotting.

Expected panels supported by the exported tables:
  - Main Fig. 1e: guide UMI/cell in an additional cell type (K562)
  - Supplement: guide UMI/cell across RT primers (CS1, DC1, Dual, no_guide)
  - Supplement: guide UMI/cell stratified by inferred cell type (K562 vs hiPSC)
  - Supplement: iPSC-guide vs K562-guide specificity/mixing scatter
  - Supplement: guide-detection metrics and guide-balance/skew summaries
  - Supplement: transcriptome QC and transcriptome saturation by sample

The script does not make plots. It writes tables and a bundle zip.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

try:
    from scipy.stats import mannwhitneyu, skew as scipy_skew
except Exception:  # pragma: no cover
    mannwhitneyu = None
    scipy_skew = None

try:
    import scanpy as sc
    import anndata as ad
except Exception:  # pragma: no cover
    sc = None
    ad = None


###############################################################################
# General helpers
###############################################################################


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".tsv") or path.endswith(".tsv.gz") or path.endswith(".txt") or path.endswith(".txt.gz"):
        return pd.read_csv(path, sep="\t", **kwargs)
    return pd.read_csv(path, **kwargs)


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def normalize_cell_id(x: Any) -> str:
    s = str(x)
    s = re.sub(r"__s\d+$", "", s)
    return s


def parse_sample_order(config: Dict[str, Any]) -> List[str]:
    return list(config.get("sample_order", ["CS1", "DC1", "Dual", "no_guide"]))


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    idx = np.where(mask)[0]
    p_valid = p[idx]
    order = np.argsort(p_valid)
    sorted_idx = idx[order]
    sorted_p = p_valid[order]
    m = len(sorted_p)
    adj_sorted = np.empty(m, dtype=float)
    for k, pv in enumerate(sorted_p, start=1):
        adj_sorted[k - 1] = (m - k + 1) * pv
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out[sorted_idx] = adj_sorted
    return out


###############################################################################
# GEX loading and QC
###############################################################################


@dataclass
class GexData:
    X: sparse.csr_matrix
    genes: pd.DataFrame
    cells: pd.DataFrame
    cell_ids: List[str]
    gene_names: List[str]
    source_dir: str


def infer_companion_paths(dge_dir: str | Path) -> Dict[str, str]:
    d = Path(dge_dir)
    matrix_candidates = [d / "count_matrix.mtx", d / "DGE.mtx"]
    gene_candidates = [d / "all_genes.csv", d / "genes.csv"]
    cell_candidates = [d / "cell_metadata.csv", d / "cells.csv", d / "cell_metadata.tsv"]
    out = {}
    for key, candidates in [("matrix", matrix_candidates), ("genes", gene_candidates), ("cells", cell_candidates)]:
        for p in candidates:
            if p.exists():
                out[key] = str(p)
                break
        if key not in out:
            raise FileNotFoundError(f"Could not find {key} companion file in {dge_dir}. Tried: {candidates}")
    return out


def load_gex(dge_dir: str | Path, config: Dict[str, Any]) -> GexData:
    paths = infer_companion_paths(dge_dir)
    log(f"Loading GEX matrix: {paths['matrix']}")
    X = mmread(paths["matrix"]).tocsr()
    genes = read_csv_auto(paths["genes"])
    cells = read_csv_auto(paths["cells"])

    gene_name_col = config.get("gene_name_column", "gene_name")
    if gene_name_col not in genes.columns:
        # Fall back to the first column if needed.
        gene_name_col = genes.columns[0]
    genes = genes.copy()
    genes[gene_name_col] = genes[gene_name_col].astype(str)
    notna = genes[gene_name_col].notna() & (genes[gene_name_col].astype(str) != "nan")

    # Determine orientation.
    n_cells, n_genes = len(cells), len(genes)
    if X.shape == (n_cells, n_genes):
        pass
    elif X.shape == (n_genes, n_cells):
        X = X.T.tocsr()
    else:
        raise ValueError(
            f"Matrix shape {X.shape} does not match cells ({n_cells}) and genes ({n_genes}) in {dge_dir}"
        )

    keep_gene_idx = np.where(notna.values)[0]
    X = X[:, keep_gene_idx].tocsr()
    genes = genes.iloc[keep_gene_idx].reset_index(drop=True)

    cell_id_col = config.get("cell_id_column", "bc_wells")
    if cell_id_col not in cells.columns:
        cell_id_col = cells.columns[0]
    cell_ids = [normalize_cell_id(x) for x in cells[cell_id_col].astype(str).tolist()]
    gene_names = genes[gene_name_col].astype(str).tolist()
    return GexData(X=X, genes=genes, cells=cells, cell_ids=cell_ids, gene_names=gene_names, source_dir=str(dge_dir))


def row_nnz(X: sparse.csr_matrix) -> np.ndarray:
    return np.diff(X.indptr)


def compute_gex_qc(gex: GexData, config: Dict[str, Any]) -> pd.DataFrame:
    X = gex.X.tocsr()
    totals = np.asarray(X.sum(axis=1)).ravel()
    n_genes = row_nnz(X)
    gene_names = pd.Index(gex.gene_names)
    mt_mask = gene_names.astype(str).str.startswith(config.get("mito_prefix", "MT-"))
    if mt_mask.any():
        mt_counts = np.asarray(X[:, np.where(mt_mask)[0]].sum(axis=1)).ravel()
    else:
        mt_counts = np.zeros(X.shape[0], dtype=float)
    pct_mt = np.divide(mt_counts, totals, out=np.zeros_like(mt_counts, dtype=float), where=totals > 0) * 100
    obs = gex.cells.copy()
    obs["cell_id"] = gex.cell_ids
    obs["total_counts"] = totals
    obs["n_genes_by_counts"] = n_genes
    obs["pct_counts_mt"] = pct_mt
    if "sample" not in obs.columns:
        raise ValueError("cell_metadata.csv must contain a 'sample' column for this notebook-derived analysis.")
    return obs


def apply_notebook_qc(qc: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    thresholds = config.get("gex_qc", {})
    max_genes = thresholds.get("max_genes_by_counts", 5000)
    max_counts = thresholds.get("max_total_counts", 5000)
    max_pct_mt = thresholds.get("max_pct_counts_mt", 40)
    min_counts = thresholds.get("min_total_counts", None)
    min_genes = thresholds.get("min_genes_by_counts", None)
    mask = (
        (qc["n_genes_by_counts"] < max_genes)
        & (qc["total_counts"] < max_counts)
        & (qc["pct_counts_mt"] < max_pct_mt)
    )
    if min_counts is not None:
        mask &= qc["total_counts"] >= min_counts
    if min_genes is not None:
        mask &= qc["n_genes_by_counts"] >= min_genes
    out = qc.loc[mask].copy()
    return out


###############################################################################
# Cell-type assignment
###############################################################################


def log_normalized_marker_score(gex: GexData, cell_idx: np.ndarray, markers: Sequence[str]) -> np.ndarray:
    gene_to_idx = {str(g): i for i, g in enumerate(gex.gene_names)}
    idxs = [gene_to_idx[m] for m in markers if m in gene_to_idx]
    if not idxs:
        return np.zeros(len(cell_idx), dtype=float)
    X = gex.X[cell_idx, :].tocsr()
    totals = np.asarray(X.sum(axis=1)).ravel()
    sub = X[:, idxs].toarray().astype(float)
    norm = np.divide(sub, totals[:, None], out=np.zeros_like(sub), where=totals[:, None] > 0) * 10000
    return np.log1p(norm).mean(axis=1)


def assign_cell_type(gex: GexData, qc_cells: pd.DataFrame, config: Dict[str, Any], outdir: Path) -> pd.DataFrame:
    """Assign hiPSC/K562 using marker scores, with optional Scanpy Leiden support."""
    method = config.get("cell_type_assignment", {}).get("method", "marker_score")
    k562_markers = config.get("cell_type_assignment", {}).get("k562_markers", ["GATA1", "KLF1", "HBB", "HBA1", "HBA2"])
    hipsc_markers = config.get("cell_type_assignment", {}).get("hipsc_markers", ["SOX2", "NANOG", "POU5F1", "LIN28A"])

    cell_to_idx = {c: i for i, c in enumerate(gex.cell_ids)}
    cell_idx = np.array([cell_to_idx[c] for c in qc_cells["cell_id"]], dtype=int)
    k_score = log_normalized_marker_score(gex, cell_idx, k562_markers)
    h_score = log_normalized_marker_score(gex, cell_idx, hipsc_markers)

    out = qc_cells[["cell_id", "sample", "total_counts", "n_genes_by_counts", "pct_counts_mt"]].copy()
    out["k562_marker_score"] = k_score
    out["hipsc_marker_score"] = h_score
    out["cell_type_assignment_method"] = "marker_score"
    out["cell_type"] = np.where(k_score > h_score, "K562", "hiPSC")
    out.loc[(k_score == 0) & (h_score == 0), "cell_type"] = "unassigned"

    # Optional: run Scanpy like the notebook and use marker scores to label clusters.
    if method in {"scanpy_leiden", "scanpy_leiden_marker_label"}:
        if sc is None or ad is None:
            print("WARNING: scanpy/anndata unavailable; using marker_score cell type assignment.", file=sys.stderr)
        else:
            try:
                log("Running Scanpy notebook-like Leiden clustering for cell type assignment")
                Xqc = gex.X[cell_idx, :].copy().tocsr()
                adata = ad.AnnData(Xqc)
                adata.obs = out.set_index("cell_id").copy()
                adata.var_names = pd.Index(gex.gene_names).astype(str)
                adata.var_names_make_unique()
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                adata.raw = adata.copy()
                sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.25)
                if "highly_variable" in adata.var.columns and adata.var["highly_variable"].sum() > 50:
                    adata = adata[:, adata.var["highly_variable"]].copy()
                sc.pp.regress_out(adata, ["total_counts"])
                sc.pp.scale(adata, max_value=10)
                sc.tl.pca(adata, svd_solver="arpack")
                sc.pp.neighbors(adata, n_neighbors=20, n_pcs=min(30, adata.obsm["X_pca"].shape[1]))
                sc.tl.umap(adata)
                sc.tl.leiden(adata, resolution=float(config.get("cell_type_assignment", {}).get("leiden_resolution", 0.2)))
                leiden = adata.obs["leiden"].astype(str).copy()
                cluster_scores = out.assign(leiden=leiden.values).groupby("leiden")[["k562_marker_score", "hipsc_marker_score"]].mean()
                cluster_label = np.where(cluster_scores["k562_marker_score"] > cluster_scores["hipsc_marker_score"], "K562", "hiPSC")
                label_map = dict(zip(cluster_scores.index.astype(str), cluster_label))
                out["leiden"] = leiden.values
                out["cell_type"] = out["leiden"].astype(str).map(label_map).fillna(out["cell_type"])
                out["cell_type_assignment_method"] = "scanpy_leiden_marker_label"
                # Save embeddings for optional UMAP panels.
                emb = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
                emb.insert(0, "cell_id", out["cell_id"].values)
                emb["leiden"] = out["leiden"].values
                emb["cell_type"] = out["cell_type"].values
                emb.to_csv(outdir / "umap_embedding_cell_type.tsv.gz", sep="\t", index=False, compression="gzip")
            except Exception as e:
                print(f"WARNING: Scanpy cell-type assignment failed ({e}); using marker_score assignment.", file=sys.stderr)

    out.to_csv(outdir / "cell_type_assignment.tsv.gz", sep="\t", index=False, compression="gzip")
    return out


###############################################################################
# Guide loading and metrics
###############################################################################


def normalize_guide_id(x: Any) -> str:
    s = str(x)
    s = re.sub(r"^gRNA_", "", s)
    return s


def guide_family(guide_id: str) -> str:
    if str(guide_id).startswith("iPSC_"):
        return "iPSC"
    if str(guide_id).startswith("K562_"):
        return "K562"
    if re.search(r"(?i)non.?target|ntc|negative|scramble|control", str(guide_id)):
        return "NTC"
    return "other"


def infer_guide_target(guide_id: str) -> str:
    gid = str(guide_id)
    parts = gid.split("_")
    if len(parts) >= 2 and parts[0] in {"iPSC", "K562"}:
        return parts[1]
    return parts[0]


def load_guide_long(config: Dict[str, Any], valid_cells: Iterable[str], outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    guide_path = config["guide_data_path"]
    chunksize = int(config.get("guide_chunksize", 1_000_000))
    valid = set(str(c) for c in valid_cells)
    valid_with_suffix = {c + "__s1" for c in valid}
    usecols = config.get("guide_usecols", ["gene", "cell", "count"])

    log(f"Reading guide-count table in chunks: {guide_path}")
    chunks = []
    rows_seen = 0
    rows_kept = 0
    for chunk in pd.read_csv(guide_path, sep="\t", usecols=usecols, chunksize=chunksize):
        rows_seen += len(chunk)
        chunk = chunk.rename(columns={usecols[0]: "gene", usecols[1]: "cell", usecols[2]: "count"})
        chunk["cell_id"] = chunk["cell"].map(normalize_cell_id)
        mask = chunk["cell_id"].isin(valid) | chunk["cell"].isin(valid_with_suffix)
        if not mask.any():
            continue
        sub = chunk.loc[mask, ["gene", "cell_id", "count"]].copy()
        sub["guide_id"] = sub["gene"].map(normalize_guide_id)
        sub["count"] = pd.to_numeric(sub["count"], errors="coerce").fillna(0).astype(float)
        sub = sub[sub["count"] > 0]
        rows_kept += len(sub)
        chunks.append(sub[["cell_id", "guide_id", "count"]])

    if chunks:
        long = pd.concat(chunks, ignore_index=True)
        # Sum any duplicated cell-guide rows.
        long = long.groupby(["cell_id", "guide_id"], as_index=False)["count"].sum()
    else:
        long = pd.DataFrame(columns=["cell_id", "guide_id", "count"])

    qc = pd.DataFrame([{
        "guide_data_path": guide_path,
        "rows_seen": rows_seen,
        "rows_kept_after_valid_cell_filter": rows_kept,
        "n_filtered_cell_guide_pairs": len(long),
        "n_cells_with_any_guide_row": long["cell_id"].nunique() if not long.empty else 0,
        "n_guides_seen_after_filter": long["guide_id"].nunique() if not long.empty else 0,
    }])
    long.to_csv(outdir / "filtered_cell_guide_long.tsv.gz", sep="\t", index=False, compression="gzip")
    return long, qc


def build_cell_guide_metrics(long: pd.DataFrame, cell_annot: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    guide_threshold = float(config.get("guide_threshold", 5))
    cells = cell_annot[["cell_id", "sample", "cell_type", "total_counts", "n_genes_by_counts", "pct_counts_mt"]].copy()
    if long.empty:
        for col in ["total_gRNA", "top_guide_umis", "iPSC_total", "K562_total", "n_guides_ge_threshold"]:
            cells[col] = 0.0
        cells["top_guide"] = ""
        return cells

    # Cell totals.
    total = long.groupby("cell_id")["count"].sum().rename("total_gRNA")
    top_idx = long.sort_values(["cell_id", "count"], ascending=[True, False]).groupby("cell_id", as_index=False).head(1)
    top = top_idx.set_index("cell_id")[["guide_id", "count"]].rename(columns={"guide_id": "top_guide", "count": "top_guide_umis"})
    long2 = long.copy()
    long2["guide_family"] = long2["guide_id"].map(guide_family)
    fam = long2.pivot_table(index="cell_id", columns="guide_family", values="count", aggfunc="sum", fill_value=0)
    for col in ["iPSC", "K562", "NTC", "other"]:
        if col not in fam.columns:
            fam[col] = 0.0
    fam = fam.rename(columns={"iPSC": "iPSC_total", "K562": "K562_total", "NTC": "NTC_total", "other": "other_guide_total"})
    ge = long[long["count"] >= guide_threshold].groupby("cell_id").size().rename("n_guides_ge_threshold")

    out = cells.set_index("cell_id").join(total).join(top).join(fam).join(ge).reset_index()
    numeric_fill = ["total_gRNA", "top_guide_umis", "iPSC_total", "K562_total", "NTC_total", "other_guide_total", "n_guides_ge_threshold"]
    for col in numeric_fill:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0)
    out["top_guide"] = out.get("top_guide", pd.Series(index=out.index, dtype=object)).fillna("")
    out["detected_total_gRNA_ge_threshold"] = out["total_gRNA"] >= guide_threshold
    out["detected_top_guide_ge_threshold"] = out["top_guide_umis"] >= guide_threshold
    out["expected_species_total_gRNA"] = np.where(out["cell_type"] == "K562", out["K562_total"], np.where(out["cell_type"] == "hiPSC", out["iPSC_total"], np.nan))
    out["off_species_total_gRNA"] = np.where(out["cell_type"] == "K562", out["iPSC_total"], np.where(out["cell_type"] == "hiPSC", out["K562_total"], np.nan))
    out["correct_species_guide_fraction"] = np.divide(
        out["expected_species_total_gRNA"],
        out["expected_species_total_gRNA"] + out["off_species_total_gRNA"],
        out=np.full(len(out), np.nan, dtype=float),
        where=(out["expected_species_total_gRNA"] + out["off_species_total_gRNA"]).to_numpy() > 0,
    )
    out["guide_umis_per_detected_guide"] = np.divide(
        out["total_gRNA"],
        out["n_guides_ge_threshold"],
        out=np.full(len(out), np.nan, dtype=float),
        where=out["n_guides_ge_threshold"].to_numpy() > 0,
    )
    return out


def guide_metrics_by_group(long: pd.DataFrame, cell_metrics: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    guide_threshold = float(config.get("guide_threshold", 5))
    if long.empty:
        return pd.DataFrame()
    annot = cell_metrics[["cell_id", "sample", "cell_type"]].copy()
    x = long.merge(annot, on="cell_id", how="left")
    x["guide_family"] = x["guide_id"].map(guide_family)
    x["target_gene"] = x["guide_id"].map(infer_guide_target)
    rows = (
        x.groupby(["sample", "cell_type", "guide_id", "guide_family", "target_gene"], dropna=False)
        .agg(
            total_guide_umis=("count", "sum"),
            n_cells_nonzero=("cell_id", "nunique"),
            n_cells_ge_threshold=("count", lambda s: int((s >= guide_threshold).sum())),
            median_umis_nonzero=("count", "median"),
            mean_umis_nonzero=("count", "mean"),
        )
        .reset_index()
    )
    denom = cell_metrics.groupby(["sample", "cell_type"], dropna=False)["cell_id"].nunique().rename("n_cells_group").reset_index()
    rows = rows.merge(denom, on=["sample", "cell_type"], how="left")
    rows["fraction_cells_ge_threshold"] = rows["n_cells_ge_threshold"] / rows["n_cells_group"]
    return rows


###############################################################################
# Summary tables and stats
###############################################################################


def summarize_cell_metrics(cell_metrics: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    metrics = config.get("summary_metrics", [
        "total_gRNA", "top_guide_umis", "iPSC_total", "K562_total", "expected_species_total_gRNA",
        "off_species_total_gRNA", "correct_species_guide_fraction", "total_counts", "n_genes_by_counts", "pct_counts_mt"
    ])
    rows = []
    for group_cols in [["sample"], ["cell_type"], ["sample", "cell_type"]]:
        for keys, sub in cell_metrics.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = {col: val for col, val in zip(group_cols, keys)}
            base["grouping"] = "+".join(group_cols)
            base["n_cells"] = len(sub)
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[metric], errors="coerce")
                rows.append({
                    **base,
                    "metric": metric,
                    "mean": float(vals.mean(skipna=True)),
                    "median": float(vals.median(skipna=True)),
                    "q1": float(vals.quantile(0.25)),
                    "q3": float(vals.quantile(0.75)),
                    "min": float(vals.min(skipna=True)),
                    "max": float(vals.max(skipna=True)),
                })
    return pd.DataFrame(rows)


def detection_summary(cell_metrics: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    thresholds = config.get("guide_detection_thresholds", [1, 3, 5, 10, 20, 50])
    rows = []
    metric_defs = {
        "total_gRNA": "all guides total",
        "top_guide_umis": "top guide",
        "expected_species_total_gRNA": "expected species guides total",
        "off_species_total_gRNA": "off species guides total",
    }
    for group_cols in [["sample"], ["sample", "cell_type"], ["cell_type"]]:
        for keys, sub in cell_metrics.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = {col: val for col, val in zip(group_cols, keys)}
            base["grouping"] = "+".join(group_cols)
            base["n_cells"] = len(sub)
            for metric, label in metric_defs.items():
                if metric not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[metric], errors="coerce").fillna(0)
                for thr in thresholds:
                    rows.append({
                        **base,
                        "metric": metric,
                        "metric_label": label,
                        "threshold": thr,
                        "n_detected": int((vals >= thr).sum()),
                        "fraction_detected": float((vals >= thr).mean()) if len(vals) else np.nan,
                        "percent_detected": float((vals >= thr).mean() * 100) if len(vals) else np.nan,
                    })
    return pd.DataFrame(rows)


def pairwise_primer_tests(cell_metrics: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    primers = [p for p in config.get("primer_samples", ["CS1", "DC1", "Dual"]) if p in set(cell_metrics["sample"].dropna())]
    metrics = config.get("primer_test_metrics", ["total_gRNA", "expected_species_total_gRNA", "top_guide_umis"])
    cell_types = ["all"] + sorted([x for x in cell_metrics["cell_type"].dropna().unique() if x != "unassigned"])
    rows = []
    for metric in metrics:
        if metric not in cell_metrics.columns:
            continue
        for ct in cell_types:
            sub = cell_metrics if ct == "all" else cell_metrics[cell_metrics["cell_type"] == ct]
            pairs = []
            pvals = []
            for i in range(len(primers)):
                for j in range(i + 1, len(primers)):
                    a = pd.to_numeric(sub.loc[sub["sample"] == primers[i], metric], errors="coerce").dropna().to_numpy()
                    b = pd.to_numeric(sub.loc[sub["sample"] == primers[j], metric], errors="coerce").dropna().to_numpy()
                    p = np.nan
                    if mannwhitneyu is not None and len(a) > 0 and len(b) > 0:
                        try:
                            p = float(mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue)
                        except TypeError:
                            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                    pairs.append((primers[i], primers[j], len(a), len(b), float(np.median(a)) if len(a) else np.nan, float(np.median(b)) if len(b) else np.nan))
                    pvals.append(p)
            adj = holm_adjust(pvals)
            for (a_name, b_name, n_a, n_b, med_a, med_b), p, q in zip(pairs, pvals, adj):
                rows.append({
                    "metric": metric,
                    "cell_type": ct,
                    "sample_a": a_name,
                    "sample_b": b_name,
                    "n_a": n_a,
                    "n_b": n_b,
                    "median_a": med_a,
                    "median_b": med_b,
                    "pvalue_mannwhitney_two_sided": p,
                    "qvalue_holm": q,
                })
    return pd.DataFrame(rows)


def guide_balance_by_target(cell_metrics: pd.DataFrame, guide_metrics: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if guide_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Target-level abundance based on guide IDs; particularly useful for K562/iPSC guides.
    target = (
        guide_metrics.groupby(["sample", "cell_type", "guide_family", "target_gene"], dropna=False)
        .agg(
            n_guides=("guide_id", "nunique"),
            total_guide_umis=("total_guide_umis", "sum"),
            n_cells_ge_threshold=("n_cells_ge_threshold", "sum"),
            median_fraction_cells_ge_threshold=("fraction_cells_ge_threshold", "median"),
        )
        .reset_index()
    )
    skew_rows = []
    for keys, sub in target.groupby(["sample", "cell_type", "guide_family"], dropna=False):
        vals = pd.to_numeric(sub["n_cells_ge_threshold"], errors="coerce").fillna(0)
        med = float(vals.median()) if len(vals) else np.nan
        mean = float(vals.mean()) if len(vals) else np.nan
        sk = float(vals.skew()) if len(vals) > 2 else np.nan
        if scipy_skew is not None and len(vals) > 2:
            sk = float(scipy_skew(vals))
        skew_rows.append({
            "sample": keys[0],
            "cell_type": keys[1],
            "guide_family": keys[2],
            "n_targets": int(len(vals)),
            "mean_cells_per_target": mean,
            "median_cells_per_target": med,
            "mean_to_median_cells_per_target": mean / med if med and np.isfinite(med) and med > 0 else np.nan,
            "skewness_cells_per_target": sk,
        })
    return target, pd.DataFrame(skew_rows)


###############################################################################
# Saturation curves
###############################################################################


def read_saturation_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    raw = raw.iloc[1:].copy()
    if raw.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in saturation file: {path}")
    raw = raw.iloc[:, :2]
    raw.columns = ["read_count", "transcript_count"]
    raw["read_count"] = pd.to_numeric(raw["read_count"], errors="coerce")
    raw["transcript_count"] = pd.to_numeric(raw["transcript_count"], errors="coerce")
    raw = raw.dropna().sort_values("read_count")
    return raw


def load_saturation(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    qc = []
    files = config.get("saturation_files", {})
    for sample, path in files.items():
        exists = Path(path).exists()
        qc.append({"sample": sample, "path": path, "exists": exists})
        if not exists:
            continue
        try:
            df = read_saturation_csv(path)
            df.insert(0, "sample", sample)
            rows.append(df)
        except Exception as e:
            qc[-1]["error"] = str(e)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sample", "read_count", "transcript_count"])
    return out, pd.DataFrame(qc)


###############################################################################
# Main export
###############################################################################


def write_data_dictionary(outdir: Path) -> None:
    rows = [
        ("cell_metrics.tsv.gz", "Per-QC-passing-cell guide, transcriptome QC, sample, and inferred cell-type metrics."),
        ("main_panel_K562_guide_capture.tsv.gz", "K562 cells in guide-primer samples, intended for main Fig. 1e K562 guide-capture panel."),
        ("primer_guide_capture_cell_metrics.tsv.gz", "Per-cell guide metrics across CS1, DC1, Dual, and no_guide, for primer-comparison panels."),
        ("species_specificity_cell_metrics.tsv.gz", "Per-cell iPSC-guide vs K562-guide totals for specificity/mixing scatter panels."),
        ("guide_detection_summary.tsv", "Percent of cells above guide UMI thresholds by sample and cell type."),
        ("cell_metric_summary.tsv", "Summary statistics for cell-level guide and transcriptome metrics."),
        ("guide_metrics_by_sample_celltype.tsv.gz", "Per-guide abundance and carrier counts by sample and inferred cell type."),
        ("guide_target_balance.tsv.gz", "Target-level guide detection balance/skew table."),
        ("guide_target_skew_summary.tsv", "Skewness and mean/median ratio for guide target representation."),
        ("pairwise_primer_tests.tsv", "Mann-Whitney/Holm pairwise tests across CS1, DC1, and Dual primer groups."),
        ("transcriptome_saturation.tsv", "Transcriptome saturation input parsed from tscp_counts_subsampled.csv files."),
        ("transcriptome_qc_by_sample_celltype.tsv", "Transcriptome QC summary by sample and cell type."),
        ("processing_qc.tsv", "Run-level QC, file paths, row counts, and cell counts."),
        ("cell_type_assignment.tsv.gz", "Inferred K562/hiPSC cell-type assignments and marker scores."),
        ("filtered_cell_guide_long.tsv.gz", "Filtered long cell-guide count table; useful for custom replotting."),
        ("manifest.json", "Run metadata and bundled file list."),
    ]
    pd.DataFrame(rows, columns=["file", "description"]).to_csv(outdir / "data_dictionary.tsv", sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export K562/iPSC Parse v3 circular-capture panel plot-input tables")
    ap.add_argument("--config", required=True, help="JSON config file")
    args = ap.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    outdir = ensure_dir(config.get("output_dir", "k562_parsev3_figure_export"))
    log(f"Output directory: {outdir}")

    # Load GEX and QC.
    gex = load_gex(config["dge_filtered_dir"], config)
    gex_qc_all = compute_gex_qc(gex, config)
    gex_qc = apply_notebook_qc(gex_qc_all, config)
    log(f"GEX cells before QC: {len(gex_qc_all):,}; after notebook QC: {len(gex_qc):,}")

    # Assign cell types.
    cell_type_annot = assign_cell_type(gex, gex_qc, config, outdir)

    # Load guide counts and build cell metrics.
    guide_long, guide_qc = load_guide_long(config, cell_type_annot["cell_id"].tolist(), outdir)
    cell_metrics = build_cell_guide_metrics(guide_long, cell_type_annot, config)
    sample_order = parse_sample_order(config)
    cell_metrics["sample"] = pd.Categorical(cell_metrics["sample"], categories=sample_order, ordered=True)
    cell_metrics.to_csv(outdir / "cell_metrics.tsv.gz", sep="\t", index=False, compression="gzip")

    # Panel-specific exports.
    primer_samples = config.get("primer_samples", ["CS1", "DC1", "Dual"])
    main_k562 = cell_metrics[(cell_metrics["cell_type"] == "K562") & (cell_metrics["sample"].isin(primer_samples))].copy()
    main_k562.to_csv(outdir / "main_panel_K562_guide_capture.tsv.gz", sep="\t", index=False, compression="gzip")

    primer_cells = cell_metrics[cell_metrics["sample"].isin(sample_order)].copy()
    primer_cells.to_csv(outdir / "primer_guide_capture_cell_metrics.tsv.gz", sep="\t", index=False, compression="gzip")

    specificity_cols = [
        "cell_id", "sample", "cell_type", "iPSC_total", "K562_total", "expected_species_total_gRNA",
        "off_species_total_gRNA", "correct_species_guide_fraction", "total_gRNA", "top_guide_umis",
        "total_counts", "n_genes_by_counts", "pct_counts_mt"
    ]
    cell_metrics[[c for c in specificity_cols if c in cell_metrics.columns]].to_csv(
        outdir / "species_specificity_cell_metrics.tsv.gz", sep="\t", index=False, compression="gzip"
    )

    # Summary tables.
    det = detection_summary(cell_metrics, config)
    det.to_csv(outdir / "guide_detection_summary.tsv", sep="\t", index=False)

    summ = summarize_cell_metrics(cell_metrics, config)
    summ.to_csv(outdir / "cell_metric_summary.tsv", sep="\t", index=False)

    guide_metrics = guide_metrics_by_group(guide_long, cell_metrics, config)
    guide_metrics.to_csv(outdir / "guide_metrics_by_sample_celltype.tsv.gz", sep="\t", index=False, compression="gzip")

    target_balance, skew_summary = guide_balance_by_target(cell_metrics, guide_metrics, config)
    target_balance.to_csv(outdir / "guide_target_balance.tsv.gz", sep="\t", index=False, compression="gzip")
    skew_summary.to_csv(outdir / "guide_target_skew_summary.tsv", sep="\t", index=False)

    primer_tests = pairwise_primer_tests(cell_metrics, config)
    primer_tests.to_csv(outdir / "pairwise_primer_tests.tsv", sep="\t", index=False)

    # Transcriptome saturation and transcriptome QC.
    sat, sat_qc = load_saturation(config)
    sat.to_csv(outdir / "transcriptome_saturation.tsv", sep="\t", index=False)
    sat_qc.to_csv(outdir / "transcriptome_saturation_file_qc.tsv", sep="\t", index=False)

    tx_qc = (
        cell_metrics.groupby(["sample", "cell_type"], dropna=False)
        .agg(
            n_cells=("cell_id", "nunique"),
            median_total_counts=("total_counts", "median"),
            mean_total_counts=("total_counts", "mean"),
            median_genes=("n_genes_by_counts", "median"),
            mean_genes=("n_genes_by_counts", "mean"),
            median_pct_mt=("pct_counts_mt", "median"),
        )
        .reset_index()
    )
    tx_qc.to_csv(outdir / "transcriptome_qc_by_sample_celltype.tsv", sep="\t", index=False)

    # Processing QC.
    before_counts = gex_qc_all.groupby("sample", dropna=False)["cell_id"].nunique().rename("n_cells_before_qc").reset_index()
    after_counts = gex_qc.groupby("sample", dropna=False)["cell_id"].nunique().rename("n_cells_after_qc").reset_index()
    ctype_counts = cell_metrics.groupby(["sample", "cell_type"], dropna=False)["cell_id"].nunique().rename("n_cells").reset_index()
    file_qc = pd.DataFrame([{
        "dge_filtered_dir": config["dge_filtered_dir"],
        "guide_data_path": config["guide_data_path"],
        "all_guides_path": config.get("all_guides_path", ""),
        "n_gex_cells_before_qc": len(gex_qc_all),
        "n_gex_cells_after_qc": len(gex_qc),
        "n_cell_metrics_rows": len(cell_metrics),
        "n_filtered_cell_guide_pairs": len(guide_long),
        "n_guides_seen_after_filter": guide_long["guide_id"].nunique() if not guide_long.empty else 0,
        "cell_type_assignment_method_requested": config.get("cell_type_assignment", {}).get("method", "marker_score"),
    }])
    processing_qc = pd.concat([
        file_qc,
        guide_qc.reindex(columns=file_qc.columns.union(guide_qc.columns), fill_value=np.nan)
    ], ignore_index=True, sort=False)
    processing_qc.to_csv(outdir / "processing_qc.tsv", sep="\t", index=False)
    before_counts.to_csv(outdir / "gex_cells_before_qc_by_sample.tsv", sep="\t", index=False)
    after_counts.to_csv(outdir / "gex_cells_after_qc_by_sample.tsv", sep="\t", index=False)
    ctype_counts.to_csv(outdir / "cell_type_counts_by_sample.tsv", sep="\t", index=False)

    write_data_dictionary(outdir)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": Path(__file__).name,
        "config": os.path.abspath(args.config),
        "notebook_source": "250904.Parse.v3.iPSC.K562.copy.ipynb",
        "description": "Plot-input export for K562/iPSC Parse v3 circular-capture guide and transcriptome QC panels.",
        "files": sorted([str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()]),
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    bundle_path = Path(config.get("bundle_path", outdir / "k562_parsev3_figure_panel_plot_input_bundle.zip"))
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in outdir.rglob("*"):
            if p.is_file() and p.resolve() != bundle_path.resolve():
                zf.write(p, arcname=str(p.relative_to(outdir)))
    log(f"Wrote bundle: {bundle_path}")


if __name__ == "__main__":
    main()
