#!/usr/bin/env python3
"""
b08-only Figure 1 export/plotting script for the CC-Perturb-seq manuscript.

Purpose
-------
The previous Figure 1 draft used "top guide UMIs" across all guide-like features.
For the single-guide hiPSC comparison, this is not the right quantity. This script
restricts the guide feature to the expected b08 guide in each assay:

    Circular capture: sg_b08
    CROP-seq:         b08

It writes uploadable plot-input tables and, when plotnine is installed, editable SVGs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp


# -------------------------
# Basic I/O helpers
# -------------------------

def as_path(x: str | os.PathLike[str]) -> Path:
    return Path(x).expanduser().resolve()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = as_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_csr(x: Any) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def to_1d(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def find_existing(base: Path, candidates: Sequence[str]) -> Path | None:
    for c in candidates:
        fp = base / c
        if fp.exists():
            return fp
    return None


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


def infer_column(df: pd.DataFrame, preferred: Sequence[str], fallback_contains: Sequence[str] = ()) -> str:
    for c in preferred:
        if c in df.columns:
            return c
    for token in fallback_contains:
        matches = [c for c in df.columns if token.lower() in c.lower()]
        if matches:
            return matches[0]
    if df.shape[1] == 1:
        return df.columns[0]
    raise ValueError(f"Could not infer column. Columns: {list(df.columns)}")


def read_parse_matrix(
    matrix_dir: str | os.PathLike[str],
    matrix_candidates: Sequence[str] = ("count_matrix.mtx", "DGE.mtx", "matrix.mtx"),
    feature_csv_candidates: Sequence[str] = ("all_genes.csv", "all_guides.csv", "features.csv"),
    cell_csv_candidates: Sequence[str] = ("cell_metadata.csv", "barcodes.csv", "cells.csv"),
    feature_name_preferred: Sequence[str] = ("gene_name", "guide_id", "guide_name", "name", "feature_name"),
    cell_name_preferred: Sequence[str] = ("bc_wells", "barcode", "cell_barcode", "cell", "cell_id"),
) -> dict[str, Any]:
    """Read a Parse-style MatrixMarket directory and orient matrix to cells x features."""
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
    feature_col = infer_column(feature_df, feature_name_preferred, ("gene", "guide", "name"))
    feature_names_raw = feature_df[feature_col]

    if cell_fp is not None:
        cell_df = pd.read_csv(cell_fp)
        cell_col = infer_column(cell_df, cell_name_preferred, ("bc", "barcode", "cell"))
        cell_names_raw = cell_df[cell_col]
    else:
        cell_df = None
        cell_col = None
        cell_names_raw = pd.Series([f"cell_{i}" for i in range(max(X.shape))])

    n_features = len(feature_df)
    n_cells = len(cell_df) if cell_df is not None else None

    if n_cells is not None and X.shape == (n_cells, n_features):
        pass
    elif n_cells is not None and X.shape == (n_features, n_cells):
        X = X.T.tocsr()
    elif X.shape[1] == n_features:
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

    obs.index = cell_names
    feature_df.index = feature_names

    return {
        "X": X,
        "obs": obs,
        "var": feature_df,
        "cell_names": cell_names,
        "feature_names": feature_names,
        "matrix_path": str(matrix_fp),
        "feature_path": str(feature_fp),
        "cell_path": str(cell_fp) if cell_fp is not None else None,
        "feature_name_col": feature_col,
        "cell_name_col": cell_col,
    }


# -------------------------
# QC and b08 extraction
# -------------------------

def feature_regex_mask(feature_names: Sequence[str], regex: str | None) -> np.ndarray:
    if not regex:
        return np.zeros(len(feature_names), dtype=bool)
    pat = re.compile(regex)
    return np.array([bool(pat.search(str(x))) for x in feature_names], dtype=bool)


def qc_from_matrix(X: sp.spmatrix, feature_names: Sequence[str]) -> pd.DataFrame:
    X = ensure_csr(X)
    total_counts = to_1d(X.sum(axis=1))
    n_features = np.diff(X.indptr)
    mt_mask = np.array([str(x).upper().startswith("MT-") for x in feature_names], dtype=bool)
    if mt_mask.any():
        mt_counts = to_1d(X[:, mt_mask].sum(axis=1))
        pct_mt = np.divide(
            mt_counts, total_counts,
            out=np.zeros_like(mt_counts, dtype=float),
            where=total_counts > 0,
        ) * 100
    else:
        pct_mt = np.zeros(X.shape[0], dtype=float)
    return pd.DataFrame({
        "total_counts": total_counts,
        "n_genes_by_counts": n_features,
        "pct_counts_mt": pct_mt,
    })


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


def resolve_feature_index(feature_names: Sequence[str], expected_feature: str, fallback_regex: str | None = None) -> int:
    names = np.array([str(x) for x in feature_names], dtype=object)
    exact = np.where(names == expected_feature)[0]
    if len(exact) == 1:
        return int(exact[0])
    if len(exact) > 1:
        raise ValueError(f"Expected feature {expected_feature!r} matched multiple features: {names[exact].tolist()}")

    lower = np.where(np.char.lower(names.astype(str)) == expected_feature.lower())[0]
    if len(lower) == 1:
        return int(lower[0])
    if fallback_regex:
        pat = re.compile(fallback_regex)
        hits = np.array([i for i, nm in enumerate(names) if pat.search(str(nm))], dtype=int)
        if len(hits) == 1:
            return int(hits[0])
        if len(hits) > 1:
            raise ValueError(f"Fallback regex {fallback_regex!r} matched multiple features: {names[hits].tolist()}")
    guide_like = [str(n) for n in names if ("b08" in str(n).lower() or "sg_" in str(n).lower())]
    raise ValueError(
        f"Could not find expected feature {expected_feature!r}. "
        f"b08/sg-like features seen: {guide_like[:30]}"
    )


def process_expected_b08_sample(
    *,
    matrix_dir: str | os.PathLike[str],
    dataset_id: str,
    sample_id: str,
    method: str,
    cell_type: str,
    expected_feature: str,
    expected_feature_regex: str | None = None,
    guide_exclusion_regex: str | None = None,
    detection_threshold: int = 5,
    qc_filter: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parsed = read_parse_matrix(matrix_dir)
    X = parsed["X"].tocsr()
    features = np.array(parsed["feature_names"], dtype=object).astype(str)

    expected_idx = resolve_feature_index(features, expected_feature, expected_feature_regex)

    guide_mask = feature_regex_mask(features, guide_exclusion_regex)
    guide_mask[expected_idx] = True
    gex_mask = ~guide_mask

    X_gex = X[:, gex_mask].tocsr()
    gex_features = features[gex_mask].astype(str).tolist()
    qc = qc_from_matrix(X_gex, gex_features)
    qc.index = parsed["cell_names"]

    keep = filter_cells_by_qc(qc, **dict(qc_filter or {}))
    b08_umis_all = to_1d(X[:, expected_idx].sum(axis=1)).astype(float)
    cell_names = np.array(parsed["cell_names"], dtype=object).astype(str)

    out = pd.DataFrame({
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "method": method,
        "cell_type": cell_type,
        "cell_id": cell_names[keep],
        "expected_guide": expected_feature,
        "expected_guide_umis": b08_umis_all[keep],
        "detected_ge_1": b08_umis_all[keep] >= 1,
        "detected_ge_3": b08_umis_all[keep] >= 3,
        "detected_ge_5": b08_umis_all[keep] >= 5,
        "detected_ge_10": b08_umis_all[keep] >= 10,
        "detected_ge_20": b08_umis_all[keep] >= 20,
    })
    out = pd.concat([out.reset_index(drop=True), qc.loc[cell_names[keep]].reset_index(drop=True)], axis=1)

    n = int(out.shape[0])
    det_col = f"detected_ge_{detection_threshold}"
    if det_col not in out.columns:
        out[det_col] = out["expected_guide_umis"] >= detection_threshold

    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "method": method,
        "cell_type": cell_type,
        "expected_guide": expected_feature,
        "expected_feature_index": int(expected_idx),
        "n_cells": n,
        "median_b08_umis_all_cells": float(out["expected_guide_umis"].median()) if n else np.nan,
        "mean_b08_umis_all_cells": float(out["expected_guide_umis"].mean()) if n else np.nan,
        "median_b08_umis_detected_ge_5": float(out.loc[out["detected_ge_5"], "expected_guide_umis"].median()) if out["detected_ge_5"].any() else np.nan,
        "pct_cells_b08_ge_1": float(out["detected_ge_1"].mean() * 100) if n else np.nan,
        "pct_cells_b08_ge_3": float(out["detected_ge_3"].mean() * 100) if n else np.nan,
        "pct_cells_b08_ge_5": float(out["detected_ge_5"].mean() * 100) if n else np.nan,
        "pct_cells_b08_ge_10": float(out["detected_ge_10"].mean() * 100) if n else np.nan,
        "pct_cells_b08_ge_20": float(out["detected_ge_20"].mean() * 100) if n else np.nan,
        "pct_cells_b08_at_detection_threshold": float(out[det_col].mean() * 100) if n else np.nan,
        "detection_threshold": int(detection_threshold),
        "median_gex_umis": float(out["total_counts"].median()) if n else np.nan,
        "median_genes": float(out["n_genes_by_counts"].median()) if n else np.nan,
        "pct_counts_mt_median": float(out["pct_counts_mt"].median()) if n else np.nan,
        "matrix_path": parsed["matrix_path"],
        "feature_path": parsed["feature_path"],
        "cell_path": parsed["cell_path"],
        "guide_exclusion_regex": guide_exclusion_regex,
    }
    return out, summary


# -------------------------
# Outputs
# -------------------------

def make_detection_threshold_table(cell_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_id, sample_id, method, cell_type, expected_guide), g in cell_df.groupby(
        ["dataset_id", "sample_id", "method", "cell_type", "expected_guide"],
        dropna=False,
    ):
        for threshold in [1, 3, 5, 10, 20]:
            rows.append({
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "method": method,
                "cell_type": cell_type,
                "expected_guide": expected_guide,
                "threshold": threshold,
                "n_cells": int(g.shape[0]),
                "n_detected": int((g["expected_guide_umis"] >= threshold).sum()),
                "percent_cells_detected": float((g["expected_guide_umis"] >= threshold).mean() * 100),
            })
    return pd.DataFrame(rows)


def write_data_dictionary(out_dir: Path) -> None:
    dd = pd.DataFrame([
        ("b08_cell_metrics_for_plotting.tsv.gz", "Per-cell b08 guide UMI counts after transcriptome QC filtering."),
        ("b08_summary_for_plotting.tsv", "Per-sample summary statistics for the expected b08 guide."),
        ("b08_detection_thresholds_for_plotting.tsv", "Percent cells with b08 detected across UMI thresholds."),
        ("quality_checks/b08_feature_check.tsv", "Feature matching and source path checks."),
        ("plots/*.svg", "Editable SVG plots if plotnine is available."),
    ], columns=["file", "description"])
    dd.to_csv(out_dir / "data_dictionary.tsv", sep="\t", index=False)


def make_bundle(out_dir: Path, bundle_name: str = "figure1_b08_plot_input_bundle.zip") -> Path:
    bundle = out_dir / bundle_name
    if bundle.exists():
        bundle.unlink()
    include_patterns = [
        "b08_cell_metrics_for_plotting.tsv.gz",
        "b08_summary_for_plotting.tsv",
        "b08_detection_thresholds_for_plotting.tsv",
        "data_dictionary.tsv",
        "manifest.json",
        "quality_checks/b08_feature_check.tsv",
        "plots/fig1b_b08_guide_umis.svg",
        "plots/fig1b_b08_guide_umis.png",
        "plots/fig1c_b08_detection_bar.svg",
        "plots/fig1c_b08_detection_bar.png",
        "plots/fig1c_b08_detection_threshold_curve.svg",
        "plots/fig1c_b08_detection_threshold_curve.png",
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in include_patterns:
            fp = out_dir / rel
            if fp.exists():
                zf.write(fp, arcname=rel)
    return bundle


def make_plots(out_dir: Path) -> None:
    try:
        import matplotlib as mpl
        mpl.rcParams["svg.fonttype"] = "none"
        from plotnine import (
            ggplot, aes, geom_violin, geom_boxplot, geom_text, geom_col, geom_line, geom_point,
            labs, theme_classic, theme, element_text, element_blank, element_rect,
            scale_fill_manual, scale_color_manual, scale_y_continuous, scale_x_discrete,
            position_dodge,
        )
        from mizani.transforms import log1p_trans
    except Exception as exc:
        warnings.warn(f"plotnine/matplotlib not available; skipping plots. Error: {exc!r}")
        return

    plot_dir = ensure_dir(out_dir / "plots")
    cells = pd.read_csv(out_dir / "b08_cell_metrics_for_plotting.tsv.gz", sep="\t")
    thresholds = pd.read_csv(out_dir / "b08_detection_thresholds_for_plotting.tsv", sep="\t")
    summary = pd.read_csv(out_dir / "b08_summary_for_plotting.tsv", sep="\t")

    method_order = ["Circular capture", "CROP-seq"]
    cells["method"] = pd.Categorical(cells["method"], categories=method_order, ordered=True)
    thresholds["method"] = pd.Categorical(thresholds["method"], categories=method_order, ordered=True)
    summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)

    colors = {"Circular capture": "#2C7FB8", "CROP-seq": "#D95F0E"}

    def theme_pub(width=2.9, height=2.6, legend="none"):
        return (
            theme_classic(base_size=8)
            + theme(
                figure_size=(width, height),
                axis_text=element_text(size=8, color="black"),
                axis_title=element_text(size=9, color="black"),
                plot_title=element_text(size=9, weight="bold", ha="left"),
                legend_title=element_blank(),
                legend_text=element_text(size=8),
                legend_position=legend,
                panel_border=element_rect(fill=None, color="black", size=0.4),
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
            )
        )

    n_map = summary.set_index("method")["n_cells"].to_dict()
    x_labels = [
        f"Circular\ncapture\n(n={int(n_map.get('Circular capture', 0))})",
        f"CROP-seq\n(n={int(n_map.get('CROP-seq', 0))})",
    ]

    # Fig 1b: b08 UMI distribution
    med = cells.groupby("method", observed=True)["expected_guide_umis"].median().reset_index(name="median")
    med["label"] = med["median"].map(lambda x: f"{x:.0f}")
    y_max = max(float(cells["expected_guide_umis"].max()), 1.0)
    med["y"] = y_max * 0.93
    p_b = (
        ggplot(cells, aes("method", "expected_guide_umis", fill="method"))
        + geom_violin(width=0.94, alpha=0.75, color=None)
        + geom_boxplot(width=0.18, outlier_alpha=0.08, fill="white", color="black")
        + geom_text(data=med, mapping=aes(x="method", y="y", label="label"), inherit_aes=False, size=6)
        + scale_fill_manual(values=colors)
        + scale_x_discrete(labels=x_labels)
        + scale_y_continuous(trans=log1p_trans())
        + labs(title="b  b08 guide capture in hiPSCs", x="", y="b08 guide UMIs per cell")
        + theme_pub(2.9, 2.7, "none")
    )
    p_b.save(plot_dir / "fig1b_b08_guide_umis.svg", width=2.9, height=2.7, units="in", verbose=False)
    p_b.save(plot_dir / "fig1b_b08_guide_umis.png", width=2.9, height=2.7, units="in", dpi=300, verbose=False)

    # Fig 1c option A: main detection threshold, ≥5 UMIs
    bar = thresholds.loc[thresholds["threshold"].eq(5)].copy()
    bar["label"] = bar["percent_cells_detected"].map(lambda x: f"{x:.1f}")
    p_c = (
        ggplot(bar, aes("method", "percent_cells_detected", fill="method"))
        + geom_col(width=0.7, color="black", size=0.25)
        + geom_text(aes(label="label"), va="bottom", nudge_y=2, size=6)
        + scale_fill_manual(values=colors)
        + scale_x_discrete(labels=x_labels)
        + scale_y_continuous(limits=(0, 110), breaks=[0, 25, 50, 75, 100])
        + labs(title="c  Cells with b08 guide detected", x="", y="Cells with b08 ≥5 UMIs (%)")
        + theme_pub(2.8, 2.7, "none")
    )
    p_c.save(plot_dir / "fig1c_b08_detection_bar.svg", width=2.8, height=2.7, units="in", verbose=False)
    p_c.save(plot_dir / "fig1c_b08_detection_bar.png", width=2.8, height=2.7, units="in", dpi=300, verbose=False)

    # Fig 1c option B: threshold curve
    p_curve = (
        ggplot(thresholds, aes("threshold", "percent_cells_detected", color="method"))
        + geom_line(size=0.9)
        + geom_point(size=2.2)
        + scale_color_manual(values=colors)
        + scale_y_continuous(limits=(0, 105), breaks=[0, 25, 50, 75, 100])
        + labs(title="c  b08 detection across thresholds", x="b08 UMI threshold", y="Cells detected (%)")
        + theme_pub(3.0, 2.7, "top")
    )
    p_curve.save(plot_dir / "fig1c_b08_detection_threshold_curve.svg", width=3.0, height=2.7, units="in", verbose=False)
    p_curve.save(plot_dir / "fig1c_b08_detection_threshold_curve.png", width=3.0, height=2.7, units="in", dpi=300, verbose=False)


def run(config: Mapping[str, Any]) -> Path:
    out_dir = ensure_dir(config.get("out_dir", "figure1_b08_only"))
    qc_filter = config.get("qc_filter_parse", {"min_genes": 300, "max_genes": 10000, "max_counts": 100000, "max_pct_mt": 10})
    detection_threshold = int(config.get("detection_threshold", 5))

    all_cells: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for key, ds in config.get("datasets", {}).items():
        if not ds.get("enabled", False):
            continue
        try:
            cell_df, summary = process_expected_b08_sample(
                matrix_dir=ds["matrix_dir"],
                dataset_id=ds.get("dataset_id", "ipsc_single_guide"),
                sample_id=ds["sample_id"],
                method=ds["method"],
                cell_type=ds.get("cell_type", "hiPSC"),
                expected_feature=ds["expected_feature"],
                expected_feature_regex=ds.get("expected_feature_regex"),
                guide_exclusion_regex=ds.get("guide_exclusion_regex"),
                detection_threshold=detection_threshold,
                qc_filter=qc_filter,
            )
            all_cells.append(cell_df)
            summaries.append(summary)
            checks.append({
                "dataset_key": key,
                "sample_id": ds["sample_id"],
                "expected_feature": ds["expected_feature"],
                "matched_feature_index": summary["expected_feature_index"],
                "matrix_path": summary["matrix_path"],
                "feature_path": summary["feature_path"],
                "cell_path": summary["cell_path"],
                "status": "ok",
            })
        except Exception as exc:
            errors.append({"dataset_key": key, "error": repr(exc)})
            checks.append({
                "dataset_key": key,
                "sample_id": ds.get("sample_id", key),
                "expected_feature": ds.get("expected_feature"),
                "status": "error",
                "error": repr(exc),
            })

    if not all_cells:
        raise RuntimeError(f"No b08 cell tables were generated. Errors: {errors}")

    cell_df = pd.concat(all_cells, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    threshold_df = make_detection_threshold_table(cell_df)

    cell_df.to_csv(out_dir / "b08_cell_metrics_for_plotting.tsv.gz", sep="\t", index=False, compression="gzip")
    summary_df.to_csv(out_dir / "b08_summary_for_plotting.tsv", sep="\t", index=False)
    threshold_df.to_csv(out_dir / "b08_detection_thresholds_for_plotting.tsv", sep="\t", index=False)

    qdir = ensure_dir(out_dir / "quality_checks")
    pd.DataFrame(checks).to_csv(qdir / "b08_feature_check.tsv", sep="\t", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(qdir / "b08_processing_errors.tsv", sep="\t", index=False)

    write_data_dictionary(out_dir)
    manifest = {
        "description": "b08-only plot inputs for Figure 1b/1c. Circular capture uses sg_b08; CROP-seq uses b08.",
        "out_dir": str(out_dir),
        "n_cells_table_rows": int(cell_df.shape[0]),
        "n_samples": int(summary_df.shape[0]),
        "detection_threshold": detection_threshold,
        "errors": errors,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not config.get("skip_plots", False):
        make_plots(out_dir)

    bundle = make_bundle(out_dir)
    return bundle


def default_config() -> dict[str, Any]:
    return {
        "out_dir": "/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_cc_capture_b08_only",
        "detection_threshold": 5,
        "skip_plots": False,
        "qc_filter_parse": {
            "min_genes": 300,
            "max_genes": 10000,
            "max_counts": 100000,
            "max_pct_mt": 10
        },
        "datasets": {
            "ipsc_single_cc_b08": {
                "enabled": True,
                "matrix_dir": "/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/230820_eTN2_iPSC_mixed/panmg/BS3/DGE_filtered",
                "dataset_id": "ipsc_single_guide_b08_only",
                "sample_id": "hiPSC_circular_capture_b08",
                "method": "Circular capture",
                "cell_type": "hiPSC",
                "expected_feature": "sg_b08",
                "expected_feature_regex": "^sg_b08$",
                "guide_exclusion_regex": "sg_"
            },
            "ipsc_single_crop_b08": {
                "enabled": True,
                "matrix_dir": "/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/230621_Tri_guide_iPSCs/iPSC/iPSC_guide/DGE_filtered",
                "dataset_id": "ipsc_single_guide_b08_only",
                "sample_id": "hiPSC_CROP_seq_b08",
                "method": "CROP-seq",
                "cell_type": "hiPSC",
                "expected_feature": "b08",
                "expected_feature_regex": "^b08$",
                "guide_exclusion_regex": "^(b08)$|sg_"
            }
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export and plot b08-only Figure 1b/1c inputs.")
    ap.add_argument("--config-json", default=None, help="Path to JSON config. If omitted, default Sherlock config is used.")
    ap.add_argument("--write-default-config", default=None, help="Write the default config JSON to this path and exit.")
    ap.add_argument("--skip-plots", action="store_true", help="Write tables and bundle only; do not attempt plotnine plots.")
    args = ap.parse_args()

    if args.write_default_config:
        fp = Path(args.write_default_config)
        fp.write_text(json.dumps(default_config(), indent=2))
        print(f"Wrote default config to {fp}")
        return

    if args.config_json:
        config = json.loads(Path(args.config_json).read_text())
    else:
        config = default_config()
    if args.skip_plots:
        config["skip_plots"] = True

    bundle = run(config)
    print(f"Wrote b08-only bundle: {bundle}")


if __name__ == "__main__":
    main()
