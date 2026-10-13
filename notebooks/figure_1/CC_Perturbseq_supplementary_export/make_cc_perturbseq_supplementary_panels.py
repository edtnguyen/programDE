#!/usr/bin/env python3
"""
Create plot-input tables and draft SVG/PNG panels for the CC-Perturb-seq
Figure 1 supplementary package.

This script consumes the compact plot-input bundles already generated during
this project, rather than re-reading raw matrices. It is designed to be run on
Sherlock, then the resulting output ZIP can be uploaded back for final plotting
or Illustrator refinement.

Panels generated when the matching input bundles are present:
  S1: hiPSC head-to-head QC
    - guide detection bar plot
    - transcriptome saturation / matched-depth QC
    - on-target knockdown, per-gene and/or per-guide, if knockdown bundle exists
  S2: Parse v3 K562/iPSC + RT-primer robustness
    - K562 guide capture for CS1/DC1/Dual
    - hiPSC guide capture for CS1/DC1/Dual
    - K562/hiPSC/no-guide negative control panel
    - guide detection threshold curves
    - expected-species guide specificity
    - transcriptome QC by sample/cell type
  S3: additional-cell-type generalizability
    - teloHAEC guide capture CC vs CROP
    - teloHAEC guide detection
    - teloHAEC transcriptome QC

The script also writes a missing-data report for panels that require data not
currently present in the bundles.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    mpl.rcParams["svg.fonttype"] = "none"
    from plotnine import (
        ggplot, aes, geom_violin, geom_boxplot, geom_text, geom_col,
        geom_point, geom_line, geom_linerange, geom_segment, geom_jitter,
        facet_wrap, labs, theme_classic, theme, element_text, element_blank,
        element_rect, scale_fill_manual, scale_color_manual, scale_y_continuous,
        scale_x_discrete, scale_x_continuous, position_dodge
    )
    from mizani.transforms import log1p_trans
    from mizani.formatters import comma_format
    PLOTNINE_AVAILABLE = True
except Exception as e:  # pragma: no cover
    PLOTNINE_AVAILABLE = False
    PLOTNINE_IMPORT_ERROR = str(e)


# Colors chosen to match the panels already drafted in this project.
METHOD_COLORS = {"Circular capture": "#2C7FB8", "CROP-seq": "#D95F0E"}
PRIMER_COLORS = {"CS1": "#2C7FB8", "DC1": "#D95F0E", "Dual": "#C0392B", "no_guide": "#9E9E9E"}
SIGNIF_COLORS = {"q≤0.10": "#D95F02", "NS": "#1F1F1F"}


###############################################################################
# IO helpers
###############################################################################


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_nonempty_path(path: Optional[str | Path]) -> bool:
    return bool(path) and Path(path).exists()


def read_table_path(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix in [".tsv", ".txt"] or path.name.endswith(".tsv.gz") else ","
    if path.name.endswith(".gz"):
        return pd.read_csv(path, sep=sep, compression="gzip")
    return pd.read_csv(path, sep=sep)


def read_table_from_bundle(bundle: str | Path, member: str) -> pd.DataFrame:
    """Read a TSV/CSV member from a zip bundle or an extracted directory."""
    bundle = Path(bundle)
    if not bundle.exists():
        raise FileNotFoundError(str(bundle))

    if bundle.is_dir():
        path = bundle / member
        if not path.exists():
            # try basename fallback
            candidates = list(bundle.rglob(Path(member).name))
            if not candidates:
                raise FileNotFoundError(f"{member} not found in {bundle}")
            path = candidates[0]
        return read_table_path(path)

    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        if member not in names:
            # tolerate bundles where the table is nested under one leading folder
            suffix_hits = [n for n in names if n.endswith(member)]
            basename_hits = [n for n in names if Path(n).name == Path(member).name]
            hits = suffix_hits or basename_hits
            if not hits:
                raise FileNotFoundError(f"{member} not found in {bundle}")
            member = hits[0]
        with zf.open(member) as fh:
            if member.endswith(".gz"):
                return pd.read_csv(gzip.open(fh, "rt"), sep="\t")
            if member.endswith(".tsv") or member.endswith(".txt"):
                return pd.read_csv(fh, sep="\t")
            return pd.read_csv(fh)


def try_read(bundle: Optional[str], member: str, missing: list, panel: str, required_for: str) -> Optional[pd.DataFrame]:
    if not is_nonempty_path(bundle):
        missing.append({"panel": panel, "required_for": required_for, "missing": "bundle_path", "path_or_member": str(bundle or "")})
        return None
    try:
        return read_table_from_bundle(bundle, member)
    except Exception as e:
        missing.append({"panel": panel, "required_for": required_for, "missing": "member", "path_or_member": f"{bundle} :: {member}", "error": str(e)})
        return None


def write_table(df: pd.DataFrame, outdir: Path, name: str) -> Path:
    path = outdir / name
    if name.endswith(".gz"):
        df.to_csv(path, sep="\t", index=False, compression="gzip")
    else:
        df.to_csv(path, sep="\t", index=False)
    return path


###############################################################################
# Plot helpers
###############################################################################


def base_theme(fig_size=(3.2, 2.8), legend="none"):
    if not PLOTNINE_AVAILABLE:
        return None
    return (
        theme_classic(base_size=8)
        + theme(
            figure_size=fig_size,
            axis_text=element_text(size=8, color="black"),
            axis_title=element_text(size=9, color="black"),
            plot_title=element_text(size=10, weight="bold", ha="left"),
            plot_subtitle=element_text(size=8, ha="left"),
            legend_title=element_blank(),
            legend_text=element_text(size=8),
            legend_position=legend,
            panel_border=element_rect(fill=None, color="black", size=0.4),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            strip_background=element_rect(fill="#F2F2F2", color="#D0D0D0"),
            strip_text=element_text(size=8, weight="bold"),
        )
    )


def save_plot(p, outdir: Path, stem: str, width: float, height: float) -> None:
    if not PLOTNINE_AVAILABLE:
        return
    p.save(outdir / f"{stem}.svg", width=width, height=height, units="in", verbose=False)
    p.save(outdir / f"{stem}.png", width=width, height=height, units="in", dpi=300, verbose=False)


def violin_panel(df: pd.DataFrame, x: str, y: str, fill: str, title: str, ylab: str,
                 outdir: Path, stem: str, order: Sequence[str], colors: Dict[str, str],
                 subtitle: str = "", width: float = 3.2, height: float = 2.8,
                 facet: Optional[str] = None, include_n: bool = True, log_y: bool = True) -> pd.DataFrame:
    d = df.copy()
    d = d[d[x].isin(order)].copy()
    d[x] = pd.Categorical(d[x], categories=list(order), ordered=True)
    d = d[pd.notna(d[y])].copy()

    group_cols = ([facet] if facet else []) + [x]
    summ = d.groupby(group_cols, observed=True)[y].agg(["count", "median", "max", "mean"]).reset_index()
    summ["label"] = summ["median"].map(lambda v: f"{v:.0f}" if pd.notna(v) and abs(v - round(v)) < 1e-9 else (f"{v:.1f}" if pd.notna(v) else "NA"))
    summ["y_pos"] = summ["max"] * 0.92

    if PLOTNINE_AVAILABLE:
        xlabels = {v: (f"{v}\n(n={int(summ.loc[summ[x] == v, 'count'].sum())})" if include_n else v) for v in order}
        p = (
            ggplot(d, aes(x, y, fill=fill))
            + geom_violin(width=0.95, alpha=0.78, color=None)
            + geom_boxplot(width=0.18, outlier_alpha=0.08, fill="white", color="black")
            + geom_text(data=summ, mapping=aes(x=x, y="y_pos", label="label"), inherit_aes=False, size=5.8)
            + scale_fill_manual(values={k: colors[k] for k in order if k in colors})
            + scale_x_discrete(labels=[xlabels[v] for v in order])
            + labs(title=title, subtitle=subtitle, x="", y=ylab)
            + base_theme((width, height), legend="none")
        )
        if log_y:
            p = p + scale_y_continuous(trans=log1p_trans())
        if facet:
            p = p + facet_wrap(f"~{facet}", nrow=1)
        save_plot(p, outdir, stem, width, height)
    return summ


###############################################################################
# Panel builders
###############################################################################


def build_hiPSC_detection(cfg: Dict[str, Any], outdir: Path, missing: list) -> None:
    panel = "S1_hiPSC_guide_detection"
    b08 = cfg["inputs"].get("b08_bundle")
    fig1 = cfg["inputs"].get("figure1_bundle")

    df = None
    if is_nonempty_path(b08):
        df = try_read(b08, "b08_summary_for_plotting.tsv", missing, panel, "b08 detection summary")
        if df is not None and "pct_cells_b08_ge_5" in df.columns:
            out = df[["sample_id", "method", "cell_type", "expected_guide", "n_cells", "pct_cells_b08_ge_5"]].copy()
            out["metric"] = "expected guide >5 UMIs"
            out["percent"] = out["pct_cells_b08_ge_5"]
    if df is None and is_nonempty_path(fig1):
        m = try_read(fig1, "panel_specific/fig1c_assignment_metrics.tsv", missing, panel, "fallback guide detection metrics")
        if m is not None:
            out = m[m["metric"].isin(["pct_cells_any_guide", "pct_guides_detected_min_cells"])].copy()
            out["percent"] = out["percent"].astype(float)
    if 'out' not in locals():
        missing.append({"panel": panel, "required_for": "hiPSC guide detection", "missing": "usable_detection_table", "path_or_member": "b08_bundle or figure1_bundle"})
        return
    write_table(out, outdir, "S1_hiPSC_guide_detection.tsv")

    if PLOTNINE_AVAILABLE:
        if "pct_cells_b08_ge_5" in out.columns:
            out["method"] = pd.Categorical(out["method"], categories=["Circular capture", "CROP-seq"], ordered=True)
            out["percent_label"] = out["percent"].map(lambda x: f"{x:.1f}")
            p = (
                ggplot(out, aes("method", "percent", fill="method"))
                + geom_col(width=0.7, color="black", size=0.25, show_legend=False)
                + geom_text(aes(label="percent_label"), va="bottom", size=6, nudge_y=2)
                + scale_fill_manual(values=METHOD_COLORS)
                + scale_y_continuous(limits=(0, 110), breaks=[0, 25, 50, 75, 100])
                + scale_x_discrete(labels=["Circular\ncapture", "CROP-seq"])
                + labs(title="Guide detection in hiPSCs", x="", y="Cells with guide >5 UMIs (%)")
                + base_theme((2.8, 2.7), legend="none")
            )
            save_plot(p, outdir, "S1_hiPSC_guide_detection", 2.8, 2.7)


def build_transcriptome_downsampling(cfg: Dict[str, Any], outdir: Path, missing: list) -> None:
    panel = "S1_transcriptome_saturation"
    bundle = cfg["inputs"].get("fig1d_downsampling_bundle")
    df = try_read(bundle, "transcriptome_downsampling_summary.tsv.gz", missing, panel, "transcriptome downsampling")
    if df is None:
        return
    write_table(df, outdir, "S1_transcriptome_downsampling_summary.tsv.gz")

    # Column-name robust plotting.
    x_candidates = ["assigned_reads_per_cell", "mean_assigned_reads_per_cell", "reads_per_cell", "sampled_assigned_reads_per_cell"]
    y_candidates = ["median_umis_per_cell", "median_transcriptome_umis_per_cell", "median_umi_per_cell"]
    gene_candidates = ["median_genes_per_cell", "median_features_per_cell", "median_genes_detected_per_cell"]
    xcol = next((c for c in x_candidates if c in df.columns), None)
    ycol = next((c for c in y_candidates if c in df.columns), None)
    gcol = next((c for c in gene_candidates if c in df.columns), None)
    if not (xcol and ycol and "method" in df.columns):
        missing.append({"panel": panel, "required_for": "plot transcriptome saturation", "missing": "expected_columns", "path_or_member": f"need x={x_candidates}, y={y_candidates}"})
        return
    d = df.copy()
    if "feature_set" in d.columns:
        if "all_non_guide" in set(d["feature_set"].astype(str)):
            d = d[d["feature_set"].astype(str) == "all_non_guide"].copy()
    d = d[pd.notna(d[xcol]) & pd.notna(d[ycol])]
    if PLOTNINE_AVAILABLE and not d.empty:
        d["method"] = pd.Categorical(d["method"], categories=["Circular capture", "CROP-seq"], ordered=True)
        p = (
            ggplot(d, aes(xcol, ycol, color="method"))
            + geom_line(size=1.0)
            + geom_point(size=1.8)
            + scale_color_manual(values=METHOD_COLORS)
            + scale_x_continuous(labels=comma_format())
            + scale_y_continuous(labels=comma_format())
            + labs(title="Transcriptome saturation in hiPSCs", x="Assigned transcriptome reads per cell", y="Median transcriptome UMIs per cell")
            + base_theme((3.4, 2.8), legend="top")
        )
        save_plot(p, outdir, "S1_transcriptome_umi_saturation", 3.4, 2.8)
        if gcol:
            p2 = (
                ggplot(d, aes(xcol, gcol, color="method"))
                + geom_line(size=1.0)
                + geom_point(size=1.8)
                + scale_color_manual(values=METHOD_COLORS)
                + scale_x_continuous(labels=comma_format())
                + scale_y_continuous(labels=comma_format())
                + labs(title="Transcriptome gene saturation in hiPSCs", x="Assigned transcriptome reads per cell", y="Median genes detected per cell")
                + base_theme((3.4, 2.8), legend="top")
            )
            save_plot(p2, outdir, "S1_transcriptome_gene_saturation", 3.4, 2.8)


def build_knockdown(cfg: Dict[str, Any], outdir: Path, missing: list) -> None:
    panel = "S1_on_target_knockdown"
    gene_bundle = cfg["inputs"].get("knockdown_per_gene_bundle")
    guide_bundle = cfg["inputs"].get("knockdown_per_guide_bundle")

    gene = try_read(gene_bundle, "knockdown_per_gene.tsv.gz", missing, panel, "per-gene knockdown",) if is_nonempty_path(gene_bundle) else None
    guide = try_read(guide_bundle, "knockdown_per_guide.tsv.gz", missing, panel, "per-guide knockdown",) if is_nonempty_path(guide_bundle) else None

    if gene is not None:
        write_table(gene, outdir, "S1_knockdown_per_gene.tsv.gz")
        d = gene[gene.get("sample_id", "") == cfg.get("knockdown_sample_id", "hiPSC_10gene_pilot")].copy()
        if not d.empty and PLOTNINE_AVAILABLE:
            if "gene_rank_by_control_expression" not in d.columns and "control_mean_log_normalized_expr" in d.columns:
                d = d.sort_values("control_mean_log_normalized_expr").reset_index(drop=True)
                d["gene_rank_by_control_expression"] = np.arange(1, len(d)+1)
            d = d.sort_values("gene_rank_by_control_expression")
            order = d["target_gene"].tolist()
            d["target_gene"] = pd.Categorical(d["target_gene"], categories=order, ordered=True)
            d["signif_label"] = np.where(d["significant_knockdown_q10"].fillna(False), "q≤0.10", "NS")
            if "control_sd_log_normalized_expr" in d.columns:
                d["ctrl_ymin"] = (d["control_mean_log_normalized_expr"] - d["control_sd_log_normalized_expr"]).clip(lower=0)
                d["ctrl_ymax"] = d["control_mean_log_normalized_expr"] + d["control_sd_log_normalized_expr"]
            else:
                d["ctrl_ymin"] = d["control_mean_log_normalized_expr"]
                d["ctrl_ymax"] = d["control_mean_log_normalized_expr"]
            p = (
                ggplot(d, aes(x="target_gene"))
                + geom_segment(aes(x="target_gene", xend="target_gene", y="control_mean_log_normalized_expr", yend="carrier_mean_log_normalized_expr"), color="#9E9E9E", size=0.7)
                + geom_linerange(aes(ymin="ctrl_ymin", ymax="ctrl_ymax"), color="#C7C7C7", size=1.2)
                + geom_point(aes(y="control_mean_log_normalized_expr"), color="#6E6E6E", fill="white", size=2.4, stroke=0.6)
                + geom_point(aes(y="carrier_mean_log_normalized_expr", color="signif_label"), size=2.7)
                + scale_color_manual(values=SIGNIF_COLORS)
                + labs(title="On-target knockdown", x="Target gene", y="Mean target-gene expression\nlog1p(counts per 10k)")
                + base_theme((6.8, 2.8), legend="top")
                + theme(axis_text_x=element_text(rotation=45, ha="right", size=7, color="black"))
            )
            save_plot(p, outdir, "S1_on_target_knockdown_per_gene", 6.8, 2.8)
    if guide is not None:
        write_table(guide, outdir, "S1_knockdown_per_guide.tsv.gz")


def build_k562_parsev3(cfg: Dict[str, Any], outdir: Path, missing: list) -> None:
    panel = "S2_ParseV3_K562_iPSC_primer"
    bundle = cfg["inputs"].get("k562_parsev3_bundle")
    cm = try_read(bundle, "primer_guide_capture_cell_metrics.tsv.gz", missing, panel, "cell-level guide metrics")
    det = try_read(bundle, "guide_detection_summary.tsv", missing, panel, "guide detection threshold curves")
    sp = try_read(bundle, "species_specificity_cell_metrics.tsv.gz", missing, panel, "species specificity")
    tx = try_read(bundle, "transcriptome_qc_by_sample_celltype.tsv", missing, panel, "transcriptome QC")
    pw = try_read(bundle, "pairwise_primer_tests.tsv", missing, panel, "pairwise primer tests")

    sample_order = ["CS1", "DC1", "Dual"]
    all_order = ["CS1", "DC1", "Dual", "no_guide"]

    if cm is not None:
        write_table(cm, outdir, "S2_parsev3_cell_guide_metrics.tsv.gz")
        # Main K562 panel.
        k562 = cm[cm["cell_type"] == "K562"].copy()
        if not k562.empty:
            summ = violin_panel(k562, "sample", "top_guide_umis", "sample", "Guide capture in K562 cells", "Top guide UMIs per cell", outdir, "S2_K562_guide_capture", sample_order, PRIMER_COLORS, "Parse Evercode v3 circular capture", 3.35, 2.95, include_n=True)
            write_table(summ, outdir, "S2_K562_guide_capture_summary.tsv")
        hipsc = cm[cm["cell_type"] == "hiPSC"].copy()
        if not hipsc.empty:
            summ = violin_panel(hipsc, "sample", "top_guide_umis", "sample", "Guide capture in hiPSC cells", "Top guide UMIs per cell", outdir, "S2_hiPSC_guide_capture", sample_order, PRIMER_COLORS, "Parse Evercode v3 circular capture", 3.35, 2.95, include_n=True)
            write_table(summ, outdir, "S2_hiPSC_guide_capture_summary.tsv")
        both = cm[cm["cell_type"].isin(["K562", "hiPSC"])].copy()
        if not both.empty:
            summ = violin_panel(both, "sample", "top_guide_umis", "sample", "Guide capture across cell types and primer designs", "Top guide UMIs per cell", outdir, "S2_all_celltypes_primer_capture", all_order, PRIMER_COLORS, "No-guide sample is a negative control", 6.2, 2.95, facet="cell_type", include_n=False)
            write_table(summ, outdir, "S2_all_celltypes_primer_capture_summary.tsv")

    if det is not None:
        write_table(det, outdir, "S2_parsev3_guide_detection_summary.tsv")
        d = det[(det["grouping"] == "sample+cell_type") & (det["metric"] == "top_guide_umis") & (det["cell_type"].isin(["K562", "hiPSC"])) & (det["sample"].isin(sample_order))].copy()
        if not d.empty and PLOTNINE_AVAILABLE:
            d["sample"] = pd.Categorical(d["sample"], categories=sample_order, ordered=True)
            d["cell_type"] = pd.Categorical(d["cell_type"], categories=["K562", "hiPSC"], ordered=True)
            p = (
                ggplot(d, aes("threshold", "percent_detected", color="sample"))
                + geom_line(size=1.0)
                + geom_point(size=1.7)
                + scale_color_manual(values={k: PRIMER_COLORS[k] for k in sample_order})
                + scale_x_continuous(breaks=sorted(d["threshold"].unique()))
                + scale_y_continuous(limits=(0, 100), breaks=[0, 25, 50, 75, 100])
                + facet_wrap("~cell_type", nrow=1)
                + labs(title="Guide detection across UMI thresholds", x="Top guide UMI threshold", y="Cells detected (%)")
                + base_theme((5.2, 2.95), legend="top")
            )
            save_plot(p, outdir, "S2_detection_threshold_curves", 5.2, 2.95)

    if sp is not None:
        write_table(sp, outdir, "S2_species_specificity_cell_metrics.tsv.gz")
        d = sp[(sp["cell_type"].isin(["K562", "hiPSC"])) & (sp["sample"].isin(sample_order))].copy()
        d = d[pd.notna(d["correct_species_guide_fraction"])]
        if not d.empty:
            d["sample"] = pd.Categorical(d["sample"], categories=sample_order, ordered=True)
            d["cell_type"] = pd.Categorical(d["cell_type"], categories=["K562", "hiPSC"], ordered=True)
            summ = d.groupby(["cell_type", "sample"], observed=True)["correct_species_guide_fraction"].agg(["count", "median", "max"]).reset_index()
            write_table(summ, outdir, "S2_species_specificity_summary.tsv")
            if PLOTNINE_AVAILABLE:
                ann = summ.copy(); ann["label"] = ann["median"].map(lambda x: f"{x:.2f}"); ann["y"] = 0.965
                p = (
                    ggplot(d, aes("sample", "correct_species_guide_fraction", fill="sample"))
                    + geom_violin(width=0.95, alpha=0.8, color=None)
                    + geom_boxplot(width=0.18, outlier_alpha=0.08, fill="white", color="black")
                    + geom_text(data=ann, mapping=aes(x="sample", y="y", label="label"), inherit_aes=False, size=5.5)
                    + scale_fill_manual(values={k: PRIMER_COLORS[k] for k in sample_order})
                    + scale_y_continuous(limits=(0, 1.0), breaks=[0, .25, .5, .75, 1.0])
                    + facet_wrap("~cell_type", nrow=1)
                    + labs(title="Expected-species guide specificity", x="", y="Expected-species guide fraction")
                    + base_theme((5.2, 2.95), legend="none")
                )
                save_plot(p, outdir, "S2_species_specificity", 5.2, 2.95)

    if tx is not None:
        write_table(tx, outdir, "S2_transcriptome_qc_by_sample_celltype.tsv")
        # Plot if long or wide summary is recognizable.
        d = tx.copy()
        if {"sample", "cell_type", "metric", "median"}.issubset(d.columns):
            d = d[d["metric"].isin(["total_counts", "n_genes_by_counts", "pct_counts_mt"])]
            if not d.empty and PLOTNINE_AVAILABLE:
                d["sample"] = pd.Categorical(d["sample"], categories=all_order, ordered=True)
                p = (
                    ggplot(d, aes("sample", "median", fill="sample"))
                    + geom_col(width=.72, color="black", size=.25)
                    + scale_fill_manual(values={k: PRIMER_COLORS[k] for k in all_order if k in PRIMER_COLORS})
                    + facet_wrap("~metric", scales="free_y")
                    + labs(title="Parse v3 transcriptome QC", x="", y="Median per cell")
                    + base_theme((6.2, 2.95), legend="none")
                    + theme(axis_text_x=element_text(rotation=45, ha="right", color="black"))
                )
                save_plot(p, outdir, "S2_transcriptome_qc", 6.2, 2.95)
    if pw is not None:
        write_table(pw, outdir, "S2_pairwise_primer_tests.tsv")


def build_telohaec(cfg: Dict[str, Any], outdir: Path, missing: list) -> None:
    panel = "S3_teloHAEC_generalizability"
    bundle = cfg["inputs"].get("figure1_bundle")
    full = try_read(bundle, "panel_specific/fig1e_other_celltypes_guide_umis_full.tsv.gz", missing, panel, "teloHAEC guide capture")
    cells = try_read(bundle, "cell_metrics_for_plotting.tsv.gz", missing, panel, "teloHAEC cell metrics")

    if full is not None:
        d = full[full["sample_id"].isin(["teloHAEC_circular_capture", "teloHAEC_CROP_seq"])].copy()
        write_table(d, outdir, "S3_teloHAEC_guide_capture_cell_metrics.tsv.gz")
        if not d.empty:
            d["method"] = pd.Categorical(d["method"], categories=["Circular capture", "CROP-seq"], ordered=True)
            summ = violin_panel(d, "method", "guide_umis_top", "method", "Guide capture in teloHAECs", "Top guide UMIs per cell", outdir, "S3_teloHAEC_guide_capture", ["Circular capture", "CROP-seq"], METHOD_COLORS, width=3.15, height=2.85, include_n=True)
            write_table(summ, outdir, "S3_teloHAEC_guide_capture_summary.tsv")
            # Detection table at thresholds.
            rows = []
            for (sample_id, method), sub in d.groupby(["sample_id", "method"], observed=True):
                for th in [1, 3, 5, 10, 20, 50]:
                    rows.append({"sample_id": sample_id, "method": method, "threshold": th, "n_cells": len(sub), "n_detected": int((sub["guide_umis_top"] >= th).sum()), "percent_detected": 100 * float((sub["guide_umis_top"] >= th).mean())})
            det = pd.DataFrame(rows)
            write_table(det, outdir, "S3_teloHAEC_guide_detection_thresholds.tsv")
            if PLOTNINE_AVAILABLE:
                p = (
                    ggplot(det, aes("threshold", "percent_detected", color="method"))
                    + geom_line(size=1.0)
                    + geom_point(size=1.8)
                    + scale_color_manual(values=METHOD_COLORS)
                    + scale_x_continuous(breaks=[1, 3, 5, 10, 20, 50])
                    + scale_y_continuous(limits=(0, 100), breaks=[0, 25, 50, 75, 100])
                    + labs(title="Guide detection in teloHAECs", x="Top guide UMI threshold", y="Cells detected (%)")
                    + base_theme((3.2, 2.85), legend="top")
                )
                save_plot(p, outdir, "S3_teloHAEC_guide_detection_thresholds", 3.2, 2.85)

    if cells is not None:
        d = cells[cells["sample_id"].isin(["teloHAEC_circular_capture", "teloHAEC_CROP_seq"])].copy()
        if not d.empty:
            write_table(d, outdir, "S3_teloHAEC_transcriptome_cell_metrics.tsv.gz")
            rows = []
            for metric in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
                if metric in d.columns:
                    for (sample_id, method), sub in d.groupby(["sample_id", "method"], observed=True):
                        rows.append({"sample_id": sample_id, "method": method, "metric": metric, "n_cells": len(sub), "median": float(np.median(sub[metric])), "mean": float(np.mean(sub[metric]))})
            qc = pd.DataFrame(rows)
            write_table(qc, outdir, "S3_teloHAEC_transcriptome_qc_summary.tsv")
            if PLOTNINE_AVAILABLE and not qc.empty:
                qc["method"] = pd.Categorical(qc["method"], categories=["Circular capture", "CROP-seq"], ordered=True)
                p = (
                    ggplot(qc, aes("method", "median", fill="method"))
                    + geom_col(width=.7, color="black", size=.25)
                    + scale_fill_manual(values=METHOD_COLORS)
                    + facet_wrap("~metric", scales="free_y")
                    + scale_x_discrete(labels=["Circular\ncapture", "CROP-seq"])
                    + labs(title="teloHAEC transcriptome QC", x="", y="Median per cell")
                    + base_theme((5.2, 2.85), legend="none")
                )
                save_plot(p, outdir, "S3_teloHAEC_transcriptome_qc", 5.2, 2.85)


###############################################################################
# Missing data report
###############################################################################


def write_missing_data_report(outdir: Path, missing_runtime: list) -> None:
    planned_missing = [
        {
            "panel": "S1_guide_read_mapping_QC",
            "required_for": "% barcode-valid guide reads and % uniquely mapped guide reads",
            "missing": "guide-library read/mapping summary tables",
            "path_or_member": "Need per-library guide read QC: total reads, barcode-valid reads, mapped reads, uniquely mapped reads. Examples: kb/kite logs, alignment summary, or custom CRISPR library mapping summary.",
        },
        {
            "panel": "S1_read_origin",
            "required_for": "% guideRNA reads from oligo-dT vs guide RT primer",
            "missing": "read-origin or primer-source table",
            "path_or_member": "Need counts of guide-supporting reads by primer/source: oligo-dT GEX readout vs guide-specific RT/circular-capture library.",
        },
        {
            "panel": "S1_or_later_replicate_correlations",
            "required_for": "replicate correlations",
            "missing": "replicate guide-count or perturbation-effect matrices",
            "path_or_member": "Need per-guide counts/effects per replicate, with replicate labels.",
        },
        {
            "panel": "S3_primary_mouse_hepatocytes",
            "required_for": "claim that CC-Perturb-seq works in primary mouse hepatocytes",
            "missing": "hepatocyte GEX/guide count data or plot-input bundle",
            "path_or_member": "Need cell_metrics-like table or raw processed GEX+guide matrices for primary mouse hepatocytes.",
        },
        {
            "panel": "S1a_S2a_schematics",
            "required_for": "expanded schematic and Parse v3/K562 experiment design",
            "missing": "schematic artwork",
            "path_or_member": "Requires Illustrator/BioRender-style schematic rather than count data.",
        },
    ]
    df = pd.DataFrame(missing_runtime + planned_missing)
    if df.empty:
        df = pd.DataFrame(columns=["panel", "required_for", "missing", "path_or_member", "error"])
    df.to_csv(outdir / "missing_data_report.tsv", sep="\t", index=False)


###############################################################################
# Main
###############################################################################


def make_bundle(outdir: Path, bundle_path: Path) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(outdir.rglob("*")):
            if p.is_file() and p.resolve() != bundle_path.resolve():
                zf.write(p, arcname=str(p.relative_to(outdir)))


def write_data_dictionary(outdir: Path) -> None:
    rows = [
        ("S1_hiPSC_guide_detection.tsv", "hiPSC circular-capture vs CROP-seq guide detection table."),
        ("S1_transcriptome_downsampling_summary.tsv.gz", "Matched-depth transcriptome downsampling / saturation summary."),
        ("S1_knockdown_per_gene.tsv.gz", "Per-gene on-target knockdown table, when available."),
        ("S2_parsev3_cell_guide_metrics.tsv.gz", "Parse v3 K562/iPSC cell-level guide metrics."),
        ("S2_K562_guide_capture_summary.tsv", "Summary statistics for K562 CS1/DC1/Dual guide-capture panel."),
        ("S2_species_specificity_summary.tsv", "Expected-species guide specificity summary."),
        ("S3_teloHAEC_guide_capture_cell_metrics.tsv.gz", "teloHAEC circular capture vs CROP-seq guide-capture cell metrics."),
        ("S3_teloHAEC_guide_detection_thresholds.tsv", "teloHAEC detection threshold table."),
        ("S3_teloHAEC_transcriptome_qc_summary.tsv", "teloHAEC transcriptome QC summary."),
        ("missing_data_report.tsv", "Panels/data still needed or skipped because inputs were unavailable."),
        ("*.svg / *.png", "Draft panels generated if plotnine is installed."),
    ]
    pd.DataFrame(rows, columns=["file", "description"]).to_csv(outdir / "data_dictionary.tsv", sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make CC-Perturb-seq supplementary plot-input tables and draft panels")
    ap.add_argument("--config", required=True, help="JSON config")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    outdir = ensure_dir(cfg.get("output_dir", "cc_perturbseq_supplementary_panels"))
    log(f"Output directory: {outdir}")
    if not PLOTNINE_AVAILABLE:
        log(f"plotnine not available; tables only. Import error: {PLOTNINE_IMPORT_ERROR}")

    missing = []
    build_hiPSC_detection(cfg, outdir, missing)
    build_transcriptome_downsampling(cfg, outdir, missing)
    build_knockdown(cfg, outdir, missing)
    build_k562_parsev3(cfg, outdir, missing)
    build_telohaec(cfg, outdir, missing)
    write_missing_data_report(outdir, missing)
    write_data_dictionary(outdir)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": Path(__file__).name,
        "config": os.path.abspath(args.config),
        "plotnine_available": PLOTNINE_AVAILABLE,
        "output_dir": str(outdir),
        "files": sorted([str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()]),
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    bundle_path = Path(cfg.get("bundle_path", outdir / "cc_perturbseq_supplementary_panels_bundle.zip"))
    make_bundle(outdir, bundle_path)
    log(f"Wrote {bundle_path}")


if __name__ == "__main__":
    main()
