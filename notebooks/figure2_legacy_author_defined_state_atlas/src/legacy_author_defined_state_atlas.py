#!/usr/bin/env python3
"""Build a coarse CC-Perturb-seq state atlas from the user's previous author-defined
Leiden-res-0.5 cluster labels.

This intentionally bypasses the control-only cluster review workflow. It does not recluster
or use target-to-program effects. State labels are fixed by the mapping recovered from
`ipsc-ec.Hippo.figure.ipynb`:
0 artEC, 1 iPSC, 2 Neuro-iPSC, 3 PrimStreak, 4 Lat-Meso,
5 Fibro-Mesen, 6 Pre-EC, 7 Neu-Mesen, 8 Others.

Scientific caveat: those legacy cluster boundaries were originally learned from all cells,
not from pure controls alone. The outputs document that explicitly.
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import yaml
from matplotlib.gridspec import GridSpec

try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

DAY_ORDER = ["D0", "D1", "D2", "D3"]


def normalize_day(x: Any) -> str:
    s = str(x).strip().upper().replace("DAY", "D").replace(" ", "")
    return {
        "0": "D0", "0.0": "D0", "D0": "D0",
        "1": "D1", "1.0": "D1", "D1": "D1",
        "2": "D2", "2.0": "D2", "D2": "D2",
        "3": "D3", "3.0": "D3", "D3": "D3",
    }.get(s, s)


def resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if value in (None, "", "null"):
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path) -> Tuple[Dict[str, Any], Path]:
    p = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(p.read_text())
    base = p.parent.parent if p.parent.name == "config" else p.parent
    path_keys = [
        "query_h5ad", "guide_metadata_tsv", "legacy_cluster_state_map_tsv",
        "state_color_map_tsv", "day_color_map_tsv", "output_dir",
    ]
    for key in path_keys:
        if key in cfg:
            resolved = resolve_path(base, cfg.get(key))
            cfg[key] = str(resolved) if resolved is not None else None
    return cfg, base


def ensure_csr(x: Any) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(np.asarray(x))


def read_color_map(path: str, key_col: str) -> Dict[str, str]:
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    d = pd.read_csv(p, sep=sep)
    color_col = "color" if "color" in d.columns else "hex"
    return dict(zip(d[key_col].astype(str), d[color_col].astype(str)))


def read_cluster_map(path: str) -> pd.DataFrame:
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    d = pd.read_csv(p, sep=sep, dtype={"source_cluster": str})
    required = {"source_cluster", "assigned_state"}
    if not required.issubset(d.columns):
        raise ValueError(f"Cluster map must contain {required}; found {list(d.columns)}")
    if d["source_cluster"].duplicated().any():
        raise ValueError("Cluster map has duplicated source_cluster values")
    return d


def select_pure_controls(query: ad.AnnData, cfg: Mapping[str, Any], outdir: Path):
    key = cfg.get("query_guide_assignment_obsm", "guide_assignment")
    names_key = cfg.get("query_guide_names_uns", "guide_names")
    if key not in query.obsm:
        raise KeyError(f"{key!r} absent from query.obsm; available={list(query.obsm.keys())}")
    if names_key not in query.uns:
        raise KeyError(f"{names_key!r} absent from query.uns; available={list(query.uns.keys())}")
    assign = ensure_csr(query.obsm[key])
    guides = list(map(str, query.uns[names_key]))
    if assign.shape[1] != len(guides):
        raise ValueError("guide_assignment columns do not match guide_names length")

    meta = pd.read_csv(cfg["guide_metadata_tsv"], sep="\t")
    if "guide_id" not in meta.columns:
        raise ValueError("Guide metadata requires guide_id")
    meta = meta.drop_duplicates("guide_id").set_index(meta["guide_id"].astype(str), drop=False)
    missing = [g for g in guides if g not in meta.index]
    if missing:
        pd.Series(missing, name="guide_id").to_csv(
            outdir / "guides_missing_from_metadata.tsv", sep="\t", index=False
        )
        raise ValueError(
            f"{len(missing)} query guides missing from metadata; "
            "see guides_missing_from_metadata.tsv"
        )

    m = meta.loc[guides]
    if "type" in m.columns:
        types = m["type"].astype(str).str.lower().to_numpy()
        control_cols = np.isin(types, ["safe-targeting", "non-targeting", "ntc"])
        real_cols = ~control_cols
    else:
        real_cols = m["targeting"].fillna(False).astype(bool).to_numpy()
        control_cols = ~real_cols

    n_control = np.asarray(assign[:, control_cols].sum(axis=1)).ravel()
    n_real = np.asarray(assign[:, real_cols].sum(axis=1)).ravel()
    pure = (n_control > 0) & (n_real == 0)
    mixed = (n_control > 0) & (n_real > 0)

    qc = pd.DataFrame({
        "cell_id": query.obs_names.astype(str),
        "n_control_guides": n_control,
        "n_real_targeting_guides": n_real,
        "is_pure_control": pure,
        "mixed_control_and_real": mixed,
    })
    qc.to_csv(outdir / "control_cell_definition_qc.tsv", sep="\t", index=False)
    definition = (
        "At least one safe-targeting or non-targeting guide and zero real-targeting guides. "
        "Cells carrying control and real-targeting guides together are excluded from the pure-control subset."
    )
    return pure, definition, qc


def state_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population, d in assignments.groupby("population", observed=True):
        for (day, rep), sub in d.groupby(["observed_day", "biological_replicate"], observed=True):
            counts = sub["assigned_state"].value_counts()
            total = int(len(sub))
            for state, n in counts.items():
                rows.append({
                    "population": population,
                    "observed_day": day,
                    "biological_replicate": rep,
                    "assigned_state": state,
                    "n_cells": int(n),
                    "fraction": float(n / total) if total else np.nan,
                    "row_total": total,
                })
    return pd.DataFrame(rows)


def descriptive_top_markers(
    query: ad.AnnData,
    states: np.ndarray,
    cfg: Mapping[str, Any],
    top_n: int,
) -> pd.DataFrame:
    layer = cfg.get("marker_expression_layer", "X")
    if layer in (None, "", "X"):
        x = query.X
    elif layer == "raw":
        if query.raw is None:
            raise ValueError("marker_expression_layer=raw but adata.raw is absent")
        x = query.raw.X
    else:
        if layer not in query.layers:
            raise KeyError(f"Layer {layer!r} absent; available={list(query.layers.keys())}")
        x = query.layers[layer]
    x = ensure_csr(x).astype(np.float32)

    sym_col = cfg.get("query_gene_symbol_col")
    symbols = (
        query.var[sym_col].astype(str).to_numpy()
        if sym_col and sym_col in query.var.columns
        else query.var_names.astype(str).to_numpy()
    )

    max_per_state = int(cfg.get("max_cells_per_state_for_marker_summary", 12000))
    rng = np.random.default_rng(int(cfg.get("random_seed", 20260727)))
    rows = []
    all_idx = np.arange(query.n_obs)

    for state in pd.unique(states):
        inside = all_idx[states == state]
        outside = all_idx[states != state]
        if len(inside) > max_per_state:
            inside = rng.choice(inside, max_per_state, replace=False)
        if len(outside) > max_per_state:
            outside = rng.choice(outside, max_per_state, replace=False)
        mean_in = np.asarray(x[inside].mean(axis=0)).ravel()
        mean_out = np.asarray(x[outside].mean(axis=0)).ravel()
        score = mean_in - mean_out
        order = np.argsort(score)[::-1][:top_n]
        for rank, j in enumerate(order, start=1):
            rows.append({
                "state": str(state),
                "rank": rank,
                "gene": str(symbols[j]),
                "mean_expression_in_state": float(mean_in[j]),
                "mean_expression_outside_state": float(mean_out[j]),
                "mean_difference": float(score[j]),
                "marker_summary_type": "descriptive mean-expression difference; not a DE test",
            })
    return pd.DataFrame(rows)


def _sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, max_n, replace=False))


def add_state_labels(ax, emb: np.ndarray, states: np.ndarray, colors: Mapping[str, str]):
    texts = []
    for state in pd.unique(states):
        mask = states == state
        if mask.sum() == 0:
            continue
        center = np.nanmedian(emb[mask], axis=0)
        text = ax.text(
            center[0], center[1], str(state),
            ha="center", va="center", fontsize=9.0, fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=colors.get(str(state), "#999999"), linewidth=0.8, alpha=0.88),
            zorder=20,
        )
        texts.append(text)
    if adjust_text is not None and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.45, alpha=0.7),
            expand=(1.08, 1.15), force_text=(0.25, 0.35), max_move=20,
            ensure_inside_axes=True, time_lim=1.0,
        )


def plot_candidate(
    emb: np.ndarray,
    states: np.ndarray,
    days: np.ndarray,
    colors: Mapping[str, str],
    state_order: list[str],
    outbase: Path,
    title: str,
    subtitle: str,
    cfg: Mapping[str, Any],
    composition: Optional[pd.DataFrame] = None,
):
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    if composition is None:
        fig, ax = plt.subplots(figsize=(float(cfg.get("figure_width_inches", 8.4)), float(cfg.get("figure_height_inches", 6.2))))
        axes = [ax]
    else:
        fig = plt.figure(figsize=(float(cfg.get("figure_width_inches", 8.4)), float(cfg.get("figure_height_inches", 6.2))))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[5.0, 1.0], hspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        ax_comp = fig.add_subplot(gs[1, 0])
        axes = [ax, ax_comp]

    max_points = int(cfg.get("max_points_display", 120000))
    idx = _sample_indices(len(states), max_points, int(cfg.get("random_seed", 20260727)))
    size = float(cfg.get("point_size_all_cells", 1.8))
    alpha = float(cfg.get("point_alpha_all_cells", 0.55))
    rasterized = bool(cfg.get("rasterize_points", True))

    # Draw state groups in stable order.
    used_order = [s for s in state_order if s in set(states)] + [s for s in pd.unique(states) if s not in state_order]
    for state in used_order:
        mask = states[idx] == state
        ax.scatter(
            emb[idx][mask, 0], emb[idx][mask, 1], s=size,
            c=colors.get(state, "#BDBDBD"), alpha=alpha,
            linewidths=0, rasterized=rasterized, label=state,
        )

    add_state_labels(ax, emb, states, colors)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=14.5, fontweight="bold", pad=12)
    ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, color="#444444")
    ax.legend(
        frameon=False, fontsize=8.0, ncol=2, loc="upper right",
        markerscale=3.0, handletextpad=0.25, columnspacing=0.8,
    )

    if composition is not None:
        comp = composition.copy()
        comp = comp.groupby(["observed_day", "assigned_state"], as_index=False)["n_cells"].sum()
        totals = comp.groupby("observed_day")["n_cells"].transform("sum")
        comp["fraction"] = comp["n_cells"] / totals
        pivot = comp.pivot(index="observed_day", columns="assigned_state", values="fraction").fillna(0)
        pivot = pivot.reindex(index=DAY_ORDER, fill_value=0)
        pivot = pivot.reindex(columns=used_order, fill_value=0)
        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for state in used_order:
            vals = pivot[state].to_numpy()
            ax_comp.bar(x, vals, bottom=bottom, color=colors.get(state, "#BDBDBD"), width=0.72, linewidth=0)
            bottom += vals
        ax_comp.set_xticks(x, pivot.index, fontsize=8.5)
        ax_comp.set_ylim(0, 1)
        ax_comp.set_yticks([0, 0.5, 1], ["0", "50", "100"])
        ax_comp.set_ylabel("Cells (%)", fontsize=8.5)
        ax_comp.set_title("Cross-sectional state composition by observed collection day", loc="left", fontsize=9.0, pad=2)
        ax_comp.spines["top"].set_visible(False)
        ax_comp.spines["right"].set_visible(False)
        ax_comp.tick_params(labelsize=8)

    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(outbase.with_suffix("." + ext), dpi=350 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_legacy_author_atlas(config_path: str | Path) -> Path:
    cfg, _ = load_config(config_path)
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    qpath = Path(cfg["query_h5ad"])
    if not qpath.exists():
        raise FileNotFoundError(f"Query H5AD does not exist: {qpath}")
    query = sc.read_h5ad(qpath)

    required_obs = [cfg["query_day_col"], cfg["query_replicate_col"], cfg["query_cluster_col"]]
    missing_obs = [c for c in required_obs if c not in query.obs.columns]
    if missing_obs:
        raise KeyError(f"Missing required query.obs columns: {missing_obs}; available={list(query.obs.columns)}")
    umap_key = cfg.get("query_umap_key", "X_umap")
    if umap_key not in query.obsm:
        raise KeyError(f"{umap_key!r} absent from query.obsm; available={list(query.obsm.keys())}")

    cluster_map_df = read_cluster_map(cfg["legacy_cluster_state_map_tsv"])
    cluster_map = dict(zip(cluster_map_df["source_cluster"].astype(str), cluster_map_df["assigned_state"].astype(str)))
    clusters = query.obs[cfg["query_cluster_col"]].astype(str).to_numpy()
    missing_clusters = sorted(set(clusters) - set(cluster_map))
    if missing_clusters:
        raise ValueError(f"Legacy state map is missing query clusters: {missing_clusters}")
    states = np.asarray([cluster_map[c] for c in clusters], dtype=object)

    days = query.obs[cfg["query_day_col"]].map(normalize_day).to_numpy()
    reps = query.obs[cfg["query_replicate_col"]].astype(str).to_numpy()
    emb = np.asarray(query.obsm[umap_key], dtype=float)[:, :2]
    pure_control, control_definition, _ = select_pure_controls(query, cfg, outdir)

    # Export per-cell assignments, including one-hot probabilities for downstream sensitivity code.
    assignments = pd.DataFrame({
        "cell_id": query.obs_names.astype(str),
        "observed_day": days,
        "biological_replicate": reps,
        "is_pure_control": pure_control,
        "source_cluster": clusters,
        "assigned_state": states,
        "assignment_confidence": 1.0,
        "assignment_entropy": 0.0,
        "atlas_embedding_1": emb[:, 0],
        "atlas_embedding_2": emb[:, 1],
        "state_definition_source": "legacy author-defined leiden_res_0.5 clustering learned from all CC-Perturb-seq cells",
        "state_mapping_source": "ipsc-ec.Hippo.figure.ipynb cluster_mapping",
    })
    state_order = list(cfg.get("state_order", []))
    for state in state_order:
        assignments[f"probability__{state}"] = (states == state).astype(float)
    assignments.to_csv(outdir / "control_state_assignments.tsv", sep="\t", index=False)

    # Review/audit table: no manual review is required for this legacy-label workflow.
    review_rows = []
    for cluster in sorted(pd.unique(clusters), key=lambda x: (len(str(x)), str(x))):
        mask = clusters == cluster
        day_comp = pd.Series(days[mask]).value_counts(normalize=True).to_dict()
        rep_comp = pd.Series(reps[mask]).value_counts(normalize=True).to_dict()
        review_rows.append({
            "source_cluster": cluster,
            "approved_state": cluster_map[str(cluster)],
            "n_cells": int(mask.sum()),
            "day_composition_json": json.dumps({str(k): float(v) for k, v in day_comp.items()}),
            "replicate_composition_json": json.dumps({str(k): float(v) for k, v in rep_comp.items()}),
            "annotation_confidence": "legacy author-approved",
            "annotation_source": "ipsc-ec.Hippo.figure.ipynb",
            "review_required": False,
            "notes": "Previous author-defined mapping from leiden_res_0.5; original boundaries were learned from all cells.",
        })
    pd.DataFrame(review_rows).to_csv(outdir / "control_state_annotation_review.tsv", sep="\t", index=False)
    cluster_map_df.to_csv(outdir / "legacy_cluster_state_map_used.tsv", sep="\t", index=False)

    # Counts by day and replicate for all cells and pure controls.
    count_input = []
    for population, mask in [
        ("all_CC-Perturb-seq_cells", np.ones(query.n_obs, dtype=bool)),
        ("pure_controls", pure_control),
    ]:
        d = assignments.loc[mask, ["observed_day", "biological_replicate", "assigned_state"]].copy()
        d["population"] = population
        count_input.append(d)
    counts = state_counts(pd.concat(count_input, ignore_index=True))
    counts.to_csv(outdir / "state_counts_by_day_and_replicate.tsv", sep="\t", index=False)

    # Colors.
    state_colors = read_color_map(cfg["state_color_map_tsv"], "state")
    day_colors = read_color_map(cfg["day_color_map_tsv"], "day")
    pd.DataFrame({
        "state": state_order,
        "color": [state_colors.get(s, "#BDBDBD") for s in state_order],
        "source": "package state_color_map.tsv",
    }).to_csv(outdir / "state_color_map.tsv", sep="\t", index=False)

    # Descriptive markers (optional, not used to define labels).
    if bool(cfg.get("compute_top_markers", True)):
        markers = descriptive_top_markers(
            query, states, cfg, int(cfg.get("top_markers_per_state", 25))
        )
    else:
        markers = pd.DataFrame(columns=["state", "rank", "gene"])
    markers.to_csv(outdir / "state_marker_summary.tsv", sep="\t", index=False)

    # Figures.
    candA_mask = pure_control
    candA_counts = counts[counts["population"] == "pure_controls"]
    plot_candidate(
        emb[candA_mask], states[candA_mask], days[candA_mask], state_colors,
        state_order, outdir / "Fig2_control_defined_state_atlas_candidateA",
        "Pure control cells in the author-defined coarse-state atlas",
        f"Legacy author-defined Leiden-res-0.5 labels; n={int(candA_mask.sum()):,} pure control cells.",
        cfg, composition=candA_counts,
    )
    all_counts = counts[counts["population"] == "all_CC-Perturb-seq_cells"]
    plot_candidate(
        emb, states, days, state_colors, state_order,
        outdir / "Fig2_control_defined_state_atlas_candidateB",
        "Author-defined coarse states resolve the endothelial differentiation landscape",
        f"Fixed legacy Leiden-res-0.5 labels applied to all CC-Perturb-seq cells; n={query.n_obs:,} cells.",
        cfg, composition=all_counts,
    )

    selected = str(cfg.get("selected_main_candidate", "all_cells"))
    source_base = outdir / (
        "Fig2_control_defined_state_atlas_candidateA"
        if selected == "pure_controls" else "Fig2_control_defined_state_atlas_candidateB"
    )
    for ext in ["png", "pdf", "svg"]:
        shutil.copyfile(
            source_base.with_suffix("." + ext),
            outdir / ("Fig2_control_defined_state_atlas." + ext),
        )

    # QC and notes.
    qc_rows = []
    for population, mask in [
        ("all_CC-Perturb-seq_cells", np.ones(query.n_obs, dtype=bool)),
        ("pure_controls", pure_control),
    ]:
        qc_rows.append({
            "population": population,
            "n_cells": int(mask.sum()),
            "n_states": int(pd.Series(states[mask]).nunique()),
            "state_definition_source": "legacy leiden_res_0.5 all-cell clustering",
            "state_mapping_source": "ipsc-ec.Hippo.figure.ipynb",
            "review_bypassed_by_user_request": True,
        })
    pd.DataFrame(qc_rows).to_csv(outdir / "state_atlas_qc.tsv", sep="\t", index=False)

    legend = (
        "**Figure 2. Author-defined coarse states resolve the endothelial differentiation landscape.** "
        "Cells are displayed in the existing CC-Perturb-seq UMAP and colored using the previously author-defined "
        "Leiden-resolution-0.5 cluster labels (artEC, iPSC, Neuro-iPSC, PrimStreak, Lat-Meso, Fibro-Mesen, "
        "Pre-EC, Neu-Mesen and Others). The lower strip summarizes cross-sectional state composition by observed "
        "collection day. Pure controls are defined as cells carrying at least one safe/non-targeting guide and no "
        "real-targeting guides. The display is cross-sectional and does not represent lineage tracing."
    )
    (outdir / "draft_panel_legend.md").write_text(legend + "\n")
    methods = (
        "# Methods note\n\n"
        "The state labels were taken directly from the previous author-defined mapping of `leiden_res_0.5` in "
        "`ipsc-ec.Hippo.figure.ipynb`: 0 artEC, 1 iPSC, 2 Neuro-iPSC, 3 PrimStreak, 4 Lat-Meso, "
        "5 Fibro-Mesen, 6 Pre-EC, 7 Neu-Mesen and 8 Others. No target-to-program effects were used and no "
        "reclustering was performed in this workflow. The existing UMAP coordinates were used unchanged. "
        "However, the original Leiden boundaries were learned from the full CC-Perturb-seq dataset rather than "
        "from pure controls alone; this differs from the stricter control-defined-atlas design. Pure controls were "
        f"identified for descriptive subset plots using: {control_definition}\n"
    )
    (outdir / "methods_note.md").write_text(methods)
    (outdir / "software_versions.json").write_text(json.dumps({
        "python": sys.version,
        "platform": platform.platform(),
        "scanpy": sc.__version__,
        "anndata": ad.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }, indent=2))
    return outdir
