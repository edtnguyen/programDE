#!/usr/bin/env python3
"""Ota-inspired rare-regulator → program → PAH analysis.

This workflow distinguishes three evidence types:

1. gamma_log_or:
   A quantitative, directional rare-variant burden effect. This is the primary
   input for the Ota-style model.

2. association_z_proxy:
   A positive association-strength proxy reconstructed from a published
   two-sided gene-level p-value. It is not an effect size and is used only in
   explicitly labelled exploratory analyses.

3. descriptive carrier counts:
   Curation evidence only; never treated as gamma.

The script performs:
- curation/review workbook generation;
- IP_GFP beta matrix extraction;
- univariate program–burden regressions with gene-label permutations;
- nested leave-one-gene-out ridge modelling with program selection inside
  every training fold;
- full-procedure gamma-label permutations;
- integration with common-variant program burden and human PAEC validation;
- network-ready tables, GraphML, PDF/PNG, and HTML graph.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


DEFAULT_RUN_DIR = Path(
    "/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/"
    "analysis/250507_ePAH1_NovaSeq/CRISPR_count/"
    "IP_GV_unified_sceptre_cnmf_no_celltype/PerturbNMF/results/"
    "IP_GV_skcnmf_K30_dt2_0_n100"
)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_gene(value: object) -> str:
    return str(value).strip().upper()


def normalize_program(value: object) -> str:
    return re.sub(r"\.0$", "", str(value).strip())


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", **kwargs)
    if frame.empty:
        raise ValueError(f"Empty table: {path}")
    return frame


def derive_association_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    published_p = pd.to_numeric(
        output["published_association_p"],
        errors="coerce",
    )
    output["association_z_proxy"] = np.nan
    valid = published_p.notna() & published_p.gt(0) & published_p.le(1)
    output.loc[valid, "association_z_proxy"] = norm.isf(
        published_p.loc[valid].to_numpy() / 2.0
    )
    output["association_z_proxy_direction"] = np.where(
        valid,
        "positive_risk_enrichment",
        "",
    )
    return output


def load_ip_crt(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    required = {"target_name", "program_name", "beta_clr"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"CRT lacks columns: {sorted(missing)}")

    frame["gene"] = frame["target_name"].map(normalize_gene)
    frame["program"] = frame["program_name"].map(normalize_program)
    frame["beta_clr"] = pd.to_numeric(frame["beta_clr"], errors="coerce")
    for column in ["p_skew", "primary_q_BH", "p_empirical"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def prepare_review(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    curation = derive_association_proxy(read_table(args.curation))
    curation["gene"] = curation["gene"].map(normalize_gene)
    curation["primary_model_eligible"] = (
        curation["primary_model_eligible"].map(parse_bool)
    )
    curation["sensitivity_only"] = (
        curation["sensitivity_only"].map(parse_bool)
    )
    curation["quantitative_effect_selected"] = (
        curation["quantitative_effect_selected"].map(parse_bool)
    )

    crt = load_ip_crt(args.crt_ip)
    programs = sorted(
        crt["program"].dropna().unique(),
        key=lambda value: int(value),
    )
    genes_in_crt = set(crt["gene"])

    curation["present_in_IP_CRT"] = curation["gene"].isin(genes_in_crt)
    curation["n_programs_in_IP_CRT"] = curation["gene"].map(
        crt.groupby("gene")["program"].nunique()
    ).fillna(0).astype(int)

    beta_matrix = (
        crt.pivot_table(
            index="gene",
            columns="program",
            values="beta_clr",
            aggfunc="first",
        )
        .reindex(columns=programs)
        .sort_index()
    )
    beta_matrix.to_csv(
        output_dir / "IP_GFP_target_by_program_beta_clr.tsv",
        sep="\t",
    )

    portal = read_table(args.portal_template)
    portal["gene"] = portal["gene"].map(normalize_gene)

    workbook = output_dir / "rare_variant_curation_review.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        curation.to_excel(writer, sheet_name="Gene_curation", index=False)
        portal.to_excel(writer, sheet_name="Portal_results", index=False)
        beta_matrix.reset_index().to_excel(
            writer,
            sheet_name="IP_beta_matrix",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "item": "Primary quantitative input",
                    "details": (
                        "gamma_log_or and gamma_se from a directional "
                        "gene-burden result."
                    ),
                },
                {
                    "item": "Exploratory proxy",
                    "details": (
                        "association_z_proxy is reconstructed from published "
                        "p-values and is not an effect size."
                    ),
                },
                {
                    "item": "Primary exclusions",
                    "details": (
                        "EIF2AK4 is recessive/PVOD-PCH; SMAD4 is disputed; "
                        "NOTCH1, TP53 and CDKN1A are controls."
                    ),
                },
                {
                    "item": "Minimum recommended genes",
                    "details": (
                        "At least six quantitative gamma estimates; eight or "
                        "more is preferable for the joint model."
                    ),
                },
            ]
        ).to_excel(writer, sheet_name="Instructions", index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = cell.font.copy(bold=True)
            for column in sheet.columns:
                width = 10
                for cell in list(column)[:1000]:
                    if cell.value is not None:
                        width = max(width, min(len(str(cell.value)) + 2, 55))
                sheet.column_dimensions[column[0].column_letter].width = width

    reviewed = output_dir / "rare_variant_effects_reviewed.tsv"
    curation.to_csv(reviewed, sep="\t", index=False)
    portal.to_csv(output_dir / "portal_results_long.tsv", sep="\t", index=False)

    status = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_curated_genes": int(len(curation)),
        "n_primary_eligible": int(curation["primary_model_eligible"].sum()),
        "n_quantitative_ready": int(
            (
                curation["quantitative_effect_selected"]
                & pd.to_numeric(curation["gamma_log_or"], errors="coerce").notna()
            ).sum()
        ),
        "programs": programs,
        "workbook": str(workbook),
        "reviewed_table": str(reviewed),
    }
    (output_dir / "CURATION_REVIEW_COMPLETE.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    print(json.dumps(status, indent=2))


def import_portal(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    curation = read_table(args.curation)
    portal = read_table(input_path)

    portal.columns = [str(column).strip() for column in portal.columns]
    lower = {column.lower(): column for column in portal.columns}

    def find(candidates, required=True):
        for candidate in candidates:
            if candidate.lower() in lower:
                return lower[candidate.lower()]
        if required:
            raise KeyError(
                f"None of {candidates} occurs in {portal.columns.tolist()}."
            )
        return None

    gene_col = find(["gene", "gene_symbol", "symbol", "gene_name"])
    beta_col = find(["beta", "effect", "log_or", "log_odds", "estimate"])
    se_col = find(["se", "standard_error", "sebeta", "std_error"])
    p_col = find(["p_value", "pval", "p", "p_value_burden"], required=False)
    endpoint_col = find(
        ["phenotype_id", "endpoint", "phenocode", "trait", "phenotype"],
        required=False,
    )
    mask_col = find(
        ["variant_mask", "mask", "annotation", "burden_set"],
        required=False,
    )

    if args.endpoint and endpoint_col is not None:
        portal = portal.loc[
            portal[endpoint_col].astype(str).str.contains(
                args.endpoint,
                case=False,
                regex=False,
                na=False,
            )
        ].copy()
    if args.mask and mask_col is not None:
        portal = portal.loc[
            portal[mask_col].astype(str).str.contains(
                args.mask,
                case=False,
                regex=False,
                na=False,
            )
        ].copy()

    normalized = pd.DataFrame(
        {
            "gene": portal[gene_col].map(normalize_gene),
            "source": args.source,
            "release": args.release,
            "phenotype_id": (
                portal[endpoint_col].astype(str)
                if endpoint_col is not None
                else args.endpoint
            ),
            "phenotype_description": args.phenotype_description,
            "variant_mask": (
                portal[mask_col].astype(str)
                if mask_col is not None
                else args.mask
            ),
            "test": args.test,
            "beta": pd.to_numeric(portal[beta_col], errors="coerce"),
            "se": pd.to_numeric(portal[se_col], errors="coerce"),
            "p_value": (
                pd.to_numeric(portal[p_col], errors="coerce")
                if p_col is not None
                else np.nan
            ),
            "case_count": np.nan,
            "control_count": np.nan,
            "effect_allele_or_burden_definition": args.definition,
            "independent_source": args.independent_source,
            "selected_for_gene": False,
            "source_url": args.source_url,
            "notes": args.notes,
        }
    )
    normalized = normalized.loc[
        normalized["gene"].isin(curation["gene"].map(normalize_gene))
    ].dropna(subset=["beta", "se"])
    normalized = normalized.loc[normalized["se"] > 0]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, sep="\t", index=False)
    print(normalized.to_string(index=False))
    print(f"\nWrote: {output}")


def apply_selected_portal(args: argparse.Namespace) -> None:
    curation = read_table(args.curation)
    curation["gene"] = curation["gene"].map(normalize_gene)
    portal = read_table(args.portal_results)
    portal["gene"] = portal["gene"].map(normalize_gene)
    portal["selected_for_gene"] = portal["selected_for_gene"].map(parse_bool)
    selected = portal.loc[portal["selected_for_gene"]].copy()

    duplicates = selected["gene"].duplicated(keep=False)
    if duplicates.any():
        duplicated_genes = sorted(selected.loc[duplicates, "gene"].unique())
        raise ValueError(
            "Select at most one portal row per gene unless you first "
            f"meta-analyze independent cohorts: {duplicated_genes}"
        )

    selected = selected.set_index("gene")
    curation = curation.set_index("gene")

    for gene, row in selected.iterrows():
        if gene not in curation.index:
            continue
        curation.loc[gene, "gamma_log_or"] = row["beta"]
        curation.loc[gene, "gamma_se"] = row["se"]
        curation.loc[gene, "gamma_p"] = row.get("p_value", np.nan)
        curation.loc[gene, "gamma_source"] = row.get("source", "")
        curation.loc[gene, "gamma_phenotype"] = row.get(
            "phenotype_description",
            row.get("phenotype_id", ""),
        )
        curation.loc[gene, "gamma_mask"] = row.get("variant_mask", "")
        curation.loc[gene, "gamma_case_count"] = row.get("case_count", "")
        curation.loc[gene, "gamma_control_count"] = row.get(
            "control_count",
            "",
        )
        curation.loc[gene, "quantitative_effect_selected"] = True

    curation = curation.reset_index()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    curation.to_csv(output, sep="\t", index=False)
    print(f"Wrote: {output}")


def weighted_linear_fit(x, y, weights=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    weights = np.asarray(weights, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    sqrt_w = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(
        X * sqrt_w[:, None],
        y * sqrt_w,
        rcond=None,
    )
    fitted = X @ coef
    residual = y - fitted
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "fitted": fitted,
        "residual": residual,
    }


def univariate_program_tests(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for program in X.columns:
        values = X[program].to_numpy(dtype=float)
        observed = weighted_linear_fit(values, y, weights)
        slope = observed["slope"]
        permuted = np.empty(n_permutations, dtype=float)
        for index in range(n_permutations):
            permuted[index] = weighted_linear_fit(
                values,
                rng.permutation(y),
                weights,
            )["slope"]
        p_value = float(
            (1 + np.sum(np.abs(permuted) >= abs(slope)))
            / (n_permutations + 1)
        )
        rho = spearmanr(values, y).statistic if len(y) >= 3 else np.nan
        rows.append(
            {
                "program": str(program),
                "slope": slope,
                "intercept": observed["intercept"],
                "spearman_rho": float(rho),
                "permutation_p": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["BH_FDR"] = multipletests(
        result["permutation_p"],
        method="fdr_bh",
    )[1]
    return result.sort_values("permutation_p")


def weighted_rank_programs(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    scores = []
    for column in range(X.shape[1]):
        slope = weighted_linear_fit(
            X[:, column],
            y,
            weights,
        )["slope"]
        scores.append(abs(slope))
    return np.argsort(-np.asarray(scores))


def fit_ridge_selected(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    n_programs: int,
):
    order = weighted_rank_programs(X_train, y_train, w_train)
    selected = order[: min(n_programs, X_train.shape[1])]

    scaler = StandardScaler()
    train_selected = scaler.fit_transform(X_train[:, selected])
    test_selected = scaler.transform(X_test[:, selected])

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(
        train_selected,
        y_train,
        sample_weight=w_train,
    )
    prediction = model.predict(test_selected)
    return prediction, selected, scaler, model


def choose_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alphas: list[float],
    n_program_grid: list[int],
):
    best = None
    for n_programs in n_program_grid:
        for alpha in alphas:
            predictions = np.empty(len(y), dtype=float)
            for held_out in range(len(y)):
                train = np.arange(len(y)) != held_out
                prediction, *_ = fit_ridge_selected(
                    X[train],
                    y[train],
                    weights[train],
                    X[~train],
                    alpha,
                    n_programs,
                )
                predictions[held_out] = prediction[0]
            mse = np.average((y - predictions) ** 2, weights=weights)
            candidate = {
                "mse": float(mse),
                "alpha": float(alpha),
                "n_programs": int(n_programs),
            }
            if best is None or candidate["mse"] < best["mse"]:
                best = candidate
    assert best is not None
    return best


def nested_loo_predictions(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alphas: list[float],
    n_program_grid: list[int],
):
    predictions = np.empty(len(y), dtype=float)
    selected_details = []
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        best = choose_hyperparameters(
            X[train],
            y[train],
            weights[train],
            alphas,
            n_program_grid,
        )
        prediction, selected, *_ = fit_ridge_selected(
            X[train],
            y[train],
            weights[train],
            X[~train],
            best["alpha"],
            best["n_programs"],
        )
        predictions[held_out] = prediction[0]
        selected_details.append(
            {
                "held_out_index": held_out,
                **best,
                "selected_indices": selected.tolist(),
            }
        )
    return predictions, selected_details


def prediction_metrics(y, predictions):
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    pearson = (
        pearsonr(y, predictions).statistic
        if len(y) >= 3 and np.std(y) > 0 and np.std(predictions) > 0
        else np.nan
    )
    spearman = (
        spearmanr(y, predictions).statistic
        if len(y) >= 3
        else np.nan
    )
    rmse = math.sqrt(mean_squared_error(y, predictions))
    if len(np.unique(np.sign(y))) >= 2:
        sign_accuracy = float(np.mean(np.sign(y) == np.sign(predictions)))
    else:
        sign_accuracy = np.nan
    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "rmse": float(rmse),
        "sign_accuracy": sign_accuracy,
    }


def permutation_worker(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alphas: list[float],
    n_program_grid: list[int],
):
    rng = np.random.default_rng(seed)
    permuted_y = rng.permutation(y)
    predictions, _ = nested_loo_predictions(
        X,
        permuted_y,
        weights,
        alphas,
        n_program_grid,
    )
    return prediction_metrics(permuted_y, predictions)


def detect_common_genetics(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    frame = read_table(path)
    if "program" not in frame:
        return None
    frame["program"] = frame["program"].map(normalize_program)
    p_candidates = [
        "MAGMA_expr_conditioned_p",
        "matched_permutation_p",
        "MAGMA_linear_p",
        "MAGMA_top50_p",
        "MAGMA_top300_p",
    ]
    present = [column for column in p_candidates if column in frame]
    if present:
        frame["common_genetic_min_p"] = frame[present].apply(
            pd.to_numeric,
            errors="coerce",
        ).min(axis=1)
    return frame


def detect_human_validation(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    frame = read_table(path)
    if "program" not in frame:
        return None
    frame["program"] = frame["program"].map(normalize_program)
    return frame


def make_graph(
    output_dir: Path,
    effects: pd.DataFrame,
    beta_long: pd.DataFrame,
    univariate: pd.DataFrame,
    joint_coefficients: pd.DataFrame,
    common: pd.DataFrame | None,
    human: pd.DataFrame | None,
    q_threshold: float,
):
    graph = nx.DiGraph()
    graph.add_node("PAH", node_type="trait", label="PAH")
    graph.add_node(
        "PAH_common_variant_risk",
        node_type="genetic_axis",
        label="PAH common-variant risk",
    )
    graph.add_edge(
        "PAH_common_variant_risk",
        "PAH",
        edge_type="risk_axis",
    )

    common_lookup = (
        common.set_index("program")
        if common is not None
        else pd.DataFrame()
    )
    human_lookup = (
        human.set_index("program")
        if human is not None
        else pd.DataFrame()
    )

    selected_programs = set(
        joint_coefficients.loc[
            joint_coefficients["coefficient"].abs() > 1e-12,
            "program",
        ].astype(str)
    )
    selected_programs.update(
        univariate.loc[univariate["BH_FDR"] < q_threshold, "program"].astype(str)
    )

    for program in sorted(selected_programs, key=int):
        attributes = {
            "node_type": "program",
            "label": f"P{program}",
        }
        if not common_lookup.empty and program in common_lookup.index:
            row = common_lookup.loc[program]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for column in [
                "common_genetic_min_p",
                "MAGMA_expr_conditioned_p",
                "matched_permutation_p",
            ]:
                if column in row:
                    attributes[column] = float(row[column])
        if not human_lookup.empty and program in human_lookup.index:
            row = human_lookup.loc[program]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for column in ["NES", "GSEA_FDR", "hedges_g"]:
                if column in row and pd.notna(row[column]):
                    attributes[f"human_{column}"] = float(row[column])
        graph.add_node(f"P{program}", **attributes)

    for row in effects.itertuples(index=False):
        graph.add_node(
            row.gene,
            node_type="rare_gene",
            label=row.gene,
            gamma=float(row.gamma),
            gamma_se=float(row.gamma_se) if np.isfinite(row.gamma_se) else np.nan,
            source=str(row.gamma_source),
        )
        graph.add_edge(
            row.gene,
            "PAH",
            edge_type="observed_rare_variant_effect",
            effect=float(row.gamma),
        )

    # Keep strong gene→program edges for readability.
    beta = beta_long.loc[
        beta_long["program"].astype(str).isin(selected_programs)
        & beta_long["gene"].isin(effects["gene"])
    ].copy()
    beta["abs_beta"] = beta["beta_clr"].abs()
    beta = (
        beta.sort_values(["gene", "abs_beta"], ascending=[True, False])
        .groupby("gene", observed=True)
        .head(4)
    )
    for row in beta.itertuples(index=False):
        graph.add_edge(
            row.gene,
            f"P{row.program}",
            edge_type="perturbation_regulation",
            beta_clr=float(row.beta_clr),
            p_skew=float(row.p_skew) if np.isfinite(row.p_skew) else np.nan,
        )

    for row in joint_coefficients.itertuples(index=False):
        if str(row.program) not in selected_programs:
            continue
        graph.add_edge(
            f"P{row.program}",
            "PAH",
            edge_type="joint_program_to_trait",
            coefficient=float(row.coefficient),
        )
        if (
            common is not None
            and not common_lookup.empty
            and str(row.program) in common_lookup.index
        ):
            common_row = common_lookup.loc[str(row.program)]
            if isinstance(common_row, pd.DataFrame):
                common_row = common_row.iloc[0]
            p_value = common_row.get("common_genetic_min_p", np.nan)
            graph.add_edge(
                f"P{row.program}",
                "PAH_common_variant_risk",
                edge_type="common_variant_program_support",
                p_value=float(p_value) if pd.notna(p_value) else np.nan,
            )

    nx.write_graphml(
        graph,
        output_dir / "rare_regulator_program_PAH.graphml",
    )

    nodes = pd.DataFrame(
        [
            {"node": node, **attributes}
            for node, attributes in graph.nodes(data=True)
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": source, "target": target, **attributes}
            for source, target, attributes in graph.edges(data=True)
        ]
    )
    nodes.to_csv(output_dir / "network_nodes.tsv", sep="\t", index=False)
    edges.to_csv(output_dir / "network_edges.tsv", sep="\t", index=False)

    plt.figure(figsize=(14, 10))
    position = nx.spring_layout(graph, seed=14, k=1.2)
    node_colors = []
    for _, attributes in graph.nodes(data=True):
        node_colors.append(
            {
                "rare_gene": "#4C78A8",
                "program": "#F58518",
                "trait": "#E45756",
                "genetic_axis": "#72B7B2",
            }.get(attributes.get("node_type"), "#999999")
        )
    edge_colors = [
        "#555555" if data.get("edge_type") == "perturbation_regulation"
        else "#AA3377"
        for _, _, data in graph.edges(data=True)
    ]
    nx.draw_networkx(
        graph,
        position,
        labels={
            node: attributes.get("label", node)
            for node, attributes in graph.nodes(data=True)
        },
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=1400,
        font_size=8,
        arrows=True,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "rare_regulator_program_PAH.pdf")
    plt.savefig(output_dir / "rare_regulator_program_PAH.png", dpi=220)
    plt.close()

    network = Network(height="850px", width="100%", directed=True)
    for node, attributes in graph.nodes(data=True):
        network.add_node(
            node,
            label=attributes.get("label", node),
            title=json.dumps(attributes, default=str, indent=2),
            color={
                "rare_gene": "#4C78A8",
                "program": "#F58518",
                "trait": "#E45756",
                "genetic_axis": "#72B7B2",
            }.get(attributes.get("node_type"), "#999999"),
        )
    for source, target, attributes in graph.edges(data=True):
        network.add_edge(
            source,
            target,
            title=json.dumps(attributes, default=str, indent=2),
            value=max(abs(float(attributes.get("beta_clr", attributes.get("coefficient", 1.0)))), 0.2),
        )
    network.write_html(
        str(output_dir / "rare_regulator_program_PAH.html"),
        notebook=False,
        open_browser=False,
    )


def fit_model(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    curation = derive_association_proxy(read_table(args.curation))
    curation["gene"] = curation["gene"].map(normalize_gene)
    curation["primary_model_eligible"] = (
        curation["primary_model_eligible"].map(parse_bool)
    )
    curation["sensitivity_only"] = (
        curation["sensitivity_only"].map(parse_bool)
    )
    curation["quantitative_effect_selected"] = (
        curation["quantitative_effect_selected"].map(parse_bool)
    )

    if args.effect_mode == "gamma":
        curation["gamma"] = pd.to_numeric(
            curation["gamma_log_or"],
            errors="coerce",
        )
        curation["gamma_se_numeric"] = pd.to_numeric(
            curation["gamma_se"],
            errors="coerce",
        )
        selected = curation.loc[
            curation["primary_model_eligible"]
            & curation["quantitative_effect_selected"]
            & curation["gamma"].notna()
        ].copy()
        analysis_label = "quantitative_directional_gamma"
    else:
        if not args.allow_association_proxy:
            raise ValueError(
                "association_z_proxy mode requires "
                "--allow-association-proxy because it is not an effect-size model."
            )
        curation["gamma"] = pd.to_numeric(
            curation["association_z_proxy"],
            errors="coerce",
        )
        curation["gamma_se_numeric"] = np.nan
        selected = curation.loc[
            curation["primary_model_eligible"]
            & curation["gamma"].notna()
        ].copy()
        analysis_label = "exploratory_association_strength_proxy"

    if len(selected) < args.minimum_genes:
        missing = curation.loc[
            curation["primary_model_eligible"]
            & ~curation["gene"].isin(selected["gene"]),
            ["gene", "curation_status", "gamma_source"],
        ]
        message = (
            f"Only {len(selected)} genes are ready; at least "
            f"{args.minimum_genes} are required.\n"
            "Fill gamma_log_or/gamma_se, set quantitative_effect_selected=TRUE, "
            "then rerun.\n\nMissing primary genes:\n"
            + missing.to_string(index=False)
        )
        (output_dir / "NEEDS_MORE_QUANTITATIVE_EFFECTS.txt").write_text(message)
        raise RuntimeError(message)

    crt = load_ip_crt(args.crt_ip)
    crt = crt.loc[crt["gene"].isin(selected["gene"])].copy()

    programs = sorted(
        crt["program"].unique(),
        key=lambda value: int(value),
    )
    if args.prespecified_programs is not None:
        prespecified = read_table(args.prespecified_programs)
        prespecified = prespecified.loc[
            prespecified["include"].map(parse_bool)
        ]
        requested = set(
            prespecified["program"].map(normalize_program)
        )
        programs = [program for program in programs if program in requested]

    beta_matrix = (
        crt.pivot_table(
            index="gene",
            columns="program",
            values="beta_clr",
            aggfunc="first",
        )
        .reindex(index=selected["gene"], columns=programs)
    )
    if beta_matrix.isna().any(axis=None):
        raise ValueError(
            "The selected gene×program beta matrix contains missing values."
        )

    selected = selected.set_index("gene").loc[beta_matrix.index].reset_index()
    y = selected["gamma"].to_numpy(dtype=float)
    se = selected["gamma_se_numeric"].to_numpy(dtype=float)
    weights = np.where(
        np.isfinite(se) & (se > 0),
        1.0 / se**2,
        1.0,
    )
    X = beta_matrix.to_numpy(dtype=float)

    selected_effects = pd.DataFrame(
        {
            "gene": selected["gene"],
            "gamma": y,
            "gamma_se": se,
            "weight": weights,
            "gamma_source": selected.get("gamma_source", ""),
            "analysis_label": analysis_label,
        }
    )
    selected_effects.to_csv(
        output_dir / "selected_rare_variant_effects.tsv",
        sep="\t",
        index=False,
    )
    beta_matrix.to_csv(
        output_dir / "selected_gene_by_program_beta.tsv",
        sep="\t",
    )

    univariate = univariate_program_tests(
        beta_matrix,
        y,
        weights,
        args.univariate_permutations,
        args.seed,
    )
    univariate.to_csv(
        output_dir / "univariate_program_rare_effect_tests.tsv",
        sep="\t",
        index=False,
    )

    alphas = [float(value) for value in args.alphas.split(",")]
    n_program_grid = [
        value
        for value in range(1, min(args.max_programs, X.shape[1]) + 1)
    ]
    predictions, fold_details = nested_loo_predictions(
        X,
        y,
        weights,
        alphas,
        n_program_grid,
    )
    observed_metrics = prediction_metrics(y, predictions)

    full_best = choose_hyperparameters(
        X,
        y,
        weights,
        alphas,
        n_program_grid,
    )
    _, selected_indices, scaler, model = fit_ridge_selected(
        X,
        y,
        weights,
        X[:1],
        full_best["alpha"],
        full_best["n_programs"],
    )
    coefficients = np.zeros(X.shape[1], dtype=float)
    coefficients[selected_indices] = model.coef_ / scaler.scale_

    coefficient_table = pd.DataFrame(
        {
            "program": programs,
            "coefficient": coefficients,
            "selected_full_model": [
                index in set(selected_indices)
                for index in range(len(programs))
            ],
        }
    ).sort_values(
        "coefficient",
        key=lambda values: values.abs(),
        ascending=False,
    )
    coefficient_table.to_csv(
        output_dir / "joint_program_coefficients.tsv",
        sep="\t",
        index=False,
    )

    loo = selected_effects.copy()
    loo["loo_prediction"] = predictions
    loo["residual"] = loo["gamma"] - loo["loo_prediction"]
    loo.to_csv(
        output_dir / "leave_one_gene_out_predictions.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(fold_details).to_json(
        output_dir / "nested_LOO_fold_details.json",
        orient="records",
        indent=2,
    )

    seeds = [args.seed + 100_000 + index for index in range(args.model_permutations)]
    permuted_metrics = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(permutation_worker)(
            seed,
            X,
            y,
            weights,
            alphas,
            n_program_grid,
        )
        for seed in seeds
    )
    permutation_table = pd.DataFrame(permuted_metrics)
    permutation_table.to_csv(
        output_dir / "joint_model_permutation_null.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    permutation_p = {
        "pearson": float(
            (1 + np.sum(
                permutation_table["pearson"].abs()
                >= abs(observed_metrics["pearson"])
            ))
            / (len(permutation_table) + 1)
        )
        if np.isfinite(observed_metrics["pearson"])
        else np.nan,
        "spearman": float(
            (1 + np.sum(
                permutation_table["spearman"].abs()
                >= abs(observed_metrics["spearman"])
            ))
            / (len(permutation_table) + 1)
        )
        if np.isfinite(observed_metrics["spearman"])
        else np.nan,
        "rmse": float(
            (1 + np.sum(
                permutation_table["rmse"]
                <= observed_metrics["rmse"]
            ))
            / (len(permutation_table) + 1)
        ),
    }

    common = detect_common_genetics(args.common_program_burden)
    human = detect_human_validation(args.human_validation)
    make_graph(
        output_dir,
        selected_effects,
        crt,
        univariate,
        coefficient_table,
        common,
        human,
        args.graph_univariate_fdr,
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_label": analysis_label,
        "n_genes": int(len(selected)),
        "n_programs_available": int(X.shape[1]),
        "full_model_alpha": full_best["alpha"],
        "full_model_n_programs": full_best["n_programs"],
        "observed_LOO_metrics": observed_metrics,
        "permutation_p_values": permutation_p,
        "univariate_permutations": args.univariate_permutations,
        "joint_model_permutations": args.model_permutations,
        "effect_mode": args.effect_mode,
    }
    (output_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with pd.ExcelWriter(
        output_dir / "rare_regulator_program_model_review.xlsx",
        engine="openpyxl",
    ) as writer:
        selected_effects.to_excel(
            writer,
            sheet_name="Rare_effects",
            index=False,
        )
        beta_matrix.reset_index().to_excel(
            writer,
            sheet_name="Beta_matrix",
            index=False,
        )
        univariate.to_excel(
            writer,
            sheet_name="Univariate_programs",
            index=False,
        )
        loo.to_excel(
            writer,
            sheet_name="LOO_predictions",
            index=False,
        )
        coefficient_table.to_excel(
            writer,
            sheet_name="Joint_coefficients",
            index=False,
        )
        pd.DataFrame(
            [
                {"metric": key, "observed": value, "permutation_p": permutation_p.get(key)}
                for key, value in observed_metrics.items()
            ]
        ).to_excel(
            writer,
            sheet_name="Model_metrics",
            index=False,
        )
        if common is not None:
            common.to_excel(
                writer,
                sheet_name="Common_program_risk",
                index=False,
            )
        if human is not None:
            human.to_excel(
                writer,
                sheet_name="Human_PAH_programs",
                index=False,
            )

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = cell.font.copy(bold=True)
            for column in sheet.columns:
                width = 10
                for cell in list(column)[:2000]:
                    if cell.value is not None:
                        width = max(width, min(len(str(cell.value)) + 2, 55))
                sheet.column_dimensions[column[0].column_letter].width = width

    # Diagnostics.
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y, predictions)
    low = min(y.min(), predictions.min())
    high = max(y.max(), predictions.max())
    axis.plot([low, high], [low, high], linewidth=1)
    for gene, observed, predicted in zip(selected["gene"], y, predictions):
        axis.annotate(gene, (observed, predicted), fontsize=8)
    axis.set_xlabel("Observed rare-variant effect")
    axis.set_ylabel("Nested LOO prediction")
    axis.set_title("Rare-regulator joint program model")
    figure.tight_layout()
    figure.savefig(output_dir / "LOO_observed_vs_predicted.pdf")
    figure.savefig(output_dir / "LOO_observed_vs_predicted.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, max(5, 0.3 * len(coefficient_table))))
    plot = coefficient_table.sort_values("coefficient")
    axis.barh("P" + plot["program"].astype(str), plot["coefficient"])
    axis.axvline(0, linewidth=1)
    axis.set_xlabel("Joint model coefficient")
    axis.set_ylabel("Program")
    axis.set_title("Selected program effects on rare-variant PAH burden")
    figure.tight_layout()
    figure.savefig(output_dir / "joint_program_coefficients.pdf")
    figure.savefig(output_dir / "joint_program_coefficients.png", dpi=220)
    plt.close(figure)

    interpretation = f"""Rare-regulator → program → PAH model
====================================

Analysis label
--------------
{analysis_label}

Genes used: {len(selected)}
Programs considered: {X.shape[1]}

Observed nested LOO metrics
---------------------------
Pearson:  {observed_metrics['pearson']}
Spearman: {observed_metrics['spearman']}
RMSE:     {observed_metrics['rmse']}
Sign accuracy: {observed_metrics['sign_accuracy']}

Permutation p values
--------------------
Pearson:  {permutation_p['pearson']}
Spearman: {permutation_p['spearman']}
RMSE:     {permutation_p['rmse']}

Important interpretation
------------------------
A quantitative gamma analysis is the primary Ota-inspired analysis.

The association_z_proxy mode is exploratory only. It converts published
association p-values to positive Z magnitudes and therefore tests whether
program regulation predicts strength of published evidence, not a causal
effect size.

For a binary disease, most genuine damaging-variant effects are expected to
increase risk. Sign prediction can therefore be uninformative; correlation and
out-of-sample prediction of effect magnitude are more useful.

EIF2AK4 is excluded from the primary dominant model because its established
PVOD/PCH mechanism is recessive. SMAD4 is excluded because ClinGen classifies
the PAH relationship as disputed. NOTCH1, TP53, and CDKN1A are controls.

With a small targeted gene panel, all results are hypothesis-generating.
Program selection is repeated inside every leave-one-gene-out training fold,
and the full procedure is repeated after gamma-label permutations.
"""
    (output_dir / "HOW_TO_INTERPRET.txt").write_text(interpretation)

    required = [
        output_dir / "rare_regulator_program_model_review.xlsx",
        output_dir / "leave_one_gene_out_predictions.tsv",
        output_dir / "joint_program_coefficients.tsv",
        output_dir / "rare_regulator_program_PAH.graphml",
        output_dir / "rare_regulator_program_PAH.html",
        output_dir / "model_summary.json",
        output_dir / "HOW_TO_INTERPRET.txt",
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "Missing model outputs:\n" + "\n".join(missing)
        )

    print(json.dumps(summary, indent=2))
    print(f"\nWrote model to: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    review = subparsers.add_parser("prepare-review")
    review.add_argument("--curation", type=Path, required=True)
    review.add_argument("--portal-template", type=Path, required=True)
    review.add_argument("--crt-ip", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    review.set_defaults(func=prepare_review)

    importer = subparsers.add_parser("import-portal")
    importer.add_argument("--input", type=Path, required=True)
    importer.add_argument("--curation", type=Path, required=True)
    importer.add_argument("--source", required=True)
    importer.add_argument("--release", default="")
    importer.add_argument("--endpoint", default="")
    importer.add_argument("--phenotype-description", default="")
    importer.add_argument("--mask", default="")
    importer.add_argument("--test", default="burden")
    importer.add_argument("--definition", default="")
    importer.add_argument("--independent-source", action="store_true")
    importer.add_argument("--source-url", default="")
    importer.add_argument("--notes", default="")
    importer.add_argument("--output", type=Path, required=True)
    importer.set_defaults(func=import_portal)

    apply_portal = subparsers.add_parser("apply-selected-portal")
    apply_portal.add_argument("--curation", type=Path, required=True)
    apply_portal.add_argument("--portal-results", type=Path, required=True)
    apply_portal.add_argument("--output", type=Path, required=True)
    apply_portal.set_defaults(func=apply_selected_portal)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--curation", type=Path, required=True)
    fit.add_argument("--crt-ip", type=Path, required=True)
    fit.add_argument("--common-program-burden", type=Path, default=None)
    fit.add_argument("--human-validation", type=Path, default=None)
    fit.add_argument("--prespecified-programs", type=Path, default=None)
    fit.add_argument(
        "--effect-mode",
        choices=["gamma", "association_z_proxy"],
        default="gamma",
    )
    fit.add_argument("--allow-association-proxy", action="store_true")
    fit.add_argument("--minimum-genes", type=int, default=6)
    fit.add_argument("--max-programs", type=int, default=5)
    fit.add_argument(
        "--alphas",
        default="0.01,0.1,1,10,100",
    )
    fit.add_argument("--univariate-permutations", type=int, default=20000)
    fit.add_argument("--model-permutations", type=int, default=20000)
    fit.add_argument("--n-jobs", type=int, default=8)
    fit.add_argument("--seed", type=int, default=20260623)
    fit.add_argument("--graph-univariate-fdr", type=float, default=0.10)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.set_defaults(func=fit_model)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
