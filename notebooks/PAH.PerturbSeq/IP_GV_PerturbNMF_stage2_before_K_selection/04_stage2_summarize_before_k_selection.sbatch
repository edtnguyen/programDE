#!/usr/bin/env bash
#SBATCH --job-name=IPGV_s2_summary
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_stage2_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
PILOT_ROOT="${BASE}/PerturbNMF/pilot_array_n20"
RUN_DIR="${PILOT_ROOT}/merged"
INFERENCE_DIR="${RUN_DIR}/Inference"
EVALUATION_DIR="${RUN_DIR}/Evaluation"
STAGE2_ROOT="${PILOT_ROOT}/stage2_pre_k_selection"
SUMMARY_DIR="${STAGE2_ROOT}/summary_before_k_selection"

K_VALUES=(5 10 15 20 25 30 40 50)
THRESHOLDS=(0.2 2.0)

for K in "${K_VALUES[@]}"; do
    for T in "${THRESHOLDS[@]}"; do
        TAG="${T//./_}"
        MARKER="${STAGE2_ROOT}/metrics/K${K}_dt${TAG}_COMPLETE.txt"
        [[ -s "${MARKER}" ]] || {
            echo "Metrics marker missing: ${MARKER}" >&2
            exit 1
        }
    done
done

mkdir -p "${SUMMARY_DIR}"

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
export MPLBACKEND=Agg
unset PYTHONSTARTUP || true

echo "============================================================"
echo "Summarize all Stage 2 models before K selection"
echo "Started:       $(date)"
echo "Job ID:        ${SLURM_JOB_ID:-interactive}"
echo "Summary dir:   ${SUMMARY_DIR}"
echo "============================================================"

python - \
    "${RUN_DIR}" \
    "${SUMMARY_DIR}" \
    "${#K_VALUES[@]}" \
    "${K_VALUES[@]}" \
    "${#THRESHOLDS[@]}" \
    "${THRESHOLDS[@]}" <<'PY'
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from cnmf import load_df_from_npz

args = list(sys.argv[1:])
run_dir = Path(args.pop(0))
summary_dir = Path(args.pop(0))

n_k = int(args.pop(0))
k_values = [int(args.pop(0)) for _ in range(n_k)]

n_t = int(args.pop(0))
thresholds = [float(args.pop(0)) for _ in range(n_t)]

if args:
    raise ValueError(f"Unexpected trailing arguments: {args}")

summary_dir.mkdir(parents=True, exist_ok=True)
inference = run_dir / "Inference"
evaluation = run_dir / "Evaluation"

FDR = 0.05
TOP_TERMS = 30
MODEL_ORDER = [
    (k, t)
    for k in k_values
    for t in thresholds
]
MODEL_LABELS = [
    f"K{k}, dt={t:g}"
    for k, t in MODEL_ORDER
]

def threshold_tag(value: float) -> str:
    return str(float(value)).replace(".", "_")

def read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")

def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan

def adjusted_p_column(df: pd.DataFrame) -> str:
    for candidate in [
        "Adjusted P-value",
        "adj_pval",
        "FDR q-val",
        "adjusted_p_value",
    ]:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"No adjusted-p column found. Columns: {df.columns.tolist()}"
    )

# -----------------------
# cNMF K-selection stats
# -----------------------
k_stats_npz = inference / "Inference.k_selection_stats.df.npz"
k_stats = load_df_from_npz(k_stats_npz).copy()
if "K" in k_stats.columns:
    k_stats_table = k_stats.copy()
    k_stats_table["K"] = pd.to_numeric(
        k_stats_table["K"], errors="coerce"
    )
    if k_stats_table["K"].isna().any():
        raise ValueError("K-selection table has non-numeric K values.")
    k_stats_table = k_stats_table.set_index("K", drop=False)
else:
    k_stats_table = k_stats.copy()
    k_stats_table.index = pd.to_numeric(
        pd.Index(k_stats_table.index), errors="coerce"
    )
    if pd.isna(k_stats_table.index).any():
        raise ValueError("K-selection index contains non-numeric values.")
    k_stats_table.index.name = "K"
    k_stats_table.insert(0, "K", k_stats_table.index.astype(float))

k_stats_table.to_csv(
    summary_dir / "k_selection_stats.tsv",
    sep="\t",
    index=False,
)

# -----------------------
# Stage 2 model summaries
# -----------------------
model_rows = []
go_all = []
reactome_all = []
top_gene_rows = []
perturb_replication_rows = []
categorical_all = []

for k, threshold in MODEL_ORDER:
    tag = threshold_tag(threshold)
    folder = evaluation / f"{k}_{tag}"

    go = read_tsv(folder / f"{k}_GO_term_enrichment.txt")
    reactome = read_tsv(folder / f"{k}_geneset_enrichment.txt")
    expvar = read_tsv(folder / f"{k}_Explained_Variance.txt")
    expvar_summary = read_tsv(
        folder / f"{k}_Explained_Variance_Summary.txt"
    )
    categorical = read_tsv(
        folder / f"{k}_categorical_association_results.txt"
    )
    ip = read_tsv(
        folder / f"{k}_perturbation_association_results_IP_GFP.txt"
    )
    gv = read_tsv(
        folder / f"{k}_perturbation_association_results_GV_GFP.txt"
    )

    # Enrichment summaries
    go_p = adjusted_p_column(go)
    react_p = adjusted_p_column(reactome)

    go = go.copy()
    go["K"] = k
    go["threshold"] = threshold
    go["model"] = f"K{k}_dt{tag}"
    go_all.append(go)

    reactome = reactome.copy()
    reactome["K"] = k
    reactome["threshold"] = threshold
    reactome["model"] = f"K{k}_dt{tag}"
    reactome_all.append(reactome)

    go_sig = go.loc[pd.to_numeric(go[go_p], errors="coerce") < FDR]
    react_sig = reactome.loc[
        pd.to_numeric(reactome[react_p], errors="coerce") < FDR
    ]

    go_sig_programs = go_sig["program_name"].astype(str).nunique()
    react_sig_programs = react_sig["program_name"].astype(str).nunique()

    # Mouse/sample association. Apply BH across K programs within each model.
    cat_p_candidates = [
        c for c in categorical.columns
        if c.endswith("kruskall_wallis_pval")
    ]
    if len(cat_p_candidates) != 1:
        raise ValueError(
            f"Expected one categorical p-value column in {folder}; "
            f"found {cat_p_candidates}."
        )
    cat_p_col = cat_p_candidates[0]
    cat_p = pd.to_numeric(categorical[cat_p_col], errors="coerce").fillna(1.0)
    cat_q = multipletests(cat_p.to_numpy(), method="fdr_bh")[1]
    categorical = categorical.copy()
    categorical["sample_BH_FDR"] = cat_q
    categorical["K"] = k
    categorical["threshold"] = threshold
    categorical["model"] = f"K{k}_dt{tag}"
    categorical_all.append(categorical)
    n_mouse_assoc = int(np.sum(cat_q < FDR))

    # Perturbation replication across the two mice.
    key_cols = ["target_name", "program_name"]
    required = set(key_cols + ["log2FC", "adj_pval"])
    for label, df in [("IP_GFP", ip), ("GV_GFP", gv)]:
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(
                f"{label}, K={k}, dt={threshold}: missing {sorted(missing)}"
            )

    merged = ip[key_cols + ["log2FC", "adj_pval"]].merge(
        gv[key_cols + ["log2FC", "adj_pval"]],
        on=key_cols,
        how="inner",
        suffixes=("_IP_GFP", "_GV_GFP"),
    )
    merged["K"] = k
    merged["threshold"] = threshold
    merged["model"] = f"K{k}_dt{tag}"
    merged["same_sign"] = (
        np.sign(merged["log2FC_IP_GFP"])
        == np.sign(merged["log2FC_GV_GFP"])
    )
    merged["significant_both"] = (
        (merged["adj_pval_IP_GFP"] < FDR)
        & (merged["adj_pval_GV_GFP"] < FDR)
    )
    merged["replicated_same_sign"] = (
        merged["significant_both"] & merged["same_sign"]
    )
    perturb_replication_rows.append(merged)

    finite = merged[
        np.isfinite(merged["log2FC_IP_GFP"])
        & np.isfinite(merged["log2FC_GV_GFP"])
    ]
    if len(finite) >= 3:
        perturb_rho = float(
            spearmanr(
                finite["log2FC_IP_GFP"],
                finite["log2FC_GV_GFP"],
            ).statistic
        )
    else:
        perturb_rho = np.nan

    ip_sig_fraction = safe_fraction(
        int(np.sum(pd.to_numeric(ip["adj_pval"], errors="coerce") < FDR)),
        len(ip),
    )
    gv_sig_fraction = safe_fraction(
        int(np.sum(pd.to_numeric(gv["adj_pval"], errors="coerce") < FDR)),
        len(gv),
    )

    # Top genes per program.
    spectra_path = (
        inference
        / f"Inference.gene_spectra_score.k_{k}.dt_{tag}.txt"
    )
    spectra = pd.read_csv(spectra_path, sep="\t", index_col=0)
    for program, row in spectra.iterrows():
        top = row.sort_values(ascending=False).head(20)
        for rank, (gene, score) in enumerate(top.items(), start=1):
            top_gene_rows.append(
                {
                    "K": k,
                    "threshold": threshold,
                    "model": f"K{k}_dt{tag}",
                    "program_name": str(program),
                    "rank": rank,
                    "gene": str(gene),
                    "score": float(score),
                }
            )

    exp_row = expvar_summary.iloc[0].to_dict()
    model_rows.append(
        {
            "K": k,
            "threshold": threshold,
            "model": f"K{k}_dt{tag}",
            "n_programs": k,
            "total_explained_variance": float(exp_row.get("Total", np.nan)),
            "sum_individual_variance": float(exp_row.get("Sum", np.nan)),
            "median_individual_variance": float(
                exp_row.get("Median", np.nan)
            ),
            "max_individual_variance": float(exp_row.get("Max", np.nan)),
            "GO_significant_programs": go_sig_programs,
            "GO_significant_program_fraction": safe_fraction(
                go_sig_programs, k
            ),
            "GO_significant_unique_terms": int(
                go_sig["Term"].astype(str).nunique()
            ),
            "Reactome_significant_programs": react_sig_programs,
            "Reactome_significant_program_fraction": safe_fraction(
                react_sig_programs, k
            ),
            "Reactome_significant_unique_terms": int(
                react_sig["Term"].astype(str).nunique()
            ),
            "mouse_associated_programs_BH_FDR_0.05": n_mouse_assoc,
            "mouse_associated_program_fraction": safe_fraction(
                n_mouse_assoc, k
            ),
            "IP_GFP_significant_target_program_fraction": ip_sig_fraction,
            "GV_GFP_significant_target_program_fraction": gv_sig_fraction,
            "cross_mouse_effect_spearman": perturb_rho,
            "cross_mouse_sign_concordance": float(
                merged["same_sign"].mean()
            ) if len(merged) else np.nan,
            "cross_mouse_replicated_same_sign_pairs": int(
                merged["replicated_same_sign"].sum()
            ),
            "cross_mouse_pairs_tested": int(len(merged)),
        }
    )

model_summary = pd.DataFrame(model_rows).sort_values(
    ["K", "threshold"]
)
model_summary.to_csv(
    summary_dir / "model_selection_metrics.tsv",
    sep="\t",
    index=False,
)

go_all_df = pd.concat(go_all, ignore_index=True)
reactome_all_df = pd.concat(reactome_all, ignore_index=True)
categorical_all_df = pd.concat(categorical_all, ignore_index=True)
perturb_all_df = pd.concat(perturb_replication_rows, ignore_index=True)
top_genes_df = pd.DataFrame(top_gene_rows)

go_all_df.to_csv(
    summary_dir / "GO_enrichment_all_models.tsv",
    sep="\t",
    index=False,
)
reactome_all_df.to_csv(
    summary_dir / "Reactome_enrichment_all_models.tsv",
    sep="\t",
    index=False,
)
categorical_all_df.to_csv(
    summary_dir / "mouse_association_all_models.tsv",
    sep="\t",
    index=False,
)
perturb_all_df.to_csv(
    summary_dir / "perturbation_replication_across_mice.tsv",
    sep="\t",
    index=False,
)
top_genes_df.to_csv(
    summary_dir / "top20_genes_all_models.tsv",
    sep="\t",
    index=False,
)

# -----------------------
# Excel review workbook
# -----------------------
def top_enrichment_table(df: pd.DataFrame, n_per_program: int = 5):
    pcol = adjusted_p_column(df)
    return (
        df.sort_values(["model", "program_name", pcol])
        .groupby(["model", "program_name"], observed=True)
        .head(n_per_program)
    )

with pd.ExcelWriter(
    summary_dir / "model_selection_review.xlsx",
    engine="openpyxl",
) as writer:
    model_summary.to_excel(writer, sheet_name="model_summary", index=False)
    k_stats_table.to_excel(
        writer, sheet_name="k_selection", index=False
    )
    top_genes_df.to_excel(writer, sheet_name="top20_genes", index=False)
    top_enrichment_table(go_all_df).to_excel(
        writer, sheet_name="top_GO_per_program", index=False
    )
    top_enrichment_table(reactome_all_df).to_excel(
        writer, sheet_name="top_Reactome_per_program", index=False
    )
    perturb_all_df.loc[
        perturb_all_df["replicated_same_sign"]
    ].to_excel(
        writer, sheet_name="replicated_perturbations", index=False
    )

# -----------------------
# Plot helpers
# -----------------------
def save_line_plot(
    data: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    stem: str,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    for threshold, group in data.groupby("threshold", sort=True):
        group = group.sort_values("K")
        ax.plot(
            group["K"],
            group[y],
            marker="o",
            label=f"density threshold {threshold:g}",
        )
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(summary_dir / f"{stem}.pdf")
    fig.savefig(summary_dir / f"{stem}.png", dpi=200)
    plt.close(fig)

save_line_plot(
    model_summary,
    "total_explained_variance",
    "Total explained variance",
    "cNMF total explained variance",
    "explained_variance_across_K",
)
save_line_plot(
    model_summary,
    "GO_significant_program_fraction",
    "Fraction of programs",
    "Programs with at least one significant GO term",
    "GO_program_fraction_across_K",
)
save_line_plot(
    model_summary,
    "Reactome_significant_program_fraction",
    "Fraction of programs",
    "Programs with at least one significant Reactome term",
    "Reactome_program_fraction_across_K",
)
save_line_plot(
    model_summary,
    "mouse_associated_program_fraction",
    "Fraction of programs",
    "Programs associated with mouse identity (BH FDR < 0.05)",
    "mouse_association_across_K",
)
save_line_plot(
    model_summary,
    "cross_mouse_effect_spearman",
    "Spearman correlation",
    "Perturbation log2FC concordance across mice",
    "perturbation_concordance_across_K",
)

# K-selection statistics can have slightly different capitalization.
numeric_k_stats = k_stats_table.select_dtypes(include=[np.number])
for column in numeric_k_stats.columns:
    if column == "K":
        continue
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        k_stats_table["K"].astype(float),
        numeric_k_stats[column].astype(float),
        marker="o",
    )
    ax.set_xlabel("K")
    ax.set_ylabel(str(column))
    ax.set_title(f"cNMF K-selection metric: {column}")
    fig.tight_layout()
    safe = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in str(column)
    )
    fig.savefig(summary_dir / f"k_selection_{safe}.pdf")
    fig.savefig(summary_dir / f"k_selection_{safe}.png", dpi=200)
    plt.close(fig)

def enrichment_heatmap(
    df: pd.DataFrame,
    stem: str,
    title: str,
    top_n: int = TOP_TERMS,
):
    pcol = adjusted_p_column(df)
    temp = df.copy()
    temp[pcol] = pd.to_numeric(temp[pcol], errors="coerce")
    temp["score"] = -np.log10(temp[pcol].clip(lower=1e-300))
    temp["is_sig"] = temp[pcol] < FDR

    model_term = (
        temp.groupby(["Term", "model"], observed=True)
        .agg(
            score=("score", "max"),
            is_sig=("is_sig", "max"),
        )
        .reset_index()
    )
    term_rank = (
        model_term.groupby("Term", observed=True)
        .agg(
            significant_models=("is_sig", "sum"),
            max_score=("score", "max"),
        )
        .sort_values(
            ["significant_models", "max_score"],
            ascending=False,
        )
        .head(top_n)
    )
    terms = term_rank.index.tolist()
    matrix = (
        model_term.loc[model_term["Term"].isin(terms)]
        .pivot(index="Term", columns="model", values="score")
        .reindex(index=terms)
        .reindex(
            columns=[
                f"K{k}_dt{threshold_tag(t)}"
                for k, t in MODEL_ORDER
            ]
        )
        .fillna(0.0)
        .clip(upper=50)
    )
    matrix.to_csv(
        summary_dir / f"{stem}_matrix.tsv",
        sep="\t",
    )

    fig_height = max(8, 0.32 * len(matrix.index))
    fig_width = max(12, 0.75 * len(matrix.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(MODEL_LABELS, rotation=90)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Term")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Maximum -log10(adjusted P) across programs")
    fig.tight_layout()
    fig.savefig(summary_dir / f"{stem}.pdf")
    fig.savefig(summary_dir / f"{stem}.png", dpi=200)
    plt.close(fig)

enrichment_heatmap(
    go_all_df,
    "GO_terms_across_K_heatmap",
    "Top GO biological-process terms across K and density thresholds",
)
enrichment_heatmap(
    reactome_all_df,
    "Reactome_terms_across_K_heatmap",
    "Top Reactome terms across K and density thresholds",
)

# -----------------------
# Multi-page review PDF
# -----------------------
review_pdf = summary_dir / "pre_K_selection_review.pdf"
plot_files = [
    "explained_variance_across_K.png",
    "GO_program_fraction_across_K.png",
    "Reactome_program_fraction_across_K.png",
    "mouse_association_across_K.png",
    "perturbation_concordance_across_K.png",
    "GO_terms_across_K_heatmap.png",
    "Reactome_terms_across_K_heatmap.png",
]
plot_files.extend(
    sorted(p.name for p in summary_dir.glob("k_selection_*.png"))
)

with PdfPages(review_pdf) as pdf:
    for filename in plot_files:
        path = summary_dir / filename
        if not path.is_file():
            continue
        image = plt.imread(path)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(path.stem)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# Human-readable checklist; intentionally no automatic composite K choice.
notes = f"""Pre-K selection outputs
=======================

Review these jointly rather than selecting K from one metric:

1. cNMF stability and prediction error:
   k_selection_stats.tsv and k_selection_*.pdf

2. Model fit:
   explained_variance_across_K.pdf

3. Biological interpretability:
   GO_program_fraction_across_K.pdf
   Reactome_program_fraction_across_K.pdf
   GO_terms_across_K_heatmap.pdf
   Reactome_terms_across_K_heatmap.pdf
   top20_genes_all_models.tsv

4. Mouse/replicate dependence:
   mouse_association_across_K.pdf
   Lower is generally preferable unless the mice differ biologically.

5. Exploratory perturbation reproducibility:
   perturbation_concordance_across_K.pdf
   perturbation_replication_across_mice.tsv

The Stage 2A perturbation test is a per-cell Mann-Whitney analysis run
separately in IP_GFP and GV_GFP. Because this experiment has high guide
multiplicity, treat it as an exploratory K-selection diagnostic, not the
final perturbation-effect test. Run CRT after selecting K.
"""
(summary_dir / "HOW_TO_REVIEW_BEFORE_K_SELECTION.txt").write_text(notes)

required_outputs = [
    summary_dir / "model_selection_metrics.tsv",
    summary_dir / "model_selection_review.xlsx",
    summary_dir / "pre_K_selection_review.pdf",
    summary_dir / "GO_terms_across_K_heatmap.pdf",
    summary_dir / "Reactome_terms_across_K_heatmap.pdf",
    summary_dir / "perturbation_replication_across_mice.tsv",
]
missing = [
    str(path)
    for path in required_outputs
    if not path.is_file() or path.stat().st_size == 0
]
if missing:
    raise RuntimeError(
        "Missing summary outputs:\n" + "\n".join(missing)
    )

print(model_summary.to_string(index=False))
print(f"\nWrote review outputs under: {summary_dir}")
PY

cat > "${SUMMARY_DIR}/STAGE2_PRE_K_SELECTION_COMPLETE.txt" <<EOF
Stage 2A pre-K evaluation completed successfully.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
run_dir=${RUN_DIR}
summary_dir=${SUMMARY_DIR}
K_values=${K_VALUES[*]}
thresholds=${THRESHOLDS[*]}
EOF

echo
echo "============================================================"
echo "STAGE 2 PRE-K SUMMARY COMPLETED: $(date)"
echo "Review PDF:"
echo "  ${SUMMARY_DIR}/pre_K_selection_review.pdf"
echo "Model table:"
echo "  ${SUMMARY_DIR}/model_selection_metrics.tsv"
echo "Workbook:"
echo "  ${SUMMARY_DIR}/model_selection_review.xlsx"
echo "============================================================"
