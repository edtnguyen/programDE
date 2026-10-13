#!/usr/bin/env bash
#SBATCH --job-name=IP_GV_cNMF_pilot
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -Eeuo pipefail
trap 'echo "ERROR at line ${LINENO}; exit code $?" >&2' ERR

START_SECONDS=$SECONDS

# -----------------------------
# User paths and run parameters
# -----------------------------
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_sk_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
SOURCE_H5AD="${BASE}/combined/IP_GFP_GV_GFP_guide_positive_cNMF_input.h5ad"

WORK_ROOT="${BASE}/PerturbNMF"
INPUT_DIR="${WORK_ROOT}/input"
OUTPUT_DIRECTORY="${WORK_ROOT}/results"
PREPARED_H5AD="${INPUT_DIR}/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"
GUIDE_ANNOTATION="${INPUT_DIR}/IP_GFP_GV_GFP_guide_annotation.tsv"
PREP_SUMMARY="${INPUT_DIR}/IP_GFP_GV_GFP_PerturbNMF_input_summary.json"

RUN_NAME="IP_GV_skcnmf_pilot_n20"
K_VALUES=(5 10 15 20 25 30 40 50)
SEL_THRESHOLDS=(0.2 2.0)
NUM_ITER=20
NUM_HVG=3000
SEED=14

# Set REBUILD_INPUT=1 before sbatch only when the source H5AD has changed.
REBUILD_INPUT="${REBUILD_INPUT:-0}"

PIPELINE="${REPO}/src/Stage1_Inference/sk-cNMF/Slurm_Version/sk-cNMF_batch_inference_pipeline.py"
VALIDATOR="${REPO}/.claude/skills/perturbNMF-runner/scripts/validate_data.py"
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIRECTORY}" "${WORK_ROOT}/tmp"
export TMPDIR="${WORK_ROOT}/tmp/${SLURM_JOB_ID:-interactive}"
mkdir -p "${TMPDIR}"

source "${CONDA_SH}"
conda activate "${ENV}"

# Prevent incompatible packages under ~/.local from entering this environment.
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
unset PYTHONSTARTUP || true

# One cNMF replicate is run at a time; BLAS may use these CPU threads.
export OMP_NUM_THREADS="${NTHREADS}"
export OPENBLAS_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export BLIS_NUM_THREADS="${NTHREADS}"
export NUMEXPR_NUM_THREADS="${NTHREADS}"

echo "============================================================"
echo "IP/GV PerturbNMF Stage 1 pilot"
echo "Started:          $(date)"
echo "Job ID:           ${SLURM_JOB_ID:-interactive}"
echo "Node:             ${SLURMD_NODENAME:-$(hostname)}"
echo "Python:           $(which python)"
echo "Python version:   $(python --version 2>&1)"
echo "Source H5AD:      ${SOURCE_H5AD}"
echo "Prepared H5AD:    ${PREPARED_H5AD}"
echo "Output directory: ${OUTPUT_DIRECTORY}"
echo "Run name:         ${RUN_NAME}"
echo "K values:         ${K_VALUES[*]}"
echo "Replicates/K:     ${NUM_ITER}"
echo "HVGs:             ${NUM_HVG}"
echo "Thresholds:       ${SEL_THRESHOLDS[*]}"
echo "============================================================"

[[ -r "${SOURCE_H5AD}" ]] || {
    echo "Missing source H5AD: ${SOURCE_H5AD}" >&2
    exit 1
}
[[ -r "${PIPELINE}" ]] || {
    echo "Missing PerturbNMF pipeline: ${PIPELINE}" >&2
    exit 1
}

# -------------------------------------------------------------------------
# Prepare the exact AnnData format expected by PerturbNMF.
# X remains raw counts. PCA/UMAP are computed on a temporary log-normalized
# copy and then copied into obsm.
# -------------------------------------------------------------------------
if [[ "${REBUILD_INPUT}" == "1" || ! -s "${PREPARED_H5AD}" ]]; then
    echo
    echo "=== Preparing PerturbNMF input ==="

    python - \
        "${SOURCE_H5AD}" \
        "${PREPARED_H5AD}" \
        "${GUIDE_ANNOTATION}" \
        "${PREP_SUMMARY}" \
        "${NTHREADS}" <<'PY'
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

source = Path(sys.argv[1])
output = Path(sys.argv[2])
guide_annotation_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
n_threads = int(sys.argv[5])

sc.settings.n_jobs = n_threads
sc.settings.verbosity = 2

adata = ad.read_h5ad(source)
print(f"Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

if not adata.obs_names.is_unique:
    raise ValueError("obs_names are not unique.")
if not adata.var_names.is_unique:
    raise ValueError("var_names are not unique.")
if "sample_id" not in adata.obs:
    raise KeyError("Expected obs['sample_id'].")

# Preserve raw counts in X and standardize sparse representation.
if sp.issparse(adata.X):
    adata.X = adata.X.tocsr().astype(np.float32)
    x_values = adata.X.data
else:
    adata.X = np.asarray(adata.X, dtype=np.float32)
    x_values = adata.X.ravel()

if x_values.size:
    if not np.isfinite(x_values).all():
        raise ValueError("X contains NaN or infinite values.")
    if x_values.min() < 0:
        raise ValueError("X contains negative values; expected raw counts.")
    max_fractional = float(np.max(np.abs(x_values - np.rint(x_values))))
else:
    max_fractional = 0.0

if max_fractional > 1e-3:
    raise ValueError(
        f"X is not integer-like raw counts; maximum fractional deviation is "
        f"{max_fractional:.6g}."
    )

# IP_GFP and GV_GFP are the two biological-replicate mouse labels.
adata.obs["sample"] = pd.Categorical(adata.obs["sample_id"].astype(str))
adata.obs["mouse"] = adata.obs["sample"].copy()

# Use the SCEPTRE assignment matrix, retaining guide order from uns.
source_assignment_key = (
    "guide_assignment"
    if "guide_assignment" in adata.obsm
    else "guide_assignments"
    if "guide_assignments" in adata.obsm
    else None
)
if source_assignment_key is None:
    raise KeyError(
        "Neither obsm['guide_assignment'] nor obsm['guide_assignments'] exists."
    )

raw_assignment = adata.obsm[source_assignment_key]
if sp.issparse(raw_assignment):
    guide_assignment = raw_assignment.tocsr().copy()
else:
    guide_assignment = sp.csr_matrix(np.asarray(raw_assignment))

guide_assignment.eliminate_zeros()
guide_assignment.data = np.ones(guide_assignment.nnz, dtype=np.int8)

if "guide_names" not in adata.uns or "guide_targets" not in adata.uns:
    raise KeyError("Expected uns['guide_names'] and uns['guide_targets'].")

guide_names = np.asarray(adata.uns["guide_names"]).astype(str)
guide_targets = np.asarray(adata.uns["guide_targets"]).astype(str)

if guide_assignment.shape != (adata.n_obs, len(guide_names)):
    raise ValueError(
        "Guide assignment shape mismatch: "
        f"{guide_assignment.shape} versus "
        f"({adata.n_obs}, {len(guide_names)})."
    )
if len(guide_targets) != len(guide_names):
    raise ValueError("guide_targets and guide_names have different lengths.")

n_assigned = np.asarray(guide_assignment.sum(axis=1)).ravel().astype(int)
if np.any(n_assigned == 0):
    raise ValueError(
        f"{int(np.sum(n_assigned == 0))} cells have no SCEPTRE-assigned guide "
        "in this guide-positive input."
    )

adata.obsm["guide_assignment"] = guide_assignment
adata.uns["guide_names"] = guide_names
adata.uns["guide_targets"] = guide_targets

# Ensure the compiled result has an explicit gene-symbol column.
adata.var["symbol"] = adata.var_names.astype(str)

# Compute plotting embeddings from a temporary normalized copy.
embedding = ad.AnnData(
    X=adata.X.copy(),
    obs=adata.obs[["sample"]].copy(),
    var=pd.DataFrame(index=adata.var_names.copy()),
)
sc.pp.filter_genes(embedding, min_cells=3)
sc.pp.normalize_total(embedding, target_sum=1e4)
sc.pp.log1p(embedding)

n_hvg = min(3000, embedding.n_vars)
sc.pp.highly_variable_genes(
    embedding,
    n_top_genes=n_hvg,
    flavor="seurat",
    subset=True,
)
sc.pp.scale(embedding, max_value=10)

n_pcs = min(50, embedding.n_obs - 1, embedding.n_vars - 1)
if n_pcs < 2:
    raise ValueError("Too few cells or genes to calculate PCA.")

sc.tl.pca(
    embedding,
    n_comps=n_pcs,
    svd_solver="arpack",
    random_state=14,
)
sc.pp.neighbors(
    embedding,
    n_neighbors=min(15, embedding.n_obs - 1),
    n_pcs=min(30, n_pcs),
    random_state=14,
)
sc.tl.umap(embedding, random_state=14)

adata.obsm["X_pca"] = np.asarray(embedding.obsm["X_pca"], dtype=np.float32)
adata.obsm["X_umap"] = np.asarray(embedding.obsm["X_umap"], dtype=np.float32)

# Avoid AnnData index-name/column-name collisions during writing.
adata.obs.index.name = None
adata.var.index.name = None

sample_counts = (
    adata.obs["sample"].astype(str).value_counts().sort_index().to_dict()
)
assignment_counts = pd.Series(n_assigned).value_counts().sort_index().to_dict()

adata.uns["perturbnmf_preparation"] = {
    "source_h5ad": str(source),
    "expression_matrix": "raw RNA counts",
    "assignment_method": "SCEPTRE",
    "guide_assignment_key": "guide_assignment",
    "pca_umap_input": "temporary normalize_total(1e4), log1p, 3000 HVGs",
    "random_seed": 14,
}

control_labels = {
    "non-targeting",
    "non_targeting",
    "nontargeting",
    "ntc",
    "control",
    "safe-targeting",
    "safe_targeting",
}
targeting = [
    str(target).strip().lower() not in control_labels
    for target in guide_targets
]
guide_annotation = pd.DataFrame(
    {
        "guide_names": guide_names,
        "guide_targets": guide_targets,
        "targeting": targeting,
    },
    index=guide_names,
)
guide_annotation.index.name = "guide_id"

summary = {
    "source_h5ad": str(source),
    "prepared_h5ad": str(output),
    "n_cells": int(adata.n_obs),
    "n_genes": int(adata.n_vars),
    "n_guides": int(len(guide_names)),
    "n_unique_targets": int(len(set(guide_targets))),
    "sample_counts": {str(k): int(v) for k, v in sample_counts.items()},
    "assigned_guides_per_cell": {
        str(k): int(v) for k, v in assignment_counts.items()
    },
    "mean_assigned_guides_per_cell": float(n_assigned.mean()),
    "fraction_multiguide": float(np.mean(n_assigned > 1)),
    "x_max_fractional_deviation": max_fractional,
    "x_is_sparse": bool(sp.issparse(adata.X)),
    "x_pca_shape": list(map(int, adata.obsm["X_pca"].shape)),
    "x_umap_shape": list(map(int, adata.obsm["X_umap"].shape)),
    "guide_assignment_shape": list(map(int, guide_assignment.shape)),
}

output.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(output, compression="gzip")
guide_annotation.to_csv(guide_annotation_path, sep="\t")
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps(summary, indent=2))
print(f"Wrote: {output}")
print(f"Wrote: {guide_annotation_path}")
print(f"Wrote: {summary_path}")
PY
else
    echo
    echo "=== Reusing prepared input ==="
    echo "${PREPARED_H5AD}"
fi

# -------------------------------------------------------------------------
# Validate the prepared object using the validator at the pinned checkout.
# -------------------------------------------------------------------------
echo
echo "=== Validating prepared input ==="
python "${VALIDATOR}" \
    --counts_fn "${PREPARED_H5AD}" \
    --categorical_key sample \
    --guide_names_key guide_names \
    --guide_targets_key guide_targets \
    --guide_assignment_key guide_assignment

# -------------------------------------------------------------------------
# Pilot K sweep. This performs:
# prepare -> factorize -> combine -> K selection -> consensus -> diagnostics.
#
# It intentionally does not compile 16 redundant H5MU files or query MyGene.
# The selected K/threshold will be compiled in the final 100-replicate run.
# -------------------------------------------------------------------------
echo
echo "=== Running sk-cNMF pilot K sweep ==="
python "${PIPELINE}" \
    --counts_fn "${PREPARED_H5AD}" \
    --output_directory "${OUTPUT_DIRECTORY}" \
    --run_name "${RUN_NAME}" \
    --species mouse \
    --K "${K_VALUES[@]}" \
    --numiter "${NUM_ITER}" \
    --numhvgenes "${NUM_HVG}" \
    --seed "${SEED}" \
    --init random \
    --loss frobenius \
    --algo cd \
    --max_NMF_iter 1000 \
    --tol 1e-4 \
    --sel_threshs "${SEL_THRESHOLDS[@]}" \
    --categorical_key sample \
    --guide_names_key guide_names \
    --guide_targets_key guide_targets \
    --guide_assignment_key guide_assignment \
    --gene_names_key symbol \
    --run_factorize \
    --run_refit \
    --run_diagnostic_plots \
    --skip_existing

RUN_DIR="${OUTPUT_DIRECTORY}/${RUN_NAME}"
K_SELECTION="${RUN_DIR}/Inference/Inference.k_selection.png"
K_STATS="${RUN_DIR}/Inference/Inference.k_selection_stats.df.npz"

[[ -s "${K_SELECTION}" ]] || {
    echo "Expected K-selection plot was not created: ${K_SELECTION}" >&2
    exit 1
}
[[ -s "${K_STATS}" ]] || {
    echo "Expected K-selection statistics were not created: ${K_STATS}" >&2
    exit 1
}

cat > "${RUN_DIR}/NEXT_STEP.txt" <<EOF
Pilot completed.

Review:
  ${K_SELECTION}
  ${RUN_DIR}/Inference/diagnosis_plots/
  ${RUN_DIR}/Inference/Inference.clustering.k_<K>.dt_0_2.png
  ${RUN_DIR}/Inference/Inference.clustering.k_<K>.dt_2_0.png

After choosing K and density threshold, submit:
  sbatch run_ip_gv_perturbnmf_final.sbatch <K> <threshold>

Example only:
  sbatch run_ip_gv_perturbnmf_final.sbatch 30 0.2
EOF

ELAPSED=$((SECONDS - START_SECONDS))
echo
echo "============================================================"
echo "PILOT COMPLETED: $(date)"
echo "Elapsed seconds: ${ELAPSED}"
echo "K-selection plot:"
echo "  ${K_SELECTION}"
echo "Diagnostic directory:"
echo "  ${RUN_DIR}/Inference/diagnosis_plots"
echo "============================================================"
