#!/usr/bin/env bash
#SBATCH --job-name=IPGV_cNMF_prep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

START_SECONDS=$SECONDS

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_sk_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
SOURCE_H5AD="${BASE}/combined/IP_GFP_GV_GFP_guide_positive_cNMF_input.h5ad"

WORK_ROOT="${BASE}/PerturbNMF"
INPUT_DIR="${WORK_ROOT}/input"
PREPARED_H5AD="${INPUT_DIR}/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"
GUIDE_ANNOTATION="${INPUT_DIR}/IP_GFP_GV_GFP_guide_annotation.tsv"
PREP_SUMMARY="${INPUT_DIR}/IP_GFP_GV_GFP_PerturbNMF_input_summary.json"

PILOT_ROOT="${WORK_ROOT}/pilot_array_n20"
SEED_DIR="${PILOT_ROOT}/seeds"
PREP_MARKER="${PILOT_ROOT}/PREPARATION_COMPLETE.txt"

K_VALUES=(5 10 15 20 25 30 40 50)
NUM_ITER=20
SEED=14

# Submit with --export=ALL,REBUILD_INPUT=1 only after the source H5AD changes.
REBUILD_INPUT="${REBUILD_INPUT:-0}"

VALIDATOR="${REPO}/.claude/skills/perturbNMF-runner/scripts/validate_data.py"
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${INPUT_DIR}" "${PILOT_ROOT}" "${SEED_DIR}" "${WORK_ROOT}/tmp"
export TMPDIR="${WORK_ROOT}/tmp/${SLURM_JOB_ID:-interactive}"
mkdir -p "${TMPDIR}"

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
unset PYTHONSTARTUP || true

export OMP_NUM_THREADS="${NTHREADS}"
export OPENBLAS_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export BLIS_NUM_THREADS="${NTHREADS}"
export NUMEXPR_NUM_THREADS="${NTHREADS}"

echo "============================================================"
echo "IP/GV PerturbNMF array pilot: input preparation"
echo "Started:       $(date)"
echo "Job ID:        ${SLURM_JOB_ID:-interactive}"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "Python:        $(which python)"
echo "Source H5AD:   ${SOURCE_H5AD}"
echo "Prepared H5AD: ${PREPARED_H5AD}"
echo "Pilot root:    ${PILOT_ROOT}"
echo "============================================================"

[[ -r "${SOURCE_H5AD}" ]] || {
    echo "Missing source H5AD: ${SOURCE_H5AD}" >&2
    exit 1
}
[[ -r "${VALIDATOR}" ]] || {
    echo "Missing PerturbNMF validator: ${VALIDATOR}" >&2
    exit 1
}

if [[ "${REBUILD_INPUT}" == "1" || ! -s "${PREPARED_H5AD}" ]]; then
    echo
    echo "=== Preparing PerturbNMF input H5AD ==="

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

# Preserve raw RNA counts in X.
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
        "X is not integer-like raw counts; maximum fractional deviation is "
        f"{max_fractional:.6g}."
    )

# The two sample IDs are the biological-replicate mouse labels.
adata.obs["sample"] = pd.Categorical(adata.obs["sample_id"].astype(str))
adata.obs["mouse"] = adata.obs["sample"].copy()

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
        f"{guide_assignment.shape} versus ({adata.n_obs}, {len(guide_names)})."
    )
if len(guide_targets) != len(guide_names):
    raise ValueError("guide_targets and guide_names have different lengths.")

n_assigned = np.asarray(guide_assignment.sum(axis=1)).ravel().astype(int)
if np.any(n_assigned == 0):
    raise ValueError(
        f"{int(np.sum(n_assigned == 0))} cells have no SCEPTRE-assigned guide "
        "in the guide-positive input."
    )

adata.obsm["guide_assignment"] = guide_assignment
adata.uns["guide_names"] = guide_names
adata.uns["guide_targets"] = guide_targets
adata.var["symbol"] = adata.var_names.astype(str)

# PCA and UMAP are only metadata for downstream visualization. Their input is
# a temporary normalized copy; adata.X remains raw counts.
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
    "pca_umap_input": "normalize_total(1e4), log1p, 3000 HVGs",
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
    echo "=== Reusing existing prepared H5AD ==="
    echo "${PREPARED_H5AD}"
fi

echo
echo "=== Generating deterministic K-specific NMF seed files ==="
python - \
    "${SEED_DIR}" \
    "${NUM_ITER}" \
    "${SEED}" \
    "${K_VALUES[@]}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

seed_dir = Path(sys.argv[1])
n_iter = int(sys.argv[2])
master_seed = int(sys.argv[3])
k_values = sorted({int(x) for x in sys.argv[4:]})

seed_dir.mkdir(parents=True, exist_ok=True)

# This exactly matches sk-cNMF's legacy np.random.seed + np.random.randint
# sequence for a single prepare() call with all K values.
np.random.seed(master_seed)
all_seeds = np.random.randint(
    low=1,
    high=(2**31) - 1,
    size=len(k_values) * n_iter,
)

rows = []
for k_index, k in enumerate(k_values):
    start = k_index * n_iter
    stop = start + n_iter
    k_seeds = all_seeds[start:stop].astype(np.int64)
    np.save(seed_dir / f"nmf_seeds_K{k}.npy", k_seeds)
    for iteration, seed in enumerate(k_seeds):
        rows.append(
            {
                "K": k,
                "iter": iteration,
                "nmf_seed": int(seed),
                "master_seed": master_seed,
            }
        )

manifest = pd.DataFrame(rows)
manifest.to_csv(seed_dir / "nmf_seed_manifest.tsv", sep="\t", index=False)

if len(manifest) != len(k_values) * n_iter:
    raise RuntimeError("Seed manifest has the wrong number of rows.")

print(manifest.groupby("K").size())
print(f"Wrote seed files and manifest under {seed_dir}")
PY

echo
echo "=== Validating prepared input ==="
python "${VALIDATOR}" \
    --counts_fn "${PREPARED_H5AD}" \
    --categorical_key sample \
    --guide_names_key guide_names \
    --guide_targets_key guide_targets \
    --guide_assignment_key guide_assignment

cat > "${PREP_MARKER}" <<EOF
Preparation completed successfully.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
source_h5ad=${SOURCE_H5AD}
prepared_h5ad=${PREPARED_H5AD}
K_values=${K_VALUES[*]}
numiter=${NUM_ITER}
seed=${SEED}
seed_manifest=${SEED_DIR}/nmf_seed_manifest.tsv
EOF

ELAPSED=$((SECONDS - START_SECONDS))
echo
echo "============================================================"
echo "PREPARATION COMPLETED: $(date)"
echo "Elapsed seconds: ${ELAPSED}"
echo "Marker: ${PREP_MARKER}"
echo "============================================================"
