#!/usr/bin/env bash
#SBATCH --job-name=IPGV_s2_preflight
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_stage2_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
PILOT_ROOT="${WORK_ROOT}/pilot_array_n20"
RUN_DIR="${PILOT_ROOT}/merged"
INFERENCE_DIR="${RUN_DIR}/Inference"
STAGE2_ROOT="${PILOT_ROOT}/stage2_pre_k_selection"
MARKER="${STAGE2_ROOT}/STAGE2_PREFLIGHT_COMPLETE.txt"

PREPARED_H5AD="${WORK_ROOT}/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"
X_NORMALIZED="${INFERENCE_DIR}/cnmf_tmp/Inference.norm_counts.h5ad"
K_STATS="${INFERENCE_DIR}/Inference.k_selection_stats.df.npz"

K_VALUES=(5 10 15 20 25 30 40 50)
THRESHOLDS=(0.2 2.0)

mkdir -p "${STAGE2_ROOT}"

[[ -x "${ENV}/bin/python" ]] || {
    echo "Stage 2 environment is missing: ${ENV}" >&2
    echo "Run setup_ip_gv_perturbnmf_stage2_env_mamba.sh first." >&2
    exit 1
}

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
export MPLBACKEND=Agg
unset PYTHONSTARTUP || true

echo "============================================================"
echo "IP/GV Stage 2 preflight"
echo "Started:       $(date)"
echo "Job ID:        ${SLURM_JOB_ID:-interactive}"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "Python:        $(which python)"
echo "Pilot root:    ${PILOT_ROOT}"
echo "============================================================"

for required in \
    "${PREPARED_H5AD}" \
    "${X_NORMALIZED}" \
    "${K_STATS}" \
    "${INFERENCE_DIR}/Inference.k_selection.png"
do
    [[ -s "${required}" ]] || {
        echo "Missing pilot output: ${required}" >&2
        exit 1
    }
done

for cache in \
    "${REPO}/src/Stage2_Evaluation/Resources/Reactome_2022_mouse.json" \
    "${REPO}/src/Stage2_Evaluation/Resources/GO_Biological_Process_2023_mouse.json"
do
    [[ -s "${cache}" ]] || {
        echo "Missing mouse enrichment cache: ${cache}" >&2
        echo "Rerun the Stage 2 setup script." >&2
        exit 1
    }
done

for K in "${K_VALUES[@]}"; do
    for T in "${THRESHOLDS[@]}"; do
        TAG="${T//./_}"
        for required in \
            "${INFERENCE_DIR}/Inference.usages.k_${K}.dt_${TAG}.consensus.txt" \
            "${INFERENCE_DIR}/Inference.gene_spectra_score.k_${K}.dt_${TAG}.txt"
        do
            [[ -s "${required}" ]] || {
                echo "Missing consensus output: ${required}" >&2
                exit 1
            }
        done
    done
done

python - \
    "${PREPARED_H5AD}" \
    "${X_NORMALIZED}" \
    "${REPO}" <<'PY'
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

prepared_path = Path(sys.argv[1])
norm_path = Path(sys.argv[2])
repo = Path(sys.argv[3])

prepared = ad.read_h5ad(prepared_path, backed="r")
norm = ad.read_h5ad(norm_path, backed="r")

print("Prepared shape:", prepared.shape)
print("Normalized cNMF shape:", norm.shape)

if prepared.n_obs != norm.n_obs:
    raise ValueError("Prepared and normalized objects have different cell counts.")
if "sample" not in prepared.obs:
    raise KeyError("Prepared input lacks obs['sample'].")
if "guide_assignment" not in prepared.obsm:
    raise KeyError("Prepared input lacks obsm['guide_assignment'].")
if "guide_names" not in prepared.uns or "guide_targets" not in prepared.uns:
    raise KeyError("Prepared input lacks guide metadata.")

samples = sorted(prepared.obs["sample"].astype(str).unique().tolist())
if samples != ["GV_GFP", "IP_GFP"]:
    raise ValueError(f"Unexpected sample labels: {samples}")

assignment = prepared.obsm["guide_assignment"]
if assignment.shape[0] != prepared.n_obs:
    raise ValueError("Guide assignment cell dimension is incorrect.")
if assignment.shape[1] != len(prepared.uns["guide_names"]):
    raise ValueError("Guide assignment guide dimension is incorrect.")

prepared.file.close()
norm.file.close()

for library in ["Reactome_2022", "GO_Biological_Process_2023"]:
    cache = (
        repo
        / "src/Stage2_Evaluation/Resources"
        / f"{library}_mouse.json"
    )
    content = json.loads(cache.read_text())
    if not isinstance(content, dict) or len(content) == 0:
        raise ValueError(f"Invalid enrichment cache: {cache}")
    print(f"{library}: {len(content):,} cached terms")

from Stage1_Inference.src import compile_results
from Stage2_Evaluation.A_Metrics.src import (
    compute_categorical_association,
    compute_explained_variance,
    compute_geneset_enrichment,
    compute_perturbation_association,
)

print("Stage 1 compile import: OK")
print("Stage 2 metric imports: OK")
PY

cat > "${MARKER}" <<EOF
Stage 2 preflight completed successfully.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
pilot_root=${PILOT_ROOT}
run_dir=${RUN_DIR}
prepared_h5ad=${PREPARED_H5AD}
normalized_h5ad=${X_NORMALIZED}
K_values=${K_VALUES[*]}
thresholds=${THRESHOLDS[*]}
EOF

echo
echo "STAGE 2 PREFLIGHT PASSED"
echo "Marker: ${MARKER}"
