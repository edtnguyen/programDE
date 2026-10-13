#!/usr/bin/env bash
#SBATCH --job-name=IPGV_s2_metrics
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-15%4

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_stage2_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
PILOT_ROOT="${BASE}/PerturbNMF/pilot_array_n20"
RUN_NAME="merged"
RUN_DIR="${PILOT_ROOT}/${RUN_NAME}"
STAGE2_ROOT="${PILOT_ROOT}/stage2_pre_k_selection"

K_VALUES=(5 10 15 20 25 30 40 50)
THRESHOLDS=(0.2 2.0)

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this file as a SLURM array.}"
N_THRESH="${#THRESHOLDS[@]}"
K_INDEX=$((TASK_ID / N_THRESH))
T_INDEX=$((TASK_ID % N_THRESH))

if (( K_INDEX < 0 || K_INDEX >= ${#K_VALUES[@]} )); then
    echo "Invalid array task index: ${TASK_ID}" >&2
    exit 2
fi

K="${K_VALUES[$K_INDEX]}"
THRESHOLD="${THRESHOLDS[$T_INDEX]}"
TAG="${THRESHOLD//./_}"

H5MU="${RUN_DIR}/Inference/adata/cNMF_${K}_${TAG}.h5mu"
X_NORMALIZED="${RUN_DIR}/Inference/cnmf_tmp/Inference.norm_counts.h5ad"
COMPILE_MARKER="${STAGE2_ROOT}/compile/K${K}_dt${TAG}_COMPLETE.txt"
MARKER="${STAGE2_ROOT}/metrics/K${K}_dt${TAG}_COMPLETE.txt"

PIPELINE="${REPO}/src/Stage2_Evaluation/A_Metrics/Slurm_Version/cNMF_evaluation_pipeline.py"
OUTPUT_FOLDER="${RUN_DIR}/Evaluation/${K}_${TAG}"

[[ -s "${COMPILE_MARKER}" ]] || {
    echo "Compile marker missing: ${COMPILE_MARKER}" >&2
    exit 1
}
[[ -s "${H5MU}" ]] || {
    echo "Compiled H5MU missing: ${H5MU}" >&2
    exit 1
}
[[ -s "${X_NORMALIZED}" ]] || {
    echo "Normalized cNMF matrix missing: ${X_NORMALIZED}" >&2
    exit 1
}

mkdir -p "$(dirname "${MARKER}")" "${OUTPUT_FOLDER}" "${STAGE2_ROOT}/tmp"
export TMPDIR="${STAGE2_ROOT}/tmp/metrics_${SLURM_ARRAY_JOB_ID:-array}_${TASK_ID}"
mkdir -p "${TMPDIR}"

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
export MPLBACKEND=Agg
unset PYTHONSTARTUP || true

# Joblib performs the program-level parallelism. Keep nested BLAS pools at 1.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-8}"

echo "============================================================"
echo "PerturbNMF Stage 2A metrics"
echo "Started:        $(date)"
echo "Job ID:         ${SLURM_JOB_ID:-interactive}"
echo "Array task:     ${TASK_ID}"
echo "K:              ${K}"
echo "Threshold:      ${THRESHOLD}"
echo "H5MU:           ${H5MU}"
echo "Output folder:  ${OUTPUT_FOLDER}"
echo "============================================================"

python "${PIPELINE}" \
    --out_dir "${PILOT_ROOT}" \
    --run_name "${RUN_NAME}" \
    --K "${K}" \
    --sel_threshs "${THRESHOLD}" \
    --X_normalized_path "${X_NORMALIZED}" \
    --Perform_categorical \
    --Perform_perturbation \
    --Perform_geneset \
    --Perform_explained_variance \
    --data_key rna \
    --prog_key cNMF \
    --categorical_key sample \
    --organism mouse \
    --gene_names_key symbol \
    --guide_names_key guide_names \
    --guide_targets_key guide_targets \
    --guide_assignment_key guide_assignment \
    --guide_annotation_key non-targeting \
    --FDR_method BH \
    --n_top 300 \
    --use_cache \
    --skip_existing

echo
echo "=== Verifying Stage 2 outputs ==="

REQUIRED=(
    "${OUTPUT_FOLDER}/${K}_categorical_association_results.txt"
    "${OUTPUT_FOLDER}/${K}_categorical_association_posthoc.txt"
    "${OUTPUT_FOLDER}/${K}_perturbation_association_results_IP_GFP.txt"
    "${OUTPUT_FOLDER}/${K}_perturbation_association_results_GV_GFP.txt"
    "${OUTPUT_FOLDER}/${K}_geneset_enrichment.txt"
    "${OUTPUT_FOLDER}/${K}_GO_term_enrichment.txt"
    "${OUTPUT_FOLDER}/${K}_Explained_Variance.txt"
    "${OUTPUT_FOLDER}/${K}_Explained_Variance_Summary.txt"
)

for required in "${REQUIRED[@]}"; do
    [[ -s "${required}" ]] || {
        echo "Missing or empty Stage 2 output: ${required}" >&2
        exit 1
    }
done

python - "${OUTPUT_FOLDER}" "${K}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

folder = Path(sys.argv[1])
k = int(sys.argv[2])

def read_tsv(name):
    path = folder / name
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise ValueError(f"Output is empty: {path}")
    return df

go = read_tsv(f"{k}_GO_term_enrichment.txt")
reactome = read_tsv(f"{k}_geneset_enrichment.txt")
expvar = read_tsv(f"{k}_Explained_Variance.txt")
cat = read_tsv(f"{k}_categorical_association_results.txt")
ip = read_tsv(f"{k}_perturbation_association_results_IP_GFP.txt")
gv = read_tsv(f"{k}_perturbation_association_results_GV_GFP.txt")

if expvar.shape[0] != k:
    raise ValueError(
        f"Expected {k} explained-variance rows; found {expvar.shape[0]}."
    )

for label, df in [("GO", go), ("Reactome", reactome)]:
    required = {"program_name", "Term", "Adjusted P-value"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{label} output lacks columns: {sorted(missing)}")

for label, df in [("IP_GFP", ip), ("GV_GFP", gv)]:
    required = {"target_name", "program_name", "log2FC", "adj_pval"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            f"{label} perturbation output lacks columns: {sorted(missing)}"
        )

print("GO rows:", len(go))
print("Reactome rows:", len(reactome))
print("Explained-variance rows:", len(expvar))
print("Categorical rows:", len(cat))
print("IP perturbation rows:", len(ip))
print("GV perturbation rows:", len(gv))
print("Stage 2 output validation: OK")
PY

cat > "${MARKER}" <<EOF
Stage 2 metrics completed successfully.
date=$(date --iso-8601=seconds)
array_job_id=${SLURM_ARRAY_JOB_ID:-not_set}
array_task_id=${TASK_ID}
job_id=${SLURM_JOB_ID:-interactive}
K=${K}
threshold=${THRESHOLD}
output_folder=${OUTPUT_FOLDER}
EOF

echo "STAGE 2 METRICS COMPLETED: ${MARKER}"
