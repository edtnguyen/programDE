#!/usr/bin/env bash
#SBATCH --job-name=IP_GV_cNMF_final
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

# Usage:
#   sbatch run_ip_gv_perturbnmf_final.sbatch <K> <density_threshold>
#
# Example only, after inspecting the pilot:
#   sbatch run_ip_gv_perturbnmf_final.sbatch 30 0.2

FINAL_K="${1:-}"
FINAL_THRESHOLD="${2:-}"

if [[ -z "${FINAL_K}" || -z "${FINAL_THRESHOLD}" ]]; then
    echo "Usage: sbatch $0 <K> <density_threshold>" >&2
    echo "Example: sbatch $0 30 0.2" >&2
    exit 2
fi
if ! [[ "${FINAL_K}" =~ ^[1-9][0-9]*$ ]]; then
    echo "K must be a positive integer; received '${FINAL_K}'." >&2
    exit 2
fi
if ! [[ "${FINAL_THRESHOLD}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Density threshold must be numeric; received '${FINAL_THRESHOLD}'." >&2
    exit 2
fi

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_sk_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
PREPARED_H5AD="${WORK_ROOT}/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"
OUTPUT_DIRECTORY="${WORK_ROOT}/results"

# Canonicalize with the Bash printf builtin because PerturbNMF names
# threshold 2 as "2_0".
printf -v FINAL_THRESHOLD_CANON "%g" "${FINAL_THRESHOLD}"
if [[ "${FINAL_THRESHOLD_CANON}" != *.*       && "${FINAL_THRESHOLD_CANON}" != *e*       && "${FINAL_THRESHOLD_CANON}" != *E* ]]; then
    FINAL_THRESHOLD_CANON="${FINAL_THRESHOLD_CANON}.0"
fi
FINAL_THRESHOLD="${FINAL_THRESHOLD_CANON}"
THRESHOLD_TAG="${FINAL_THRESHOLD//./_}"
RUN_NAME="IP_GV_skcnmf_K${FINAL_K}_dt${THRESHOLD_TAG}_n100"

NUM_ITER=100
NUM_HVG=3000
SEED=14

# Set RUN_MYGENE_ANNOTATION=1 at submission only when online gene-description
# annotation is desired. It is not required for cNMF or Stage 2.
RUN_MYGENE_ANNOTATION="${RUN_MYGENE_ANNOTATION:-0}"

PIPELINE="${REPO}/src/Stage1_Inference/sk-cNMF/Slurm_Version/sk-cNMF_batch_inference_pipeline.py"
VALIDATOR="${REPO}/.claude/skills/perturbNMF-runner/scripts/validate_data.py"
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"

[[ -s "${PREPARED_H5AD}" ]] || {
    echo "Prepared input is missing: ${PREPARED_H5AD}" >&2
    echo "Run the pilot script first." >&2
    exit 1
}

mkdir -p "${OUTPUT_DIRECTORY}" "${WORK_ROOT}/tmp"
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
echo "IP/GV PerturbNMF selected-K run"
echo "Started:          $(date)"
echo "Job ID:           ${SLURM_JOB_ID:-interactive}"
echo "Node:             ${SLURMD_NODENAME:-$(hostname)}"
echo "Prepared H5AD:    ${PREPARED_H5AD}"
echo "Output directory: ${OUTPUT_DIRECTORY}"
echo "Run name:         ${RUN_NAME}"
echo "K:                ${FINAL_K}"
echo "Replicates:       ${NUM_ITER}"
echo "HVGs:             ${NUM_HVG}"
echo "Threshold:        ${FINAL_THRESHOLD}"
echo "MyGene annotate:  ${RUN_MYGENE_ANNOTATION}"
echo "============================================================"

echo
echo "=== Revalidating prepared input ==="
python "${VALIDATOR}" \
    --counts_fn "${PREPARED_H5AD}" \
    --categorical_key sample \
    --guide_names_key guide_names \
    --guide_targets_key guide_targets \
    --guide_assignment_key guide_assignment

echo
echo "=== Running selected-K sk-cNMF ==="
python "${PIPELINE}" \
    --counts_fn "${PREPARED_H5AD}" \
    --output_directory "${OUTPUT_DIRECTORY}" \
    --run_name "${RUN_NAME}" \
    --species mouse \
    --K "${FINAL_K}" \
    --numiter "${NUM_ITER}" \
    --numhvgenes "${NUM_HVG}" \
    --seed "${SEED}" \
    --init random \
    --loss frobenius \
    --algo cd \
    --max_NMF_iter 1000 \
    --tol 1e-4 \
    --sel_threshs "${FINAL_THRESHOLD}" \
    --categorical_key sample \
    --guide_names_key guide_names \
    --guide_targets_key guide_targets \
    --guide_assignment_key guide_assignment \
    --gene_names_key symbol \
    --run_factorize \
    --run_refit \
    --skip_existing

RUN_DIR="${OUTPUT_DIRECTORY}/${RUN_NAME}"

# Compile only the selected K/threshold into H5MU, then generate diagnostics.
# This avoids creating one full RNA-containing H5MU for every pilot model.
echo
echo "=== Compiling selected model and diagnostics ==="
python - \
    "${RUN_DIR}" \
    "${FINAL_K}" \
    "${FINAL_THRESHOLD}" \
    "${RUN_MYGENE_ANNOTATION}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

from Stage1_Inference.src import (
    annotate_genes_to_excel,
    compile_results,
    get_top_indices_fast,
)
from Stage1_Inference.src.plot_diagnostics import generate_all_plots

run_dir = Path(sys.argv[1])
k = int(sys.argv[2])
threshold = float(sys.argv[3])
run_annotation = sys.argv[4] == "1"
threshold_tag = str(threshold).replace(".", "_")

compile_results(
    str(run_dir),
    "Inference",
    components=[k],
    sel_threshs=[threshold],
    guide_names_key="guide_names",
    guide_targets_key="guide_targets",
    categorical_key="sample",
    guide_assignment_key="guide_assignment",
    gene_names_key="symbol",
)

generate_all_plots(
    run_dir=str(run_dir),
    run_name="Inference",
    K_list=[k],
    sel_thresh_list=[threshold],
    categorical_key="sample",
)

if run_annotation:
    spectra_path = (
        run_dir
        / "Inference"
        / f"Inference.gene_spectra_score.k_{k}.dt_{threshold_tag}.txt"
    )
    annotation_dir = run_dir / "Inference" / "Annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    spectra = pd.read_csv(spectra_path, sep="\t", index_col=0)
    top_genes = get_top_indices_fast(spectra, gene_num=300)
    annotate_genes_to_excel(
        top_genes,
        species="mouse",
        output_file=str(annotation_dir / f"{k}_{threshold_tag}.xlsx"),
    )
PY

H5MU="${RUN_DIR}/Inference/adata/cNMF_${FINAL_K}_${THRESHOLD_TAG}.h5mu"
[[ -s "${H5MU}" ]] || {
    echo "Expected selected-model H5MU was not created: ${H5MU}" >&2
    exit 1
}

echo
echo "=== Validating selected-model H5MU ==="
python "${VALIDATOR}" \
    --counts_fn "${H5MU}" \
    --data_key rna \
    --prog_key cNMF \
    --categorical_key sample \
    --guide_names_key guide_names \
    --guide_targets_key guide_targets \
    --guide_assignment_key guide_assignment

cat > "${RUN_DIR}/RUN_COMPLETE.txt" <<EOF
Selected-K cNMF run completed.

Input:
  ${PREPARED_H5AD}

Parameters:
  K=${FINAL_K}
  density_threshold=${FINAL_THRESHOLD}
  numiter=${NUM_ITER}
  numhvgenes=${NUM_HVG}
  solver=cd
  loss=frobenius
  tol=1e-4
  seed=${SEED}

Primary Stage 2 input:
  ${H5MU}

Diagnostics:
  ${RUN_DIR}/Inference/diagnosis_plots/
EOF

ELAPSED=$((SECONDS - START_SECONDS))
echo
echo "============================================================"
echo "FINAL STAGE 1 RUN COMPLETED: $(date)"
echo "Elapsed seconds: ${ELAPSED}"
echo "Stage 2 input:"
echo "  ${H5MU}"
echo "Diagnostics:"
echo "  ${RUN_DIR}/Inference/diagnosis_plots"
echo "============================================================"
