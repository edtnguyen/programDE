#!/usr/bin/env bash
#SBATCH --job-name=IPGV_K30_summary
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

RUN_DIR="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"

# SLURM executes a copied batch script from /var/spool/slurmd, so
# BASH_SOURCE[0] does not point to the original bundle directory.
# The submission helper exports WORKBOOK_SCRIPT_DIR explicitly.
if [[ -n "${WORKBOOK_SCRIPT_DIR:-}" ]]; then
    SCRIPT_DIR="$(
        cd -- "${WORKBOOK_SCRIPT_DIR}"
        pwd
    )"
elif [[ -n "${SLURM_SUBMIT_DIR:-}"         && -r "${SLURM_SUBMIT_DIR}/build_ip_gv_k30_dt2_workbook.py" ]]; then
    SCRIPT_DIR="$(
        cd -- "${SLURM_SUBMIT_DIR}"
        pwd
    )"
else
    echo "Could not locate build_ip_gv_k30_dt2_workbook.py." >&2
    echo "Submit with submit_ip_gv_k30_dt2_workbook.sh, or set:" >&2
    echo "  sbatch --export=ALL,WORKBOOK_SCRIPT_DIR=/path/to/bundle ..." >&2
    exit 1
fi

PYTHON_SCRIPT="${SCRIPT_DIR}/build_ip_gv_k30_dt2_workbook.py"

OUTPUT_DIR="${RUN_DIR}/Interpretation/Summary/K30_dt2_0"
LOG_DIR="${OUTPUT_DIR}/logs"
OUTPUT_XLSX="${OUTPUT_DIR}/IP_GV_K30_dt2_PerturbNMF_summary.xlsx"

mkdir -p "${LOG_DIR}"

# When submitted through this helper, override generic SBATCH log names with:
#   sbatch --output="${LOG_DIR}/summary_%j.out" \
#          --error="${LOG_DIR}/summary_%j.err" \
#          run_ip_gv_k30_dt2_workbook.sbatch

[[ -x "${ENV}/bin/python" ]] || {
    echo "Stage 2 environment is missing: ${ENV}" >&2
    exit 1
}
[[ -r "${PYTHON_SCRIPT}" ]] || {
    echo "Workbook builder is missing: ${PYTHON_SCRIPT}" >&2
    exit 1
}

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
unset PYTHONSTARTUP || true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "Build final IP/GV K30, dt=2.0 PerturbNMF workbook"
echo "Started:   $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-interactive}"
echo "Node:      ${SLURMD_NODENAME:-$(hostname)}"
echo "Python:    $(which python)"
echo "Script dir:${SCRIPT_DIR}"
echo "Builder:   ${PYTHON_SCRIPT}"
echo "Run dir:   ${RUN_DIR}"
echo "Output:    ${OUTPUT_XLSX}"
echo "============================================================"

python "${PYTHON_SCRIPT}" \
    --run-dir "${RUN_DIR}" \
    --k 30 \
    --density-threshold 2.0 \
    --crt-variant rna_qc_guides \
    --top-genes 300 \
    --summary-top-genes 30 \
    --fdr 0.05 \
    --specificity-fdr 0.10 \
    --output-dir "${OUTPUT_DIR}" \
    --output-name "$(basename "${OUTPUT_XLSX}")" \
    --overwrite

[[ -s "${OUTPUT_XLSX}" ]] || {
    echo "Expected workbook was not created: ${OUTPUT_XLSX}" >&2
    exit 1
}

echo
echo "WORKBOOK COMPLETE"
echo "${OUTPUT_XLSX}"
