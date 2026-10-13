#!/usr/bin/env bash
#SBATCH --job-name=IPGV_PAH_GSA_resume
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_gwas_program_burden_20260621"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
FINAL_RUN="${WORK_ROOT}/results/IP_GV_skcnmf_K30_dt2_0_n100"

GWAS_ACCESSION="GCST007228"
OUTPUT_ROOT="${FINAL_RUN}/Interpretation/PAH_GWAS_program_burden/${GWAS_ACCESSION}"
MAGMA_GENE_MANIFEST="${OUTPUT_ROOT}/magma_gene/MAGMA_GENE_ANALYSIS.json"
PROGRAM_INPUT_DIR="${OUTPUT_ROOT}/program_inputs"
MAGMA_PROGRAM_DIR="${OUTPUT_ROOT}/magma_program"
SUMMARY_DIR="${OUTPUT_ROOT}/summary"
HUMAN_VALIDATION="${FINAL_RUN}/Interpretation/Human_validation_GSE185479/summary/K30_dt2_human_validation_all_programs.tsv"

CODE_DIR="${PAH_GWAS_RESUME_CODE_DIR:?PAH_GWAS_RESUME_CODE_DIR is required. Submit with the helper.}"
R_PROGRAM="${CODE_DIR}/run_magma_program_tests_fixed.R"
PIPELINE="${OUTPUT_ROOT}/code/pah_gwas_program_burden.py"

for required in \
    "${ENV}/bin/python" \
    "${ENV}/bin/magma" \
    "${MAGMA_GENE_MANIFEST}" \
    "${PROGRAM_INPUT_DIR}/PROGRAM_INPUTS_COMPLETE.json" \
    "${PROGRAM_INPUT_DIR}/program_gene_covariates_target_and_MHC_excluded.txt" \
    "${PROGRAM_INPUT_DIR}/program_gene_covariates_target_excluded_MHC_included.txt" \
    "${PROGRAM_INPUT_DIR}/program_top50_target_and_MHC_excluded.sets" \
    "${PROGRAM_INPUT_DIR}/program_top300_target_and_MHC_excluded.sets" \
    "${PROGRAM_INPUT_DIR}/program_top50_target_excluded_MHC_included.sets" \
    "${PROGRAM_INPUT_DIR}/program_top300_target_excluded_MHC_included.sets" \
    "${R_PROGRAM}" \
    "${PIPELINE}"
do
    [[ -s "${required}" || -x "${required}" ]] || {
        echo "Missing required resume input: ${required}" >&2
        exit 1
    }
done

mkdir -p "${MAGMA_PROGRAM_DIR}" "${SUMMARY_DIR}"

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true
export R_LIBS_USER="${ENV}/lib/R/library"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

GENES_RAW="$(
    python - "${MAGMA_GENE_MANIFEST}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["genes_raw"])
PY
)"

[[ -s "${GENES_RAW}" ]] || {
    echo "MAGMA genes.raw is missing: ${GENES_RAW}" >&2
    exit 1
}

echo "============================================================"
echo "Resume PAH GWAS program tests and summary"
echo "Started:     $(date)"
echo "Job ID:      ${SLURM_JOB_ID:-interactive}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "genes.raw:   ${GENES_RAW}"
echo "program dir: ${MAGMA_PROGRAM_DIR}"
echo "============================================================"

echo
echo "=== 1. Corrected MAGMA continuous and categorical program tests ==="
Rscript "${R_PROGRAM}" \
    "${GENES_RAW}" \
    "${PROGRAM_INPUT_DIR}/program_gene_covariates_target_and_MHC_excluded.txt" \
    "${PROGRAM_INPUT_DIR}/program_gene_covariates_target_excluded_MHC_included.txt" \
    "${PROGRAM_INPUT_DIR}/program_top50_target_and_MHC_excluded.sets" \
    "${PROGRAM_INPUT_DIR}/program_top300_target_and_MHC_excluded.sets" \
    "${PROGRAM_INPUT_DIR}/program_top50_target_excluded_MHC_included.sets" \
    "${PROGRAM_INPUT_DIR}/program_top300_target_excluded_MHC_included.sets" \
    "${MAGMA_PROGRAM_DIR}"

echo
echo "=== 2. Rebuild final summary only ==="
SUMMARY_ARGS=(
    summarize
    --program-input-dir "${PROGRAM_INPUT_DIR}"
    --magma-results-dir "${MAGMA_PROGRAM_DIR}"
    --top-driver-genes 25
    --output-dir "${SUMMARY_DIR}"
)
if [[ -s "${HUMAN_VALIDATION}" ]]; then
    SUMMARY_ARGS+=(--human-validation "${HUMAN_VALIDATION}")
fi

python "${PIPELINE}" "${SUMMARY_ARGS[@]}"

for required in \
    "${SUMMARY_DIR}/PAH_GWAS_program_burden_results.tsv" \
    "${SUMMARY_DIR}/PAH_GWAS_K30_program_burden_review.xlsx" \
    "${SUMMARY_DIR}/PAH_GWAS_program_burden_review.pdf" \
    "${SUMMARY_DIR}/PAH_GWAS_program_driver_genes.tsv" \
    "${SUMMARY_DIR}/HOW_TO_INTERPRET.txt" \
    "${SUMMARY_DIR}/ANALYSIS_COMPLETE.json"
do
    [[ -s "${required}" ]] || {
        echo "Missing final output: ${required}" >&2
        exit 1
    }
done

echo
echo "PAH GWAS PROGRAM-TEST RESUME COMPLETED"
echo "${SUMMARY_DIR}/PAH_GWAS_K30_program_burden_review.xlsx"
