#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Submission failed at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

SBATCH_SCRIPT="${SCRIPT_DIR}/resume_pah_gwas_program_tests_v2.sbatch"
R_SCRIPT="${SCRIPT_DIR}/run_magma_program_tests_v2.R"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
OUTPUT_ROOT="${FINAL_RUN}/Interpretation/PAH_GWAS_program_burden/GCST007228"
CODE_DIR="${OUTPUT_ROOT}/code_resume_v2"
LOG_DIR="${OUTPUT_ROOT}/logs"

for required in "${SBATCH_SCRIPT}" "${R_SCRIPT}"; do
    [[ -r "${required}" ]] || {
        echo "Missing resume-v2 file: ${required}" >&2
        exit 1
    }
done

mkdir -p "${CODE_DIR}" "${LOG_DIR}"
install -m 0755 "${R_SCRIPT}" "${CODE_DIR}/run_magma_program_tests_v2.R"

sha256sum \
    "${CODE_DIR}/run_magma_program_tests_v2.R" \
    > "${CODE_DIR}/SHA256SUMS_program_test_v2.txt"

raw="$(
    sbatch --parsable \
        --export="ALL,PAH_GWAS_RESUME_V2_CODE_DIR=${CODE_DIR}" \
        --output="${LOG_DIR}/program_tests_v2_%j.out" \
        --error="${LOG_DIR}/program_tests_v2_%j.err" \
        "${SBATCH_SCRIPT}"
)"
job_id="${raw%%;*}"

cat <<EOF

Submitted MAGMA v1.10 program tests
-----------------------------------
Job ID: ${job_id}

Logs:
  ${LOG_DIR}/program_tests_v2_${job_id}.out
  ${LOG_DIR}/program_tests_v2_${job_id}.err

Final review:
  ${OUTPUT_ROOT}/summary/PAH_GWAS_K30_program_burden_review.xlsx

Monitor:
  squeue -j ${job_id}
EOF
