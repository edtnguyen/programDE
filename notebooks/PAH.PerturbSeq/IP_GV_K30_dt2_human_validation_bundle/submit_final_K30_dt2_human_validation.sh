#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Submission failed at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

RUN_SCRIPT="${SCRIPT_DIR}/run_final_K30_dt2_human_validation.sbatch"
PY_PIPELINE="${SCRIPT_DIR}/pah_external_validation.py"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_final_K30_dt2_human_validation.py"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_external_validation_20260620"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
FINAL_RUN="${WORK_ROOT}/results/IP_GV_skcnmf_K30_dt2_0_n100"
OUTPUT_ROOT="${FINAL_RUN}/Interpretation/Human_validation_GSE185479"
LOG_DIR="${OUTPUT_ROOT}/logs"
CODE_DIR="${OUTPUT_ROOT}/code"

HUMAN_PREP_MARKER="${WORK_ROOT}/pilot_array_n20/external_validation_human_PAH/GSE185479/processed/HUMAN_DATA_PREPARATION_COMPLETE.txt"
RESOURCE_MARKER="${WORK_ROOT}/pilot_array_n20/external_validation_human_PAH/resources/RESOURCE_DOWNLOAD_COMPLETE.txt"

for required in \
    "${RUN_SCRIPT}" \
    "${PY_PIPELINE}" \
    "${SUMMARY_SCRIPT}"
do
    [[ -r "${required}" ]] || {
        echo "Missing bundle file: ${required}" >&2
        exit 1
    }
done

[[ -x "${ENV}/bin/python" ]] || {
    echo "Human-validation environment is missing: ${ENV}" >&2
    echo "Run setup_pah_external_validation_env_mamba.sh from the earlier bundle." >&2
    exit 1
}

[[ -s "${RESOURCE_MARKER}" && -s "${HUMAN_PREP_MARKER}" ]] || {
    echo "The shared GSE185479 resources/processed data are missing." >&2
    echo "Run the earlier download and preparation workflow first:" >&2
    echo "  bash 00_download_GSE185479_and_MGI.sh" >&2
    echo "  sbatch 01_prepare_GSE185479.sbatch" >&2
    exit 1
}

mkdir -p "${LOG_DIR}" "${CODE_DIR}"

install -m 0755 "${PY_PIPELINE}" \
    "${CODE_DIR}/pah_external_validation.py"
install -m 0755 "${SUMMARY_SCRIPT}" \
    "${CODE_DIR}/summarize_final_K30_dt2_human_validation.py"

sha256sum \
    "${CODE_DIR}/pah_external_validation.py" \
    "${CODE_DIR}/summarize_final_K30_dt2_human_validation.py" \
    > "${CODE_DIR}/SHA256SUMS.txt"

raw="$(
    sbatch --parsable \
        --export="ALL,PAH_FINAL_CODE_DIR=${CODE_DIR}" \
        --output="${LOG_DIR}/human_validation_%j.out" \
        --error="${LOG_DIR}/human_validation_%j.err" \
        "${RUN_SCRIPT}"
)"
job_id="${raw%%;*}"

cat <<EOF

Submitted final K30/dt2.0 human validation
-----------------------------------------
Job ID: ${job_id}

Logs:
  ${LOG_DIR}/human_validation_${job_id}.out
  ${LOG_DIR}/human_validation_${job_id}.err

Output:
  ${OUTPUT_ROOT}/summary/K30_dt2_human_validation_review.xlsx
  ${OUTPUT_ROOT}/summary/K30_dt2_human_validation_review.pdf

Monitor:
  squeue -j ${job_id}
EOF
