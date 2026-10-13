#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Submission failed at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

PREPARE_SCRIPT="${SCRIPT_DIR}/01_prepare_GSE214742.sbatch"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/02_analyze_SOX17_programs.sbatch"
PYTHON_SCRIPT="${SCRIPT_DIR}/sox17_mechanism.py"
R_SCRIPT="${SCRIPT_DIR}/run_edger_GSE214742.R"
METADATA="${SCRIPT_DIR}/GSE214742_samples.tsv"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/sox17_mechanism_20260622"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/SOX17_mechanism_GSE214742"
RESOURCE_MARKER="${ROOT}/resources/RESOURCE_DOWNLOAD_COMPLETE.txt"
CODE_DIR="${ROOT}/code"
LOG_DIR="${ROOT}/logs"

[[ -x "${ENV}/bin/python" ]] || {
    echo "Environment is missing: ${ENV}" >&2
    echo "Run setup_sox17_mechanism_env_mamba.sh first." >&2
    exit 1
}
[[ -s "${RESOURCE_MARKER}" ]] || {
    echo "Resource marker is missing: ${RESOURCE_MARKER}" >&2
    echo "Run 00_download_GSE214742_and_MGI.sh first." >&2
    exit 1
}

for required in \
    "${PREPARE_SCRIPT}" \
    "${ANALYSIS_SCRIPT}" \
    "${PYTHON_SCRIPT}" \
    "${R_SCRIPT}" \
    "${METADATA}"
do
    [[ -r "${required}" ]] || {
        echo "Missing bundle file: ${required}" >&2
        exit 1
    }
done

mkdir -p "${CODE_DIR}" "${LOG_DIR}"

install -m 0755 "${PYTHON_SCRIPT}" "${CODE_DIR}/sox17_mechanism.py"
install -m 0755 "${R_SCRIPT}" "${CODE_DIR}/run_edger_GSE214742.R"
install -m 0644 "${METADATA}" "${CODE_DIR}/GSE214742_samples.tsv"

sha256sum \
    "${CODE_DIR}/sox17_mechanism.py" \
    "${CODE_DIR}/run_edger_GSE214742.R" \
    "${CODE_DIR}/GSE214742_samples.tsv" \
    > "${CODE_DIR}/SHA256SUMS.txt"

submit_job() {
    local raw
    raw="$(sbatch --parsable "$@")"
    printf '%s\n' "${raw%%;*}"
}

PREP_JOB="$(
    submit_job \
        --export="ALL,SOX17_CODE_DIR=${CODE_DIR}" \
        --output="${LOG_DIR}/prepare_%j.out" \
        --error="${LOG_DIR}/prepare_%j.err" \
        "${PREPARE_SCRIPT}"
)"

ANALYSIS_JOB="$(
    submit_job \
        --dependency="afterok:${PREP_JOB}" \
        --export="ALL,SOX17_CODE_DIR=${CODE_DIR}" \
        --output="${LOG_DIR}/analysis_%j.out" \
        --error="${LOG_DIR}/analysis_%j.err" \
        "${ANALYSIS_SCRIPT}"
)"

cat <<EOF

Submitted SOX17 variant-to-program workflow
-------------------------------------------
Preparation job: ${PREP_JOB}
Analysis job:    ${ANALYSIS_JOB}

Logs:
  ${LOG_DIR}/prepare_${PREP_JOB}.out
  ${LOG_DIR}/prepare_${PREP_JOB}.err
  ${LOG_DIR}/analysis_${ANALYSIS_JOB}.out
  ${LOG_DIR}/analysis_${ANALYSIS_JOB}.err

Final review:
  ${ROOT}/analysis/SOX17_variant_to_program_to_PAH_review.xlsx
  ${ROOT}/analysis/SOX17_variant_to_program_to_PAH_review.pdf

Monitor:
  squeue -j ${PREP_JOB},${ANALYSIS_JOB}
EOF
