#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Submission failed at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

PREPARE="${SCRIPT_DIR}/01_prepare_GSE185479.sbatch"
VALIDATE="${SCRIPT_DIR}/02_validate_all_models_array.sbatch"
SUMMARY="${SCRIPT_DIR}/03_summarize_external_validation.sbatch"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_external_validation_20260620"
BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
EXTERNAL_ROOT="${BASE}/PerturbNMF/pilot_array_n20/external_validation_human_PAH"
RESOURCE_MARKER="${EXTERNAL_ROOT}/resources/RESOURCE_DOWNLOAD_COMPLETE.txt"
LOG_DIR="${EXTERNAL_ROOT}/logs"

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

[[ -x "${ENV}/bin/python" ]] || {
    echo "External-validation environment is missing: ${ENV}" >&2
    echo "Run setup_pah_external_validation_env_mamba.sh first." >&2
    exit 1
}
[[ -s "${RESOURCE_MARKER}" ]] || {
    echo "Downloaded-resource marker is missing: ${RESOURCE_MARKER}" >&2
    echo "Run 00_download_GSE185479_and_MGI.sh first." >&2
    exit 1
}
if ! [[ "${MAX_CONCURRENT}" =~ ^([1-9]|1[0-6])$ ]]; then
    echo "MAX_CONCURRENT must be 1 through 16." >&2
    exit 2
fi

for script in "${PREPARE}" "${VALIDATE}" "${SUMMARY}"; do
    [[ -r "${script}" ]] || {
        echo "Missing script: ${script}" >&2
        exit 1
    }
done

mkdir -p "${LOG_DIR}"

submit_job() {
    local raw
    raw="$(sbatch --parsable "$@")"
    printf '%s\n' "${raw%%;*}"
}

PREP_JOB="$(
    submit_job \
        --output="${LOG_DIR}/prepare_%j.out" \
        --error="${LOG_DIR}/prepare_%j.err" \
        "${PREPARE}"
)"

ARRAY_JOB="$(
    submit_job \
        --dependency="afterok:${PREP_JOB}" \
        --array="0-15%${MAX_CONCURRENT}" \
        --output="${LOG_DIR}/model_%A_%a.out" \
        --error="${LOG_DIR}/model_%A_%a.err" \
        "${VALIDATE}"
)"

SUMMARY_JOB="$(
    submit_job \
        --dependency="afterok:${ARRAY_JOB}" \
        --output="${LOG_DIR}/summary_%j.out" \
        --error="${LOG_DIR}/summary_%j.err" \
        "${SUMMARY}"
)"

cat <<EOF

Submitted human PAH external validation
---------------------------------------
GSE preparation: ${PREP_JOB}
Model array:     ${ARRAY_JOB}  (0-15%${MAX_CONCURRENT})
Summary:         ${SUMMARY_JOB}

Logs:
  ${LOG_DIR}

Monitor:
  squeue -j ${PREP_JOB},${ARRAY_JOB},${SUMMARY_JOB}
  squeue -r -j ${ARRAY_JOB}

Final review:
  ${EXTERNAL_ROOT}/summary_before_K_selection/
      human_PAH_external_validation_review.pdf
EOF
