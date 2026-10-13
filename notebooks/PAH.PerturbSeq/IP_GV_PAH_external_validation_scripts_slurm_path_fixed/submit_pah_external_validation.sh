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
CODE_DIR="${EXTERNAL_ROOT}/code"

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

# Stage non-SBATCH support files in a persistent analysis directory. Slurm
# copies batch scripts into /var/spool/slurmd, so sibling-file lookups from
# BASH_SOURCE are not reliable inside a running job.
mkdir -p "${LOG_DIR}" "${CODE_DIR}"
install -m 0755 "${SCRIPT_DIR}/pah_external_validation.py" \
    "${CODE_DIR}/pah_external_validation.py"
install -m 0755 "${SCRIPT_DIR}/run_edger_GSE185479.R" \
    "${CODE_DIR}/run_edger_GSE185479.R"
install -m 0644 "${SCRIPT_DIR}/GSE185479_samples.tsv" \
    "${CODE_DIR}/GSE185479_samples.tsv"

for required_code in \
    "${CODE_DIR}/pah_external_validation.py" \
    "${CODE_DIR}/run_edger_GSE185479.R" \
    "${CODE_DIR}/GSE185479_samples.tsv"
do
    [[ -s "${required_code}" ]] || {
        echo "Failed to stage support file: ${required_code}" >&2
        exit 1
    }
done

sha256sum \
    "${CODE_DIR}/pah_external_validation.py" \
    "${CODE_DIR}/run_edger_GSE185479.R" \
    "${CODE_DIR}/GSE185479_samples.tsv" \
    > "${CODE_DIR}/SHA256SUMS.txt"

submit_job() {
    local raw
    raw="$(sbatch --parsable "$@")"
    printf '%s\n' "${raw%%;*}"
}

PREP_JOB="$(
    submit_job \
        --export=ALL,PAH_EXT_CODE_DIR="${CODE_DIR}" \
        --output="${LOG_DIR}/prepare_%j.out" \
        --error="${LOG_DIR}/prepare_%j.err" \
        "${PREPARE}"
)"

ARRAY_JOB="$(
    submit_job \
        --dependency="afterok:${PREP_JOB}" \
        --export=ALL,PAH_EXT_CODE_DIR="${CODE_DIR}" \
        --array="0-15%${MAX_CONCURRENT}" \
        --output="${LOG_DIR}/model_%A_%a.out" \
        --error="${LOG_DIR}/model_%A_%a.err" \
        "${VALIDATE}"
)"

SUMMARY_JOB="$(
    submit_job \
        --dependency="afterok:${ARRAY_JOB}" \
        --export=ALL,PAH_EXT_CODE_DIR="${CODE_DIR}" \
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

Staged code:
  ${CODE_DIR}

Logs:
  ${LOG_DIR}

Monitor:
  squeue -j ${PREP_JOB},${ARRAY_JOB},${SUMMARY_JOB}
  squeue -r -j ${ARRAY_JOB}

Final review:
  ${EXTERNAL_ROOT}/summary_before_K_selection/
      human_PAH_external_validation_review.pdf
EOF
