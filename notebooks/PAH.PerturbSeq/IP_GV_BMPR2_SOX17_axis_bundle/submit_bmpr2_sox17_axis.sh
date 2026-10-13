#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Submission failed at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

PREP="${SCRIPT_DIR}/01_prepare_BMPR2_datasets.sbatch"
ANALYSIS="${SCRIPT_DIR}/02_analyze_BMPR2_SOX17_axis.sbatch"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/bmpr2_sox17_axis_20260623"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/BMPR2_SOX17_axis"
RESOURCE_MARKER="${ROOT}/resources/RESOURCE_DOWNLOAD_COMPLETE.txt"
CODE_DIR="${ROOT}/code"
LOG_DIR="${ROOT}/logs"

[[ -x "${ENV}/bin/python" ]] || {
    echo "Environment is missing: ${ENV}" >&2
    echo "Run setup_bmpr2_sox17_axis_mamba.sh first." >&2
    exit 1
}
[[ -s "${RESOURCE_MARKER}" ]] || {
    echo "Resource marker is missing: ${RESOURCE_MARKER}" >&2
    echo "Run 00_download_BMPR2_GEO.sh first." >&2
    exit 1
}

mkdir -p "${CODE_DIR}" "${LOG_DIR}"

for source in \
    bmpr2_sox17_axis.py \
    prepare_GSE70456.R \
    prepare_GSE157231.R \
    GSE70456_samples.tsv \
    GSE157231_samples.tsv
do
    [[ -r "${SCRIPT_DIR}/${source}" ]] || {
        echo "Missing bundle file: ${source}" >&2
        exit 1
    }
done

install -m 0755 "${SCRIPT_DIR}/bmpr2_sox17_axis.py" \
    "${CODE_DIR}/bmpr2_sox17_axis.py"
install -m 0755 "${SCRIPT_DIR}/prepare_GSE70456.R" \
    "${CODE_DIR}/prepare_GSE70456.R"
install -m 0755 "${SCRIPT_DIR}/prepare_GSE157231.R" \
    "${CODE_DIR}/prepare_GSE157231.R"
install -m 0644 "${SCRIPT_DIR}/GSE70456_samples.tsv" \
    "${CODE_DIR}/GSE70456_samples.tsv"
install -m 0644 "${SCRIPT_DIR}/GSE157231_samples.tsv" \
    "${CODE_DIR}/GSE157231_samples.tsv"

sha256sum "${CODE_DIR}"/* > "${CODE_DIR}/SHA256SUMS.txt"

submit_job() {
    local raw
    raw="$(sbatch --parsable "$@")"
    printf '%s\n' "${raw%%;*}"
}

PREP_JOB="$(
    submit_job \
        --export="ALL,BMPR2_AXIS_CODE_DIR=${CODE_DIR}" \
        --output="${LOG_DIR}/prepare_%j.out" \
        --error="${LOG_DIR}/prepare_%j.err" \
        "${PREP}"
)"

ANALYSIS_JOB="$(
    submit_job \
        --dependency="afterok:${PREP_JOB}" \
        --export="ALL,BMPR2_AXIS_CODE_DIR=${CODE_DIR}" \
        --output="${LOG_DIR}/analysis_%j.out" \
        --error="${LOG_DIR}/analysis_%j.err" \
        "${ANALYSIS}"
)"

cat <<EOF

Submitted BMPR2–SOX17 directional-axis workflow
------------------------------------------------
Preparation job: ${PREP_JOB}
Analysis job:    ${ANALYSIS_JOB}

Logs:
  ${LOG_DIR}/prepare_${PREP_JOB}.out
  ${LOG_DIR}/prepare_${PREP_JOB}.err
  ${LOG_DIR}/analysis_${ANALYSIS_JOB}.out
  ${LOG_DIR}/analysis_${ANALYSIS_JOB}.err

Final review:
  ${ROOT}/analysis/BMPR2_SOX17_PAH_axis_review.xlsx
  ${ROOT}/analysis/BMPR2_SOX17_PAH_axis_review.pdf
EOF
