#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Submission failed at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

PREP_SCRIPT="${SCRIPT_DIR}/01_prepare_ip_gv_perturbnmf_array.sbatch"
ARRAY_SCRIPT="${SCRIPT_DIR}/02_run_ip_gv_perturbnmf_k_array.sbatch"
MERGE_SCRIPT="${SCRIPT_DIR}/03_merge_ip_gv_perturbnmf_array.sbatch"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
PILOT_ROOT="${BASE}/PerturbNMF/pilot_array_n20"
LOG_DIR="${PILOT_ROOT}/logs"

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
ARRAY_SPEC="0-7%${MAX_CONCURRENT}"

for script in "${PREP_SCRIPT}" "${ARRAY_SCRIPT}" "${MERGE_SCRIPT}"; do
    [[ -r "${script}" ]] || {
        echo "Missing script: ${script}" >&2
        exit 1
    }
done

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-8]$ ]]; then
    echo "MAX_CONCURRENT must be an integer from 1 through 8." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

submit_job() {
    local raw
    raw="$(sbatch --parsable "$@")"
    # Federated Slurm may return JOBID;CLUSTER. Dependencies need JOBID.
    printf '%s\n' "${raw%%;*}"
}

echo "Submitting preparation job..."
PREP_JOB_ID="$(
    submit_job \
        --output="${LOG_DIR}/prepare_%j.out" \
        --error="${LOG_DIR}/prepare_%j.err" \
        "${PREP_SCRIPT}"
)"

echo "Submitting K array after successful preparation..."
ARRAY_JOB_ID="$(
    submit_job \
        --dependency="afterok:${PREP_JOB_ID}" \
        --array="${ARRAY_SPEC}" \
        --output="${LOG_DIR}/K_%A_%a.out" \
        --error="${LOG_DIR}/K_%A_%a.err" \
        "${ARRAY_SCRIPT}"
)"

echo "Submitting merge/refit job after every array task succeeds..."
MERGE_JOB_ID="$(
    submit_job \
        --dependency="afterok:${ARRAY_JOB_ID}" \
        --output="${LOG_DIR}/merge_%j.out" \
        --error="${LOG_DIR}/merge_%j.err" \
        "${MERGE_SCRIPT}"
)"

cat <<EOF

Submitted IP/GV PerturbNMF array pilot
--------------------------------------
Preparation job: ${PREP_JOB_ID}
K array job:     ${ARRAY_JOB_ID}  (${ARRAY_SPEC})
Merge job:       ${MERGE_JOB_ID}

K mapping:
  task 0 -> K=5
  task 1 -> K=10
  task 2 -> K=15
  task 3 -> K=20
  task 4 -> K=25
  task 5 -> K=30
  task 6 -> K=40
  task 7 -> K=50

Logs:
  ${LOG_DIR}

Monitor:
  squeue -j ${PREP_JOB_ID},${ARRAY_JOB_ID},${MERGE_JOB_ID}
  squeue -r -j ${ARRAY_JOB_ID}

After completion:
  ${PILOT_ROOT}/merged/Inference/Inference.k_selection.png
EOF
