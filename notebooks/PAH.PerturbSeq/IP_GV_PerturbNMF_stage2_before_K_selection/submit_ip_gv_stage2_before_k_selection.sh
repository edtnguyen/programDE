#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Submission failed at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

PREFLIGHT="${SCRIPT_DIR}/01_stage2_preflight.sbatch"
COMPILE="${SCRIPT_DIR}/02_stage2_compile_models_array.sbatch"
METRICS="${SCRIPT_DIR}/03_stage2_metrics_array.sbatch"
SUMMARY="${SCRIPT_DIR}/04_stage2_summarize_before_k_selection.sbatch"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_stage2_20260620"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
STAGE2_ROOT="${BASE}/PerturbNMF/pilot_array_n20/stage2_pre_k_selection"
LOG_DIR="${STAGE2_ROOT}/logs"

MAX_COMPILE_CONCURRENT="${MAX_COMPILE_CONCURRENT:-4}"
MAX_METRICS_CONCURRENT="${MAX_METRICS_CONCURRENT:-4}"

[[ -x "${ENV}/bin/python" ]] || {
    echo "Stage 2 environment is missing: ${ENV}" >&2
    echo "First run: bash setup_ip_gv_perturbnmf_stage2_env_mamba.sh" >&2
    exit 1
}

for script in "${PREFLIGHT}" "${COMPILE}" "${METRICS}" "${SUMMARY}"; do
    [[ -r "${script}" ]] || {
        echo "Missing script: ${script}" >&2
        exit 1
    }
done

for value in "${MAX_COMPILE_CONCURRENT}" "${MAX_METRICS_CONCURRENT}"; do
    if ! [[ "${value}" =~ ^([1-9]|1[0-6])$ ]]; then
        echo "Concurrency caps must be integers from 1 through 16." >&2
        exit 2
    fi
done

mkdir -p "${LOG_DIR}"

submit_job() {
    local raw
    raw="$(sbatch --parsable "$@")"
    printf '%s\n' "${raw%%;*}"
}

echo "Submitting Stage 2 preflight..."
PREFLIGHT_JOB="$(
    submit_job \
        --output="${LOG_DIR}/preflight_%j.out" \
        --error="${LOG_DIR}/preflight_%j.err" \
        "${PREFLIGHT}"
)"

echo "Submitting 16-model compile array..."
COMPILE_JOB="$(
    submit_job \
        --dependency="afterok:${PREFLIGHT_JOB}" \
        --array="0-15%${MAX_COMPILE_CONCURRENT}" \
        --output="${LOG_DIR}/compile_%A_%a.out" \
        --error="${LOG_DIR}/compile_%A_%a.err" \
        "${COMPILE}"
)"

echo "Submitting 16-model Stage 2 metrics array..."
METRICS_JOB="$(
    submit_job \
        --dependency="afterok:${COMPILE_JOB}" \
        --array="0-15%${MAX_METRICS_CONCURRENT}" \
        --output="${LOG_DIR}/metrics_%A_%a.out" \
        --error="${LOG_DIR}/metrics_%A_%a.err" \
        "${METRICS}"
)"

echo "Submitting cross-K summary job..."
SUMMARY_JOB="$(
    submit_job \
        --dependency="afterok:${METRICS_JOB}" \
        --output="${LOG_DIR}/summary_%j.out" \
        --error="${LOG_DIR}/summary_%j.err" \
        "${SUMMARY}"
)"

cat <<EOF

Submitted Stage 2A evaluation before K selection
------------------------------------------------
Preflight job: ${PREFLIGHT_JOB}
Compile array: ${COMPILE_JOB}  (0-15%${MAX_COMPILE_CONCURRENT})
Metrics array: ${METRICS_JOB}  (0-15%${MAX_METRICS_CONCURRENT})
Summary job:   ${SUMMARY_JOB}

Array mapping:
  task = 2 * K_index + threshold_index

  0  K=5   dt=0.2       1  K=5   dt=2.0
  2  K=10  dt=0.2       3  K=10  dt=2.0
  4  K=15  dt=0.2       5  K=15  dt=2.0
  6  K=20  dt=0.2       7  K=20  dt=2.0
  8  K=25  dt=0.2       9  K=25  dt=2.0
  10 K=30  dt=0.2       11 K=30  dt=2.0
  12 K=40  dt=0.2       13 K=40  dt=2.0
  14 K=50  dt=0.2       15 K=50  dt=2.0

Logs:
  ${LOG_DIR}

Monitor:
  squeue -j ${PREFLIGHT_JOB},${COMPILE_JOB},${METRICS_JOB},${SUMMARY_JOB}
  squeue -r -j ${COMPILE_JOB}
  squeue -r -j ${METRICS_JOB}

Final review:
  ${STAGE2_ROOT}/summary_before_k_selection/pre_K_selection_review.pdf
EOF
