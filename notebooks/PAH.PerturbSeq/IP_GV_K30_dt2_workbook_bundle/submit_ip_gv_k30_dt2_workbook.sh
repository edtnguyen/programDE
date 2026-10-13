#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_ip_gv_k30_dt2_workbook.sbatch"

RUN_DIR="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
LOG_DIR="${RUN_DIR}/Interpretation/Summary/K30_dt2_0/logs"

mkdir -p "${LOG_DIR}"

# Pass the original bundle directory explicitly. SLURM otherwise runs a
# temporary copy of the SBATCH script from /var/spool/slurmd.
cd "${SCRIPT_DIR}"

raw="$(
    sbatch --parsable \
        --chdir="${SCRIPT_DIR}" \
        --export="ALL,WORKBOOK_SCRIPT_DIR=${SCRIPT_DIR}" \
        --output="${LOG_DIR}/summary_%j.out" \
        --error="${LOG_DIR}/summary_%j.err" \
        "${SBATCH_SCRIPT}"
)"
job_id="${raw%%;*}"

echo "Submitted workbook job: ${job_id}"
echo "Logs:"
echo "  ${LOG_DIR}/summary_${job_id}.out"
echo "  ${LOG_DIR}/summary_${job_id}.err"
