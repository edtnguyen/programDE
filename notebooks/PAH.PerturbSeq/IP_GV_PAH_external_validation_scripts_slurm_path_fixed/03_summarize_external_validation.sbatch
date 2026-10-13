#!/usr/bin/env bash
#SBATCH --job-name=IPGV_ext_summary
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_external_validation_20260620"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
EXTERNAL_ROOT="${BASE}/PerturbNMF/pilot_array_n20/external_validation_human_PAH"
VALIDATION_ROOT="${EXTERNAL_ROOT}/models"
SUMMARY_DIR="${EXTERNAL_ROOT}/summary_before_K_selection"

# Slurm executes a copied batch script from /var/spool/slurmd. Support
# files therefore live in a persistent directory staged by the submit wrapper.
CODE_DIR="${PAH_EXT_CODE_DIR:-${EXTERNAL_ROOT}/code}"
PY_SCRIPT="${CODE_DIR}/pah_external_validation.py"

[[ -s "${PY_SCRIPT}" ]] || {
    echo "Missing staged support file: ${PY_SCRIPT}" >&2
    exit 1
}

K_VALUES=(5 10 15 20 25 30 40 50)
THRESHOLDS=(0.2 2.0)

for K in "${K_VALUES[@]}"; do
    for THRESHOLD in "${THRESHOLDS[@]}"; do
        TAG="${THRESHOLD//./_}"
        MARKER="${VALIDATION_ROOT}/K${K}_dt${TAG}/MODEL_EXTERNAL_VALIDATION_COMPLETE.txt"
        [[ -s "${MARKER}" ]] || {
            echo "Missing model completion marker: ${MARKER}" >&2
            exit 1
        }
    done
done

mkdir -p "${SUMMARY_DIR}"

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

python "${PY_SCRIPT}" summarize \
    --validation-root "${VALIDATION_ROOT}" \
    --output-dir "${SUMMARY_DIR}" \
    --k-values "${K_VALUES[@]}" \
    --thresholds "${THRESHOLDS[@]}" \
    --top-programs-per-model 5

cat > "${SUMMARY_DIR}/EXTERNAL_VALIDATION_COMPLETE.txt" <<EOF
Human PAH external-validation summary completed successfully.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
validation_root=${VALIDATION_ROOT}
summary_dir=${SUMMARY_DIR}
K_values=${K_VALUES[*]}
thresholds=${THRESHOLDS[*]}
EOF

echo
echo "============================================================"
echo "EXTERNAL VALIDATION SUMMARY COMPLETED"
echo "Review PDF:"
echo "  ${SUMMARY_DIR}/human_PAH_external_validation_review.pdf"
echo "Workbook:"
echo "  ${SUMMARY_DIR}/human_PAH_external_validation_review.xlsx"
echo "============================================================"
