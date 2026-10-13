#!/usr/bin/env bash
#SBATCH --job-name=rare_gene_review
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/rare_regulator_program_20260623"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/Rare_regulator_program_model"
REVIEW_DIR="${ROOT}/curation"

CRT_IP="${FINAL_RUN}/Evaluation/30_2_0/CRT_notebook/discovery/rna_qc_guides/IP_GFP/target_program_CRT.tsv"

CODE_DIR="${RARE_REGULATOR_CODE_DIR:?RARE_REGULATOR_CODE_DIR is required.}"
PIPELINE="${CODE_DIR}/rare_regulator_program_model.py"
CURATION="${CODE_DIR}/rare_variant_effects_reviewed.tsv"
PORTAL="${CODE_DIR}/portal_results_long.tsv"

for required in \
    "${ENV}/bin/python" \
    "${PIPELINE}" \
    "${CURATION}" \
    "${PORTAL}" \
    "${CRT_IP}"
do
    [[ -s "${required}" || -x "${required}" ]] || {
        echo "Missing required input: ${required}" >&2
        exit 1
    }
done

mkdir -p "${REVIEW_DIR}"

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

python "${PIPELINE}" prepare-review \
    --curation "${CURATION}" \
    --portal-template "${PORTAL}" \
    --crt-ip "${CRT_IP}" \
    --output-dir "${REVIEW_DIR}"

echo
echo "CURATION REVIEW COMPLETE"
echo "${REVIEW_DIR}/rare_variant_curation_review.xlsx"
echo
echo "Next:"
echo "1. Review/edit ${REVIEW_DIR}/rare_variant_effects_reviewed.tsv"
echo "2. Fill/select portal rows in ${REVIEW_DIR}/portal_results_long.tsv"
echo "3. Run the fitting job after at least six quantitative gamma effects exist."
