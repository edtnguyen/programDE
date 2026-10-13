#!/usr/bin/env bash
#SBATCH --job-name=IPGV_K30_human
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_external_validation_20260620"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"

FINAL_RUN="${WORK_ROOT}/results/IP_GV_skcnmf_K30_dt2_0_n100"
FINAL_SPECTRA="${FINAL_RUN}/Inference/Inference.gene_spectra_score.k_30.dt_2_0.txt"
PREPARED_H5AD="${WORK_ROOT}/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"
CRT_IP="${FINAL_RUN}/Evaluation/30_2_0/CRT_notebook/discovery/rna_qc_guides/IP_GFP/target_program_CRT.tsv"

# Reuse the already downloaded and processed human cohort from the K sweep.
SHARED_HUMAN_ROOT="${WORK_ROOT}/pilot_array_n20/external_validation_human_PAH"
RESOURCE_DIR="${SHARED_HUMAN_ROOT}/resources"
HUMAN_DIR="${SHARED_HUMAN_ROOT}/GSE185479/processed"

ORTHOLOGS="${RESOURCE_DIR}/HOM_ProteinCoding.rpt"
HUMAN_LOGCPM="${HUMAN_DIR}/GSE185479_pseudobulk_logCPM.tsv.gz"
HUMAN_METADATA="${HUMAN_DIR}/GSE185479_pseudobulk_metadata.tsv"
HUMAN_DE="${HUMAN_DIR}/GSE185479_edgeR_PAH_vs_healthy.tsv.gz"
HUMAN_PREP_MARKER="${HUMAN_DIR}/HUMAN_DATA_PREPARATION_COMPLETE.txt"

OUTPUT_ROOT="${FINAL_RUN}/Interpretation/Human_validation_GSE185479"
VALIDATION_DIR="${OUTPUT_ROOT}/model"
SUMMARY_DIR="${OUTPUT_ROOT}/summary"
MARKER="${OUTPUT_ROOT}/FINAL_HUMAN_VALIDATION_COMPLETE.txt"

CODE_DIR="${PAH_FINAL_CODE_DIR:?PAH_FINAL_CODE_DIR is required. Submit with the helper.}"
PIPELINE="${CODE_DIR}/pah_external_validation.py"
SUMMARY_SCRIPT="${CODE_DIR}/summarize_final_K30_dt2_human_validation.py"

for required in \
    "${ENV}/bin/python" \
    "${PIPELINE}" \
    "${SUMMARY_SCRIPT}" \
    "${HUMAN_PREP_MARKER}" \
    "${FINAL_SPECTRA}" \
    "${PREPARED_H5AD}" \
    "${CRT_IP}" \
    "${ORTHOLOGS}" \
    "${HUMAN_LOGCPM}" \
    "${HUMAN_METADATA}" \
    "${HUMAN_DE}"
do
    [[ -s "${required}" || -x "${required}" ]] || {
        echo "Missing required input: ${required}" >&2
        exit 1
    }
done

mkdir -p "${VALIDATION_DIR}" "${SUMMARY_DIR}" "${OUTPUT_ROOT}/tmp"
export TMPDIR="${OUTPUT_ROOT}/tmp/${SLURM_JOB_ID:-interactive}"
mkdir -p "${TMPDIR}"

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "Final K30/dt2.0 mouse-to-human PAH validation"
echo "Started:         $(date)"
echo "Job ID:          ${SLURM_JOB_ID:-interactive}"
echo "Node:            ${SLURMD_NODENAME:-$(hostname)}"
echo "Final spectra:   ${FINAL_SPECTRA}"
echo "Human dataset:   GSE185479"
echo "Validation dir:  ${VALIDATION_DIR}"
echo "Summary dir:     ${SUMMARY_DIR}"
echo "============================================================"

python "${PIPELINE}" validate-model \
    --spectra "${FINAL_SPECTRA}" \
    --k 30 \
    --threshold 2.0 \
    --orthologs "${ORTHOLOGS}" \
    --prepared-h5ad "${PREPARED_H5AD}" \
    --human-logcpm "${HUMAN_LOGCPM}" \
    --human-metadata "${HUMAN_METADATA}" \
    --human-de "${HUMAN_DE}" \
    --output-dir "${VALIDATION_DIR}" \
    --top-n 300 \
    --min-genes 15 \
    --gsea-permutations 5000 \
    --threads "${SLURM_CPUS_PER_TASK:-4}" \
    --seed 14

python "${SUMMARY_SCRIPT}" \
    --validation-dir "${VALIDATION_DIR}" \
    --crt-ip "${CRT_IP}" \
    --output-dir "${SUMMARY_DIR}" \
    --model-label "IP_GV_K30_dt2_0_n100" \
    --storey-fdr 0.10 \
    --seed 1729 \
    --overwrite

REQUIRED_OUTPUTS=(
    "${VALIDATION_DIR}/external_validation_summary.tsv"
    "${VALIDATION_DIR}/program_PAH_preranked_GSEA.tsv"
    "${VALIDATION_DIR}/program_donor_score_effects.tsv"
    "${SUMMARY_DIR}/K30_dt2_human_validation_review.xlsx"
    "${SUMMARY_DIR}/K30_dt2_human_validation_review.pdf"
    "${SUMMARY_DIR}/IP_q10_target_program_human_support.tsv"
)

for required in "${REQUIRED_OUTPUTS[@]}"; do
    [[ -s "${required}" ]] || {
        echo "Missing final validation output: ${required}" >&2
        exit 1
    }
done

cat > "${MARKER}" <<EOF
Final K30/dt2.0 human PAH validation completed successfully.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
final_spectra=${FINAL_SPECTRA}
human_dataset=GSE185479
validation_dir=${VALIDATION_DIR}
summary_dir=${SUMMARY_DIR}
EOF

echo
echo "FINAL HUMAN VALIDATION COMPLETED"
echo "Workbook:"
echo "  ${SUMMARY_DIR}/K30_dt2_human_validation_review.xlsx"
echo "Review PDF:"
echo "  ${SUMMARY_DIR}/K30_dt2_human_validation_review.pdf"
echo "Focused target-program table:"
echo "  ${SUMMARY_DIR}/IP_q10_target_program_human_support.tsv"
