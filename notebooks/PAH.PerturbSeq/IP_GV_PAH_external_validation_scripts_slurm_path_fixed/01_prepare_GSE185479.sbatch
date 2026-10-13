#!/usr/bin/env bash
#SBATCH --job-name=IPGV_ext_prepare
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_external_validation_20260620"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
EXTERNAL_ROOT="${BASE}/PerturbNMF/pilot_array_n20/external_validation_human_PAH"
RESOURCE_DIR="${EXTERNAL_ROOT}/resources"
RAW_DIR="${EXTERNAL_ROOT}/GSE185479/raw"
HUMAN_DIR="${EXTERNAL_ROOT}/GSE185479/processed"
LOG_DIR="${EXTERNAL_ROOT}/logs"

# Slurm executes a copied batch script from /var/spool/slurmd. Support
# files therefore live in a persistent directory staged by the submit wrapper.
CODE_DIR="${PAH_EXT_CODE_DIR:-${EXTERNAL_ROOT}/code}"
PY_SCRIPT="${CODE_DIR}/pah_external_validation.py"
R_SCRIPT="${CODE_DIR}/run_edger_GSE185479.R"
SAMPLE_METADATA="${CODE_DIR}/GSE185479_samples.tsv"

RESOURCE_MARKER="${RESOURCE_DIR}/RESOURCE_DOWNLOAD_COMPLETE.txt"
MARKER="${HUMAN_DIR}/HUMAN_DATA_PREPARATION_COMPLETE.txt"

for required_code in "${PY_SCRIPT}" "${R_SCRIPT}" "${SAMPLE_METADATA}"; do
    [[ -s "${required_code}" ]] || {
        echo "Missing staged support file: ${required_code}" >&2
        exit 1
    }
done

[[ -s "${RESOURCE_MARKER}" ]] || {
    echo "Resource marker is missing: ${RESOURCE_MARKER}" >&2
    echo "Run 00_download_GSE185479_and_MGI.sh first." >&2
    exit 1
}

mkdir -p "${HUMAN_DIR}" "${LOG_DIR}" "${EXTERNAL_ROOT}/tmp"
export TMPDIR="${EXTERNAL_ROOT}/tmp/prepare_${SLURM_JOB_ID:-interactive}"
mkdir -p "${TMPDIR}"

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "============================================================"
echo "Prepare GSE185479 human PAEC external-validation data"
echo "Started:     $(date)"
echo "Job ID:      ${SLURM_JOB_ID:-interactive}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Raw dir:     ${RAW_DIR}"
echo "Output dir:  ${HUMAN_DIR}"
echo "============================================================"

python "${PY_SCRIPT}" prepare-gse \
    --raw-dir "${RAW_DIR}" \
    --sample-metadata "${SAMPLE_METADATA}" \
    --output-dir "${HUMAN_DIR}" \
    --min-genes 200 \
    --min-counts 500 \
    --max-pct-mt 20 \
    --min-cells-per-gene 3

Rscript "${R_SCRIPT}" \
    "${HUMAN_DIR}/GSE185479_pseudobulk_counts.tsv.gz" \
    "${HUMAN_DIR}/GSE185479_pseudobulk_metadata.tsv" \
    "${HUMAN_DIR}/GSE185479_edgeR_PAH_vs_healthy.tsv.gz" \
    "${HUMAN_DIR}/GSE185479_pseudobulk_logCPM.tsv.gz" \
    "${HUMAN_DIR}"

for required in \
    "${HUMAN_DIR}/GSE185479_PAEC_QC_counts.h5ad" \
    "${HUMAN_DIR}/GSE185479_PAEC_log1p.h5ad" \
    "${HUMAN_DIR}/GSE185479_pseudobulk_counts.tsv.gz" \
    "${HUMAN_DIR}/GSE185479_pseudobulk_logCPM.tsv.gz" \
    "${HUMAN_DIR}/GSE185479_edgeR_PAH_vs_healthy.tsv.gz" \
    "${HUMAN_DIR}/GSE185479_edgeR_MDS.pdf"
do
    [[ -s "${required}" ]] || {
        echo "Missing expected preparation output: ${required}" >&2
        exit 1
    }
done

cat > "${MARKER}" <<EOF
Human external-validation data prepared successfully.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
dataset=GSE185479
processed_dir=${HUMAN_DIR}
EOF

echo
echo "HUMAN DATA PREPARATION COMPLETED"
echo "Marker: ${MARKER}"
