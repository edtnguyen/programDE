#!/usr/bin/env bash
#SBATCH --job-name=GSE214742_prep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/sox17_mechanism_20260622"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/SOX17_mechanism_GSE214742"
RESOURCE_DIR="${ROOT}/resources"
RAW_DIR="${ROOT}/GSE214742/raw"
PROCESSED_DIR="${ROOT}/GSE214742/processed"
DE_DIR="${ROOT}/GSE214742/edgeR"

CODE_DIR="${SOX17_CODE_DIR:?SOX17_CODE_DIR is required. Submit with the helper.}"
PIPELINE="${CODE_DIR}/sox17_mechanism.py"
R_SCRIPT="${CODE_DIR}/run_edger_GSE214742.R"
METADATA="${CODE_DIR}/GSE214742_samples.tsv"

for required in \
    "${ENV}/bin/python" \
    "${RESOURCE_DIR}/RESOURCE_DOWNLOAD_COMPLETE.txt" \
    "${RESOURCE_DIR}/gencode.v28.annotation.gtf.gz" \
    "${PIPELINE}" \
    "${R_SCRIPT}" \
    "${METADATA}"
do
    [[ -s "${required}" || -x "${required}" ]] || {
        echo "Missing required input: ${required}" >&2
        exit 1
    }
done

mkdir -p "${PROCESSED_DIR}" "${DE_DIR}" "${ROOT}/tmp"
export TMPDIR="${ROOT}/tmp/prepare_${SLURM_JOB_ID:-interactive}"
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
echo "Prepare GSE214742 and run paired edgeR contrasts"
echo "Started: $(date)"
echo "Job ID:  ${SLURM_JOB_ID:-interactive}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "Root:    ${ROOT}"
echo "============================================================"

python "${PIPELINE}" prepare \
    --raw-dir "${RAW_DIR}" \
    --metadata "${METADATA}" \
    --gencode-gtf "${RESOURCE_DIR}/gencode.v28.annotation.gtf.gz" \
    --output-dir "${PROCESSED_DIR}"

Rscript "${R_SCRIPT}" \
    "${PROCESSED_DIR}/GSE214742_gene_counts.tsv.gz" \
    "${PROCESSED_DIR}/GSE214742_metadata.tsv" \
    "${DE_DIR}"

for required in \
    "${PROCESSED_DIR}/GSE214742_PREPARATION_COMPLETE.json" \
    "${DE_DIR}/EDGER_COMPLETE.json" \
    "${DE_DIR}/GSE214742_all_samples_logCPM.tsv.gz" \
    "${DE_DIR}/siSOX17_vs_negativeSIRNA_edgeR.tsv.gz" \
    "${DE_DIR}/Promoter_Repression_vs_BFP_Repression_edgeR.tsv.gz" \
    "${DE_DIR}/Signal1_Repression_vs_BFP_Repression_edgeR.tsv.gz" \
    "${DE_DIR}/Promoter_Activation_vs_BFP_Activation_edgeR.tsv.gz"
do
    [[ -s "${required}" ]] || {
        echo "Missing preparation output: ${required}" >&2
        exit 1
    }
done

cat > "${ROOT}/GSE214742_PREP_AND_DE_COMPLETE.txt" <<EOF
GSE214742 preparation and edgeR completed.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
processed_dir=${PROCESSED_DIR}
edgeR_dir=${DE_DIR}
EOF

echo
echo "GSE214742 PREPARATION COMPLETED"
