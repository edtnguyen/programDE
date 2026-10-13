#!/usr/bin/env bash
#SBATCH --job-name=SOX17_programs
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/sox17_mechanism_20260622"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
FINAL_RUN="${WORK_ROOT}/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/SOX17_mechanism_GSE214742"

RESOURCE_DIR="${ROOT}/resources"
PROCESSED_DIR="${ROOT}/GSE214742/processed"
DE_DIR="${ROOT}/GSE214742/limma"
OUTPUT_DIR="${ROOT}/analysis"

SPECTRA="${FINAL_RUN}/Inference/Inference.gene_spectra_score.k_30.dt_2_0.txt"
PREPARED_H5AD="${WORK_ROOT}/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"
MOUSE_CRT_IP="${FINAL_RUN}/Evaluation/30_2_0/CRT_notebook/discovery/rna_qc_guides/IP_GFP/target_program_CRT.tsv"
HUMAN_PAH_PROGRAMS="${FINAL_RUN}/Interpretation/Human_validation_GSE185479/summary/K30_dt2_human_validation_all_programs.tsv"
HUMAN_PAH_DE="${WORK_ROOT}/pilot_array_n20/external_validation_human_PAH/GSE185479/processed/GSE185479_edgeR_PAH_vs_healthy.tsv.gz"

CODE_DIR="${SOX17_CODE_DIR:?SOX17_CODE_DIR is required. Submit with the helper.}"
PIPELINE="${CODE_DIR}/sox17_mechanism.py"

for required in \
    "${ENV}/bin/python" \
    "${ROOT}/GSE214742_PREP_AND_DE_COMPLETE.txt" \
    "${PIPELINE}" \
    "${SPECTRA}" \
    "${PREPARED_H5AD}" \
    "${MOUSE_CRT_IP}" \
    "${HUMAN_PAH_PROGRAMS}" \
    "${RESOURCE_DIR}/HOM_ProteinCoding.rpt" \
    "${PROCESSED_DIR}/GSE214742_metadata.tsv" \
    "${PROCESSED_DIR}/GSE214742_log2_abundance_per_million.tsv.gz"
do
    [[ -s "${required}" || -x "${required}" ]] || {
        echo "Missing required input: ${required}" >&2
        exit 1
    }
done

mkdir -p "${OUTPUT_DIR}" "${ROOT}/tmp"
export TMPDIR="${ROOT}/tmp/analysis_${SLURM_JOB_ID:-interactive}"
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
echo "SOX17 variant-to-program-to-PAH analysis"
echo "Started: $(date)"
echo "Job ID:  ${SLURM_JOB_ID:-interactive}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "Output:  ${OUTPUT_DIR}"
echo "============================================================"

ARGS=(
    analyze
    --spectra "${SPECTRA}"
    --k 30
    --mgi-report "${RESOURCE_DIR}/HOM_ProteinCoding.rpt"
    --prepared-h5ad "${PREPARED_H5AD}"
    --metadata "${PROCESSED_DIR}/GSE214742_metadata.tsv"
    --logcpm "${PROCESSED_DIR}/GSE214742_log2_abundance_per_million.tsv.gz"
    --de-dir "${DE_DIR}"
    --mouse-crt-ip "${MOUSE_CRT_IP}"
    --human-pah-programs "${HUMAN_PAH_PROGRAMS}"
    --top-n 300
    --min-genes 15
    --gsea-permutations 5000
    --threads "${SLURM_CPUS_PER_TASK:-8}"
    --seed 14
    --output-dir "${OUTPUT_DIR}"
)

if [[ -s "${HUMAN_PAH_DE}" ]]; then
    ARGS+=(--human-pah-de "${HUMAN_PAH_DE}")
fi

python "${PIPELINE}" "${ARGS[@]}"

for required in \
    "${OUTPUT_DIR}/SOX17_variant_to_program_to_PAH_review.xlsx" \
    "${OUTPUT_DIR}/SOX17_variant_to_program_to_PAH_review.pdf" \
    "${OUTPUT_DIR}/SOX17_directional_program_chain.tsv" \
    "${OUTPUT_DIR}/SOX17_expression_QC.tsv" \
    "${OUTPUT_DIR}/ANALYSIS_COMPLETE.json"
do
    [[ -s "${required}" ]] || {
        echo "Missing SOX17 output: ${required}" >&2
        exit 1
    }
done

echo
echo "SOX17 MECHANISM ANALYSIS COMPLETED"
echo "Workbook:"
echo "  ${OUTPUT_DIR}/SOX17_variant_to_program_to_PAH_review.xlsx"
echo "Review PDF:"
echo "  ${OUTPUT_DIR}/SOX17_variant_to_program_to_PAH_review.pdf"
echo "Directional table:"
echo "  ${OUTPUT_DIR}/SOX17_directional_program_chain.tsv"
