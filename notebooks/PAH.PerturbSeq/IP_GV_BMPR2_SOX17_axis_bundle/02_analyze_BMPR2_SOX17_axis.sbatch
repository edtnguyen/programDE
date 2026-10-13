#!/usr/bin/env bash
#SBATCH --job-name=BMPR2_SOX17_axis
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
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/bmpr2_sox17_axis_20260623"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
FINAL_RUN="${WORK_ROOT}/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/BMPR2_SOX17_axis"
OUTPUT_DIR="${ROOT}/analysis"

SPECTRA="${FINAL_RUN}/Inference/Inference.gene_spectra_score.k_30.dt_2_0.txt"
PREPARED_H5AD="${WORK_ROOT}/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"
MOUSE_CRT_IP="${FINAL_RUN}/Evaluation/30_2_0/CRT_notebook/discovery/rna_qc_guides/IP_GFP/target_program_CRT.tsv"
SOX17_RESULTS="${FINAL_RUN}/Interpretation/SOX17_mechanism_GSE214742/analysis/SOX17_directional_program_chain.tsv"
HUMAN_PAH_PROGRAMS="${FINAL_RUN}/Interpretation/Human_validation_GSE185479/summary/K30_dt2_human_validation_all_programs.tsv"

CODE_DIR="${BMPR2_AXIS_CODE_DIR:?BMPR2_AXIS_CODE_DIR is required.}"
PIPELINE="${CODE_DIR}/bmpr2_sox17_axis.py"

for required in \
    "${ENV}/bin/python" \
    "${ROOT}/BMPR2_DATA_PREPARATION_COMPLETE.txt" \
    "${PIPELINE}" \
    "${SPECTRA}" \
    "${PREPARED_H5AD}" \
    "${MOUSE_CRT_IP}" \
    "${SOX17_RESULTS}" \
    "${HUMAN_PAH_PROGRAMS}" \
    "${ROOT}/resources/HOM_ProteinCoding.rpt"
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

python "${PIPELINE}" \
    --spectra "${SPECTRA}" \
    --k 30 \
    --mgi-report "${ROOT}/resources/HOM_ProteinCoding.rpt" \
    --prepared-h5ad "${PREPARED_H5AD}" \
    --gse70456-dir "${ROOT}/GSE70456" \
    --gse157231-dir "${ROOT}/GSE157231" \
    --mouse-crt-ip "${MOUSE_CRT_IP}" \
    --sox17-results "${SOX17_RESULTS}" \
    --human-pah-programs "${HUMAN_PAH_PROGRAMS}" \
    --top-n 300 \
    --min-genes 15 \
    --gsea-permutations 5000 \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --seed 14 \
    --output-dir "${OUTPUT_DIR}"

for required in \
    "${OUTPUT_DIR}/BMPR2_SOX17_PAH_axis_review.xlsx" \
    "${OUTPUT_DIR}/BMPR2_SOX17_PAH_axis_review.pdf" \
    "${OUTPUT_DIR}/BMPR2_SOX17_PAH_directional_axis.tsv" \
    "${OUTPUT_DIR}/ANALYSIS_COMPLETE.json"
do
    [[ -s "${required}" ]] || {
        echo "Missing final output: ${required}" >&2
        exit 1
    }
done

echo "BMPR2–SOX17 AXIS ANALYSIS COMPLETED"
echo "${OUTPUT_DIR}/BMPR2_SOX17_PAH_axis_review.xlsx"
