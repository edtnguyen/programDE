#!/usr/bin/env bash
#SBATCH --job-name=IPGV_PAH_GWAS
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_gwas_program_burden_20260621"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
FINAL_RUN="${WORK_ROOT}/results/IP_GV_skcnmf_K30_dt2_0_n100"

GWAS_ACCESSION="${GWAS_ACCESSION:-GCST007228}"
GWAS_N="${GWAS_N:-11744}"
GWAS_LABEL="${GWAS_LABEL:-Rhodes_2019_clinical_PAH}"

OUTPUT_ROOT="${FINAL_RUN}/Interpretation/PAH_GWAS_program_burden/${GWAS_ACCESSION}"
GWAS_DIR="${OUTPUT_ROOT}/gwas"
MAGMA_GENE_DIR="${OUTPUT_ROOT}/magma_gene"
PROGRAM_INPUT_DIR="${OUTPUT_ROOT}/program_inputs"
MAGMA_PROGRAM_DIR="${OUTPUT_ROOT}/magma_program"
SUMMARY_DIR="${OUTPUT_ROOT}/summary"
LOG_DIR="${OUTPUT_ROOT}/logs"
TMP_ROOT="${OUTPUT_ROOT}/tmp/${SLURM_JOB_ID:-interactive}"

DOWNLOAD_MANIFEST="${GWAS_DIR}/GWAS_DOWNLOAD_COMPLETE.json"
FORMATTED_GWAS="${GWAS_DIR}/${GWAS_ACCESSION}.MAGMA.formatted.tsv.gz"
MAGMA_GENE_MANIFEST="${MAGMA_GENE_DIR}/MAGMA_GENE_ANALYSIS.json"

SPECTRA="${FINAL_RUN}/Inference/Inference.gene_spectra_score.k_30.dt_2_0.txt"
PREPARED_H5AD="${WORK_ROOT}/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"

EXTERNAL_ROOT="${WORK_ROOT}/pilot_array_n20/external_validation_human_PAH"
MGI_REPORT="${EXTERNAL_ROOT}/resources/HOM_ProteinCoding.rpt"
HUMAN_LOGCPM="${EXTERNAL_ROOT}/GSE185479/processed/GSE185479_pseudobulk_logCPM.tsv.gz"
HUMAN_VALIDATION="${FINAL_RUN}/Interpretation/Human_validation_GSE185479/summary/K30_dt2_human_validation_all_programs.tsv"

CODE_DIR="${PAH_GWAS_CODE_DIR:?PAH_GWAS_CODE_DIR is required. Submit with the helper.}"
PIPELINE="${CODE_DIR}/pah_gwas_program_burden.py"
R_GENE="${CODE_DIR}/run_magma_gene_analysis.R"
R_PROGRAM="${CODE_DIR}/run_magma_program_tests.R"

for required in \
    "${ENV}/bin/python" \
    "${DOWNLOAD_MANIFEST}" \
    "${FORMATTED_GWAS}" \
    "${SPECTRA}" \
    "${PREPARED_H5AD}" \
    "${MGI_REPORT}" \
    "${HUMAN_LOGCPM}" \
    "${PIPELINE}" \
    "${R_GENE}" \
    "${R_PROGRAM}"
do
    [[ -s "${required}" || -x "${required}" ]] || {
        echo "Missing required input: ${required}" >&2
        exit 1
    }
done

mkdir -p \
    "${MAGMA_GENE_DIR}" \
    "${PROGRAM_INPUT_DIR}" \
    "${MAGMA_PROGRAM_DIR}" \
    "${SUMMARY_DIR}" \
    "${LOG_DIR}" \
    "${TMP_ROOT}"

export TMPDIR="${TMP_ROOT}"

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true
export R_LIBS_USER="${ENV}/lib/R/library"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

GENOME_BUILD="$(
    python - "${DOWNLOAD_MANIFEST}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["genome_build"])
PY
)"

MANIFEST_N="$(
    python - "${DOWNLOAD_MANIFEST}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["sample_size"])
PY
)"

if [[ "${MANIFEST_N}" != "${GWAS_N}" ]]; then
    echo "WARNING: manifest N=${MANIFEST_N}, requested N=${GWAS_N}." >&2
    GWAS_N="${MANIFEST_N}"
fi

echo "============================================================"
echo "PAH GWAS × K30 program-burden analysis"
echo "Started:       $(date)"
echo "Job ID:        ${SLURM_JOB_ID:-interactive}"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "GWAS:          ${GWAS_ACCESSION}"
echo "GWAS label:    ${GWAS_LABEL}"
echo "N:             ${GWAS_N}"
echo "Genome build:  ${GENOME_BUILD}"
echo "Output root:   ${OUTPUT_ROOT}"
echo "============================================================"

echo
echo "=== 1. MAGMA SNP-to-gene and gene-level analysis ==="
Rscript "${R_GENE}" \
    "${FORMATTED_GWAS}" \
    "${GENOME_BUILD}" \
    "${GWAS_N}" \
    "${MAGMA_GENE_DIR}/cache" \
    "${MAGMA_GENE_MANIFEST}" \
    "${FORCE_MAGMA_GENE_ANALYSIS:-false}"

GENES_OUT="$(
    python - "${MAGMA_GENE_MANIFEST}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["genes_out"])
PY
)"
GENES_RAW="$(
    python - "${MAGMA_GENE_MANIFEST}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["genes_raw"])
PY
)"

[[ -s "${GENES_OUT}" && -s "${GENES_RAW}" ]] || {
    echo "MAGMA gene files are missing." >&2
    exit 1
}

echo
echo "=== 2. Map final cNMF program genes and run matched burden ==="
python "${PIPELINE}" build-program-inputs \
    --spectra "${SPECTRA}" \
    --k 30 \
    --mgi-report "${MGI_REPORT}" \
    --prepared-h5ad "${PREPARED_H5AD}" \
    --human-logcpm "${HUMAN_LOGCPM}" \
    --genes-out "${GENES_OUT}" \
    --top-n 300 \
    --n-permutations "${N_PERMUTATIONS:-20000}" \
    --seed 20260621 \
    --output-dir "${PROGRAM_INPUT_DIR}"

echo
echo "=== 3. MAGMA continuous and categorical program tests ==="
Rscript "${R_PROGRAM}" \
    "${GENES_RAW}" \
    "${PROGRAM_INPUT_DIR}/program_gene_covariates_target_and_MHC_excluded.txt" \
    "${PROGRAM_INPUT_DIR}/program_gene_covariates_target_excluded_MHC_included.txt" \
    "${PROGRAM_INPUT_DIR}/program_top50_target_and_MHC_excluded.sets" \
    "${PROGRAM_INPUT_DIR}/program_top300_target_and_MHC_excluded.sets" \
    "${PROGRAM_INPUT_DIR}/program_top50_target_excluded_MHC_included.sets" \
    "${PROGRAM_INPUT_DIR}/program_top300_target_excluded_MHC_included.sets" \
    "${MAGMA_PROGRAM_DIR}"

echo
echo "=== 4. Summarize and plot ==="
SUMMARY_ARGS=(
    --program-input-dir "${PROGRAM_INPUT_DIR}"
    --magma-results-dir "${MAGMA_PROGRAM_DIR}"
    --top-driver-genes 25
    --output-dir "${SUMMARY_DIR}"
)
if [[ -s "${HUMAN_VALIDATION}" ]]; then
    SUMMARY_ARGS+=(--human-validation "${HUMAN_VALIDATION}")
fi

python "${PIPELINE}" summarize "${SUMMARY_ARGS[@]}"

REQUIRED_OUTPUTS=(
    "${SUMMARY_DIR}/PAH_GWAS_program_burden_results.tsv"
    "${SUMMARY_DIR}/PAH_GWAS_K30_program_burden_review.xlsx"
    "${SUMMARY_DIR}/PAH_GWAS_program_burden_review.pdf"
    "${SUMMARY_DIR}/PAH_GWAS_program_driver_genes.tsv"
    "${SUMMARY_DIR}/HOW_TO_INTERPRET.txt"
    "${SUMMARY_DIR}/ANALYSIS_COMPLETE.json"
)
for required in "${REQUIRED_OUTPUTS[@]}"; do
    [[ -s "${required}" ]] || {
        echo "Missing final output: ${required}" >&2
        exit 1
    }
done

cat > "${OUTPUT_ROOT}/ANALYSIS_RUN_COMPLETE.txt" <<EOF
PAH GWAS program-burden analysis completed.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
GWAS_accession=${GWAS_ACCESSION}
GWAS_label=${GWAS_LABEL}
sample_size=${GWAS_N}
genome_build=${GENOME_BUILD}
final_spectra=${SPECTRA}
summary_dir=${SUMMARY_DIR}
EOF

echo
echo "PAH GWAS PROGRAM-BURDEN ANALYSIS COMPLETED"
echo "Workbook:"
echo "  ${SUMMARY_DIR}/PAH_GWAS_K30_program_burden_review.xlsx"
echo "Review PDF:"
echo "  ${SUMMARY_DIR}/PAH_GWAS_program_burden_review.pdf"
echo "Program table:"
echo "  ${SUMMARY_DIR}/PAH_GWAS_program_burden_results.tsv"
