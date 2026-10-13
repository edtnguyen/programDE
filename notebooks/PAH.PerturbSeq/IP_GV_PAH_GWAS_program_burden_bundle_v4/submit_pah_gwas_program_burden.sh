#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Submission failed at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

RUN_SCRIPT="${SCRIPT_DIR}/run_pah_gwas_program_burden.sbatch"
PYTHON_PIPELINE="${SCRIPT_DIR}/pah_gwas_program_burden.py"
R_GENE="${SCRIPT_DIR}/run_magma_gene_analysis.R"
R_PROGRAM="${SCRIPT_DIR}/run_magma_program_tests.R"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_gwas_program_burden_20260621"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"

GWAS_ACCESSION="${GWAS_ACCESSION:-GCST007228}"
GWAS_N="${GWAS_N:-11744}"
GWAS_LABEL="${GWAS_LABEL:-Rhodes_2019_clinical_PAH}"

OUTPUT_ROOT="${FINAL_RUN}/Interpretation/PAH_GWAS_program_burden/${GWAS_ACCESSION}"
GWAS_MARKER="${OUTPUT_ROOT}/gwas/GWAS_DOWNLOAD_COMPLETE.json"
CODE_DIR="${OUTPUT_ROOT}/code"
LOG_DIR="${OUTPUT_ROOT}/logs"

[[ -x "${ENV}/bin/python" ]] || {
    echo "Environment is missing: ${ENV}" >&2
    echo "Run setup_pah_gwas_program_burden_mamba.sh first." >&2
    exit 1
}
[[ -s "${GWAS_MARKER}" ]] || {
    echo "GWAS download marker is missing: ${GWAS_MARKER}" >&2
    echo "Run 01_download_pah_gwas.sh first with the same GWAS variables." >&2
    exit 1
}

for required in \
    "${RUN_SCRIPT}" \
    "${PYTHON_PIPELINE}" \
    "${R_GENE}" \
    "${R_PROGRAM}"
do
    [[ -r "${required}" ]] || {
        echo "Missing bundle file: ${required}" >&2
        exit 1
    }
done

mkdir -p "${CODE_DIR}" "${LOG_DIR}"

install -m 0755 "${PYTHON_PIPELINE}" \
    "${CODE_DIR}/pah_gwas_program_burden.py"
install -m 0755 "${R_GENE}" \
    "${CODE_DIR}/run_magma_gene_analysis.R"
install -m 0755 "${R_PROGRAM}" \
    "${CODE_DIR}/run_magma_program_tests.R"

sha256sum \
    "${CODE_DIR}/pah_gwas_program_burden.py" \
    "${CODE_DIR}/run_magma_gene_analysis.R" \
    "${CODE_DIR}/run_magma_program_tests.R" \
    > "${CODE_DIR}/SHA256SUMS.txt"

raw="$(
    sbatch --parsable \
        --export="ALL,PAH_GWAS_CODE_DIR=${CODE_DIR},GWAS_ACCESSION=${GWAS_ACCESSION},GWAS_N=${GWAS_N},GWAS_LABEL=${GWAS_LABEL}" \
        --output="${LOG_DIR}/program_burden_%j.out" \
        --error="${LOG_DIR}/program_burden_%j.err" \
        "${RUN_SCRIPT}"
)"
job_id="${raw%%;*}"

cat <<EOF

Submitted PAH GWAS program-burden analysis
------------------------------------------
Job ID:      ${job_id}
GWAS:        ${GWAS_ACCESSION}
Label:       ${GWAS_LABEL}
Sample size: ${GWAS_N}

Logs:
  ${LOG_DIR}/program_burden_${job_id}.out
  ${LOG_DIR}/program_burden_${job_id}.err

Final outputs:
  ${OUTPUT_ROOT}/summary/PAH_GWAS_K30_program_burden_review.xlsx
  ${OUTPUT_ROOT}/summary/PAH_GWAS_program_burden_review.pdf
  ${OUTPUT_ROOT}/summary/PAH_GWAS_program_burden_results.tsv

Monitor:
  squeue -j ${job_id}
EOF
