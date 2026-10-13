#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_gwas_program_burden_20260621"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"

GWAS_ACCESSION="${GWAS_ACCESSION:-GCST007228}"
GWAS_N="${GWAS_N:-11744}"
GWAS_LABEL="${GWAS_LABEL:-Rhodes_2019_clinical_PAH}"

OUTPUT_ROOT="${FINAL_RUN}/Interpretation/PAH_GWAS_program_burden/${GWAS_ACCESSION}"
GWAS_DIR="${OUTPUT_ROOT}/gwas"

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"
PIPELINE="${SCRIPT_DIR}/pah_gwas_program_burden.py"

[[ -x "${ENV}/bin/python" ]] || {
    echo "GWAS program-burden environment is missing: ${ENV}" >&2
    echo "Run setup_pah_gwas_program_burden_mamba.sh first." >&2
    exit 1
}
[[ -r "${PIPELINE}" ]] || {
    echo "Missing pipeline: ${PIPELINE}" >&2
    exit 1
}

mkdir -p "${GWAS_DIR}"

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

echo "============================================================"
echo "Download and format PAH GWAS"
echo "Accession:   ${GWAS_ACCESSION}"
echo "Sample size: ${GWAS_N}"
echo "Label:       ${GWAS_LABEL}"
echo "Output:      ${GWAS_DIR}"
echo "============================================================"

python "${PIPELINE}" download-gwas \
    --accession "${GWAS_ACCESSION}" \
    --label "${GWAS_LABEL}" \
    --sample-size "${GWAS_N}" \
    --output-dir "${GWAS_DIR}"

[[ -s "${GWAS_DIR}/GWAS_DOWNLOAD_COMPLETE.json" ]] || {
    echo "GWAS download marker was not created." >&2
    exit 1
}

echo
echo "GWAS DOWNLOAD AND FORMATTING PASSED"
echo "${GWAS_DIR}/GWAS_DOWNLOAD_COMPLETE.json"
