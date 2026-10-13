#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Submission failed at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

PREP="${SCRIPT_DIR}/01_prepare_rare_variant_review.sbatch"
FIT="${SCRIPT_DIR}/02_fit_rare_regulator_model.sbatch"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/rare_regulator_program_20260623"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/Rare_regulator_program_model"
CODE_DIR="${ROOT}/code"
LOG_DIR="${ROOT}/logs"

[[ -x "${ENV}/bin/python" ]] || {
    echo "Environment missing: ${ENV}" >&2
    echo "Run setup_rare_regulator_program_mamba.sh first." >&2
    exit 1
}

mkdir -p "${CODE_DIR}" "${LOG_DIR}"

for file in \
    rare_regulator_program_model.py \
    rare_variant_effects_reviewed.tsv \
    portal_results_long.tsv \
    prespecified_programs.tsv \
    target_genes.txt
do
    [[ -r "${SCRIPT_DIR}/${file}" ]] || {
        echo "Missing bundle file: ${file}" >&2
        exit 1
    }
done

install -m 0755 "${SCRIPT_DIR}/rare_regulator_program_model.py" \
    "${CODE_DIR}/rare_regulator_program_model.py"
install -m 0644 "${SCRIPT_DIR}/rare_variant_effects_reviewed.tsv" \
    "${CODE_DIR}/rare_variant_effects_reviewed.tsv"
install -m 0644 "${SCRIPT_DIR}/portal_results_long.tsv" \
    "${CODE_DIR}/portal_results_long.tsv"
install -m 0644 "${SCRIPT_DIR}/prespecified_programs.tsv" \
    "${CODE_DIR}/prespecified_programs.tsv"
install -m 0644 "${SCRIPT_DIR}/target_genes.txt" \
    "${CODE_DIR}/target_genes.txt"

raw="$(
    sbatch --parsable \
        --export="ALL,RARE_REGULATOR_CODE_DIR=${CODE_DIR}" \
        --output="${LOG_DIR}/curation_%j.out" \
        --error="${LOG_DIR}/curation_%j.err" \
        "${PREP}"
)"
job_id="${raw%%;*}"

cat <<EOF

Submitted rare-variant curation review
--------------------------------------
Job ID: ${job_id}

Logs:
  ${LOG_DIR}/curation_${job_id}.out
  ${LOG_DIR}/curation_${job_id}.err

Review:
  ${ROOT}/curation/rare_variant_curation_review.xlsx
  ${ROOT}/curation/rare_variant_effects_reviewed.tsv
  ${ROOT}/curation/portal_results_long.tsv

The quantitative model is intentionally not submitted automatically.
First add/select directional gamma estimates for at least six primary genes.

Then submit:
  sbatch \
    --export=ALL,RARE_REGULATOR_CODE_DIR=${CODE_DIR} \
    --output=${LOG_DIR}/model_%j.out \
    --error=${LOG_DIR}/model_%j.err \
    ${FIT}
EOF
