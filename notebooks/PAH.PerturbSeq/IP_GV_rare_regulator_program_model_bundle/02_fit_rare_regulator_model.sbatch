#!/usr/bin/env bash
#SBATCH --job-name=rare_program_model
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/rare_regulator_program_20260623"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/Rare_regulator_program_model"
REVIEW_DIR="${ROOT}/curation"
OUTPUT_DIR="${ROOT}/model"

CRT_IP="${FINAL_RUN}/Evaluation/30_2_0/CRT_notebook/discovery/rna_qc_guides/IP_GFP/target_program_CRT.tsv"
COMMON_BURDEN="${FINAL_RUN}/Interpretation/PAH_GWAS_program_burden/GCST007228/summary/PAH_GWAS_program_burden_results.tsv"
HUMAN_VALIDATION="${FINAL_RUN}/Interpretation/Human_validation_GSE185479/summary/K30_dt2_human_validation_all_programs.tsv"

CODE_DIR="${RARE_REGULATOR_CODE_DIR:?RARE_REGULATOR_CODE_DIR is required.}"
PIPELINE="${CODE_DIR}/rare_regulator_program_model.py"
CURATION="${REVIEW_DIR}/rare_variant_effects_reviewed.tsv"

for required in \
    "${ENV}/bin/python" \
    "${PIPELINE}" \
    "${CURATION}" \
    "${CRT_IP}"
do
    [[ -s "${required}" || -x "${required}" ]] || {
        echo "Missing required input: ${required}" >&2
        exit 1
    }
done

mkdir -p "${OUTPUT_DIR}" "${ROOT}/tmp"
export TMPDIR="${ROOT}/tmp/model_${SLURM_JOB_ID:-interactive}"
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

ARGS=(
    fit
    --curation "${CURATION}"
    --crt-ip "${CRT_IP}"
    --effect-mode "${EFFECT_MODE:-gamma}"
    --minimum-genes "${MINIMUM_GENES:-6}"
    --max-programs "${MAX_PROGRAMS:-5}"
    --univariate-permutations "${UNIVARIATE_PERMUTATIONS:-20000}"
    --model-permutations "${MODEL_PERMUTATIONS:-20000}"
    --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
    --seed 20260623
    --output-dir "${OUTPUT_DIR}"
)

if [[ "${EFFECT_MODE:-gamma}" == "association_z_proxy" ]]; then
    ARGS+=(--allow-association-proxy)
fi
if [[ -s "${COMMON_BURDEN}" ]]; then
    ARGS+=(--common-program-burden "${COMMON_BURDEN}")
fi
if [[ -s "${HUMAN_VALIDATION}" ]]; then
    ARGS+=(--human-validation "${HUMAN_VALIDATION}")
fi
if [[ "${USE_PRESPECIFIED_PROGRAMS:-0}" == "1" ]]; then
    ARGS+=(
        --prespecified-programs
        "${CODE_DIR}/prespecified_programs.tsv"
    )
fi

python "${PIPELINE}" "${ARGS[@]}"

echo
echo "RARE-REGULATOR PROGRAM MODEL COMPLETED"
echo "${OUTPUT_DIR}/rare_regulator_program_model_review.xlsx"
echo "${OUTPUT_DIR}/rare_regulator_program_PAH.html"
