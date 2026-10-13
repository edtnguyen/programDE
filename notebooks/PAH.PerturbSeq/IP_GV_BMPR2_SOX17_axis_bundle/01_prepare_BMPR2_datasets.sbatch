#!/usr/bin/env bash
#SBATCH --job-name=BMPR2_GEO_prep
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
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/bmpr2_sox17_axis_20260623"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/BMPR2_SOX17_axis"
RESOURCE_DIR="${ROOT}/resources"
GSE70456_DIR="${ROOT}/GSE70456"
GSE157231_DIR="${ROOT}/GSE157231"

CODE_DIR="${BMPR2_AXIS_CODE_DIR:?BMPR2_AXIS_CODE_DIR is required.}"
R70456="${CODE_DIR}/prepare_GSE70456.R"
R157231="${CODE_DIR}/prepare_GSE157231.R"
META70456="${CODE_DIR}/GSE70456_samples.tsv"
META157231="${CODE_DIR}/GSE157231_samples.tsv"

for required in \
    "${ENV}/bin/R" \
    "${RESOURCE_DIR}/RESOURCE_DOWNLOAD_COMPLETE.txt" \
    "${RESOURCE_DIR}/GSE157231_hypoxia_BMPR2.geneCounts.matrix.txt.gz" \
    "${R70456}" \
    "${R157231}" \
    "${META70456}" \
    "${META157231}"
do
    [[ -s "${required}" || -x "${required}" ]] || {
        echo "Missing required input: ${required}" >&2
        exit 1
    }
done

mkdir -p "${GSE70456_DIR}" "${GSE157231_DIR}" "${ROOT}/tmp"
export TMPDIR="${ROOT}/tmp/prep_${SLURM_JOB_ID:-interactive}"
mkdir -p "${TMPDIR}"

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

Rscript "${R70456}" \
    "${META70456}" \
    "${GSE70456_DIR}/geo_cache" \
    "${GSE70456_DIR}"

Rscript "${R157231}" \
    "${RESOURCE_DIR}/GSE157231_hypoxia_BMPR2.geneCounts.matrix.txt.gz" \
    "${META157231}" \
    "${GSE157231_DIR}"

for required in \
    "${GSE70456_DIR}/GSE70456_COMPLETE.json" \
    "${GSE70456_DIR}/BMPR2_siRNA_baseline_limma.tsv.gz" \
    "${GSE70456_DIR}/BMPR2_siRNA_PMA_limma.tsv.gz" \
    "${GSE157231_DIR}/GSE157231_COMPLETE.json" \
    "${GSE157231_DIR}/BMPR2_siRNA_normoxia_edgeR.tsv.gz" \
    "${GSE157231_DIR}/BMPR2_siRNA_6h_hypoxia_edgeR.tsv.gz" \
    "${GSE157231_DIR}/BMPR2_siRNA_24h_hypoxia_edgeR.tsv.gz"
do
    [[ -s "${required}" ]] || {
        echo "Missing preparation output: ${required}" >&2
        exit 1
    }
done

cat > "${ROOT}/BMPR2_DATA_PREPARATION_COMPLETE.txt" <<EOF
BMPR2 datasets prepared.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
GSE70456=${GSE70456_DIR}
GSE157231=${GSE157231_DIR}
EOF

echo "BMPR2 DATA PREPARATION COMPLETED"
