#!/usr/bin/env bash
#SBATCH --job-name=IPGV_K30_stage2
#SBATCH --output=IPGV_K30_stage2_%j.out
#SBATCH --error=IPGV_K30_stage2_%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -Eeuo pipefail

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_stage2_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

OUT_DIR="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype/PerturbNMF/results"
RUN_NAME="IP_GV_skcnmf_K30_dt2_0_n100"

PIPELINE="${REPO}/src/Stage2_Evaluation/A_Metrics/Slurm_Version/cNMF_evaluation_pipeline.py"
X_NORMALIZED="${OUT_DIR}/${RUN_NAME}/Inference/cnmf_tmp/Inference.norm_counts.h5ad"

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
export MPLBACKEND=Agg

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python "${PIPELINE}" \
    --out_dir "${OUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --K 30 \
    --sel_threshs 2.0 \
    --X_normalized_path "${X_NORMALIZED}" \
    --Perform_categorical \
    --Perform_geneset \
    --Perform_explained_variance \
    --data_key rna \
    --prog_key cNMF \
    --categorical_key sample \
    --organism mouse \
    --gene_names_key symbol \
    --guide_names_key guide_names \
    --guide_targets_key guide_targets \
    --guide_assignment_key guide_assignment \
    --guide_annotation_key non-targeting \
    --FDR_method BH \
    --n_top 300 \
    --use_cache \
    --skip_existing
