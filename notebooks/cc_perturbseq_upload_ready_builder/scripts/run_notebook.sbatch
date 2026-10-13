#!/bin/bash
#SBATCH --job-name=igvf_ready
#SBATCH --partition=engreitz
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/igvf_ready_%j.out
#SBATCH --error=logs/igvf_ready_%j.err

set -euo pipefail

PACKAGE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_PREFIX=/oak/stanford/groups/engreitz/Users/tri/envs/igvf-upload-ready

mkdir -p "$PACKAGE_ROOT/logs"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

cd "$PACKAGE_ROOT"

jupyter nbconvert \
  --to notebook \
  --execute notebooks/01_make_igvf_upload_ready_files.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output 01_make_igvf_upload_ready_files.executed.ipynb \
  --output-dir notebooks
