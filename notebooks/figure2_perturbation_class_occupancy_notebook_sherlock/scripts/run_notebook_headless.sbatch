#!/bin/bash
#SBATCH --job-name=fig2_occ_nb
#SBATCH --output=logs/fig2_occ_nb_%j.out
#SBATCH --error=logs/fig2_occ_nb_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

set -euo pipefail
PACKAGE_DIR=${PACKAGE_DIR:-$(pwd)}
mkdir -p "$PACKAGE_DIR/logs"
cd "$PACKAGE_DIR"

jupyter nbconvert \
  --to notebook \
  --execute notebooks/01_generate_perturbation_class_occupancy.ipynb \
  --output 01_generate_perturbation_class_occupancy.executed.ipynb \
  --ExecutePreprocessor.timeout=-1
