#!/bin/bash
#SBATCH --job-name=occ_models
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=occ_models.%j.out
#SBATCH --error=occ_models.%j.err

set -euo pipefail
PACKAGE_DIR=${PACKAGE_DIR:-$PWD}
CONFIG=${CONFIG:-$PACKAGE_DIR/config/occupancy_export_config.yaml}
EXPORT_DIR=${EXPORT_DIR:-$PACKAGE_DIR/occupancy_compact_export}
OUTPUT_DIR=${OUTPUT_DIR:-$PACKAGE_DIR/occupancy_analysis_results}
cd "$PACKAGE_DIR"
python scripts/01_fit_occupancy_and_plot.py \
  --config "$CONFIG" \
  --export-dir "$EXPORT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --n-jobs ${SLURM_CPUS_PER_TASK:-8}
