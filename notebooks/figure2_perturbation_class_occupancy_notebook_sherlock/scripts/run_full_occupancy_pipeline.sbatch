#!/bin/bash
#SBATCH --job-name=fig2_occ
#SBATCH --output=logs/fig2_occ_%j.out
#SBATCH --error=logs/fig2_occ_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

set -euo pipefail

PACKAGE_DIR=${PACKAGE_DIR:-$(pwd)}
CONFIG=${CONFIG:-$PACKAGE_DIR/config/perturbation_class_occupancy_config.yaml}
N_JOBS=${N_JOBS:-8}

mkdir -p "$PACKAGE_DIR/logs"
cd "$PACKAGE_DIR"

python scripts/00_export_compact_occupancy_inputs.py --config "$CONFIG"
python scripts/01_fit_occupancy_and_plot.py \
  --config "$CONFIG" \
  --export-dir "$PACKAGE_DIR/occupancy_compact_export" \
  --output-dir "$PACKAGE_DIR/occupancy_analysis_results" \
  --n-jobs "$N_JOBS"

zip -r occupancy_analysis_results.zip occupancy_analysis_results
