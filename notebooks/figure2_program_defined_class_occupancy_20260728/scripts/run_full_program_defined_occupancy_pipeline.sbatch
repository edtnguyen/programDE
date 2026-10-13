#!/bin/bash
#SBATCH --job-name=prog_occ
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/prog_occ_%j.out
#SBATCH --error=logs/prog_occ_%j.err

set -euo pipefail
PACKAGE_DIR=${PACKAGE_DIR:-$(pwd)}
CONFIG=${CONFIG:-$PACKAGE_DIR/config/program_defined_class_occupancy_config.yaml}
mkdir -p "$PACKAGE_DIR/logs"
cd "$PACKAGE_DIR"

python scripts/00_export_program_defined_occupancy_inputs.py --config "$CONFIG"
python scripts/01_fit_program_defined_occupancy_and_plot.py --config "$CONFIG"
