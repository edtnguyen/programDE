#!/bin/bash
#SBATCH --job-name=occ_export
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=occ_export.%j.out
#SBATCH --error=occ_export.%j.err

set -euo pipefail
PACKAGE_DIR=${PACKAGE_DIR:-$PWD}
CONFIG=${CONFIG:-$PACKAGE_DIR/config/occupancy_export_config.yaml}
cd "$PACKAGE_DIR"
python scripts/00_export_compact_occupancy_inputs.py --config "$CONFIG"
