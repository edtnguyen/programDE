#!/bin/bash
#SBATCH --job-name=legacy_state_atlas
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=legacy_state_atlas_%j.out
#SBATCH --error=legacy_state_atlas_%j.err

set -euo pipefail
PACKAGE_DIR=${PACKAGE_DIR:-$PWD}
CONFIG=${CONFIG:-$PACKAGE_DIR/config/legacy_author_state_atlas_config.local.yaml}
cd "$PACKAGE_DIR"
python scripts/01_run_legacy_author_state_atlas.py --config "$CONFIG"
