#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/cc_perturbseq_supplementary_config.json}"

echo "[$(date)] Running CC-Perturb-seq supplementary panel export"
echo "Config: ${CONFIG}"
python "${SCRIPT_DIR}/make_cc_perturbseq_supplementary_panels.py" --config "${CONFIG}"
