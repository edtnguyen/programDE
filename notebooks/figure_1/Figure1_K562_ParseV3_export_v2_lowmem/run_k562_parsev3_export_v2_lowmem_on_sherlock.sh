#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/k562_parsev3_figure_panel_config_v2_lowmem.json}"

echo "[$(date)] Running K562/iPSC Parse v3 figure-panel export, v2 low-memory"
echo "Config: ${CONFIG}"

python "${SCRIPT_DIR}/export_k562_parsev3_figure_panels_v2_lowmem.py" --config "${CONFIG}"
