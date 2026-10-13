#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/k562_parsev3_figure_panel_config.json}"

echo "[$(date)] Running K562/iPSC Parse v3 figure-panel export"
echo "Config: ${CONFIG}"

python "${SCRIPT_DIR}/export_k562_parsev3_figure_panels.py" --config "${CONFIG}"
