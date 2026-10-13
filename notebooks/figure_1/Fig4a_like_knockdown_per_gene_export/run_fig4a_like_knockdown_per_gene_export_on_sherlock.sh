#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/fig4a_like_knockdown_per_gene_config.json}"

echo "[$(date)] Running Fig. 4A-like per-gene knockdown export"
echo "Config: ${CONFIG}"

python "${SCRIPT_DIR}/export_fig4a_like_knockdown_per_gene.py" --config "${CONFIG}"
