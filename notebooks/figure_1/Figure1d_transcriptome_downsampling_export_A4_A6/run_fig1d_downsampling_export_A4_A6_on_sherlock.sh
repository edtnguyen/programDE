#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/fig1d_transcriptome_downsampling_config_A4_A6.json}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG}" >&2
  exit 1
fi

echo "[$(date)] Running Figure 1d transcriptome downsampling export: Circular capture A4-A6"
echo "Config: ${CONFIG}"

python "${SCRIPT_DIR}/export_fig1d_transcriptome_downsampling_A4_A6.py" --config "${CONFIG}"

echo
echo "Output bundle:"
echo "/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_A4_A6/fig1d_transcriptome_downsampling_plot_input_bundle.zip"
