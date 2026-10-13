#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/fig1d_transcriptome_downsampling_config_v4.json}"

echo "[$(date)] Running Figure 1d transcriptome downsampling export v4.1"
echo "Config: ${CONFIG}"

python "${SCRIPT_DIR}/export_fig1d_transcriptome_downsampling_v4.py" --config "${CONFIG}"

echo
echo "Output bundle:"
echo "/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_v4/fig1d_transcriptome_downsampling_plot_input_bundle.zip"
