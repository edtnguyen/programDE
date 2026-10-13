#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="${1:-fig1d_transcriptome_downsampling_config_v3.json}"
echo "[$(date)] Running Figure 1d transcriptome downsampling export v3"
echo "Config: $(pwd)/$CONFIG"
python export_fig1d_transcriptome_downsampling_v3.py --config "$CONFIG"
