#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-figure1_b08_only_config.json}"

echo "[b08-only] Running with config: ${CONFIG}"
python figure1_b08_only_export_and_plot.py --config-json "${CONFIG}"

echo
echo "[b08-only] Done. Upload this ZIP if you want final formatting in ChatGPT:"
echo "/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_cc_capture_b08_only/figure1_b08_plot_input_bundle.zip"
