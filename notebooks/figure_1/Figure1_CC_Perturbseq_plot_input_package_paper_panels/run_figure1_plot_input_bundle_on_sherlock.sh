#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere. The script changes to the directory containing this package.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# The teloHAEC h5ad/kite paths and iPSC pilot guide-annotation paths are prefilled.
# This wrapper first checks paths, then reprocesses enabled datasets and writes figure1_plot_input_bundle.zip.

CONFIG_JSON="${1:-figure1_config_template.json}"
MAX_ROWS_PER_PANEL_STRATUM="${MAX_ROWS_PER_PANEL_STRATUM:-50000}"

python figure1_cc_perturbseq_pipeline.py \
  --config-json "${CONFIG_JSON}" \
  --check-paths

python figure1_cc_perturbseq_pipeline.py \
  --config-json "${CONFIG_JSON}" \
  --skip-plots \
  --max-rows-per-panel-stratum "${MAX_ROWS_PER_PANEL_STRATUM}"
