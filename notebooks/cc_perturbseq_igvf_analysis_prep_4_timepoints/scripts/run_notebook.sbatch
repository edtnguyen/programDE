#!/bin/bash
#SBATCH --job-name=igvf_analysis_audit
#SBATCH --partition=engreitz
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/igvf_analysis_audit_%j.out
#SBATCH --error=logs/igvf_analysis_audit_%j.err

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV=/oak/stanford/groups/engreitz/Users/tri/envs/igvf-analysis-prep
RUN_DIR="$ROOT/runs/${SLURM_JOB_ID}"
mkdir -p "$ROOT/logs" "$RUN_DIR"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$ENV/bin/jupyter" nbconvert \
  --to notebook \
  --execute notebooks/00_prepare_igvf_analysis_files.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output "00_prepare_igvf_analysis_files.executed.ipynb" \
  --output-dir "$RUN_DIR"

echo "Executed notebook: $RUN_DIR/00_prepare_igvf_analysis_files.executed.ipynb"
