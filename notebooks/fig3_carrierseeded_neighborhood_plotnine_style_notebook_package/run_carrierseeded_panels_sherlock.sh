#!/bin/bash
#SBATCH --job-name=fig3_carrierseeded_panels
#SBATCH --partition=engreitz,owners,normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=fig3_carrierseeded_panels.%j.out
#SBATCH --error=fig3_carrierseeded_panels.%j.err

set -euo pipefail

# Edit this block to match your environment.
# module load python/3.10
# source /path/to/your/conda.sh
# conda activate your_scanpy_env

python make_fig3_carrierseeded_neighborhood_panels.py
