#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
from legacy_author_defined_state_atlas import run_legacy_author_atlas

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(HERE / "config" / "legacy_author_state_atlas_config.local.yaml"))
args = parser.parse_args()
outdir = run_legacy_author_atlas(args.config)
print(f"Completed. Outputs written to: {outdir}")
