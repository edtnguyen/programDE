#!/usr/bin/env python
"""Read-only audit of four active and five prior-upload example IGVF AnalysisSets."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from igvf_analysis_prep import audit_analysis_sets, load_tsv, save_tsv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "portal_audit")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    examples = load_tsv(ROOT / "config" / "portal_examples.tsv")
    accessions = examples["analysis_set_accession"].tolist()
    if not os.environ.get("IGVF_API_KEY") or not os.environ.get("IGVF_SECRET_KEY"):
        print("IGVF credentials are not set. Released records may still be readable; unreleased records will require both variables.")

    sets, files, audits = audit_analysis_sets(accessions)
    save_tsv(sets, args.out / "portal_analysis_sets.tsv")
    save_tsv(files, args.out / "portal_files.tsv")
    save_tsv(audits, args.out / "portal_audits.tsv")

    print(f"AnalysisSets: {len(sets)}")
    print(f"Files:        {len(files)}")
    print(f"Audit items:  {len(audits)}")
    if not audits.empty:
        print("\nAudit levels:")
        print(audits["audit_level"].value_counts().to_string())
        blockers = audits[audits["audit_level"].isin(["ERROR", "NOT_COMPLIANT"])]
        print(f"\nBlocker-level items: {len(blockers)}")
        if len(blockers):
            cols = ["accession", "object_type", "audit_level", "audit_category", "audit_detail"]
            print(blockers[cols].to_string(index=False))
    print(f"\nOutputs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
