#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/PAH_GWAS_program_burden/GCST007228"

GWAS_GZ="${ROOT}/gwas/GCST007228.MAGMA.formatted.tsv.gz"
MANIFEST="${ROOT}/gwas/GWAS_DOWNLOAD_COMPLETE.json"
CACHE="${ROOT}/magma_gene/cache"
DIRECT="${CACHE}/direct_magma"

[[ -s "${GWAS_GZ}" ]] || {
    echo "Missing formatted GWAS: ${GWAS_GZ}" >&2
    exit 1
}
[[ -s "${MANIFEST}" ]] || {
    echo "Missing manifest: ${MANIFEST}" >&2
    exit 1
}

echo "============================================================"
echo "Repair malformed optional column in MAGMA p-value file"
echo "Input: ${GWAS_GZ}"
echo "============================================================"

python - "${GWAS_GZ}" "${MANIFEST}" <<'PY'
from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import sys

source = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])

required = ["SNP", "CHR", "BP", "P", "N", "Z"]
temporary = source.with_name(source.name + ".fixed.part")
backup = source.with_name(source.name + ".before_column_fix")

with gzip.open(source, "rt", newline="") as handle:
    reader = csv.reader(handle, delimiter="\t")
    header = next(reader)

print("Original header:", header)

if header == required:
    print("The formatted GWAS already has the corrected six-column schema.")
else:
    missing = [column for column in required if column not in header]
    if missing:
        raise RuntimeError(
            f"The source file lacks required columns: {missing}; header={header}"
        )

    indices = [header.index(column) for column in required]

    n_input = 0
    n_output = 0
    n_bad = 0
    first_bad: list[tuple[int, list[str]]] = []

    with gzip.open(source, "rt", newline="") as input_handle, gzip.open(
        temporary,
        "wt",
        newline="",
    ) as output_handle:
        reader = csv.reader(input_handle, delimiter="\t")
        source_header = next(reader)

        writer = csv.writer(
            output_handle,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writerow(required)

        for line_number, fields in enumerate(reader, start=2):
            n_input += 1

            values = []
            bad = False
            for index in indices:
                value = fields[index].strip() if index < len(fields) else ""
                if value == "":
                    bad = True
                values.append(value)

            if bad:
                n_bad += 1
                if len(first_bad) < 10:
                    first_bad.append((line_number, fields))
                continue

            writer.writerow(values)
            n_output += 1

    if n_output < 100_000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Only {n_output:,} complete rows would remain after repair."
        )

    # Validate every output line has exactly six non-empty fields.
    with gzip.open(temporary, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        fixed_header = next(reader)
        if fixed_header != required:
            raise RuntimeError(f"Unexpected repaired header: {fixed_header}")

        checked = 0
        for line_number, fields in enumerate(reader, start=2):
            checked += 1
            if len(fields) != len(required) or any(value == "" for value in fields):
                raise RuntimeError(
                    f"Malformed repaired row at line {line_number}: {fields}"
                )

    if not backup.exists():
        shutil.copy2(source, backup)

    os.replace(temporary, source)

    print(f"Input data rows:    {n_input:,}")
    print(f"Output data rows:   {n_output:,}")
    print(f"Dropped bad rows:   {n_bad:,}")
    if first_bad:
        print("First dropped rows:")
        for line_number, fields in first_bad:
            print(f"  line {line_number}: {fields}")

    print("Backup:", backup)

manifest = json.loads(manifest_path.read_text())
manifest["formatted_file"] = str(source)
manifest.setdefault("repairs", []).append(
    {
        "repair": "drop_optional_BETA_OR_LOGOR_column",
        "final_columns": required,
        "reason": (
            "MAGMA requires every p-value-file row to have the same number "
            "of fields as the header. Missing optional effect estimates "
            "produced trailing empty seventh fields."
        ),
    }
)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

print("Final file:", source)
print("Final schema:", required)
PY

echo
echo "=== Remove stale decompressed and failed MAGMA outputs ==="
rm -f -- \
    "${CACHE}/GCST007228.MAGMA.formatted.tsv" \
    "${DIRECT}/GCST007228.MAGMA.formatted.35UP.10DOWN.genes.out" \
    "${DIRECT}/GCST007228.MAGMA.formatted.35UP.10DOWN.genes.raw"

# Preserve the valid 88-MB genes.annot file. The PAT-free R script can reuse it
# unless FORCE_MAGMA_GENE_ANALYSIS=true is supplied.

echo
echo "=== Verify corrected file around the previously failing row ==="
gzip -dc "${GWAS_GZ}" | sed -n '1p;5908p' | awk -F $'\t' '{
    print "fields=" NF "\t" $0
}'

BAD_COUNT="$(
    gzip -dc "${GWAS_GZ}" \
    | awk -F $'\t' '
        NR == 1 {
            expected = NF
            next
        }
        NF != expected || NF != 6 {
            bad += 1
        }
        END {
            print bad + 0
        }
    '
)"

if [[ "${BAD_COUNT}" -ne 0 ]]; then
    echo "Corrected file still contains ${BAD_COUNT} malformed rows." >&2
    exit 1
fi

echo
echo "FORMATTED GWAS REPAIR PASSED"
echo
echo "Resubmit from the v4 bundle directory with:"
echo
echo "  FORCE_MAGMA_GENE_ANALYSIS=false \\"
echo "      bash submit_pah_gwas_program_burden.sh"
echo
echo "The valid existing .genes.annot file will be reused."
