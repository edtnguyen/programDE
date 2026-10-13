#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/PAH_GWAS_program_burden/GCST007228"

GWAS_DIR="${ROOT}/gwas"
CACHE="${ROOT}/magma_gene/cache"
MANIFEST="${GWAS_DIR}/GWAS_DOWNLOAD_COMPLETE.json"

[[ -s "${MANIFEST}" ]] || {
    echo "Missing manifest: ${MANIFEST}" >&2
    exit 1
}

echo "============================================================"
echo "MAGMA status-2 diagnostic"
echo "Root: ${ROOT}"
echo "============================================================"

echo
echo "=== 1. GWAS manifest ==="
python - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())

print("source_url:", data.get("source_url"))
print("genome_build:", data.get("genome_build"))
print("sample_size:", data.get("sample_size"))
print("formatted_file:", data.get("formatted_file"))
fmt = data.get("format_summary", {})
print("formatted_rows:", fmt.get("formatted_rows"))
print("rsid_fraction:", fmt.get("rsid_fraction"))
print("selected_columns:", fmt.get("columns"))
PY

echo
echo "=== 2. Existing MAGMA logs ==="
mapfile -t LOGS < <(
    find "${CACHE}" -type f -name '*.log' -printf '%T@ %p\n' \
    | sort -n \
    | cut -d' ' -f2-
)

if [[ "${#LOGS[@]}" -eq 0 ]]; then
    echo "No MAGMA .log files were found under ${CACHE}"
else
    for LOG in "${LOGS[@]}"; do
        echo
        echo "----- ${LOG} -----"
        tail -n 160 "${LOG}"
    done
fi

echo
echo "=== 3. Locate formatted GWAS and 1000G BIM ==="
FORMATTED="$(
    python - "${MANIFEST}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["formatted_file"])
PY
)"

BIM="$(
    find "${CACHE}" -type f -name 'g1000_eur.bim' -print -quit
)"

[[ -s "${FORMATTED}" ]] || {
    echo "Missing formatted GWAS: ${FORMATTED}" >&2
    exit 1
}
[[ -s "${BIM}" ]] || {
    echo "Missing 1000G BIM under ${CACHE}" >&2
    exit 1
}

echo "Formatted GWAS: ${FORMATTED}"
echo "1000G BIM:      ${BIM}"

echo
echo "=== 4. Lightweight SNP-ID and position overlap check ==="
python - "${FORMATTED}" "${BIM}" <<'PY'
from __future__ import annotations

import csv
import gzip
from pathlib import Path
import re
import sys

gwas_path = Path(sys.argv[1])
bim_path = Path(sys.argv[2])
MAX_GWAS = 250_000

opener = gzip.open if gwas_path.suffix in {".gz", ".bgz"} else open

sample_ids: set[str] = set()
sample_positions: set[tuple[str, str]] = set()
total = 0
rsids = 0

with opener(gwas_path, "rt", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    required = {"SNP", "CHR", "BP", "P"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise RuntimeError(f"GWAS file lacks columns: {sorted(missing)}")

    for row in reader:
        snp = str(row["SNP"]).strip()
        chromosome = str(row["CHR"]).removeprefix("chr").strip()
        position = str(row["BP"]).strip()
        if not snp or not chromosome or not position:
            continue

        sample_ids.add(snp)
        sample_positions.add((chromosome, position))
        total += 1
        rsids += bool(re.fullmatch(r"rs\d+", snp))

        if total >= MAX_GWAS:
            break

exact_id_overlap = 0
position_overlap = 0
reference_rows = 0
reference_ids_seen: set[str] = set()
reference_positions_seen: set[tuple[str, str]] = set()

with bim_path.open() as handle:
    for line in handle:
        fields = line.rstrip("\n").split()
        if len(fields) < 4:
            continue
        chromosome, snp, _, position = fields[:4]
        reference_rows += 1

        if snp in sample_ids and snp not in reference_ids_seen:
            exact_id_overlap += 1
            reference_ids_seen.add(snp)

        key = (chromosome.removeprefix("chr"), position)
        if key in sample_positions and key not in reference_positions_seen:
            position_overlap += 1
            reference_positions_seen.add(key)

print(f"GWAS rows sampled: {total:,}")
print(f"Unique GWAS SNP IDs sampled: {len(sample_ids):,}")
print(f"Sampled IDs matching ^rs[0-9]+$: {rsids / max(total, 1):.3%}")
print(f"1000G BIM rows scanned: {reference_rows:,}")
print(
    "Exact SNP-ID overlap in sample: "
    f"{exact_id_overlap:,}/{len(sample_ids):,} "
    f"({exact_id_overlap / max(len(sample_ids), 1):.3%})"
)
print(
    "CHR:BP overlap in sample: "
    f"{position_overlap:,}/{len(sample_positions):,} "
    f"({position_overlap / max(len(sample_positions), 1):.3%})"
)

if exact_id_overlap < 10_000:
    print(
        "\nDIAGNOSIS_FLAG=LOW_SNP_ID_OVERLAP\n"
        "The GWAS SNP identifiers probably do not match the 1000G reference."
    )
else:
    print(
        "\nDIAGNOSIS_FLAG=SNP_ID_OVERLAP_APPEARS_USABLE\n"
        "Inspect the MAGMA log above for a different failure."
    )
PY

echo
echo "=== 5. Direct-MAGMA output inventory ==="
find "${CACHE}/direct_magma" -maxdepth 1 -type f \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %12s %p\n' \
    2>/dev/null \
    | sort || true

echo
echo "DIAGNOSTIC COMPLETE"
