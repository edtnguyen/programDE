#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/BMPR2_SOX17_axis"
RESOURCE_DIR="${ROOT}/resources"

COUNTS_157231="${RESOURCE_DIR}/GSE157231_hypoxia_BMPR2.geneCounts.matrix.txt.gz"
MGI_REPORT="${RESOURCE_DIR}/HOM_ProteinCoding.rpt"

COUNTS_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE157nnn/GSE157231/suppl/GSE157231_hypoxia_BMPR2.geneCounts.matrix.txt.gz"
MGI_URL="https://www.informatics.jax.org/downloads/reports/HOM_ProteinCoding.rpt"

EXISTING_MGI="${BASE}/PerturbNMF/pilot_array_n20/external_validation_human_PAH/resources/HOM_ProteinCoding.rpt"

mkdir -p "${RESOURCE_DIR}"

download() {
    local url="$1"
    local output="$2"
    if [[ -s "${output}" ]]; then
        echo "Reusing ${output}"
        return
    fi
    local tmp="${output}.part"
    rm -f -- "${tmp}"
    curl \
        --location \
        --fail \
        --retry 10 \
        --retry-delay 10 \
        --connect-timeout 30 \
        --max-time 1800 \
        --output "${tmp}" \
        "${url}"
    mv "${tmp}" "${output}"
}

download "${COUNTS_URL}" "${COUNTS_157231}"
gzip -t "${COUNTS_157231}"

if [[ -s "${EXISTING_MGI}" ]]; then
    cp -f "${EXISTING_MGI}" "${MGI_REPORT}"
else
    download "${MGI_URL}" "${MGI_REPORT}"
fi

awk -F $'\t' '
    NF != 6 { bad += 1 }
    END {
        if (NR < 10000 || bad > 0) exit 1
    }
' "${MGI_REPORT}" || {
    echo "MGI ortholog report failed validation." >&2
    exit 1
}

sha256sum "${COUNTS_157231}" "${MGI_REPORT}" \
    > "${RESOURCE_DIR}/SHA256SUMS.txt"

cat > "${RESOURCE_DIR}/RESOURCE_DOWNLOAD_COMPLETE.txt" <<EOF
BMPR2 mechanism resources downloaded.
date=$(date --iso-8601=seconds)
GSE157231_counts=${COUNTS_157231}
MGI_report=${MGI_REPORT}
GSE70456_download=performed_later_by_GEOquery
EOF

echo "BMPR2 RESOURCE DOWNLOAD PASSED"
