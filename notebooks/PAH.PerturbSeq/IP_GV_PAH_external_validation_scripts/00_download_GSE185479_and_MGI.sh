#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
EXTERNAL_ROOT="${BASE}/PerturbNMF/pilot_array_n20/external_validation_human_PAH"
RESOURCE_DIR="${EXTERNAL_ROOT}/resources"
RAW_DIR="${EXTERNAL_ROOT}/GSE185479/raw"

GSE_TAR="${RESOURCE_DIR}/GSE185479_RAW.tar"
MGI_REPORT="${RESOURCE_DIR}/HOM_ProteinCoding.rpt"

GSE_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE185nnn/GSE185479/suppl/GSE185479_RAW.tar"
MGI_URL="https://www.informatics.jax.org/downloads/reports/HOM_ProteinCoding.rpt"

mkdir -p "${RESOURCE_DIR}" "${RAW_DIR}"

download_resumable() {
    local url="$1"
    local destination="$2"

    if [[ -s "${destination}" ]]; then
        echo "Reusing existing file: ${destination}"
        return
    fi

    if command -v curl >/dev/null 2>&1; then
        curl \
            --location \
            --fail \
            --retry 10 \
            --retry-delay 10 \
            --continue-at - \
            --output "${destination}" \
            "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget \
            --continue \
            --tries=10 \
            --timeout=60 \
            --output-document="${destination}" \
            "${url}"
    else
        echo "Neither curl nor wget is available." >&2
        exit 1
    fi
}

echo "=== Download GSE185479 processed matrices ==="
download_resumable "${GSE_URL}" "${GSE_TAR}"

[[ "$(stat -c %s "${GSE_TAR}")" -gt 300000000 ]] || {
    echo "GSE tar is unexpectedly small: ${GSE_TAR}" >&2
    exit 1
}
tar -tf "${GSE_TAR}" >/dev/null

if [[ ! -s "${RAW_DIR}/EXTRACTION_COMPLETE.txt" ]]; then
    echo "Extracting ${GSE_TAR}"
    tar -xf "${GSE_TAR}" -C "${RAW_DIR}"
    touch "${RAW_DIR}/EXTRACTION_COMPLETE.txt"
fi

N_MATRICES="$(find "${RAW_DIR}" -maxdepth 1 -type f -name 'GSM*_matrix.mtx.gz' | wc -l)"
N_FEATURES="$(find "${RAW_DIR}" -maxdepth 1 -type f -name 'GSM*_features.tsv.gz' | wc -l)"
N_BARCODES="$(find "${RAW_DIR}" -maxdepth 1 -type f -name 'GSM*_barcodes.tsv.gz' | wc -l)"

if [[ "${N_MATRICES}" -ne 6 || "${N_FEATURES}" -ne 6 || "${N_BARCODES}" -ne 6 ]]; then
    echo "Expected six matrices/features/barcodes after extraction." >&2
    echo "matrix=${N_MATRICES}, features=${N_FEATURES}, barcodes=${N_BARCODES}" >&2
    exit 1
fi

echo
echo "=== Download MGI one-to-one mouse-human ortholog report ==="
download_resumable "${MGI_URL}" "${MGI_REPORT}"

awk -F $'\t' '
    NF != 6 { bad += 1 }
    END {
        if (NR < 10000 || bad > 0) {
            exit 1
        }
    }
' "${MGI_REPORT}" || {
    echo "MGI ortholog report failed format validation." >&2
    exit 1
}

cat > "${RESOURCE_DIR}/RESOURCE_DOWNLOAD_COMPLETE.txt" <<EOF
Resources downloaded successfully.
date=$(date --iso-8601=seconds)
GSE185479_tar=${GSE_TAR}
GSE185479_raw=${RAW_DIR}
MGI_one_to_one_orthologs=${MGI_REPORT}
EOF

echo
echo "RESOURCE DOWNLOAD PASSED"
echo "External root: ${EXTERNAL_ROOT}"
