#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
FINAL_RUN="${BASE}/PerturbNMF/results/IP_GV_skcnmf_K30_dt2_0_n100"
ROOT="${FINAL_RUN}/Interpretation/SOX17_mechanism_GSE214742"
RESOURCE_DIR="${ROOT}/resources"
RAW_DIR="${ROOT}/GSE214742/raw"

GSE_TAR="${RESOURCE_DIR}/GSE214742_RAW.tar"
MGI_REPORT="${RESOURCE_DIR}/HOM_ProteinCoding.rpt"
GENCODE_GTF="${RESOURCE_DIR}/gencode.v28.annotation.gtf.gz"

GSE_URL="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE214742&format=file"
MGI_URL="https://www.informatics.jax.org/downloads/reports/HOM_ProteinCoding.rpt"
GENCODE_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_28/gencode.v28.annotation.gtf.gz"

EXISTING_MGI="${BASE}/PerturbNMF/pilot_array_n20/external_validation_human_PAH/resources/HOM_ProteinCoding.rpt"

mkdir -p "${RESOURCE_DIR}" "${RAW_DIR}"

download_resumable() {
    local url="$1"
    local destination="$2"

    if [[ -s "${destination}" ]]; then
        echo "Reusing: ${destination}"
        return
    fi

    local temporary="${destination}.part"
    rm -f -- "${temporary}"

    if command -v curl >/dev/null 2>&1; then
        curl \
            --location \
            --fail \
            --retry 10 \
            --retry-delay 10 \
            --continue-at - \
            --output "${temporary}" \
            "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget \
            --continue \
            --tries=10 \
            --timeout=60 \
            --output-document="${temporary}" \
            "${url}"
    else
        echo "Neither curl nor wget is available." >&2
        exit 1
    fi

    mv "${temporary}" "${destination}"
}

echo "=== Download GSE214742 processed RNA-seq files ==="
download_resumable "${GSE_URL}" "${GSE_TAR}"

[[ "$(stat -c %s "${GSE_TAR}")" -gt 5000000 ]] || {
    echo "GSE214742 tar is unexpectedly small: ${GSE_TAR}" >&2
    exit 1
}
tar -tf "${GSE_TAR}" >/dev/null

if [[ ! -s "${RAW_DIR}/EXTRACTION_COMPLETE.txt" ]]; then
    rm -f -- "${RAW_DIR}"/GSM*.csv.gz
    tar -xf "${GSE_TAR}" -C "${RAW_DIR}"
    touch "${RAW_DIR}/EXTRACTION_COMPLETE.txt"
fi

N_CSV="$(find "${RAW_DIR}" -maxdepth 2 -type f -name 'GSM*.csv.gz' | wc -l)"
if [[ "${N_CSV}" -ne 42 ]]; then
    echo "Expected 42 GSE214742 CSV files; found ${N_CSV}." >&2
    find "${RAW_DIR}" -maxdepth 2 -type f -printf '%p\n' | head -100 >&2
    exit 1
fi

echo
echo "=== Obtain MGI one-to-one ortholog report ==="
if [[ -s "${EXISTING_MGI}" ]]; then
    cp -f "${EXISTING_MGI}" "${MGI_REPORT}"
else
    download_resumable "${MGI_URL}" "${MGI_REPORT}"
fi

awk -F $'\t' '
    NF != 6 { bad += 1 }
    END {
        if (NR < 10000 || bad > 0) {
            exit 1
        }
    }
' "${MGI_REPORT}" || {
    echo "MGI ortholog report failed validation." >&2
    exit 1
}

echo
echo "=== Download GENCODE v28 annotation ==="
download_resumable "${GENCODE_URL}" "${GENCODE_GTF}"
gzip -t "${GENCODE_GTF}"

sha256sum \
    "${GSE_TAR}" \
    "${MGI_REPORT}" \
    "${GENCODE_GTF}" \
    > "${RESOURCE_DIR}/SHA256SUMS.txt"

cat > "${RESOURCE_DIR}/RESOURCE_DOWNLOAD_COMPLETE.txt" <<EOF
SOX17 mechanism resources downloaded successfully.
date=$(date --iso-8601=seconds)
GSE214742_tar=${GSE_TAR}
GSE214742_raw=${RAW_DIR}
MGI_report=${MGI_REPORT}
GENCODE_v28_GTF=${GENCODE_GTF}
GSE214742_CSV_files=${N_CSV}
EOF

echo
echo "RESOURCE DOWNLOAD PASSED"
echo "${RESOURCE_DIR}/RESOURCE_DOWNLOAD_COMPLETE.txt"
