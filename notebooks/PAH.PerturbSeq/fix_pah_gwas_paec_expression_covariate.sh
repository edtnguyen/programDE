#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_gwas_program_burden_20260621"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
FINAL_RUN="${WORK_ROOT}/results/IP_GV_skcnmf_K30_dt2_0_n100"
OUTPUT_ROOT="${FINAL_RUN}/Interpretation/PAH_GWAS_program_burden/GCST007228"

PROGRAM_INPUT_DIR="${OUTPUT_ROOT}/program_inputs"
PRIMARY_COVAR="${PROGRAM_INPUT_DIR}/program_gene_covariates_target_and_MHC_excluded.txt"
MHC_COVAR="${PROGRAM_INPUT_DIR}/program_gene_covariates_target_excluded_MHC_included.txt"

HUMAN_LOGCPM="${WORK_ROOT}/pilot_array_n20/external_validation_human_PAH/GSE185479/processed/GSE185479_pseudobulk_logCPM.tsv.gz"

RESOURCE_DIR="${OUTPUT_ROOT}/resources"
GENE_INFO_GZ="${RESOURCE_DIR}/Homo_sapiens.gene_info.gz"
GENE_INFO_URL="https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"

[[ -x "${ENV}/bin/python" ]] || {
    echo "Missing GWAS environment: ${ENV}" >&2
    exit 1
}
for required in "${PRIMARY_COVAR}" "${MHC_COVAR}" "${HUMAN_LOGCPM}"; do
    [[ -s "${required}" ]] || {
        echo "Missing required input: ${required}" >&2
        exit 1
    }
done

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

mkdir -p "${RESOURCE_DIR}"

download_resumable() {
    local url="$1"
    local destination="$2"

    if [[ -s "${destination}" ]]; then
        echo "Reusing: ${destination}"
        return
    fi

    local temporary="${destination}.part"
    rm -f -- "${temporary}"

    curl \
        --location \
        --fail \
        --retry 10 \
        --retry-delay 10 \
        --connect-timeout 30 \
        --max-time 900 \
        --output "${temporary}" \
        "${url}"

    mv "${temporary}" "${destination}"
}

echo "============================================================"
echo "Repair PAEC-expression covariate for MAGMA"
echo "Program inputs: ${PROGRAM_INPUT_DIR}"
echo "Human logCPM:   ${HUMAN_LOGCPM}"
echo "============================================================"

echo
echo "=== 1. Download NCBI human Entrez-to-symbol annotation ==="
download_resumable "${GENE_INFO_URL}" "${GENE_INFO_GZ}"
gzip -t "${GENE_INFO_GZ}"

echo
echo "=== 2. Rebuild a complete mean_logCPM covariate ==="
python - \
    "${PRIMARY_COVAR}" \
    "${MHC_COVAR}" \
    "${HUMAN_LOGCPM}" \
    "${GENE_INFO_GZ}" \
    "${PROGRAM_INPUT_DIR}" <<'PY'
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

primary_path = Path(sys.argv[1])
mhc_path = Path(sys.argv[2])
expression_path = Path(sys.argv[3])
gene_info_path = Path(sys.argv[4])
output_dir = Path(sys.argv[5])

program_columns = [f"P{i}" for i in range(1, 31)]
required_columns = ["GENE", "mean_logCPM", *program_columns]

expression = pd.read_csv(
    expression_path,
    sep="\t",
    index_col=0,
    compression="infer",
)
if expression.empty:
    raise RuntimeError(f"Human PAEC expression matrix is empty: {expression_path}")

expression.index = (
    expression.index.astype(str).str.strip().str.upper()
)
expression = expression.apply(pd.to_numeric, errors="coerce")
mean_expression = expression.mean(axis=1, skipna=True)
mean_expression = mean_expression.groupby(level=0).mean()
mean_expression = mean_expression.replace([np.inf, -np.inf], np.nan).dropna()

if len(mean_expression) < 5_000:
    raise RuntimeError(
        f"Only {len(mean_expression):,} finite human PAEC expression genes."
    )

# NCBI Homo_sapiens.gene_info columns:
# #tax_id, GeneID, Symbol, LocusTag, Synonyms, ...
gene_info = pd.read_csv(
    gene_info_path,
    sep="\t",
    compression="infer",
    dtype=str,
    usecols=["GeneID", "Symbol", "Synonyms"],
)
gene_info["GENE"] = (
    gene_info["GeneID"].astype(str).str.replace(r"\.0$", "", regex=True)
)
gene_info["official_symbol"] = (
    gene_info["Symbol"].astype(str).str.strip().str.upper()
)

entrez_to_official = (
    gene_info.drop_duplicates("GENE")
    .set_index("GENE")["official_symbol"]
)

# Use an alias only when the official NCBI symbol is absent from the PAEC
# matrix. This handles a limited number of legacy aliases without creating
# ambiguous many-to-one mappings.
expression_genes = set(mean_expression.index)
entrez_to_alias: dict[str, str] = {}

for row in gene_info.itertuples(index=False):
    gene_id = str(row.GeneID).replace(".0", "")
    official = str(row.Symbol).strip().upper()
    if official in expression_genes:
        continue

    aliases = str(row.Synonyms)
    if aliases in {"", "-", "nan", "None"}:
        continue

    matching = [
        alias.strip().upper()
        for alias in aliases.split("|")
        if alias.strip().upper() in expression_genes
    ]
    if len(matching) == 1:
        entrez_to_alias[gene_id] = matching[0]

finite_values = mean_expression.to_numpy(dtype=float)
floor_value = float(np.quantile(finite_values, 0.01))
median_value = float(np.median(finite_values))

reports = []

for covariate_path in [primary_path, mhc_path]:
    covariates = pd.read_csv(
        covariate_path,
        sep=r"\s+",
        dtype={"GENE": str},
    )
    missing_columns = set(required_columns).difference(covariates.columns)
    if missing_columns:
        raise RuntimeError(
            f"{covariate_path} lacks columns: {sorted(missing_columns)}"
        )

    covariates["GENE"] = (
        covariates["GENE"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    official = covariates["GENE"].map(entrez_to_official)
    aliases = covariates["GENE"].map(entrez_to_alias)

    selected_symbol = official.where(
        official.isin(expression_genes),
        aliases,
    )
    rebuilt = selected_symbol.map(mean_expression)

    observed = rebuilt.notna()
    if observed.mean() < 0.50:
        raise RuntimeError(
            f"Only {observed.mean():.1%} of MAGMA genes mapped to PAEC "
            f"expression for {covariate_path}."
        )

    # Genes absent after the PAEC pseudobulk expression filter are assigned
    # the low-expression floor. This is preferable to median imputation for
    # a conditioning covariate because absence mostly reflects no/very low
    # detectable expression. MAGMA standardizes the covariate internally.
    rebuilt = rebuilt.fillna(floor_value)

    if not np.isfinite(rebuilt.to_numpy(dtype=float)).all():
        raise RuntimeError("Rebuilt mean_logCPM still contains non-finite values.")
    if rebuilt.nunique() < 100:
        raise RuntimeError("Rebuilt mean_logCPM has insufficient variation.")

    backup = covariate_path.with_name(
        covariate_path.name + ".before_expression_fix"
    )
    if not backup.exists():
        shutil.copy2(covariate_path, backup)

    covariates["mean_logCPM"] = rebuilt.to_numpy(dtype=float)

    temporary = covariate_path.with_name(covariate_path.name + ".fixed.part")
    covariates[required_columns].to_csv(
        temporary,
        sep=" ",
        index=False,
        na_rep="NA",
        float_format="%.10g",
    )

    # Full structural validation.
    check = pd.read_csv(
        temporary,
        sep=r"\s+",
        dtype={"GENE": str},
    )
    if check.columns.tolist() != required_columns:
        raise RuntimeError(
            f"Unexpected repaired columns for {covariate_path}: "
            f"{check.columns.tolist()}"
        )
    if check.shape != covariates[required_columns].shape:
        raise RuntimeError(
            f"Repaired shape mismatch for {covariate_path}: {check.shape}"
        )
    if check["mean_logCPM"].isna().any():
        raise RuntimeError(
            f"Repaired mean_logCPM contains NA for {covariate_path}."
        )

    os.replace(temporary, covariate_path)

    reports.append(
        {
            "file": str(covariate_path),
            "n_genes": int(len(covariates)),
            "official_symbol_expression_matches": int(
                (official.isin(expression_genes)).sum()
            ),
            "alias_expression_matches": int(
                (
                    ~official.isin(expression_genes)
                    & aliases.isin(expression_genes)
                ).sum()
            ),
            "observed_expression_genes": int(observed.sum()),
            "observed_expression_fraction": float(observed.mean()),
            "low_expression_imputed_genes": int((~observed).sum()),
            "low_expression_floor_logCPM": floor_value,
            "observed_median_logCPM": median_value,
            "mean_logCPM_missing_after_repair": int(
                check["mean_logCPM"].isna().sum()
            ),
            "mean_logCPM_sd_after_repair": float(
                check["mean_logCPM"].std(ddof=1)
            ),
            "backup": str(backup),
        }
    )

report = {
    "repair": "complete_direct_human_entrez_to_PAEC_expression_covariate",
    "expression_matrix": str(expression_path),
    "gene_annotation": str(gene_info_path),
    "imputation": (
        "Genes absent from the filtered PAEC expression matrix were assigned "
        "the 1st percentile of observed mean logCPM."
    ),
    "files": reports,
}
report_path = output_dir / "PAEC_EXPRESSION_COVARIATE_REPAIR.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")

print(json.dumps(report, indent=2))
PY

echo
echo "=== 3. Verify MAGMA will retain the repaired covariate ==="
for FILE in "${PRIMARY_COVAR}" "${MHC_COVAR}"; do
    python - "${FILE}" <<'PY'
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
frame = pd.read_csv(path, sep=r"\s+")
values = pd.to_numeric(frame["mean_logCPM"], errors="coerce")

print(path)
print("  rows:", len(frame))
print("  missing:", int(values.isna().sum()))
print("  finite:", int(np.isfinite(values).sum()))
print("  unique:", int(values.nunique()))
print("  mean:", float(values.mean()))
print("  sd:", float(values.std(ddof=1)))

if values.isna().any() or not np.isfinite(values).all():
    raise RuntimeError("mean_logCPM still contains missing/non-finite values.")
if values.nunique() < 100:
    raise RuntimeError("mean_logCPM still has inadequate variation.")
PY
done

echo
echo "PAEC EXPRESSION COVARIATE REPAIR PASSED"
echo
echo "Now resubmit the MAGMA v1.10 resume bundle:"
echo
echo "  bash submit_resume_pah_gwas_program_tests_v2.sh"
