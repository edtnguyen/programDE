#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_stage2_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"
PREPARED_H5AD="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype/PerturbNMF/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"

[[ -x "${ENV}/bin/python" ]] || {
    echo "Stage 2 environment is missing: ${ENV}" >&2
    exit 1
}
[[ -d "${REPO}/src/Stage2_Evaluation" ]] || {
    echo "PerturbNMF checkout is missing Stage 2 code: ${REPO}" >&2
    exit 1
}
[[ -s "${PREPARED_H5AD}" ]] || {
    echo "Prepared PerturbNMF input is missing: ${PREPARED_H5AD}" >&2
    exit 1
}

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
unset PYTHONSTARTUP || true

echo "============================================================"
echo "Repair and resume IP/GV PerturbNMF Stage 2 setup"
echo "Environment: ${ENV}"
echo "Python:      $(which python)"
echo "Repository:  ${REPO}"
echo "============================================================"

echo
echo "=== 1. Apply idempotent Stage 2 patches ==="
python - "${REPO}" <<'PY'
from pathlib import Path
import re
import sys

repo = Path(sys.argv[1])


def replace_exact_or_confirm(
    path: Path,
    old: str,
    new: str,
    description: str,
) -> None:
    text = path.read_text()
    if new in text:
        print(f"Already patched: {description}")
        return
    if old not in text:
        raise RuntimeError(
            f"Could not locate either the old or new text for "
            f"{description!r} in {path}"
        )
    path.write_text(text.replace(old, new, 1))
    print(f"Patched: {description}")


# 1. CPU cNMF import for explained variance.
replace_exact_or_confirm(
    repo
    / "src/Stage2_Evaluation/A_Metrics/src/explained_variance.py",
    "from torch_cnmf import cNMF",
    "from cnmf import cNMF",
    "CPU cNMF import for explained variance",
)

# 2. torch_cnmf is optional in Utilities.
replace_exact_or_confirm(
    repo / "src/Stage2_Evaluation/A_Metrics/src/Utilities.py",
    "import torch_cnmf",
    """try:
    import torch_cnmf
except ImportError:
    torch_cnmf = None""",
    "optional torch_cnmf import in Utilities",
)

# 3. tangermeme is optional because motif evaluation is not run here.
replace_exact_or_confirm(
    repo / "src/Stage2_Evaluation/A_Metrics/src/enrichment_motif.py",
    """from tangermeme.io import read_meme, extract_loci
from tangermeme.tools.fimo import fimo""",
    """try:
    from tangermeme.io import read_meme, extract_loci
    from tangermeme.tools.fimo import fimo
except ImportError:
    read_meme = None
    extract_loci = None
    fimo = None""",
    "optional tangermeme imports",
)

# 4. Support either sparse or dense cNMF program-score matrices.
#
# The upstream source has two independent lines ending in X.toarray().
# Patch each line separately instead of matching a larger formatting-sensitive
# block. This is safe to rerun.
association = (
    repo
    / "src/Stage2_Evaluation/A_Metrics/src/"
      "association_perturbation.py"
)
text = association.read_text()

dense_marker = "test_data_ = np.asarray(test_data_)"
reference_marker = "reference_data_ = np.asarray(reference_data_)"

if dense_marker in text and reference_marker in text:
    print("Already patched: dense/sparse program-score support")
else:
    test_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)test_data_[ \t]*=[ \t]*"
        r"test_data\[:,[ \t]*program\]\.X\.toarray\(\)[ \t]*$"
    )
    reference_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)reference_data_[ \t]*=[ \t]*"
        r"reference_data\[:,[ \t]*program\]\.X\.toarray\(\)[ \t]*$"
    )

    def patch_test(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return "\n".join(
            [
                f"{indent}test_data_ = test_data[:, program].X",
                f"{indent}if sparse.issparse(test_data_):",
                f"{indent}    test_data_ = test_data_.toarray()",
                f"{indent}else:",
                f"{indent}    test_data_ = np.asarray(test_data_)",
            ]
        )

    def patch_reference(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return "\n".join(
            [
                f"{indent}reference_data_ = reference_data[:, program].X",
                f"{indent}if sparse.issparse(reference_data_):",
                f"{indent}    reference_data_ = reference_data_.toarray()",
                f"{indent}else:",
                f"{indent}    reference_data_ = np.asarray(reference_data_)",
            ]
        )

    text, n_test = test_pattern.subn(patch_test, text, count=1)
    text, n_reference = reference_pattern.subn(
        patch_reference,
        text,
        count=1,
    )

    if n_test != 1 or n_reference != 1:
        context = "\n".join(
            line
            for line in association.read_text().splitlines()
            if "test_data_" in line
            or "reference_data_" in line
            or "toarray" in line
        )
        raise RuntimeError(
            "Could not patch the two program-score extraction lines. "
            f"Matched test={n_test}, reference={n_reference}.\n"
            f"Relevant current lines:\n{context}"
        )

    association.write_text(text)
    print("Patched: dense/sparse program-score support")

# 5. Fix the upstream reassign_name typo.
pipeline = (
    repo
    / "src/Stage2_Evaluation/A_Metrics/Slurm_Version/"
      "cNMF_evaluation_pipeline.py"
)
pipeline_text = pipeline.read_text()

old_typo = "gene_names_key=args.gene_names_ke"
fixed_typo = "gene_names_key=args.gene_names_key"

if old_typo in pipeline_text:
    pipeline.write_text(
        pipeline_text.replace(old_typo, fixed_typo, 1)
    )
    print("Patched: reassign_name typo")
elif fixed_typo in pipeline_text:
    print("Already patched: reassign_name typo")
else:
    print(
        "WARNING: neither typo variant was found; "
        "continuing because --reassign_name is not used."
    )

# Compile all modified Python files before proceeding.
files_to_compile = [
    repo
    / "src/Stage2_Evaluation/A_Metrics/src/"
      "explained_variance.py",
    repo
    / "src/Stage2_Evaluation/A_Metrics/src/"
      "Utilities.py",
    repo
    / "src/Stage2_Evaluation/A_Metrics/src/"
      "enrichment_motif.py",
    association,
    pipeline,
]
for path in files_to_compile:
    compile(path.read_text(), str(path), "exec")

print("All patched Python files compile successfully.")
PY

echo
echo "=== 2. Validate Stage 2 imports ==="
python - <<'PY'
import sys
from pathlib import Path

bad = [path for path in sys.path if "/home/users/edtng/.local" in path]
if bad:
    raise RuntimeError(f"User-site contamination detected: {bad}")

import anndata
import scanpy
import mudata
import muon
import gseapy
import scikit_posthocs
import statsmodels
import cnmf

from Stage1_Inference.src import compile_results
from Stage2_Evaluation.A_Metrics.src import (
    compute_categorical_association,
    compute_explained_variance,
    compute_geneset_enrichment,
    compute_perturbation_association,
)

print("Python:", sys.version.split()[0])
print("anndata:", anndata.__version__)
print("scanpy:", scanpy.__version__)
print("mudata:", mudata.__version__)
print("muon:", getattr(muon, "__version__", "version unavailable"))
print("gseapy:", gseapy.__version__)
print("scikit-posthocs:", scikit_posthocs.__version__)
print("statsmodels:", statsmodels.__version__)
print("cnmf:", Path(cnmf.__file__).resolve())
print("Stage 1 compilation import: OK")
print("Stage 2 metric imports: OK")
PY

echo
echo "=== 3. Download or validate mouse GO/Reactome caches ==="
python - "${REPO}" "${PREPARED_H5AD}" <<'PY'
import json
import sys
from pathlib import Path

import anndata as ad

from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import (
    get_geneset,
)

repo = Path(sys.argv[1])
input_h5ad = Path(sys.argv[2])
resources = repo / "src/Stage2_Evaluation/Resources"
resources.mkdir(parents=True, exist_ok=True)

adata = ad.read_h5ad(input_h5ad, backed="r")
input_genes = {str(gene).upper() for gene in adata.var_names}
adata.file.close()

libraries = [
    "Reactome_2022",
    "GO_Biological_Process_2023",
]

for library in libraries:
    expected_cache = resources / f"{library}_mouse.json"

    if expected_cache.is_file() and expected_cache.stat().st_size > 0:
        normalized = json.loads(expected_cache.read_text())
        print(f"Reusing cache: {expected_cache}")
    else:
        genesets = get_geneset(
            organism="mouse",
            library=library,
            database="enrichr",
            use_cache=True,
        )
        if not isinstance(genesets, dict) or len(genesets) == 0:
            raise RuntimeError(
                f"No mouse gene sets returned for {library}."
            )

        normalized = {
            str(term): sorted(
                {
                    str(gene).upper()
                    for gene in genes
                    if str(gene)
                }
            )
            for term, genes in genesets.items()
        }
        expected_cache.write_text(json.dumps(normalized))
        print(f"Wrote cache: {expected_cache}")

    library_genes = {
        gene
        for genes in normalized.values()
        for gene in genes
    }
    overlap = len(input_genes.intersection(library_genes))
    print(
        f"{library}: {len(normalized):,} terms; "
        f"{overlap:,} input genes overlap."
    )
    if overlap < 100:
        raise RuntimeError(
            f"Only {overlap} input genes overlap {library}; "
            "the cache is probably incompatible with the mouse data."
        )
PY

PIPELINE="${REPO}/src/Stage2_Evaluation/A_Metrics/Slurm_Version/cNMF_evaluation_pipeline.py"

echo
echo "=== 4. Validate the Stage 2 CLI ==="
python "${PIPELINE}" --help >/tmp/ip_gv_stage2_help.txt
head -100 /tmp/ip_gv_stage2_help.txt

echo
echo "STAGE 2 SETUP PASSED"
echo "Environment: ${ENV}"
echo "PerturbNMF: ${REPO}"
