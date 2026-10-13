#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

MAMBA="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/bin/mamba"
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_stage2_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"
SKCNMF="/oak/stanford/groups/engreitz/Users/tri/Tools/sk_cNMF_perturbnmf"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
PREPARED_H5AD="${BASE}/PerturbNMF/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"

[[ -x "${MAMBA}" ]] || {
    echo "mamba not found: ${MAMBA}" >&2
    exit 1
}
[[ -d "${REPO}/.git" ]] || {
    echo "PerturbNMF checkout not found: ${REPO}" >&2
    exit 1
}
[[ -d "${SKCNMF}/src/cnmf" ]] || {
    echo "Local sk-cNMF checkout not found: ${SKCNMF}" >&2
    exit 1
}
[[ -s "${PREPARED_H5AD}" ]] || {
    echo "Prepared PerturbNMF input not found: ${PREPARED_H5AD}" >&2
    exit 1
}

echo "Using mamba: ${MAMBA}"
"${MAMBA}" --version

source "${CONDA_SH}"

echo
echo "=== 1. Create or repair the user-owned Stage 2 environment ==="
if [[ ! -x "${ENV}/bin/python" ]]; then
    "${MAMBA}" create -y -p "${ENV}" \
        --override-channels -c conda-forge \
        python=3.10 \
        numpy=1.26.4 \
        scipy=1.12 \
        pandas=2.2 \
        scikit-learn=1.5 \
        anndata=0.10.9 \
        scanpy=1.10.4 \
        mudata=0.3.1 \
        muon=0.1.7 \
        h5py \
        matplotlib \
        palettable \
        ipython \
        pyyaml \
        openpyxl \
        tqdm \
        joblib \
        statsmodels \
        requests \
        pip \
        setuptools \
        wheel
else
    echo "Environment already exists: ${ENV}"
    "${MAMBA}" install -y -p "${ENV}" \
        --override-channels -c conda-forge \
        numpy=1.26.4 \
        scipy=1.12 \
        pandas=2.2 \
        scikit-learn=1.5 \
        anndata=0.10.9 \
        scanpy=1.10.4 \
        mudata=0.3.1 \
        muon=0.1.7 \
        h5py \
        matplotlib \
        palettable \
        ipython \
        pyyaml \
        openpyxl \
        tqdm \
        joblib \
        statsmodels \
        requests \
        pip \
        setuptools \
        wheel
fi

conda activate "${ENV}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

echo
echo "=== 2. Install Python-only Stage 2 dependencies ==="
python -m pip install --upgrade \
    "gseapy==1.2.1" \
    "scikit-posthocs==0.11.4" \
    "mygene==3.2.2"

echo
echo "=== 3. Install the local EngreitzLab sk-cNMF fork ==="
python -m pip install --no-deps -e "${SKCNMF}"

echo
echo "=== 4. Patch only the user-owned PerturbNMF checkout ==="
python - "${REPO}" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1])

def replace_once(path: Path, old: str, new: str, description: str) -> None:
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new, 1))
        print(f"Patched {description}: {path}")
    elif new in text:
        print(f"Patch already present ({description}): {path}")
    else:
        raise RuntimeError(
            f"Could not find expected text for patch '{description}' in {path}"
        )

# The imported class is unused by explained_variance.py. Use the installed
# CPU cNMF package instead of requiring torch-cNMF.
explained = (
    repo
    / "src/Stage2_Evaluation/A_Metrics/src/explained_variance.py"
)
replace_once(
    explained,
    "from torch_cnmf import cNMF",
    "from cnmf import cNMF",
    "CPU cNMF import for explained variance",
)

# Utilities imports torch_cnmf but its two exported gene-renaming functions do
# not use it. Make the optional dependency non-fatal.
utilities = repo / "src/Stage2_Evaluation/A_Metrics/src/Utilities.py"
replace_once(
    utilities,
    "import torch_cnmf",
    """try:
    import torch_cnmf
except ImportError:
    torch_cnmf = None""",
    "optional torch_cnmf import in Utilities",
)

# Motif analysis is not part of this mouse pre-K workflow. Allow the metrics
# package to import without installing PyTorch/tangermeme.
motif = repo / "src/Stage2_Evaluation/A_Metrics/src/enrichment_motif.py"
replace_once(
    motif,
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

# cNMF program scores produced by compile_results are dense ndarrays. The
# upstream perturbation function calls .toarray() unconditionally.
perturb = (
    repo
    / "src/Stage2_Evaluation/A_Metrics/src/association_perturbation.py"
)
replace_once(
    perturb,
    """    test_data_ = test_data[:, program].X.toarray()
    # TODO: Resample reference data to match number of obs
    reference_data_ = reference_data[:, program].X.toarray()""",
    """    test_data_ = test_data[:, program].X
    if sparse.issparse(test_data_):
        test_data_ = test_data_.toarray()
    else:
        test_data_ = np.asarray(test_data_)

    # TODO: Resample reference data to match number of obs
    reference_data_ = reference_data[:, program].X
    if sparse.issparse(reference_data_):
        reference_data_ = reference_data_.toarray()
    else:
        reference_data_ = np.asarray(reference_data_)""",
    "dense/sparse program-score support",
)

# Fix an upstream typo even though this workflow does not use --reassign_name.
pipeline = (
    repo
    / "src/Stage2_Evaluation/A_Metrics/Slurm_Version/"
      "cNMF_evaluation_pipeline.py"
)
replace_once(
    pipeline,
    "gene_names_key=args.gene_names_ke",
    "gene_names_key=args.gene_names_key",
    "reassign_name typo",
)
PY

echo
echo "=== 5. Validate imports ==="
export PYTHONPATH="${REPO}/src"

python - <<'PY'
import sys
from pathlib import Path

bad = [p for p in sys.path if "/home/users/edtng/.local" in p]
if bad:
    raise RuntimeError(f"User-site package contamination detected: {bad}")

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
echo "=== 6. Download and normalize mouse GO/Reactome caches ==="
python - "${REPO}" "${PREPARED_H5AD}" <<'PY'
import json
import sys
from pathlib import Path

import anndata as ad

from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import get_geneset

repo = Path(sys.argv[1])
input_h5ad = Path(sys.argv[2])
resources = repo / "src/Stage2_Evaluation/Resources"
resources.mkdir(parents=True, exist_ok=True)

adata = ad.read_h5ad(input_h5ad, backed="r")
input_genes = {str(g).upper() for g in adata.var_names}
adata.file.close()

libraries = ["Reactome_2022", "GO_Biological_Process_2023"]

for library in libraries:
    gmt = get_geneset(
        organism="mouse",
        library=library,
        database="enrichr",
        use_cache=True,
    )
    if not isinstance(gmt, dict) or len(gmt) == 0:
        raise RuntimeError(f"No gene sets returned for {library}.")

    # Program gene symbols are uppercased by the PerturbNMF enrichment code.
    # Normalize cached library symbols to the same representation.
    normalized = {
        str(term): sorted({str(gene).upper() for gene in genes if str(gene)})
        for term, genes in gmt.items()
    }
    cache = resources / f"{library}_mouse.json"
    cache.write_text(json.dumps(normalized))

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
            "the downloaded library is probably incompatible."
        )

print(f"Gene-set caches written under: {resources}")
PY

PIPELINE="${REPO}/src/Stage2_Evaluation/A_Metrics/Slurm_Version/cNMF_evaluation_pipeline.py"

echo
echo "=== 7. Validate the Stage 2 CLI ==="
python "${PIPELINE}" --help >/tmp/ip_gv_stage2_help.txt
head -100 /tmp/ip_gv_stage2_help.txt

echo
echo "STAGE 2 SETUP PASSED"
echo "Environment: ${ENV}"
echo "PerturbNMF: ${REPO}"
echo
echo "Use these lines in Stage 2 jobs:"
echo "source ${CONDA_SH}"
echo "conda activate ${ENV}"
echo "export PYTHONNOUSERSITE=1"
echo "export PYTHONPATH=${REPO}/src"
