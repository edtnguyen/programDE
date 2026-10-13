#!/usr/bin/env bash
#SBATCH --job-name=IPGV_cNMF_merge
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

START_SECONDS=$SECONDS

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_sk_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
PREPARED_H5AD="${WORK_ROOT}/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"

PILOT_ROOT="${WORK_ROOT}/pilot_array_n20"
PER_K_OUTPUT="${PILOT_ROOT}/per_k"
MERGED_RUN_NAME="merged"
MERGED_RUN_DIR="${PILOT_ROOT}/${MERGED_RUN_NAME}"
SEED_MANIFEST="${PILOT_ROOT}/seeds/nmf_seed_manifest.tsv"

K_VALUES=(5 10 15 20 25 30 40 50)
SEL_THRESHOLDS=(0.2 2.0)
NUM_ITER=20
NUM_HVG=3000
SEED=14

NTHREADS="${SLURM_CPUS_PER_TASK:-8}"

[[ -s "${PREPARED_H5AD}" ]] || {
    echo "Prepared H5AD is missing: ${PREPARED_H5AD}" >&2
    exit 1
}
[[ -s "${SEED_MANIFEST}" ]] || {
    echo "Seed manifest is missing: ${SEED_MANIFEST}" >&2
    exit 1
}
for K in "${K_VALUES[@]}"; do
    MARKER="${PER_K_OUTPUT}/K${K}/K${K}_FACTORIZE_COMPLETE.txt"
    [[ -s "${MARKER}" ]] || {
        echo "Factorization marker is missing for K=${K}: ${MARKER}" >&2
        exit 1
    }
done

mkdir -p "${PILOT_ROOT}" "${PILOT_ROOT}/tmp"
export TMPDIR="${PILOT_ROOT}/tmp/merge_${SLURM_JOB_ID:-interactive}"
mkdir -p "${TMPDIR}"

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
unset PYTHONSTARTUP || true

export OMP_NUM_THREADS="${NTHREADS}"
export OPENBLAS_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export BLIS_NUM_THREADS="${NTHREADS}"
export NUMEXPR_NUM_THREADS="${NTHREADS}"

echo "============================================================"
echo "IP/GV PerturbNMF array pilot: merge and consensus"
echo "Started:        $(date)"
echo "Job ID:         ${SLURM_JOB_ID:-interactive}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "K values:       ${K_VALUES[*]}"
echo "Replicates/K:   ${NUM_ITER}"
echo "Thresholds:     ${SEL_THRESHOLDS[*]}"
echo "Merged run:     ${MERGED_RUN_DIR}"
echo "============================================================"

# Rebuild only the dedicated merged run. Per-K spectra are never removed.
case "${MERGED_RUN_DIR}" in
    "${PILOT_ROOT}/merged")
        rm -rf -- "${MERGED_RUN_DIR}"
        ;;
    *)
        echo "Refusing to remove unexpected path: ${MERGED_RUN_DIR}" >&2
        exit 1
        ;;
esac
mkdir -p "${MERGED_RUN_DIR}/Inference/cnmf_tmp"

echo
echo "=== Validating and copying per-K spectra ==="
python - \
    "${PER_K_OUTPUT}" \
    "${MERGED_RUN_DIR}" \
    "${SEED_MANIFEST}" \
    "${NUM_ITER}" \
    "${K_VALUES[@]}" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cnmf import load_df_from_npz

per_k_output = Path(sys.argv[1])
merged_run_dir = Path(sys.argv[2])
seed_manifest_path = Path(sys.argv[3])
n_iter = int(sys.argv[4])
k_values = [int(x) for x in sys.argv[5:]]

destination = merged_run_dir / "Inference" / "cnmf_tmp"
destination.mkdir(parents=True, exist_ok=True)

seed_manifest = pd.read_csv(seed_manifest_path, sep="\t")
expected_seed_rows = len(k_values) * n_iter
if len(seed_manifest) != expected_seed_rows:
    raise ValueError(
        f"Seed manifest has {len(seed_manifest)} rows; expected {expected_seed_rows}."
    )

rows = []
reference_columns = None

for k in k_values:
    expected_for_k = seed_manifest.loc[seed_manifest["K"] == k]
    if len(expected_for_k) != n_iter:
        raise ValueError(
            f"Seed manifest has {len(expected_for_k)} rows for K={k}; "
            f"expected {n_iter}."
        )

    source_tmp = per_k_output / f"K{k}" / "Inference" / "cnmf_tmp"

    for iteration in range(n_iter):
        source = (
            source_tmp
            / f"Inference.spectra.k_{k}.iter_{iteration}.df.npz"
        )
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)

        spectra = load_df_from_npz(source)
        if spectra.shape[0] != k:
            raise ValueError(
                f"{source}: expected {k} rows, found {spectra.shape[0]}."
            )
        values = spectra.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: contains NaN or infinite values.")
        if np.any(values < 0):
            raise ValueError(f"{source}: contains negative NMF weights.")

        columns = spectra.columns.astype(str).tolist()
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(
                f"{source}: HVG names/order differ from the first spectra file."
            )

        target = destination / source.name
        shutil.copy2(source, target)

        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)

        expected_seed = int(
            expected_for_k.loc[
                expected_for_k["iter"] == iteration, "nmf_seed"
            ].iloc[0]
        )
        rows.append(
            {
                "K": k,
                "iter": iteration,
                "nmf_seed": expected_seed,
                "source": str(source),
                "destination": str(target),
                "sha256": digest.hexdigest(),
                "n_programs": int(spectra.shape[0]),
                "n_hvgs": int(spectra.shape[1]),
            }
        )

manifest = pd.DataFrame(rows).sort_values(["K", "iter"])
manifest_path = merged_run_dir / "spectra_merge_manifest.tsv"
manifest.to_csv(manifest_path, sep="\t", index=False)

if len(manifest) != expected_seed_rows:
    raise RuntimeError(
        f"Copied {len(manifest)} spectra files; expected {expected_seed_rows}."
    )

print(
    f"Copied and validated {len(manifest)} spectra files with "
    f"{len(reference_columns):,} common HVGs."
)
print(f"Wrote: {manifest_path}")
PY

echo
echo "=== Running combined prepare, refit, K selection, and consensus ==="
python - \
    "${PREPARED_H5AD}" \
    "${MERGED_RUN_DIR}" \
    "${NUM_ITER}" \
    "${NUM_HVG}" \
    "${SEED}" \
    "${#K_VALUES[@]}" \
    "${K_VALUES[@]}" \
    "${#SEL_THRESHOLDS[@]}" \
    "${SEL_THRESHOLDS[@]}" <<'PY_MERGE'
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import cnmf
import yaml

from Stage1_Inference.src.plot_diagnostics import generate_all_plots

args = sys.argv[1:]
prepared_h5ad = Path(args.pop(0))
merged_run_dir = Path(args.pop(0))
n_iter = int(args.pop(0))
num_hvg = int(args.pop(0))
seed = int(args.pop(0))

n_k = int(args.pop(0))
k_values = [int(args.pop(0)) for _ in range(n_k)]

n_thresholds = int(args.pop(0))
thresholds = [float(args.pop(0)) for _ in range(n_thresholds)]

if args:
    raise ValueError(f"Unexpected trailing arguments: {args}")

inference_dir = merged_run_dir / "Inference"
inference_dir.mkdir(parents=True, exist_ok=True)

config = {
    "counts_fn": str(prepared_h5ad),
    "output_directory": str(merged_run_dir),
    "run_name": "Inference",
    "K": k_values,
    "numiter": n_iter,
    "numhvgenes": num_hvg,
    "seed": seed,
    "init": "random",
    "loss": "frobenius",
    "algo": "cd",
    "max_NMF_iter": 1000,
    "tol": 1e-4,
    "sel_threshs": thresholds,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
with (inference_dir / "config_merge.yml").open("w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

obj = cnmf.cNMF(output_dir=str(merged_run_dir), name="Inference")

# The pinned EngreitzLab sk-cNMF fork does not initialize this attribute in
# cNMF.__init__. Setting it explicitly preserves its seeded behavior.
obj.nmf_seeds = None

# prepare() writes the common normalized matrix and all-K replicate ledger.
# It does not remove the per-iteration spectra copied into cnmf_tmp above.
obj.prepare(
    counts_fn=str(prepared_h5ad),
    components=k_values,
    n_iter=n_iter,
    densify=False,
    tpm_fn=None,
    seed=seed,
    beta_loss="frobenius",
    algo="cd",
    num_highvar_genes=num_hvg,
    genes_file=None,
    tol=1e-4,
    alpha_usage=0.0,
    alpha_spectra=0.0,
    init="random",
    max_NMF_iter=1000,
    nmf_seeds=None,
)

obj.combine()
obj.k_selection_plot()

for k in k_values:
    for threshold in thresholds:
        print(f"Consensus: K={k}, density_threshold={threshold}")
        obj.consensus(
            k=k,
            density_threshold=threshold,
            show_clustering=True,
            close_clustergram_fig=True,
        )

generate_all_plots(
    run_dir=str(merged_run_dir),
    run_name="Inference",
    K_list=k_values,
    sel_thresh_list=thresholds,
    categorical_key="sample",
)

print("Combined refit, K selection, consensus, and diagnostics completed.")
PY_MERGE
echo
echo "=== Verifying merged seed ledger and key outputs ==="
python - \
    "${MERGED_RUN_DIR}" \
    "${SEED_MANIFEST}" \
    "${NUM_ITER}" \
    "${K_VALUES[@]}" <<'PY'
import sys
from pathlib import Path

import pandas as pd
from cnmf import load_df_from_npz

run_dir = Path(sys.argv[1])
seed_manifest_path = Path(sys.argv[2])
n_iter = int(sys.argv[3])
k_values = [int(x) for x in sys.argv[4:]]

inference = run_dir / "Inference"
tmp = inference / "cnmf_tmp"

params_path = tmp / "Inference.nmf_params.df.npz"
if not params_path.is_file():
    raise FileNotFoundError(params_path)

params = load_df_from_npz(params_path).copy()
params["n_components"] = params["n_components"].astype(int)
params["iter"] = params["iter"].astype(int)
params["nmf_seed"] = params["nmf_seed"].astype(int)

expected = pd.read_csv(seed_manifest_path, sep="\t")
expected = expected[["K", "iter", "nmf_seed"]].rename(
    columns={"K": "n_components"}
)
expected["n_components"] = expected["n_components"].astype(int)
expected["iter"] = expected["iter"].astype(int)
expected["nmf_seed"] = expected["nmf_seed"].astype(int)

observed = params[["n_components", "iter", "nmf_seed"]].sort_values(
    ["n_components", "iter"]
).reset_index(drop=True)
expected = expected.sort_values(
    ["n_components", "iter"]
).reset_index(drop=True)

if not observed.equals(expected):
    comparison = observed.merge(
        expected,
        on=["n_components", "iter"],
        how="outer",
        suffixes=("_observed", "_expected"),
        indicator=True,
    )
    bad = comparison[
        (comparison["_merge"] != "both")
        | (
            comparison["nmf_seed_observed"]
            != comparison["nmf_seed_expected"]
        )
    ]
    raise ValueError(
        "Merged NMF seed ledger does not match the per-K seed manifest:\n"
        + bad.head(20).to_string(index=False)
    )

required = [
    inference / "Inference.k_selection.png",
    inference / "Inference.k_selection_stats.df.npz",
]
for k in k_values:
    for threshold in (0.2, 2.0):
        tag = str(float(threshold)).replace(".", "_")
        required.extend(
            [
                inference
                / f"Inference.usages.k_{k}.dt_{tag}.consensus.txt",
                inference
                / f"Inference.gene_spectra_score.k_{k}.dt_{tag}.txt",
                inference
                / f"Inference.clustering.k_{k}.dt_{tag}.png",
            ]
        )

missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise FileNotFoundError(
        "Missing required merged outputs:\n" + "\n".join(missing)
    )

stats = load_df_from_npz(inference / "Inference.k_selection_stats.df.npz")
stats.to_csv(
    inference / "Inference.k_selection_stats.tsv",
    sep="\t",
    index=True,
)

print("Merged seed ledger matches all per-K runs.")
print(f"Verified {len(required)} key result files.")
print(stats)
PY

K_SELECTION="${MERGED_RUN_DIR}/Inference/Inference.k_selection.png"
K_STATS_TSV="${MERGED_RUN_DIR}/Inference/Inference.k_selection_stats.tsv"

cat > "${MERGED_RUN_DIR}/NEXT_STEP.txt" <<EOF
Array pilot completed successfully.

Review:
  ${K_SELECTION}
  ${K_STATS_TSV}
  ${MERGED_RUN_DIR}/Inference/diagnosis_plots/
  ${MERGED_RUN_DIR}/Inference/Inference.clustering.k_<K>.dt_0_2.png
  ${MERGED_RUN_DIR}/Inference/Inference.clustering.k_<K>.dt_2_0.png

After selecting K and density threshold, run the separate 100-replicate
selected-K script:

  sbatch run_ip_gv_perturbnmf_final.sbatch <K> <threshold>

Example syntax only:
  sbatch run_ip_gv_perturbnmf_final.sbatch 30 0.2
EOF

cat > "${MERGED_RUN_DIR}/PILOT_COMPLETE.txt" <<EOF
Array pilot completed successfully.
date=$(date --iso-8601=seconds)
job_id=${SLURM_JOB_ID:-interactive}
prepared_h5ad=${PREPARED_H5AD}
K_values=${K_VALUES[*]}
numiter=${NUM_ITER}
numhvgenes=${NUM_HVG}
density_thresholds=${SEL_THRESHOLDS[*]}
K_selection=${K_SELECTION}
K_stats=${K_STATS_TSV}
EOF

ELAPSED=$((SECONDS - START_SECONDS))
echo
echo "============================================================"
echo "ARRAY PILOT MERGE COMPLETED: $(date)"
echo "Elapsed seconds: ${ELAPSED}"
echo "K-selection plot: ${K_SELECTION}"
echo "K-selection table: ${K_STATS_TSV}"
echo "============================================================"
