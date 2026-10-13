#!/usr/bin/env python3
"""
Export plot-input tables for Figure 1d transcriptome downsampling. v4 uses Parse report cell ID order directly for tscp_assignment downsampling.

This script uses Parse/SPLiT-seq post-mapping transcript assignment tables
(tscp_assignment.csv) to approximate read-depth downsampling after mapping and
transcript assignment. It is intended for comparing transcriptome complexity
between circular capture and CROP-seq samples at matched assigned-read depth.

v4 assumptions
--------------
For these Parse outputs, report/cell IDs use bc_wells in the same order as the\ncalled-cell table: bc1_bc2_bc3 (for example, BS3 cells start with 07 or 09).\nThus the default well filter uses bc_wells component 1 and cell identity uses\nidentity_numeric, not a permutation.\n\nCore idea\n---------
Each row in tscp_assignment.csv is treated as a molecule/gene assignment with
read support in the `count` column. For each downsampling fraction f, the script
samples supporting reads for each row as Binomial(count, f). A molecule is
retained if sampled_count > 0. It then computes per-cell UMIs and genes.

Important label for manuscript/figure:
    x-axis should be "assigned transcriptome reads per cell" unless the BAM/FASTQ
    pipeline is re-run from raw reads.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

WELL_RE = re.compile(r"^\s*([A-Ha-h])\s*0?([1-9]|1[0-2])\s*$")
DNA_RE = re.compile(r"^[ACGTNacgtn]{6,20}$")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_well(well: str) -> str:
    m = WELL_RE.match(str(well))
    if not m:
        raise ValueError(f"Invalid well name: {well!r}")
    return f"{m.group(1).upper()}{int(m.group(2))}"


def well_to_plate_index(well: str) -> int:
    w = normalize_well(well)
    row = ord(w[0]) - ord("A")
    col = int(w[1:])
    return row * 12 + col


def expand_well_spec(spec: str) -> List[str]:
    """Expand specs like 'A7:A9,B1:B12,A1' into normalized wells."""
    wells: List[str] = []
    for token in re.split(r"[,;\s]+", str(spec).strip()):
        if not token:
            continue
        token = token.replace("-", ":")
        if ":" in token:
            a, b = token.split(":", 1)
            a = normalize_well(a)
            b = normalize_well(b)
            if a[0] != b[0]:
                raise ValueError(f"Range must stay in one row: {token}")
            start, end = int(a[1:]), int(b[1:])
            step = 1 if end >= start else -1
            wells.extend([f"{a[0]}{i}" for i in range(start, end + step, step)])
        else:
            wells.append(normalize_well(token))
    # preserve order but drop duplicates
    return list(dict.fromkeys(wells))


def normalize_numeric_triplet(x: object) -> str:
    """Normalize cell/barcode identifiers like 7_2_52 or 07_02_52 to 07_02_52."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    parts = s.split("_")
    if len(parts) >= 2 and all(p.isdigit() for p in parts):
        return "_".join(f"{int(p):02d}" for p in parts)
    return s




def _numeric_parts(x: object) -> Optional[List[str]]:
    """Return zero-padded numeric parts for an underscore triplet-like ID."""
    if pd.isna(x):
        return None
    parts = str(x).strip().split("_")
    if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
        return [f"{int(p):02d}" for p in parts[:3]]
    return None


def permute_numeric_triplet_series(series: pd.Series, order: str) -> pd.Series:
    """Create a DGE-style numeric triplet after permuting positions.

    order is a string such as '123' or '312'. The latter maps a tscp value
    like 01_01_07 to 07_01_01, which is often the Parse DGE cell-id order.
    """
    idxs = [int(ch) - 1 for ch in order]

    def one(x):
        parts = _numeric_parts(x)
        if parts is None:
            return normalize_numeric_triplet(x)
        return "_".join(parts[i] for i in idxs)

    return series.map(one)


def build_identity_series(df: pd.DataFrame, source: str, transform: str) -> pd.Series:
    """Build cell IDs from tscp columns using a selected representation."""
    if source not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    if transform.startswith("perm"):
        return permute_numeric_triplet_series(df[source], transform.replace("perm", ""))
    if transform == "identity_numeric":
        return df[source].map(normalize_numeric_triplet)
    if transform == "sequence":
        return df[source].astype(str).str.strip()
    return df[source].map(normalize_numeric_triplet)


def score_identity_candidates(df: pd.DataFrame, called_cells: Set[str]) -> pd.DataFrame:
    """Score candidate tscp cell-id representations against called-cell IDs."""
    rows = []
    if not called_cells or df.empty:
        return pd.DataFrame(columns=["source", "transform", "overlap_rows", "n_unique_candidate", "overlap_unique", "score"])
    candidates: List[Tuple[str, str]] = []
    for src in ["bc_wells", "bc_bcis"]:
        if src in df.columns:
            for order in ["123", "132", "213", "231", "312", "321"]:
                candidates.append((src, f"perm{order}"))
            candidates.append((src, "identity_numeric"))
    if "cell_barcode" in df.columns:
        candidates.append(("cell_barcode", "sequence"))
    for src, tr in candidates:
        ids = build_identity_series(df, src, tr)
        overlap_rows = int(ids.isin(called_cells).sum())
        unique_vals = set(v for v in ids.dropna().astype(str) if v and v.lower() != "nan")
        overlap_unique = len(unique_vals & called_cells)
        rows.append({
            "source": src,
            "transform": tr,
            "overlap_rows": overlap_rows,
            "n_unique_candidate": len(unique_vals),
            "overlap_unique": overlap_unique,
            "score": overlap_rows + 10 * overlap_unique,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["score", "overlap_rows", "overlap_unique"], ascending=False).reset_index(drop=True)
    return out

def split_component(series: pd.Series, position_1based: int) -> pd.Series:
    idx = int(position_1based) - 1
    return series.astype(str).str.split("_", expand=True).iloc[:, idx]


def read_table_flexible(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read csv/tsv with delimiter inference."""
    path = str(path)
    try:
        return pd.read_csv(path, sep=None, engine="python", nrows=nrows)
    except Exception:
        # Most Parse files are comma-separated.
        return pd.read_csv(path, sep=",", nrows=nrows)


def infer_barcode_mapping(barcode_csv: str, requested_wells: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Try to infer well -> barcode-sequence mapping from barcode_data.csv.

    The script is intentionally tolerant because Parse barcode_data.csv formats
    vary. If no sequence column can be identified, an empty mapping is returned;
    the downstream filter can then fall back to numeric bc_wells matching.
    """
    requested = set(normalize_well(w) for w in requested_wells)
    if not barcode_csv or not os.path.exists(barcode_csv):
        return pd.DataFrame(), {}

    df = read_table_flexible(barcode_csv)
    if df.empty:
        return df, {}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Identify a well column.
    well_scores = []
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.strip()
        if vals.empty:
            continue
        frac = vals.map(lambda x: bool(WELL_RE.match(x))).mean()
        n_req = vals.map(lambda x: normalize_well(x) if WELL_RE.match(x) else None).isin(requested).sum()
        name_bonus = 0.25 if "well" in col.lower() else 0
        well_scores.append((frac + name_bonus, n_req, col))
    well_scores.sort(reverse=True)
    if not well_scores or well_scores[0][0] < 0.25:
        return df, {}
    well_col = well_scores[0][2]
    df["__well_norm"] = df[well_col].map(lambda x: normalize_well(x) if WELL_RE.match(str(x)) else "")

    # Optionally restrict to barcode 1 / round 1 rows if there is an obvious round column.
    round_cols = [c for c in df.columns if any(k in c.lower() for k in ["round", "bc_round", "barcode_round", "barcode_num", "bc_num", "bc_type"])]
    candidate_df = df
    for rc in round_cols:
        vals = df[rc].astype(str).str.lower()
        mask = vals.str.fullmatch(r"(1|bc1|barcode1|barcode_1|round1|round_1|rt|rt1)", na=False)
        if mask.sum() >= max(1, len(requested) // 2):
            candidate_df = df[mask].copy()
            break

    # Identify sequence column.
    seq_scores = []
    for col in candidate_df.columns:
        if col == "__well_norm":
            continue
        vals = candidate_df[col].dropna().astype(str).str.strip()
        if vals.empty:
            continue
        frac_dna = vals.map(lambda x: bool(DNA_RE.match(x))).mean()
        n_unique = vals.nunique()
        name = col.lower()
        name_bonus = 0.0
        if "seq" in name or "sequence" in name:
            name_bonus += 0.25
        if "barcode" in name or re.search(r"\bbc\b", name):
            name_bonus += 0.15
        if "well" in name:
            name_bonus -= 0.5
        seq_scores.append((frac_dna + name_bonus, n_unique, col))
    seq_scores.sort(reverse=True)
    mapping: Dict[str, str] = {}
    if seq_scores and seq_scores[0][0] >= 0.5:
        seq_col = seq_scores[0][2]
        sub = candidate_df[candidate_df["__well_norm"].isin(requested)].copy()
        for _, row in sub.iterrows():
            seq = str(row[seq_col]).strip().upper()
            if DNA_RE.match(seq):
                mapping[str(row["__well_norm"])] = seq
    return candidate_df, mapping


def load_called_cells(paths: Sequence[str]) -> Tuple[Set[str], str, pd.DataFrame]:
    """Load called cell ids from the first existing metadata file.

    Returns a normalized set of possible cell IDs. The loader includes values
    from likely ID columns and the first column to be robust to index columns.
    """
    for path in paths:
        if not path:
            continue
        if not os.path.exists(path):
            continue
        try:
            df = read_table_flexible(path)
        except Exception as exc:
            eprint(f"WARNING: failed to read called-cell file {path}: {exc}")
            continue
        ids: Set[str] = set()
        # Include first column and named ID-like columns.
        columns = list(df.columns)
        candidate_cols = []
        for c in columns:
            cl = str(c).lower()
            if any(k in cl for k in ["cell", "barcode", "bc_bcis", "bcis"]):
                candidate_cols.append(c)
        if columns:
            candidate_cols.append(columns[0])
        for c in list(dict.fromkeys(candidate_cols)):
            vals = df[c].dropna().astype(str).map(normalize_numeric_triplet)
            ids.update(v for v in vals if v and v.lower() != "nan")
        return ids, path, df.head(10)
    return set(), "", pd.DataFrame()


def existing_paths(paths: Sequence[str]) -> List[str]:
    return [p for p in paths if p and os.path.exists(p)]


def bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x
    return x.astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])


@dataclass
class SampleConfig:
    sample_id: str
    method: str
    tscp_assignment_csv: str
    wells: str
    called_cells_path: str = ""
    called_cells_candidates: List[str] = field(default_factory=list)
    # "auto" is recommended for Parse/SPLiT tscp_assignment.csv because
    # bc_wells order in tscp files can differ from DGE/cell_metadata IDs.
    # The script scores several representations against called cells and uses
    # the best-overlap representation.
    cell_identity_column: str = "auto"
    cell_identity_transform: str = "auto"


@dataclass
class GlobalConfig:
    output_dir: str
    barcode_data_csv: str
    samples: List[SampleConfig]
    downsample_fractions: List[float]
    n_replicates: int = 5
    seed: int = 1
    chunk_size: int = 1_000_000
    bc1_cell_barcode_position: int = 1
    bc1_bc_wells_position: int = 1
    filter_mode: str = "auto"  # auto, cell_barcode_sequence, bc_wells, either
    exclude_gene_name_regex: str = r"^(?:sg_|gRNA|guide$|b08$)"
    exclude_gene_id_regex: str = r"^(?:gene_id_|id_b08$)"
    strict_called_cells: bool = False
    include_per_cell_downsampling: bool = True


def load_config(path: str) -> GlobalConfig:
    with open(path) as fh:
        d = json.load(fh)
    samples = [SampleConfig(**s) for s in d["samples"]]
    return GlobalConfig(samples=samples, **{k: v for k, v in d.items() if k != "samples"})


def make_sample_molecule_table(
    sample: SampleConfig,
    cfg: GlobalConfig,
    barcode_mapping: Dict[str, str],
    out_qc_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    wells = expand_well_spec(sample.wells)
    requested_set = set(wells)
    accepted_sequences = {barcode_mapping[w] for w in wells if w in barcode_mapping}
    accepted_plate_indices = {f"{well_to_plate_index(w):02d}" for w in wells}
    accepted_plate_indices.update({str(well_to_plate_index(w)) for w in wells})

    called_paths = []
    if sample.called_cells_path:
        called_paths.append(sample.called_cells_path)
    called_paths.extend(sample.called_cells_candidates or [])
    called_cells, called_path_used, called_head = load_called_cells(called_paths)

    required = {"bc_wells", "gene", "gene_name", "count", "exonic", "bc_bcis", "cell_barcode", "polyN"}
    usecols = lambda c: c in required

    chunks = []
    qc = {
        "sample_id": sample.sample_id,
        "method": sample.method,
        "tscp_assignment_csv": sample.tscp_assignment_csv,
        "requested_wells": ",".join(wells),
        "accepted_sequences_from_barcode_csv": ",".join(sorted(accepted_sequences)),
        "accepted_plate_indices_fallback": ",".join(sorted(accepted_plate_indices)),
        "called_cells_path_used": called_path_used,
        "n_called_cells_loaded": len(called_cells),
        "filter_mode": cfg.filter_mode,
        "bc1_cell_barcode_position": cfg.bc1_cell_barcode_position,
        "bc1_bc_wells_position": cfg.bc1_bc_wells_position,
        "rows_total_seen": 0,
        "rows_after_well_filter": 0,
        "rows_after_called_cell_filter": 0,
        "rows_after_transcriptome_filter": 0,
        "rows_with_called_cell_overlap_before_filter": 0,
        "chunks_with_selected_rows_no_called_cell_overlap": 0,
        "n_chunks": 0,
    }

    identity_choice: Optional[Tuple[str, str]] = None
    identity_scores_written = False

    eprint(f"[{now()}] Reading {sample.sample_id}: {sample.tscp_assignment_csv}")
    for chunk in pd.read_csv(sample.tscp_assignment_csv, usecols=usecols, chunksize=cfg.chunk_size):
        qc["n_chunks"] += 1
        qc["rows_total_seen"] += len(chunk)
        # Ensure required columns exist.
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing columns in {sample.tscp_assignment_csv}: {sorted(missing)}")

        # Match selected BC1 wells.
        mask_seq = pd.Series(False, index=chunk.index)
        if accepted_sequences:
            try:
                bc1_seq = split_component(chunk["cell_barcode"], cfg.bc1_cell_barcode_position).str.upper()
                mask_seq = bc1_seq.isin(accepted_sequences)
            except Exception:
                mask_seq = pd.Series(False, index=chunk.index)

        mask_well = pd.Series(False, index=chunk.index)
        try:
            bc1_well = split_component(chunk["bc_wells"], cfg.bc1_bc_wells_position).astype(str).str.lstrip("0")
            well_values = {v.lstrip("0") for v in accepted_plate_indices}
            mask_well = bc1_well.isin(well_values)
        except Exception:
            mask_well = pd.Series(False, index=chunk.index)

        if cfg.filter_mode == "cell_barcode_sequence":
            mask = mask_seq
        elif cfg.filter_mode == "bc_wells":
            mask = mask_well
        elif cfg.filter_mode == "either":
            mask = mask_seq | mask_well
        else:  # auto
            mask = mask_seq if accepted_sequences else mask_well
        chunk = chunk.loc[mask].copy()
        qc["rows_after_well_filter"] += len(chunk)
        if chunk.empty:
            continue

        # Construct a cell ID compatible with DGE/cell_metadata.
        # For Parse/SPLiT outputs, tscp_assignment.csv may encode the three rounds
        # in a different order than DGE/cell_metadata.  We therefore infer the
        # representation with maximal called-cell overlap.
        if identity_choice is None:
            if called_cells and sample.cell_identity_column.lower() == "auto":
                score_df = score_identity_candidates(chunk.head(min(len(chunk), 200000)), called_cells)
                if not score_df.empty:
                    score_df.to_csv(Path(out_qc_dir) / f"{sample.sample_id}.cell_identity_candidate_scores.tsv", sep="\t", index=False)
                    identity_scores_written = True
                    best = score_df.iloc[0]
                    if int(best["overlap_rows"]) > 0:
                        identity_choice = (str(best["source"]), str(best["transform"]))
                    else:
                        # Fall back to a Parse-friendly bc_wells ordering if no match is found.
                        identity_choice = ("bc_wells", "perm312") if "bc_wells" in chunk.columns else ("cell_barcode", "sequence")
                else:
                    identity_choice = ("bc_wells", "perm312") if "bc_wells" in chunk.columns else ("cell_barcode", "sequence")
            else:
                src = sample.cell_identity_column if sample.cell_identity_column in chunk.columns else "cell_barcode"
                tr = sample.cell_identity_transform if sample.cell_identity_transform != "auto" else ("identity_numeric" if src != "cell_barcode" else "sequence")
                identity_choice = (src, tr)
            qc["cell_identity_source_selected"] = identity_choice[0]
            qc["cell_identity_transform_selected"] = identity_choice[1]

        chunk["cell_id"] = build_identity_series(chunk, identity_choice[0], identity_choice[1])
        chunk["cell_id_source"] = identity_choice[0]
        chunk["cell_id_transform"] = identity_choice[1]

        # Apply called-cell filter when possible.
        # v3 note: many Parse tscp_assignment.csv files are sorted by raw barcode
        # combinations, so early chunks can contain valid selected wells but no
        # called cells. That is not an error. We filter silently per chunk and
        # only warn once after the full file if the total overlap is zero.
        if called_cells:
            before = len(chunk)
            m_called = chunk["cell_id"].isin(called_cells)
            n_overlap = int(m_called.sum())
            qc["rows_with_called_cell_overlap_before_filter"] += n_overlap
            if n_overlap == 0 and before > 0:
                qc["chunks_with_selected_rows_no_called_cell_overlap"] += 1
            chunk = chunk.loc[m_called].copy()
        qc["rows_after_called_cell_filter"] += len(chunk)
        if chunk.empty:
            continue

        # Exclude guide/non-transcriptome features.
        gene_name = chunk["gene_name"].fillna("").astype(str)
        gene_id = chunk["gene"].fillna("").astype(str)
        guide_mask = gene_name.str.contains(cfg.exclude_gene_name_regex, regex=True, case=False) | gene_id.str.contains(cfg.exclude_gene_id_regex, regex=True, case=False)
        chunk = chunk.loc[~guide_mask].copy()
        qc["rows_after_transcriptome_filter"] += len(chunk)
        if chunk.empty:
            continue

        chunk["read_count"] = pd.to_numeric(chunk["count"], errors="coerce").fillna(0).astype(np.int64)
        chunk = chunk[chunk["read_count"] > 0].copy()
        chunk["is_exonic"] = bool_series(chunk["exonic"])
        chunk["gene_key"] = np.where(gene_name.loc[chunk.index].astype(str).str.len() > 0, gene_name.loc[chunk.index].astype(str), gene_id.loc[chunk.index].astype(str))
        keep_cols = ["cell_id", "gene", "gene_name", "gene_key", "read_count", "is_exonic"]
        chunks.append(chunk[keep_cols])

    if chunks:
        mol = pd.concat(chunks, ignore_index=True)
    else:
        mol = pd.DataFrame(columns=["cell_id", "gene", "gene_name", "gene_key", "read_count", "is_exonic"])

    # Warn once if called-cell filtering failed globally.
    if called_cells and qc["rows_after_well_filter"] > 0 and qc["rows_after_called_cell_filter"] == 0:
        msg = (
            f"No global overlap between selected rows and called-cell IDs for {sample.sample_id} "
            f"using {qc.get('cell_identity_source_selected', '')}:{qc.get('cell_identity_transform_selected', '')}. "
            f"Check quality_checks/{sample.sample_id}.cell_identity_candidate_scores.tsv and processing_qc.tsv."
        )
        if cfg.strict_called_cells:
            raise ValueError(msg)
        eprint("WARNING:", msg)
    elif called_cells and qc["rows_after_called_cell_filter"] > 0:
        eprint(
            f"INFO: {sample.sample_id}: retained {qc['rows_after_called_cell_filter']} rows after called-cell filtering "
            f"using {qc.get('cell_identity_source_selected', '')}:{qc.get('cell_identity_transform_selected', '')}."
        )

    # Write sample-level mapping/QC.
    mapping_rows = []
    for w in wells:
        mapping_rows.append({
            "sample_id": sample.sample_id,
            "method": sample.method,
            "well": w,
            "plate_index_row_major_A1_1": well_to_plate_index(w),
            "barcode_sequence_from_barcode_data": barcode_mapping.get(w, ""),
        })
    mapping_df = pd.DataFrame(mapping_rows)
    qc.update({
        "n_molecules_retained": int(len(mol)),
        "n_cells_retained": int(mol["cell_id"].nunique()) if not mol.empty else 0,
        "n_genes_retained": int(mol["gene_key"].nunique()) if not mol.empty else 0,
        "assigned_reads_retained": int(mol["read_count"].sum()) if not mol.empty else 0,
        "n_exonic_molecules_retained": int(mol["is_exonic"].sum()) if not mol.empty else 0,
    })
    return mol, mapping_df, qc


def summarize_full_depth(mol: pd.DataFrame, sample: SampleConfig, feature_set: str) -> pd.DataFrame:
    if feature_set == "exonic_only":
        m = mol[mol["is_exonic"]].copy()
    else:
        m = mol.copy()
    if m.empty:
        return pd.DataFrame()
    cells = pd.Index(sorted(m["cell_id"].unique()))
    reads = m.groupby("cell_id")["read_count"].sum().reindex(cells, fill_value=0)
    umis = m.groupby("cell_id").size().reindex(cells, fill_value=0)
    genes = m.drop_duplicates(["cell_id", "gene_key"]).groupby("cell_id").size().reindex(cells, fill_value=0)
    return pd.DataFrame({
        "sample_id": sample.sample_id,
        "method": sample.method,
        "feature_set": feature_set,
        "cell_id": cells,
        "full_depth_assigned_reads": reads.values.astype(int),
        "full_depth_umis": umis.values.astype(int),
        "full_depth_genes": genes.values.astype(int),
    })


def downsample_one_feature_set(
    mol: pd.DataFrame,
    sample: SampleConfig,
    feature_set: str,
    fractions: Sequence[float],
    n_reps: int,
    rng: np.random.Generator,
    include_per_cell: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if feature_set == "exonic_only":
        m = mol[mol["is_exonic"]].copy()
    else:
        m = mol.copy()
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Fixed cell set for this feature set.
    cells = pd.Index(sorted(m["cell_id"].unique()))
    read_counts = m["read_count"].to_numpy(dtype=np.int64)
    cell_arr = m["cell_id"].to_numpy()
    gene_arr = m["gene_key"].to_numpy()

    summary_rows = []
    per_cell_frames = []
    n_cells = len(cells)
    total_full_reads = int(read_counts.sum())
    total_full_molecules = int(len(m))

    for frac in fractions:
        frac = float(frac)
        if not (0 < frac <= 1):
            raise ValueError(f"Downsample fraction must be in (0,1]: {frac}")
        for rep in range(1, n_reps + 1):
            if frac >= 1.0:
                sampled_counts = read_counts.copy()
            else:
                sampled_counts = rng.binomial(read_counts, frac)
            observed = sampled_counts > 0
            sampled_reads_total = int(sampled_counts.sum())
            if observed.any():
                obs = pd.DataFrame({
                    "cell_id": cell_arr[observed],
                    "gene_key": gene_arr[observed],
                    "sampled_reads_molecule": sampled_counts[observed],
                })
                cell_reads = obs.groupby("cell_id")["sampled_reads_molecule"].sum().reindex(cells, fill_value=0)
                cell_umis = obs.groupby("cell_id").size().reindex(cells, fill_value=0)
                cell_genes = obs.drop_duplicates(["cell_id", "gene_key"]).groupby("cell_id").size().reindex(cells, fill_value=0)
            else:
                cell_reads = pd.Series(0, index=cells)
                cell_umis = pd.Series(0, index=cells)
                cell_genes = pd.Series(0, index=cells)

            summary_rows.append({
                "sample_id": sample.sample_id,
                "method": sample.method,
                "feature_set": feature_set,
                "downsample_fraction": frac,
                "replicate": rep,
                "n_cells": n_cells,
                "full_depth_assigned_reads_total": total_full_reads,
                "full_depth_molecules_total": total_full_molecules,
                "sampled_assigned_reads_total": sampled_reads_total,
                "assigned_reads_per_cell": sampled_reads_total / n_cells if n_cells else np.nan,
                "median_umis_per_cell": float(np.median(cell_umis.values)),
                "mean_umis_per_cell": float(np.mean(cell_umis.values)),
                "q25_umis_per_cell": float(np.quantile(cell_umis.values, 0.25)),
                "q75_umis_per_cell": float(np.quantile(cell_umis.values, 0.75)),
                "median_genes_per_cell": float(np.median(cell_genes.values)),
                "mean_genes_per_cell": float(np.mean(cell_genes.values)),
                "q25_genes_per_cell": float(np.quantile(cell_genes.values, 0.25)),
                "q75_genes_per_cell": float(np.quantile(cell_genes.values, 0.75)),
            })
            if include_per_cell:
                per_cell_frames.append(pd.DataFrame({
                    "sample_id": sample.sample_id,
                    "method": sample.method,
                    "feature_set": feature_set,
                    "downsample_fraction": frac,
                    "replicate": rep,
                    "cell_id": cells,
                    "sampled_assigned_reads": cell_reads.values.astype(int),
                    "sampled_umis": cell_umis.values.astype(int),
                    "sampled_genes": cell_genes.values.astype(int),
                }))

    summary = pd.DataFrame(summary_rows)
    per_cell = pd.concat(per_cell_frames, ignore_index=True) if per_cell_frames else pd.DataFrame()
    return summary, per_cell


def write_tsv_gz(df: pd.DataFrame, path: str):
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description="Export transcriptome downsampling plot inputs for Fig. 1d.")
    parser.add_argument("--config", "--config-json", dest="config", required=True, help="JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "quality_checks"
    qc_dir.mkdir(exist_ok=True)

    all_requested_wells: List[str] = []
    for s in cfg.samples:
        all_requested_wells.extend(expand_well_spec(s.wells))
    barcode_df, barcode_mapping = infer_barcode_mapping(cfg.barcode_data_csv, all_requested_wells)
    if not barcode_df.empty:
        barcode_df.head(200).to_csv(qc_dir / "barcode_data_head.tsv", sep="\t", index=False)
    if barcode_mapping:
        pd.DataFrame([{"well": k, "barcode_sequence": v} for k, v in sorted(barcode_mapping.items())]).to_csv(qc_dir / "inferred_barcode_mapping.tsv", sep="\t", index=False)
    else:
        eprint("INFO: no well-to-sequence mapping inferred from barcode_data.csv; using numeric bc_wells fallback. This is expected for Parse tscp_assignment.csv when filtering by well index.")

    all_mapping = []
    all_qc = []
    baseline_tables = []
    summary_tables = []
    per_cell_tables = []

    rng = np.random.default_rng(cfg.seed)

    for sample in cfg.samples:
        mol, mapping_df, qc = make_sample_molecule_table(sample, cfg, barcode_mapping, str(qc_dir))
        all_mapping.append(mapping_df)
        all_qc.append(qc)
        if mol.empty:
            eprint(f"WARNING: no retained transcriptome molecules for {sample.sample_id}")
            continue
        # Write sample molecule summaries, not the full molecule table, to keep bundle compact.
        gene_summary = (
            mol.groupby(["gene_key", "gene_name", "gene"], dropna=False)
            .agg(n_molecules=("read_count", "size"), assigned_reads=("read_count", "sum"), n_cells=("cell_id", "nunique"))
            .reset_index()
            .sort_values("assigned_reads", ascending=False)
            .head(5000)
        )
        gene_summary.to_csv(qc_dir / f"{sample.sample_id}.top_gene_summary.tsv", sep="\t", index=False)

        for feature_set in ["all_non_guide", "exonic_only"]:
            baseline = summarize_full_depth(mol, sample, feature_set)
            if not baseline.empty:
                baseline_tables.append(baseline)
            summary, per_cell = downsample_one_feature_set(
                mol,
                sample,
                feature_set,
                cfg.downsample_fractions,
                cfg.n_replicates,
                rng,
                include_per_cell=cfg.include_per_cell_downsampling,
            )
            if not summary.empty:
                summary_tables.append(summary)
            if not per_cell.empty:
                per_cell_tables.append(per_cell)

    mapping_all = pd.concat(all_mapping, ignore_index=True) if all_mapping else pd.DataFrame()
    qc_all = pd.DataFrame(all_qc)
    baseline_all = pd.concat(baseline_tables, ignore_index=True) if baseline_tables else pd.DataFrame()
    summary_all = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    per_cell_all = pd.concat(per_cell_tables, ignore_index=True) if per_cell_tables else pd.DataFrame()

    mapping_all.to_csv(out_dir / "barcode_filter_mapping.tsv", sep="\t", index=False)
    qc_all.to_csv(out_dir / "processing_qc.tsv", sep="\t", index=False)
    if not baseline_all.empty:
        write_tsv_gz(baseline_all, out_dir / "baseline_cell_metrics.tsv.gz")
    if not summary_all.empty:
        write_tsv_gz(summary_all, out_dir / "transcriptome_downsampling_summary.tsv.gz")
    if not per_cell_all.empty:
        write_tsv_gz(per_cell_all, out_dir / "transcriptome_downsampling_per_cell.tsv.gz")

    dd = pd.DataFrame([
        {"file": "barcode_filter_mapping.tsv", "description": "Requested well set and inferred barcode sequence mapping used for each sample."},
        {"file": "processing_qc.tsv", "description": "Rows retained at each processing step and called-cell overlap diagnostics."},
        {"file": "baseline_cell_metrics.tsv.gz", "description": "Full-depth per-cell assigned reads, UMIs, and genes before downsampling."},
        {"file": "transcriptome_downsampling_summary.tsv.gz", "description": "Plot-ready saturation curves: median UMIs/genes per cell at each assigned-read depth."},
        {"file": "transcriptome_downsampling_per_cell.tsv.gz", "description": "Per-cell downsampled transcriptome metrics for optional box/violin plots."},
        {"file": "quality_checks/*.tsv", "description": "Barcode-data head and top gene summaries for diagnostics."},
    ])
    dd.to_csv(out_dir / "data_dictionary.tsv", sep="\t", index=False)

    manifest = {
        "created_at": now(),
        "script": os.path.basename(__file__),
        "config": cfg.__dict__.copy(),
        "notes": [
            "Downsampling uses post-mapping tscp_assignment.csv read support via Binomial(count, fraction).",
            "Use x-axis label 'assigned transcriptome reads per cell' unless raw BAM/FASTQ downsampling is performed.",
            "Feature set all_non_guide excludes guide-like features by regex and includes exonic and non-exonic assignments.",
            "Feature set exonic_only further restricts all_non_guide to exonic=True rows.",
        ],
    }
    manifest["config"]["samples"] = [s.__dict__ for s in cfg.samples]
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    readme = f"""# Figure 1d transcriptome downsampling plot-input bundle

Created: {now()}

Upload this entire ZIP to ChatGPT to generate the matched-depth transcriptome saturation plot.

Main plotting table:

- `transcriptome_downsampling_summary.tsv.gz`

Recommended main panel:

- x-axis: `assigned_reads_per_cell`
- y-axis: `median_umis_per_cell`
- color: `method`
- use `feature_set == 'all_non_guide'` first

Recommended supplement:

- y-axis: `median_genes_per_cell`
- optionally compare `all_non_guide` vs `exonic_only`

Important: this is post-mapping assigned-read downsampling from `tscp_assignment.csv`, not raw FASTQ/BAM remapping.
"""
    with open(out_dir / "README_upload_this_to_chatgpt.md", "w") as fh:
        fh.write(readme)

    zip_path = out_dir / "fig1d_transcriptome_downsampling_plot_input_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path == zip_path:
                continue
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(out_dir)))
    print(str(zip_path))


if __name__ == "__main__":
    main()
