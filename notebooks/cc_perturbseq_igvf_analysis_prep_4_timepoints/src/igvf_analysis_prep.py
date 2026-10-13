"""Read-only audit and local staging helpers for the 300-gene CC-Perturb-seq IGVF analysis submission.

This module intentionally contains no POST, PATCH, or upload functions. It can:
- inspect local h5ad/h5mu files;
- split a combined h5ad by timepoint without modifying the source;
- derive candidate guide-quantification and per-cell QC tables;
- convert existing CSV/TSV/XLSX tables to deterministic ``.tsv.gz`` files;
- validate staged files and calculate checksums;
- query the IGVF portal read-only for object/audit status;
- build draft portal metadata TSVs after all decision gates are resolved.

The converter is deliberately conservative. It raises actionable errors instead of
silently guessing a day column, guide matrix, schema, or AnalysisSet mapping.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import pandas as pd
import requests
import yaml

PORTAL_BASE = "https://api.data.igvf.org"
PORTAL_PUBLIC_BASE = "https://data.igvf.org"
DAY_ORDER = ("D0", "D1", "D2", "D3")
RETRY_STATUS = {429, 500, 502, 503, 504}


class PrepError(RuntimeError):
    """Raised when a conversion would require an unsupported assumption."""


class DecisionRequired(PrepError):
    """Raised when the prompt explicitly requires a human/DACC decision."""

def validate_timepoint_analysis_sets(
    mapping: Mapping[str, str],
) -> dict[str, str]:
    """Require exactly four unique AnalysisSets keyed by D0, D1, D2, and D3.

    This prevents a combined or example AnalysisSet from being used as a fifth
    active file_set and prevents timepoints from being silently mixed.
    """
    normalized = {str(k).strip().upper(): str(v).strip() for k, v in mapping.items()}
    expected_days = set(DAY_ORDER)
    observed_days = set(normalized)
    missing = sorted(expected_days - observed_days)
    extra = sorted(observed_days - expected_days)
    if missing or extra:
        raise DecisionRequired(
            "Active AnalysisSet mapping must contain exactly D0, D1, D2, and D3. "
            f"Missing={missing}; extra={extra}"
        )
    empty = [day for day in DAY_ORDER if not normalized.get(day)]
    if empty:
        raise DecisionRequired(f"Missing AnalysisSet accession for: {', '.join(empty)}")
    values = [normalized[day] for day in DAY_ORDER]
    if len(set(values)) != len(values):
        raise DecisionRequired("Each timepoint must use a distinct AnalysisSet accession")
    invalid = [value for value in values if not re.fullmatch(r"IGVFDS[A-Z0-9]+", value)]
    if invalid:
        raise DecisionRequired(
            "Unexpected AnalysisSet accession format: " + ", ".join(invalid)
        )
    return {day: normalized[day] for day in DAY_ORDER}



def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise PrepError(f"Expected a YAML mapping in {path}")
    return data


def write_yaml(data: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)
    return path


def normalize_day(value: Any) -> str | None:
    """Normalize common timepoint values to D0-D3."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper().replace("DAY", "D")
    text = re.sub(r"[^A-Z0-9]", "", text)
    aliases = {
        "0": "D0", "D0": "D0",
        "1": "D1", "D1": "D1",
        "2": "D2", "D2": "D2",
        "3": "D3", "D3": "D3",
    }
    return aliases.get(text)


def md5sum(path: str | Path, block_size: int = 16 * 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "md5sum": md5sum(path) if path.is_file() else None,
    }


def load_tsv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kwargs)


def save_tsv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")
    return path


def _detect_delimiter(path: Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    if ".tsv" in suffixes or path.suffix.lower() in {".tab"}:
        return "\t"
    if ".csv" in suffixes:
        return ","
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(32_768)
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        return "\t"


def read_table_auto(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read CSV/TSV/TXT/XLSX without changing column names.

    XLSX uses the ``calamine`` engine so this package does not mutate or depend on
    the metadata workbook's formatting.
    """
    path = Path(path)
    lower = path.name.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, sheet_name=sheet_name, engine="calamine")
    delimiter = _detect_delimiter(path)
    return pd.read_csv(path, sep=delimiter, compression="infer")


def sanitize_column_name(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed_column"


def normalize_table_columns(
    df: pd.DataFrame,
    *,
    rename_columns: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = [str(c) for c in df.columns]
    if len(set(original)) != len(original):
        dupes = sorted({c for c in original if original.count(c) > 1})
        raise PrepError(f"Duplicate input column names: {dupes}")
    normalized = [sanitize_column_name(c) if rename_columns else str(c).strip() for c in original]
    if len(set(normalized)) != len(normalized):
        raise PrepError("Column normalization would create duplicate names; inspect the source table.")
    out = df.copy()
    out.columns = normalized
    mapping = pd.DataFrame({"original_column": original, "submitted_column": normalized})
    return out, mapping


def write_deterministic_tsv_gz(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    na_rep: str = "NA",
) -> Path:
    """Write a reproducible gzip stream (mtime=0) containing a tab-delimited table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + ".", dir=output_path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                    df.to_csv(text, sep="\t", index=False, na_rep=na_rep, lineterminator="\n")
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output_path


def convert_table_to_tsv_gz(
    source_path: str | Path,
    output_path: str | Path,
    *,
    sheet_name: str | int = 0,
    rename_columns: bool = False,
    na_rep: str = "NA",
) -> tuple[Path, pd.DataFrame]:
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    df = read_table_auto(source_path, sheet_name=sheet_name)
    df, mapping = normalize_table_columns(df, rename_columns=rename_columns)
    if df.empty:
        raise PrepError(f"Refusing to stage an empty table: {source_path}")
    out = write_deterministic_tsv_gz(df, output_path, na_rep=na_rep)
    return out, mapping


def validate_tsv_gz(path: str | Path) -> dict[str, Any]:
    """Validate gzip integrity and tabular structure without loading the full file."""
    path = Path(path)
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "gzip_ok": False,
        "tab_delimited": False,
        "row_count": None,
        "column_count": None,
        "duplicate_columns": [],
        "malformed_rows": 0,
        "empty": None,
        "status": "ERROR",
        "error": "",
    }
    if not path.is_file():
        report["error"] = "missing file"
        return report
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            report["gzip_ok"] = True
            if header is None:
                report["empty"] = True
                report["error"] = "empty gzip stream"
                return report
            report["column_count"] = len(header)
            report["duplicate_columns"] = sorted({c for c in header if header.count(c) > 1})
            rows = 0
            malformed = 0
            for row in reader:
                rows += 1
                if len(row) != len(header):
                    malformed += 1
            report["row_count"] = rows
            report["malformed_rows"] = malformed
            report["empty"] = rows == 0
            report["tab_delimited"] = len(header) > 1
            errors = []
            if len(header) <= 1:
                errors.append("not recognizably tab-delimited")
            if report["duplicate_columns"]:
                errors.append("duplicate columns")
            if malformed:
                errors.append(f"{malformed} malformed rows")
            if rows == 0:
                errors.append("no data rows")
            if errors:
                report["error"] = "; ".join(errors)
            else:
                report["status"] = "PASS"
    except Exception as exc:  # gzip, Unicode, CSV errors
        report["error"] = f"{type(exc).__name__}: {exc}"
    if report["exists"]:
        report["size_bytes"] = path.stat().st_size
        report["md5sum"] = md5sum(path)
    return report



def validate_tabular_semantics(
    path: str | Path,
    *,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    """Check common numeric constraints without assuming a project-specific schema.

    The function only applies constraints when a column name clearly indicates
    a p/q/FDR/probability or correlation statistic. It reports, rather than
    renames or repairs, questionable values.
    """
    path = Path(path)
    report: dict[str, Any] = {
        "path": str(path),
        "status": "PASS",
        "range_violations": [],
        "nonnumeric_values": [],
        "nonfinite_values": [],
        "missing_tokens_observed": [],
    }
    probability_pattern = re.compile(
        r"(^|_)(pvalue|p_value|pval|p_val|qvalue|q_value|qval|q_val|fdr|false_discovery_rate|probability|posterior_probability)($|_)",
        re.IGNORECASE,
    )
    correlation_pattern = re.compile(
        r"(correlation|pearson|spearman|rho)",
        re.IGNORECASE,
    )
    numeric_pattern = re.compile(
        r"(effect|loading|log2[_-]?(fc|fold)|fold[_-]?change|score|statistic|z[_-]?score)",
        re.IGNORECASE,
    )
    missing_tokens: set[str] = set()
    range_counts: dict[str, int] = {}
    nonnumeric_counts: dict[str, int] = {}
    nonfinite_counts: dict[str, int] = {}
    try:
        for chunk in pd.read_csv(path, sep="\t", compression="gzip", chunksize=chunksize):
            for column in chunk.columns:
                name = str(column)
                series = chunk[column]
                raw_non_null = series.dropna()
                for token in raw_non_null.astype(str).unique()[:1000]:
                    if token.strip().lower() in {"na", "nan", "n/a", "null", "none", ".", ""}:
                        missing_tokens.add(token)
                if not (
                    probability_pattern.search(name)
                    or correlation_pattern.search(name)
                    or numeric_pattern.search(name)
                ):
                    continue
                numeric = pd.to_numeric(series, errors="coerce")
                nonmissing = series.notna()
                bad_numeric = int((nonmissing & numeric.isna()).sum())
                if bad_numeric:
                    nonnumeric_counts[name] = nonnumeric_counts.get(name, 0) + bad_numeric
                finite_mask = numeric.notna()
                if finite_mask.any():
                    import numpy as np
                    nonfinite = int((~np.isfinite(numeric[finite_mask].astype(float))).sum())
                    if nonfinite:
                        nonfinite_counts[name] = nonfinite_counts.get(name, 0) + nonfinite
                    vals = numeric[finite_mask].astype(float)
                    if probability_pattern.search(name):
                        bad = int(((vals < 0) | (vals > 1)).sum())
                        if bad:
                            range_counts[name] = range_counts.get(name, 0) + bad
                    elif correlation_pattern.search(name):
                        bad = int(((vals < -1) | (vals > 1)).sum())
                        if bad:
                            range_counts[name] = range_counts.get(name, 0) + bad
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["range_violations"] = [
        {"column": column, "count": count} for column, count in sorted(range_counts.items())
    ]
    report["nonnumeric_values"] = [
        {"column": column, "count": count} for column, count in sorted(nonnumeric_counts.items())
    ]
    report["nonfinite_values"] = [
        {"column": column, "count": count} for column, count in sorted(nonfinite_counts.items())
    ]
    report["missing_tokens_observed"] = sorted(missing_tokens)
    if report["range_violations"] or report["nonnumeric_values"] or report["nonfinite_values"]:
        report["status"] = "ERROR"
    return report


def infer_column_dictionary(
    table_path: str | Path,
    *,
    table_role: str,
    source_analysis_step: str,
    nrows: int = 10_000,
) -> pd.DataFrame:
    df = pd.read_csv(table_path, sep="\t", compression="gzip", nrows=nrows)
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        dtype = str(series.dtype)
        non_null = series.dropna()
        allowed = ""
        if 0 < non_null.nunique(dropna=True) <= 20:
            allowed = "|".join(map(str, sorted(non_null.astype(str).unique())))
        rows.append({
            "table_file_type": table_role,
            "column_name": column,
            "data_type": dtype,
            "units_or_scale": "TBD",
            "required": "TBD",
            "missing_value": "NA or blank; confirm",
            "description": "TBD - complete before DACC review",
            "allowed_values": allowed,
            "source_analysis_step": source_analysis_step,
        })
    return pd.DataFrame(rows)


def _import_anndata():
    try:
        import anndata as ad
    except ImportError as exc:
        raise PrepError("Install anndata before inspecting h5ad files.") from exc
    return ad


def _import_mudata():
    try:
        import mudata as mu
    except ImportError as exc:
        raise PrepError("Install mudata before inspecting h5mu files.") from exc
    return mu


def _find_day_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    exact = {str(c): c for c in frame.columns}
    lower = {str(c).lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in exact:
            return str(exact[candidate])
        if candidate.lower() in lower:
            return str(lower[candidate.lower()])
    raise DecisionRequired(
        f"Could not identify a day column. Candidates={list(candidates)}; "
        f"available obs columns={list(map(str, frame.columns))[:100]}"
    )


def inspect_h5ad(
    path: str | Path,
    *,
    day_column_candidates: Sequence[str] = ("day", "timepoint", "Day"),
) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "status": "ERROR"}
    if not path.is_file():
        result["error"] = "missing file"
        return result
    ad = _import_anndata()
    handle = None
    try:
        handle = ad.read_h5ad(path, backed="r")
        result.update({
            "n_obs": int(handle.n_obs),
            "n_vars": int(handle.n_vars),
            "obs_names_unique": bool(handle.obs_names.is_unique),
            "var_names_unique": bool(handle.var_names.is_unique),
            "obs_columns": list(map(str, handle.obs.columns)),
            "var_columns": list(map(str, handle.var.columns)),
            "obsm_keys": list(map(str, handle.obsm.keys())),
            "layers": list(map(str, handle.layers.keys())),
            "uns_keys": list(map(str, handle.uns.keys())),
            "size_bytes": path.stat().st_size,
            "md5sum": md5sum(path),
        })
        guide_keys = sorted({
            key for key in [
                *map(str, handle.obsm.keys()),
                *map(str, handle.obs.columns),
                *map(str, handle.uns.keys()),
            ]
            if re.search(r"guide|grna|sgrna", key, re.IGNORECASE)
        })
        result["guide_information_keys"] = guide_keys
        result["guide_information_present"] = bool(guide_keys)
        result["rna_matrix_present"] = bool(handle.n_obs > 0 and handle.n_vars > 0)
        try:
            day_col = _find_day_column(handle.obs, day_column_candidates)
            normalized = handle.obs[day_col].map(normalize_day)
            result["day_column"] = day_col
            result["day_values_raw"] = sorted(map(str, handle.obs[day_col].dropna().unique()))
            result["day_values_normalized"] = sorted(v for v in normalized.dropna().unique())
            result["unmapped_day_count"] = int(normalized.isna().sum())
        except DecisionRequired as exc:
            result["day_column_warning"] = str(exc)
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if handle is not None and getattr(handle, "file", None) is not None:
                handle.file.close()
        except Exception:
            pass
    return result


def split_h5ad_by_day(
    source_path: str | Path,
    output_paths: Mapping[str, str | Path],
    *,
    day_column_candidates: Sequence[str] = ("day", "timepoint", "Day"),
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create day-specific h5ad copies from one combined h5ad.

    This never edits the source file. It refuses to continue when day labels do
    not map cleanly to D0-D3 or when an expected day is absent.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    ad = _import_anndata()
    adata = ad.read_h5ad(source_path)
    day_col = _find_day_column(adata.obs, day_column_candidates)
    normalized = adata.obs[day_col].map(normalize_day)
    if normalized.isna().any():
        raw = sorted(map(str, adata.obs.loc[normalized.isna(), day_col].dropna().unique()))
        raise DecisionRequired(f"Unmapped day values in {day_col}: {raw}")
    reports = []
    for day in DAY_ORDER:
        if day not in output_paths:
            raise PrepError(f"No output path configured for {day}")
        mask = normalized.eq(day).to_numpy()
        if not mask.any():
            raise DecisionRequired(f"No cells found for {day} in {source_path}")
        output = Path(output_paths[day])
        if output.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; set overwrite=True explicitly")
        output.parent.mkdir(parents=True, exist_ok=True)
        subset = adata[mask].copy()
        subset.obs["igvf_submission_day"] = day
        subset.uns["igvf_conversion_note"] = (
            f"Day-specific subset generated from {source_path} using obs[{day_col!r}] on "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        subset.write_h5ad(output)
        reports.append({
            "day": day,
            "source_path": str(source_path),
            "output_path": str(output),
            "n_obs": int(subset.n_obs),
            "n_vars": int(subset.n_vars),
            "size_bytes": output.stat().st_size,
            "md5sum": md5sum(output),
        })
    return pd.DataFrame(reports)


def derive_per_cell_quality_report(
    h5ad_path: str | Path,
    output_path: str | Path,
    *,
    explicit_columns: Sequence[str] | None = None,
) -> tuple[Path, pd.DataFrame]:
    ad = _import_anndata()
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs.copy()
    finally:
        try:
            adata.file.close()
        except Exception:
            pass
    patterns = re.compile(
        r"(^|_)(total_counts|n_genes|pct_counts|percent|mito|mt|ribo|guide|umi|doublet|qc|batch|replicate|sample|day|cell_type)($|_)",
        re.IGNORECASE,
    )
    selected = list(explicit_columns or [])
    selected.extend(c for c in obs.columns if patterns.search(str(c)))
    selected = [c for i, c in enumerate(selected) if c in obs.columns and c not in selected[:i]]
    if not selected:
        raise DecisionRequired(
            f"No candidate QC columns found in {h5ad_path}. Available obs columns: {list(obs.columns)}"
        )
    out = obs[selected].copy()
    out.insert(0, "cell_id", obs.index.astype(str))
    path = write_deterministic_tsv_gz(out.reset_index(drop=True), output_path)
    return path, pd.DataFrame({"submitted_column": out.columns})


def _guide_names_from_adata(adata: Any, matrix: Any) -> list[str]:
    if isinstance(matrix, pd.DataFrame):
        return list(map(str, matrix.columns))
    for key in ("guide_names", "guide_ids", "guides", "gRNA_names"):
        if key in adata.uns:
            values = adata.uns[key]
            names = list(map(str, values))
            if len(names) == matrix.shape[1]:
                return names
    raise DecisionRequired(
        "Found a guide-assignment matrix but could not resolve guide names from its columns "
        "or adata.uns['guide_names'/'guide_ids'/'guides']."
    )


def derive_guide_quantifications(
    h5ad_path: str | Path,
    output_path: str | Path,
    *,
    assignment_keys: Sequence[str] = ("guide_assignment", "guide_assignments"),
    umi_keys: Sequence[str] = ("guide_umi_counts", "guide_counts"),
) -> tuple[Path, pd.DataFrame]:
    """Export non-zero cell-guide assignments in long form.

    Output columns are intentionally explicit and minimal. The resulting table
    remains a *candidate* DACC guide-quantification table until its column
    dictionary/content expectations are confirmed.
    """
    ad = _import_anndata()
    adata = ad.read_h5ad(h5ad_path)
    assignment_key = next((k for k in assignment_keys if k in adata.obsm), None)
    if assignment_key is None:
        raise DecisionRequired(
            f"None of {list(assignment_keys)} found in adata.obsm. Available: {list(adata.obsm.keys())}"
        )
    matrix = adata.obsm[assignment_key]
    names = _guide_names_from_adata(adata, matrix)
    cell_ids = list(map(str, adata.obs_names))
    umi_key = next((k for k in umi_keys if k in adata.obsm), None)
    umi_matrix = adata.obsm[umi_key] if umi_key else None

    rows: dict[str, list[Any]] = {
        "cell_id": [],
        "guide_id": [],
        "assignment_value": [],
    }
    if umi_matrix is not None:
        rows["guide_umi_count"] = []

    if isinstance(matrix, pd.DataFrame):
        arr = matrix.to_numpy()
    else:
        arr = matrix

    try:
        from scipy import sparse
        is_sparse = sparse.issparse(arr)
    except ImportError:
        is_sparse = hasattr(arr, "tocoo")

    if is_sparse:
        coo = arr.tocoo()
        for i, j, value in zip(coo.row, coo.col, coo.data):
            if value == 0:
                continue
            rows["cell_id"].append(cell_ids[int(i)])
            rows["guide_id"].append(names[int(j)])
            rows["assignment_value"].append(value)
            if umi_matrix is not None:
                rows["guide_umi_count"].append(float(umi_matrix[int(i), int(j)]))
    else:
        import numpy as np
        dense = np.asarray(arr)
        ii, jj = np.nonzero(dense)
        for i, j in zip(ii, jj):
            rows["cell_id"].append(cell_ids[int(i)])
            rows["guide_id"].append(names[int(j)])
            rows["assignment_value"].append(dense[int(i), int(j)])
            if umi_matrix is not None:
                umi_dense = umi_matrix.toarray() if hasattr(umi_matrix, "toarray") else umi_matrix
                rows["guide_umi_count"].append(float(umi_dense[int(i), int(j)]))

    out = pd.DataFrame(rows)
    if out.empty:
        raise PrepError(f"Guide assignment export is empty for {h5ad_path}")
    path = write_deterministic_tsv_gz(out, output_path)
    mapping = pd.DataFrame({"submitted_column": out.columns})
    return path, mapping


def inspect_h5mu(
    path: str | Path,
    *,
    day_column_candidates: Sequence[str] = ("day", "timepoint", "Day"),
) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "status": "ERROR"}
    if not path.is_file():
        result["error"] = "missing file"
        return result
    mu = _import_mudata()
    try:
        mdata = mu.read_h5mu(path)
        modalities = {}
        day_hits: list[dict[str, Any]] = []
        for name, mod in mdata.mod.items():
            modalities[name] = {
                "n_obs": int(mod.n_obs),
                "n_vars": int(mod.n_vars),
                "obs_columns": list(map(str, mod.obs.columns)),
                "var_columns": list(map(str, mod.var.columns)),
                "obsm_keys": list(map(str, mod.obsm.keys())),
                "varm_keys": list(map(str, mod.varm.keys())),
                "layers": list(map(str, mod.layers.keys())),
                "uns_keys": list(map(str, mod.uns.keys())),
            }
            try:
                day_col = _find_day_column(mod.obs, day_column_candidates)
                normalized = mod.obs[day_col].map(normalize_day)
                day_hits.append({
                    "location": f"mod[{name}].obs",
                    "column": day_col,
                    "values": sorted(v for v in normalized.dropna().unique()),
                    "unmapped": int(normalized.isna().sum()),
                })
            except DecisionRequired:
                pass
        try:
            day_col = _find_day_column(mdata.obs, day_column_candidates)
            normalized = mdata.obs[day_col].map(normalize_day)
            day_hits.append({
                "location": "mdata.obs",
                "column": day_col,
                "values": sorted(v for v in normalized.dropna().unique()),
                "unmapped": int(normalized.isna().sum()),
            })
        except DecisionRequired:
            pass
        distinct_days = sorted({d for hit in day_hits for d in hit["values"]})
        result.update({
            "modalities": modalities,
            "day_locations": day_hits,
            "distinct_days": distinct_days,
            "combined_across_days": len(distinct_days) > 1,
            "size_bytes": path.stat().st_size,
            "md5sum": md5sum(path),
            "status": "PASS",
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def require_h5mu_strategy(report: Mapping[str, Any], strategy: str) -> None:
    strategy = (strategy or "").strip().lower()
    if report.get("combined_across_days") and strategy in {"", "block", "block_until_dacc_decision"}:
        raise DecisionRequired(
            "The actual PerturbNMF h5mu appears to contain multiple timepoints. "
            "The workbook assumes one h5mu per day. Do not create four copies. "
            "Choose a DACC-approved combined MatrixFile strategy first."
        )


def portal_session() -> requests.Session:
    session = requests.Session()
    key = os.environ.get("IGVF_API_KEY", "")
    secret = os.environ.get("IGVF_SECRET_KEY", "")
    if key and secret:
        session.auth = (key, secret)
    session.headers.update({"Accept": "application/json"})
    return session


def portal_get(
    identifier_or_path: str,
    *,
    session: requests.Session | None = None,
    params: Mapping[str, Any] | None = None,
    timeout: int = 45,
    retries: int = 4,
) -> dict[str, Any]:
    session = session or portal_session()
    if identifier_or_path.startswith("http"):
        url = identifier_or_path
    elif identifier_or_path.startswith("/"):
        url = PORTAL_BASE + identifier_or_path
    else:
        url = f"{PORTAL_BASE}/{quote(identifier_or_path, safe='')}/"
    query = {"format": "json", **dict(params or {})}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=query, timeout=timeout)
            if response.status_code in RETRY_STATUS:
                last = requests.HTTPError(f"{response.status_code} {response.reason}")
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
                continue
            break
    raise PrepError(f"Portal GET failed for {url}: {last}")


def _audit_rows(obj: Mapping[str, Any], accession: str, obj_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = obj.get("audit") or {}
    for level, entries in audit.items():
        for entry in entries or []:
            rows.append({
                "accession": accession,
                "object_type": obj_type,
                "audit_level": level,
                "audit_category": entry.get("category", ""),
                "audit_detail": entry.get("detail", ""),
                "audit_path": entry.get("path", ""),
                "audit_name": entry.get("name", ""),
            })
    return rows


def _search_files_for_set(
    accession: str,
    *,
    session: requests.Session,
) -> list[dict[str, Any]]:
    hits: dict[str, dict[str, Any]] = {}
    filters = [
        {"file_set.accession": accession},
        {"file_set": accession},
        {"file_set": f"/analysis-sets/{accession}/"},
    ]
    for profile in ("MatrixFile", "TabularFile"):
        for extra in filters:
            params = {"type": profile, "limit": "all", **extra}
            try:
                data = portal_get("/search/", session=session, params=params)
            except PrepError:
                continue
            for obj in data.get("@graph", []):
                key = obj.get("accession") or obj.get("@id")
                if key:
                    hits[key] = obj
            if hits:
                # Keep checking the other profile, but stop redundant filters for this profile.
                break
    return list(hits.values())


def audit_analysis_sets(
    accessions: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read-only audit of AnalysisSets and linked Matrix/Tabular files."""
    session = portal_session()
    set_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for accession in accessions:
        obj = portal_get(accession, session=session)
        set_rows.append({
            "accession": obj.get("accession", accession),
            "@id": obj.get("@id", ""),
            "aliases": json.dumps(obj.get("aliases", [])),
            "status": obj.get("status", ""),
            "file_set_type": obj.get("file_set_type", ""),
            "input_file_sets": json.dumps(obj.get("input_file_sets", [])),
            "portal_url": f"{PORTAL_PUBLIC_BASE}/analysis-sets/{accession}/",
        })
        audit_rows.extend(_audit_rows(obj, accession, "AnalysisSet"))
        files = _search_files_for_set(accession, session=session)
        # Some frame modes embed files directly; include those too.
        for key in ("files", "matrix_files", "tabular_files"):
            embedded = obj.get(key) or []
            if isinstance(embedded, list):
                files.extend(x for x in embedded if isinstance(x, dict))
        unique: dict[str, dict[str, Any]] = {}
        for fobj in files:
            f_acc = fobj.get("accession") or fobj.get("@id", "")
            if not f_acc:
                continue
            # Fetch full object when search frame is sparse.
            try:
                full = portal_get(f_acc, session=session)
            except PrepError:
                full = fobj
            unique[str(f_acc)] = full
        for f_acc, fobj in unique.items():
            types = fobj.get("@type", [])
            obj_type = types[0] if types else "File"
            file_rows.append({
                "analysis_set": accession,
                "accession": fobj.get("accession", f_acc),
                "@id": fobj.get("@id", ""),
                "object_type": obj_type,
                "aliases": json.dumps(fobj.get("aliases", [])),
                "content_type": fobj.get("content_type", ""),
                "file_format": fobj.get("file_format", ""),
                "submitted_file_name": fobj.get("submitted_file_name", ""),
                "md5sum": fobj.get("md5sum", ""),
                "status": fobj.get("status", ""),
                "upload_status": fobj.get("upload_status", ""),
                "checkfiles_version": fobj.get("checkfiles_version", ""),
                "portal_url": f"{PORTAL_PUBLIC_BASE}/{fobj.get('accession', f_acc)}/",
                "audit_counts": json.dumps({k: len(v or []) for k, v in (fobj.get("audit") or {}).items()}),
            })
            audit_rows.extend(_audit_rows(fobj, fobj.get("accession", f_acc), obj_type))
    return pd.DataFrame(set_rows), pd.DataFrame(file_rows), pd.DataFrame(audit_rows)


def load_sequence_accessions_by_day(path: str | Path) -> dict[str, list[str]]:
    df = pd.read_csv(path, dtype=str)
    required = {"accession", "day"}
    missing = required - set(df.columns)
    if missing:
        raise PrepError(f"Sequence accession manifest missing columns: {sorted(missing)}")
    df["day_normalized"] = df["day"].map(normalize_day)
    if df["day_normalized"].isna().any():
        raise PrepError("Sequence accession manifest has unrecognized day labels")
    return {
        day: df.loc[df["day_normalized"].eq(day), "accession"].dropna().astype(str).tolist()
        for day in DAY_ORDER
    }


def read_source_overrides(path: str | Path) -> pd.DataFrame:
    df = load_tsv(path)
    required = {"day", "expected_file_role"}
    missing = required - set(df.columns)
    if missing:
        raise PrepError(f"Source override table missing columns: {sorted(missing)}")
    if "source_path_override" not in df.columns and "source_path" not in df.columns:
        raise PrepError(
            "Source override table must contain source_path_override or source_path"
        )
    if "source_path_override" not in df.columns:
        df = df.rename(columns={"source_path": "source_path_override"})
    df["day"] = df["day"].map(normalize_day)
    return df


def build_crosswalk(
    expected_files_path: str | Path,
    source_overrides_path: str | Path,
) -> pd.DataFrame:
    expected = load_tsv(expected_files_path)
    overrides = read_source_overrides(source_overrides_path)
    join_cols = ["day", "expected_file_role"]
    extra = [c for c in overrides.columns if c not in join_cols]
    out = expected.merge(overrides[join_cols + extra], on=join_cols, how="left", suffixes=("", "_override"))
    if "source_path_override" in out.columns:
        override = out["source_path_override"].fillna("").astype(str).str.strip()
        candidate = out.get("source_candidate", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
        out["source_path"] = override.where(override.ne(""), candidate)
    elif "source_path" not in out.columns:
        out["source_path"] = out.get("source_candidate", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    else:
        out["source_path"] = out["source_path"].fillna("").astype(str).str.strip()
    out["exists"] = out["source_path"].map(lambda p: Path(p).is_file() if p else False)
    out["size_bytes"] = out["source_path"].map(lambda p: Path(p).stat().st_size if p and Path(p).is_file() else "")
    return out


def _json_array(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        return json.dumps(value)
    text = str(value)
    try:
        parsed = json.loads(text)
        return json.dumps(parsed) if isinstance(parsed, list) else text
    except json.JSONDecodeError:
        return text


def build_metadata_tsvs(
    expected_files_path: str | Path,
    validation_report_path: str | Path,
    output_dir: str | Path,
    *,
    active_analysis_sets: Mapping[str, str],
    sequence_accession_manifest: str | Path,
) -> dict[str, Path]:
    """Build draft POST TSVs for rows validated as READY.

    No submission is performed. Rows marked already uploaded, missing, blocked,
    or decision-required are excluded.
    """
    active_analysis_sets = validate_timepoint_analysis_sets(active_analysis_sets)
    expected = load_tsv(expected_files_path)
    valid = load_tsv(validation_report_path)
    if "alias" not in valid.columns or "validation_status" not in valid.columns:
        raise PrepError("Validation report must contain alias and validation_status")
    merged = expected.merge(valid, on="alias", how="left", suffixes=("", "_validation"))
    eligible = merged[merged["validation_status"].eq("READY")].copy()
    accessions = load_sequence_accessions_by_day(sequence_accession_manifest)
    eligible["file_set"] = eligible["day"].map(active_analysis_sets)
    eligible["md5sum"] = eligible["submitted_file_name"].map(md5sum)
    for idx, row in eligible.iterrows():
        if row["profile"] == "matrix_file" and row["content_type"] == "annotated cell by gene and guide matrix":
            eligible.at[idx, "derived_from"] = json.dumps(accessions[row["day"]])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    common = [
        "aliases", "award", "lab", "md5sum", "file_format", "file_set",
        "content_type", "reference_files", "filtered", "submitter_comment",
        "description", "analysis_step_version", "derived_from", "derived_manually",
        "file_format_specifications", "submitted_file_name",
    ]
    profile_columns = {
        "matrix_file": common,
        "tabular_file": common[:7] + ["controlled_access"] + common[7:],
    }
    for profile, columns in profile_columns.items():
        part = eligible[eligible["profile"].eq(profile)].copy()
        if part.empty:
            continue
        part["aliases"] = part["alias"].map(lambda x: json.dumps([x]))
        # Preserve JSON array syntax for linked fields.
        for col in ("reference_files", "derived_from", "file_format_specifications"):
            if col in part.columns:
                part[col] = part[col].map(_json_array)
        for col in columns:
            if col not in part.columns:
                part[col] = ""
        # POST TSVs should not contain columns that are empty for every row.
        used_columns = [
            col for col in columns
            if part[col].fillna("").astype(str).str.strip().ne("").any()
        ]
        path = output_dir / f"{profile}.tsv"
        part[used_columns].to_csv(path, sep="\t", index=False, lineterminator="\n")
        outputs[profile] = path
    eligible.to_csv(output_dir / "eligible_upload_manifest.tsv", sep="\t", index=False, lineterminator="\n")
    outputs["eligible_upload_manifest"] = output_dir / "eligible_upload_manifest.tsv"
    return outputs


def dry_run_commands(
    metadata_paths: Mapping[str, str | Path],
    *,
    iu_register: str,
    python_executable: str,
) -> list[str]:
    commands = []
    for profile in ("matrix_file", "tabular_file"):
        if profile not in metadata_paths:
            continue
        commands.append(
            f"{python_executable} {iu_register} -m prod -p {profile} -d -i {metadata_paths[profile]}"
        )
    return commands


def write_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    return path


def copy_or_link(source: str | Path, destination: str | Path, *, mode: str = "copy") -> Path:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError("mode must be copy, hardlink, or symlink")
    return destination
