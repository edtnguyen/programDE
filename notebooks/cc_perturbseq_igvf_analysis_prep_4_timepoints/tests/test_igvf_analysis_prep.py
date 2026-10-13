from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from igvf_analysis_prep import (
    PrepError,
    convert_table_to_tsv_gz,
    md5sum,
    normalize_day,
    normalize_table_columns,
    validate_timepoint_analysis_sets,
    validate_tsv_gz,
    write_deterministic_tsv_gz,
)


def test_normalize_day():
    assert normalize_day("d0") == "D0"
    assert normalize_day("Day 1") == "D1"
    assert normalize_day(2) == "D2"
    assert normalize_day("D3") == "D3"
    assert normalize_day("day_4") is None


def test_deterministic_gzip(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    a = write_deterministic_tsv_gz(df, tmp_path / "a.tsv.gz")
    b = write_deterministic_tsv_gz(df, tmp_path / "b.tsv.gz")
    assert md5sum(a) == md5sum(b)
    report = validate_tsv_gz(a)
    assert report["status"] == "PASS"
    assert report["row_count"] == 2
    assert report["column_count"] == 2


def test_convert_csv(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("Gene ID,score\nA,1\nB,2\n", encoding="utf-8")
    output, mapping = convert_table_to_tsv_gz(
        source,
        tmp_path / "out.tsv.gz",
        rename_columns=True,
    )
    assert output.is_file()
    assert list(mapping["submitted_column"]) == ["Gene_ID", "score"]
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        assert handle.readline().rstrip("\n") == "Gene_ID\tscore"


def test_duplicate_column_normalization_rejected():
    df = pd.DataFrame([[1, 2]], columns=["A B", "A_B"])
    with pytest.raises(PrepError):
        normalize_table_columns(df, rename_columns=True)


def test_semantic_range_check(tmp_path: Path):
    from igvf_analysis_prep import validate_tabular_semantics

    df = pd.DataFrame({"p_value": [0.01, 1.2], "correlation": [0.4, -1.4]})
    path = write_deterministic_tsv_gz(df, tmp_path / "bad.tsv.gz")
    report = validate_tabular_semantics(path)
    assert report["status"] == "ERROR"
    assert {x["column"] for x in report["range_violations"]} == {"p_value", "correlation"}


def test_validate_timepoint_analysis_sets():
    mapping = {
        "D0": "IGVFDS5257JGQO",
        "D1": "IGVFDS3530HHEU",
        "D2": "IGVFDS7380DUFB",
        "D3": "IGVFDS6826YKCZ",
    }
    assert validate_timepoint_analysis_sets(mapping) == mapping


@pytest.mark.parametrize(
    "mapping",
    [
        {"D0": "IGVFDS1", "D1": "IGVFDS2", "D2": "IGVFDS3"},
        {
            "D0": "IGVFDS1",
            "D1": "IGVFDS2",
            "D2": "IGVFDS3",
            "D3": "IGVFDS4",
            "combined": "IGVFDS5",
        },
        {
            "D0": "IGVFDS1",
            "D1": "IGVFDS1",
            "D2": "IGVFDS3",
            "D3": "IGVFDS4",
        },
    ],
)
def test_validate_timepoint_analysis_sets_rejects_invalid(mapping):
    from igvf_analysis_prep import DecisionRequired

    with pytest.raises(DecisionRequired):
        validate_timepoint_analysis_sets(mapping)
