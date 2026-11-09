import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from insurance.prepare import clean, load_raw, main

ROW_A = {"age": 19, "sex": "female", "bmi": 27.9, "children": 0, "smoker": "yes", "region": "southwest", "charges": 1.0}
ROW_B = {"age": 18, "sex": "male", "bmi": 33.8, "children": 1, "smoker": "no", "region": "southeast", "charges": 2.0}
ROWS = [ROW_A, ROW_B, ROW_B]


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "raw" / "insurance.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(ROWS).to_csv(path, index=False)
    return path


def test_load_raw_reads_all_columns(raw_csv: Path) -> None:
    df = load_raw(raw_csv)
    assert list(df.columns) == ["age", "bmi", "children", "sex", "smoker", "region", "charges"]
    assert len(df) == 3


def test_load_raw_fails_on_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(ROWS).drop(columns=["smoker"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="smoker"):
        load_raw(path)


def test_clean_removes_duplicates_and_missing_values() -> None:
    df = pd.DataFrame(ROWS + [{**ROWS[0], "bmi": None}])
    cleaned = clean(df)
    assert len(cleaned) == 2
    assert not cleaned.isna().any().any()


def test_main_writes_processed_file(raw_csv: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "out" / "clean.csv"
    main(raw_csv, out_path)
    assert len(pd.read_csv(out_path)) == 2


def test_module_runs_as_script(raw_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "insurance.prepare")  # run a fresh copy, like `python -m` does
    runpy.run_module("insurance.prepare", run_name="__main__")
    assert (tmp_path / "data" / "processed" / "insurance_clean.csv").exists()
