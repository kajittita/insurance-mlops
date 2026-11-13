import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from insurance.monitoring import load_current, load_reference, main, run_drift_report


def write_log(path: Path, rows: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.assign(charges=1.0, model_version="1").to_csv(path, index=False)
    return path


def test_load_current_without_log_is_empty(tmp_path: Path) -> None:
    assert load_current(tmp_path / "missing.csv").empty


def test_too_few_requests_raise(data_csv: Path) -> None:
    reference = load_reference(data_csv)
    with pytest.raises(ValueError, match="at least"):
        run_drift_report(reference, reference.head(5))


def test_shifted_requests_show_drift(data_csv: Path, tmp_path: Path) -> None:
    reference = load_reference(data_csv)
    shifted = reference.assign(age=reference["age"] + 30, bmi=reference["bmi"] * 1.5)
    current = load_current(write_log(tmp_path / "log.csv", shifted))

    summary, html = run_drift_report(reference, current)

    assert summary.drifted_columns >= 2
    assert summary.current_rows == len(reference)
    assert "<html" in html.lower()


def test_main_writes_html_report(data_csv: Path, tmp_path: Path) -> None:
    log_path = write_log(tmp_path / "log.csv", load_reference(data_csv))
    report_path = tmp_path / "reports" / "drift.html"
    main(data_csv, log_path, report_path)
    assert "<html" in report_path.read_text(encoding="utf-8").lower()


def test_module_runs_as_script(data_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_log(tmp_path / "logs" / "predictions.csv", load_reference(data_csv))
    monkeypatch.delitem(sys.modules, "insurance.monitoring")  # run a fresh copy, like `python -m` does
    runpy.run_module("insurance.monitoring", run_name="__main__")
    assert (tmp_path / "reports" / "drift_report.html").exists()
