"""Data drift monitoring with Evidently: compare the logged API requests with the training data.

Run: python -m insurance.monitoring   (writes reports/drift_report.html)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from insurance.config import (
    DRIFT_REPORT_PATH,
    FEATURES,
    MIN_ROWS_FOR_DRIFT_REPORT,
    PREDICTION_LOG_PATH,
    PROCESSED_DATA_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftSummary:
    """The headline numbers of a drift report."""

    drifted_columns: int
    drift_share: float
    current_rows: int


def load_reference(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """The training data: what the model has learned is 'normal'."""
    return pd.read_csv(path).loc[:, FEATURES]


def load_current(path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    """The requests the API has received so far (empty if there were none yet)."""
    if not path.exists():
        return pd.DataFrame({column: [] for column in FEATURES})
    return pd.read_csv(path).loc[:, FEATURES]


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[DriftSummary, str]:
    """Run the Evidently data drift preset and return its summary and its HTML page."""
    if len(current) < MIN_ROWS_FOR_DRIFT_REPORT:
        raise ValueError(f"Need at least {MIN_ROWS_FOR_DRIFT_REPORT} logged requests, got {len(current)}")
    # Statistical tests suit a small sample of requests: Kolmogorov-Smirnov for numbers, chi-square for categories.
    preset = DataDriftPreset(num_method="ks", cat_method="chisquare")
    snapshot = Report([preset]).run(reference_data=reference, current_data=current)
    drift_count = next(
        metric["value"] for metric in snapshot.dict()["metrics"]
        if metric["metric_name"].startswith("DriftedColumnsCount")
    )
    summary = DriftSummary(int(drift_count["count"]), float(drift_count["share"]), len(current))
    return summary, snapshot.get_html_str(as_iframe=False)


def main(
    reference_path: Path = PROCESSED_DATA_PATH,
    log_path: Path = PREDICTION_LOG_PATH,
    report_path: Path = DRIFT_REPORT_PATH,
) -> None:
    """Build the drift report and save it as an HTML file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary, html = run_drift_report(load_reference(reference_path), load_current(log_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    logger.info(
        "%d of %d columns drifted (share %.2f), based on %d requests. Report: %s",
        summary.drifted_columns, len(FEATURES), summary.drift_share, summary.current_rows, report_path,
    )


if __name__ == "__main__":
    main()
