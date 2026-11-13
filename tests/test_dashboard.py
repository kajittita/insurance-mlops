from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from insurance.simulate import sample_requests

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def test_dashboard_asks_for_traffic_when_log_is_empty(data_csv: Path) -> None:
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert "at least" in app.info[0].value


def test_dashboard_shows_drift_metrics(data_csv: Path, tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "predictions.csv"
    log_path.parent.mkdir()
    requests = pd.DataFrame(sample_requests(data_csv, 40, shift=True))
    requests.assign(charges=1.0, model_version="1").to_csv(log_path, index=False)

    app = AppTest.from_file(APP, default_timeout=120).run()

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Drifted columns" in labels
