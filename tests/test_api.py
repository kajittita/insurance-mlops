from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression

import insurance.train as train_module
from insurance.api import create_app
from insurance.config import PREDICTION_EXPERIMENT_NAME

PERSON = {"age": 30, "sex": "female", "bmi": 25.0, "children": 1, "smoker": "no", "region": "northeast"}


@pytest.fixture
def client(data_csv: Path, tracking_uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(train_module, "candidate_models", lambda: {"linear": LinearRegression()})
    return TestClient(create_app(tracking_uri, data_csv, tmp_path / "logs" / "predictions.csv"))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_before_training_returns_503(client: TestClient) -> None:
    response = client.post("/predict", json=PERSON)
    assert response.status_code == 503


def test_train_then_predict(client: TestClient, tmp_path: Path, tracking_uri: str) -> None:
    train_response = client.post("/train")
    assert train_response.status_code == 200
    assert train_response.json()["promoted"] is True

    smoker = client.post("/predict", json={**PERSON, "smoker": "yes"}).json()
    non_smoker = client.post("/predict", json=PERSON).json()
    assert smoker["model_version"] == "1"
    assert smoker["charges"] > non_smoker["charges"]

    log = pd.read_csv(tmp_path / "logs" / "predictions.csv")
    assert len(log) == 2
    assert list(log["smoker"]) == ["yes", "no"]

    experiment = MlflowClient(tracking_uri).get_experiment_by_name(PREDICTION_EXPERIMENT_NAME)
    assert experiment is not None
    assert len(MlflowClient(tracking_uri).search_runs([experiment.experiment_id])) == 2


@pytest.mark.parametrize("bad_field", [{"age": 5}, {"bmi": 200}, {"smoker": "maybe"}, {"region": "mars"}])
def test_invalid_input_is_rejected(client: TestClient, bad_field: dict[str, object]) -> None:
    response = client.post("/predict", json={**PERSON, **bad_field})
    assert response.status_code == 422
