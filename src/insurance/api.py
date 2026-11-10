"""REST API with a /train and a /predict endpoint.

Run: uvicorn insurance.api:app --reload   then open http://127.0.0.1:8000/docs
"""

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from insurance.config import (
    CHAMPION_ALIAS,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PREDICTION_EXPERIMENT_NAME,
    PREDICTION_LOG_PATH,
    PROCESSED_DATA_PATH,
)
from insurance.train import split_features_target, train

logger = logging.getLogger(__name__)


class InsuranceInput(BaseModel):
    """One person's data. Pydantic rejects missing fields and out-of-range values with HTTP 422."""

    age: int = Field(ge=18, le=100)
    sex: Literal["female", "male"]
    bmi: float = Field(ge=10, le=80)
    children: int = Field(ge=0, le=10)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


class Prediction(BaseModel):
    """The predicted yearly medical charges and the model version that made the prediction."""

    charges: float
    model_version: str


class TrainResponse(BaseModel):
    """Summary of a training run, returned by /train."""

    best_model: str
    rmse: float
    champion_rmse: float | None
    promoted: bool
    model_version: str | None


class ChampionModel:
    """Loads the @champion model from the registry and reloads it only when the champion changes."""

    def __init__(self, client: MlflowClient) -> None:
        self.client = client
        self.version: str | None = None
        self.model: Pipeline | None = None

    def get(self) -> tuple[Pipeline, str]:
        try:
            version = str(self.client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version)
        except MlflowException as error:
            raise HTTPException(status_code=503, detail="No trained model yet, call /train first") from error
        if self.model is None or version != self.version:
            self.model = cast(Pipeline, load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"))
            self.version = version
            logger.info("Loaded champion model version %s", version)
        return self.model, version


def log_prediction(client: MlflowClient, log_path: Path, data: InsuranceInput, result: Prediction) -> None:
    """Save the request and the prediction to a CSV file (input for Evidently) and as an MLflow run."""
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **data.model_dump(), **result.model_dump()}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    experiment = client.get_experiment_by_name(PREDICTION_EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(PREDICTION_EXPERIMENT_NAME)
    run_id = client.create_run(experiment_id, run_name="predict").info.run_id
    client.log_dict(run_id, row, "request_and_prediction.json")
    client.log_metric(run_id, "prediction", result.charges)
    client.set_tag(run_id, "model_version", result.model_version)
    client.set_terminated(run_id)


def create_app(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    data_path: Path = PROCESSED_DATA_PATH,
    log_path: Path = PREDICTION_LOG_PATH,
) -> FastAPI:
    """Build the FastAPI application. The arguments make it easy to point the tests at temporary files."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    champion = ChampionModel(client)
    app = FastAPI(title="Insurance charges API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/train")
    def train_endpoint() -> TrainResponse:
        result = train(data_path, tracking_uri)
        return TrainResponse(**asdict(result))

    @app.post("/predict")
    def predict(data: InsuranceInput) -> Prediction:
        model, version = champion.get()
        x, _ = split_features_target(pd.DataFrame([{**data.model_dump(), "charges": 0.0}]))
        result = Prediction(charges=float(model.predict(x)[0]), model_version=version)
        log_prediction(client, log_path, data, result)
        return result

    return app


app = create_app()
