"""Training step: train several candidate models, track them in MLflow and
promote the best one only if it beats the current champion model.

Run: python -m insurance.train   (or: dvc repro train)
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.sklearn import load_model, log_model
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from insurance.config import (
    CATEGORICAL_FEATURES,
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    METRICS_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

# MLflow saves sklearn models with skops instead of pickle, because loading a pickle can run any code.
# skops only loads types we explicitly trust; the tree models need their internal Tree type.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


@dataclass
class TrainResult:
    """Summary of one training run."""

    best_model: str
    rmse: float
    champion_rmse: float | None
    promoted: bool
    model_version: str | None


def candidate_models() -> dict[str, BaseEstimator]:
    """The models that are compared in every training run."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """One-hot encode the categorical columns, keep the numeric ones, then fit the model."""
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the input columns from the target column."""
    x = df.loc[:, NUMERIC_FEATURES + CATEGORICAL_FEATURES].astype({column: "float64" for column in NUMERIC_FEATURES})
    return x, df.loc[:, TARGET]


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute regression metrics on the given data."""
    predictions = model.predict(x)
    return {
        "rmse": float(root_mean_squared_error(y, predictions)),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }


def champion_rmse(client: MlflowClient, x: pd.DataFrame, y: pd.Series) -> float | None:
    """RMSE of the current champion on today's test data, or None if there is no champion yet."""
    try:
        client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None
    champion = cast(Pipeline, load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"))
    return evaluate(champion, x, y)["rmse"]


def train(data_path: Path = PROCESSED_DATA_PATH, tracking_uri: str = MLFLOW_TRACKING_URI) -> TrainResult:
    """Train all candidates, log them to MLflow and promote the best one if it is better."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    df = pd.read_csv(data_path)
    test_df = df.sample(frac=TEST_SIZE, random_state=RANDOM_STATE)
    train_df = df.drop(index=test_df.index)
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    best_name, best_rmse, best_model_uri = "", float("inf"), ""
    for name, model in candidate_models().items():
        with mlflow.start_run(run_name=name):
            pipeline = build_pipeline(model)
            pipeline.fit(x_train, y_train)
            metrics = evaluate(pipeline, x_test, y_test)
            mlflow.log_param("model_type", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            model_info = log_model(
                pipeline, name="model", input_example=x_test.head(3), skops_trusted_types=TRUSTED_TYPES
            )
            logger.info("%s: rmse=%.2f mae=%.2f r2=%.3f", name, metrics["rmse"], metrics["mae"], metrics["r2"])
            if metrics["rmse"] < best_rmse:
                best_name, best_rmse, best_model_uri = name, metrics["rmse"], model_info.model_uri

    current_rmse = champion_rmse(client, x_test, y_test)
    promoted = current_rmse is None or best_rmse < current_rmse
    version: str | None = None
    if promoted:
        registered = mlflow.register_model(best_model_uri, MODEL_NAME)
        version = str(registered.version)
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
        logger.info("Promoted %s (version %s) to @%s", best_name, version, CHAMPION_ALIAS)
    else:
        logger.info("Kept the current champion: new rmse %.2f >= champion rmse %.2f", best_rmse, current_rmse)

    return TrainResult(best_name, best_rmse, current_rmse, promoted, version)


def main(metrics_path: Path = METRICS_PATH) -> None:
    """Run training and write a small metrics file for DVC."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = train()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(asdict(result), indent=2) + "\n")


if __name__ == "__main__":
    main()
