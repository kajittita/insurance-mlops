"""Central project settings: file paths, column names and MLflow settings."""

import os
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/insurance.csv")
PROCESSED_DATA_PATH = Path("data/processed/insurance_clean.csv")
METRICS_PATH = Path("metrics/metrics.json")
PREDICTION_LOG_PATH = Path("logs/predictions.csv")
DRIFT_REPORT_PATH = Path("reports/drift_report.html")

TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_COLUMNS = FEATURES + [TARGET]

TEST_SIZE = 0.2
RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "insurance-charges"
MODEL_NAME = "insurance-charges-model"
CHAMPION_ALIAS = "champion"
PREDICTION_EXPERIMENT_NAME = "insurance-predictions"

MIN_ROWS_FOR_DRIFT_REPORT = 20
