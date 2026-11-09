"""Central project settings: file paths and column names."""

from pathlib import Path

RAW_DATA_PATH = Path("data/raw/insurance.csv")
PROCESSED_DATA_PATH = Path("data/processed/insurance_clean.csv")

TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
ALL_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
