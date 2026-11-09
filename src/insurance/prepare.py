"""Data preparation step: load the raw CSV, clean it and save the result.

Run: python -m insurance.prepare   (or: dvc repro prepare)
"""

import logging
from pathlib import Path

import pandas as pd

from insurance.config import ALL_COLUMNS, PROCESSED_DATA_PATH, RAW_DATA_PATH

logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV and check that every expected column is present."""
    df = pd.read_csv(path)
    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df.loc[:, ALL_COLUMNS]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and rows with missing values."""
    cleaned = df.drop_duplicates().dropna().reset_index(drop=True)
    removed = len(df) - len(cleaned)
    if removed:
        logger.info("Removed %d duplicate or incomplete rows", removed)
    return cleaned


def main(raw_path: Path = RAW_DATA_PATH, out_path: Path = PROCESSED_DATA_PATH) -> None:
    """Run the whole preparation step."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df = clean(load_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
