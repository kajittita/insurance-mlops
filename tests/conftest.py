from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def run_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in its own empty folder, so MLflow files never end up in the project."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def data_csv(tmp_path: Path) -> Path:
    """A small synthetic dataset with the same columns as the real one."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "age": rng.integers(18, 65, n),
        "bmi": rng.uniform(18, 40, n).round(1),
        "children": rng.integers(0, 4, n),
        "sex": rng.choice(["female", "male"], n),
        "smoker": rng.choice(["yes", "no"], n),
        "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
    })
    df["charges"] = 250 * df["age"] + 20000 * (df["smoker"] == "yes") + rng.normal(0, 500, n)
    path = tmp_path / "data" / "processed" / "insurance_clean.csv"
    path.parent.mkdir(parents=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"
