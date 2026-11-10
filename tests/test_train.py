import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

import insurance.train as train_module
from insurance.train import TrainResult, build_pipeline, evaluate, split_features_target, train


def test_pipeline_learns_and_evaluates(data_csv: Path) -> None:
    x, y = split_features_target(pd.read_csv(data_csv))
    pipeline = build_pipeline(LinearRegression()).fit(x, y)
    metrics = evaluate(pipeline, x, y)
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["r2"] > 0.9


def test_first_run_promotes_and_equal_run_keeps_champion(
    data_csv: Path, tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_module, "candidate_models", lambda: {"linear": LinearRegression()})
    first = train(data_csv, tracking_uri)
    assert first.promoted and first.model_version == "1" and first.champion_rmse is None

    second = train(data_csv, tracking_uri)
    assert not second.promoted and second.model_version is None
    assert second.champion_rmse == pytest.approx(second.rmse)


def test_better_model_replaces_champion(data_csv: Path, tracking_uri: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_module, "candidate_models", lambda: {"dummy": DummyRegressor()})
    train(data_csv, tracking_uri)

    monkeypatch.setattr(train_module, "candidate_models", lambda: {"linear": LinearRegression()})
    result = train(data_csv, tracking_uri)
    assert result.promoted and result.model_version == "2"
    assert result.champion_rmse is not None and result.rmse < result.champion_rmse


def test_main_writes_metrics_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = TrainResult("linear", 1.0, None, True, "1")
    monkeypatch.setattr(train_module, "train", lambda: fake)
    metrics_path = tmp_path / "metrics.json"
    train_module.main(metrics_path)
    assert json.loads(metrics_path.read_text())["best_model"] == "linear"


def test_module_runs_as_script(data_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "insurance.train")  # run a fresh copy, like `python -m` does
    runpy.run_module("insurance.train", run_name="__main__")
    assert json.loads((tmp_path / "metrics" / "metrics.json").read_text())["promoted"] is True
