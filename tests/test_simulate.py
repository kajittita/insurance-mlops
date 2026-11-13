import json
import runpy
import sys
import urllib.request
from pathlib import Path
from types import TracebackType

import pandas as pd
import pytest

import insurance.simulate as simulate_module
from insurance.simulate import main, sample_requests, send_requests


class FakeResponse:
    """Stands in for the HTTP response, so the tests need no running API."""

    status = 200

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: type[BaseException] | BaseException | TracebackType | None) -> None:
        return None


@pytest.fixture
def sent_bodies(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    bodies: list[dict[str, object]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        assert isinstance(request.data, bytes)
        bodies.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return bodies


def test_shift_makes_people_older_and_heavier(data_csv: Path) -> None:
    normal = pd.DataFrame(sample_requests(data_csv, 30, shift=False))
    shifted = pd.DataFrame(sample_requests(data_csv, 30, shift=True))
    assert len(normal) == 30
    assert (shifted["age"] >= normal["age"]).all()
    assert shifted["bmi"].mean() > normal["bmi"].mean()


def test_send_requests_posts_every_body(sent_bodies: list[dict[str, object]], data_csv: Path) -> None:
    requests = sample_requests(data_csv, 3, shift=False)
    assert send_requests(requests, "http://api") == 3
    assert sent_bodies == requests


def test_main_uses_command_line_arguments(sent_bodies: list[dict[str, object]], data_csv: Path) -> None:
    main(["--n", "4", "--shift", "--url", "http://api"])
    assert len(sent_bodies) == 4


def test_module_runs_as_script(
    sent_bodies: list[dict[str, object]], data_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["simulate", "--n", "2"])
    monkeypatch.delitem(sys.modules, "insurance.simulate")  # run a fresh copy, like `python -m` does
    runpy.run_module("insurance.simulate", run_name="__main__")
    assert len(sent_bodies) == 2
    assert simulate_module.DEFAULT_API_URL.startswith("http")
