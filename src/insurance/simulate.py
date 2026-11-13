"""Send sample requests to the running API, so there is traffic to monitor.

Run: python -m insurance.simulate --n 50            (normal traffic)
     python -m insurance.simulate --n 50 --shift    (older, heavier people: should show drift)
"""

import argparse
import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from insurance.config import FEATURES, PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"


def sample_requests(data_path: Path, n: int, shift: bool, seed: int | None = None) -> list[dict[str, object]]:
    """Pick n random people from the dataset. With shift=True they are made older and heavier.

    Without a seed every run draws a new sample, so repeated runs do not send the same people again.
    """
    sample = pd.read_csv(data_path).loc[:, FEATURES].sample(n=n, replace=True, random_state=seed)
    if shift:
        sample["age"] = (sample["age"] + 15).clip(upper=100)
        sample["bmi"] = (sample["bmi"] * 1.2).clip(upper=80).round(2)
    return [{str(key): value for key, value in row.items()} for row in sample.to_dict(orient="records")]


def send_requests(requests: list[dict[str, object]], api_url: str) -> int:
    """POST every request to /predict and return how many succeeded."""
    succeeded = 0
    for body in requests:
        request = urllib.request.Request(
            f"{api_url}/predict",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status == 200:
                succeeded += 1
    return succeeded


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and send the requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="number of requests")
    parser.add_argument("--shift", action="store_true", help="make the data drift on purpose")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="base URL of the API")
    parser.add_argument("--seed", type=int, default=None, help="fix the random sample (for repeatable runs)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sent = send_requests(sample_requests(PROCESSED_DATA_PATH, args.n, args.shift, args.seed), args.url)
    logger.info("Sent %d requests to %s (shift=%s)", sent, args.url, args.shift)


if __name__ == "__main__":
    main()
