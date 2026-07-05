from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")


def fetch_json(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    try:
        with urlopen(url, timeout=10) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"{path} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"{path} failed: {exc.reason}") from exc

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} returned non-JSON response: {payload!r}") from exc


def main() -> int:
    checks = {
        "/health": "status",
        "/version": "app",
    }

    for path, required_key in checks.items():
        data = fetch_json(path)
        if required_key not in data:
            print(f"FAIL {path}: missing key {required_key!r}")
            return 1
        print(f"OK {path}: {data}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
