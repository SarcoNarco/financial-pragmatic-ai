#!/usr/bin/env python3
"""Send one tiny transcript to /analyze and report its latency."""

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
SAMPLE_TRANSCRIPT = (
    "CEO: Revenue grew 12 percent. "
    "CFO: Margins improved. "
    "Analyst: What drove growth?"
)


def main() -> int:
    payload = json.dumps({"transcript": SAMPLE_TRANSCRIPT}).encode("utf-8")
    request = Request(
        f"{BASE_URL}/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urlopen(request, timeout=600) as response:
            status = response.status
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        elapsed = time.perf_counter() - started
        print(f"elapsed_seconds={elapsed:.2f}")
        print(f"http_status={exc.code}")
        print(exc.read().decode("utf-8", errors="replace"))
        return 1
    except (URLError, TimeoutError) as exc:
        elapsed = time.perf_counter() - started
        print(f"elapsed_seconds={elapsed:.2f}")
        print(f"request_failed={exc}")
        return 1

    elapsed = time.perf_counter() - started
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"http_status={status}")

    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        print("response_error=invalid_json")
        return 1

    segments = result.get("segments", [])
    print(f"score={result.get('score')}")
    print(f"signal={result.get('signal')}")
    print(f"prediction={result.get('prediction')}")
    print(f"fallback_used={result.get('fallback_used')}")
    print(f"segments={len(segments) if isinstance(segments, list) else 0}")

    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
