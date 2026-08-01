#!/usr/bin/env python3
"""Send one transcript file to /analyze and report sampled-mode metadata."""

import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")


def print_result(result: dict) -> None:
    for key in (
        "score",
        "signal",
        "prediction",
        "fallback_used",
        "analysis_mode",
        "sampled",
        "segments_total",
        "segments_analyzed",
        "segment_budget",
        "sampling_note",
    ):
        print(f"{key}={result.get(key)}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: benchmark_transcript_file.py path/to/transcript.txt", file=sys.stderr)
        return 2

    path = Path(sys.argv[1]).expanduser()
    try:
        transcript = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"file_error={exc}", file=sys.stderr)
        return 2

    print(f"file_path={path}")
    print(f"character_count={len(transcript)}")
    print(f"approximate_word_count={len(transcript.split())}")

    payload = json.dumps({"transcript": transcript}).encode("utf-8")
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
        print(f"http_status={exc.code}")
        print(f"elapsed_seconds={elapsed:.2f}")
        print(f"error_body={exc.read().decode('utf-8', errors='replace')}")
        return 1
    except (URLError, TimeoutError) as exc:
        elapsed = time.perf_counter() - started
        print("http_status=request_failed")
        print(f"elapsed_seconds={elapsed:.2f}")
        print(f"error_body={exc}")
        return 1

    elapsed = time.perf_counter() - started
    print(f"http_status={status}")
    print(f"elapsed_seconds={elapsed:.2f}")
    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        print("error_body=invalid_json")
        return 1

    if not isinstance(result, dict) or result.get("error"):
        print(f"error_body={body}")
        return 1

    print_result(result)
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
