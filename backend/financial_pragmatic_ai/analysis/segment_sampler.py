"""Deterministic representative sampling for hosted full-transcript analysis."""

from __future__ import annotations

import re
from typing import Any


_SIGNAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bgrowth\b",
        r"\brevenue\b",
        r"\bmargins?\b",
        r"\bprofit\b",
        r"\boperating income\b",
        r"\bguidance\b",
        r"\boutlook\b",
        r"\bdemand\b",
        r"\bexpansion\b",
        r"\bmarket share\b",
        r"\bpricing\b",
        r"\bcosts?\b",
        r"\bpressure\b",
        r"\bdeclin(?:e|ed)\b",
        r"\bheadwinds?\b",
        r"\brisk\b",
        r"\binflation\b",
        r"\binventory\b",
        r"\bcash flow\b",
        r"\bcapex\b",
        r"\banalyst\b",
        r"\bquestions?\b",
        r"\bq\s*&\s*a\b",
        r"\bquarter\b",
        r"\byear[- ]over[- ]year\b",
        r"\bgross margin\b",
        r"\bebitda\b",
    )
)


def _evenly_spaced_indices(start: int, end: int, count: int) -> list[int]:
    """Return up to ``count`` deterministic positions in ``[start, end)``."""
    length = max(0, end - start)
    if count <= 0 or length == 0:
        return []
    if count == 1:
        return [start + (length - 1) // 2]

    return [
        start + round(position * (length - 1) / (count - 1))
        for position in range(count)
    ]


def _coverage_indices(total: int, count: int) -> list[int]:
    """Select coverage positions from the early, middle, and late thirds."""
    if count <= 0 or total <= 0:
        return []

    per_region, remainder = divmod(count, 3)
    region_counts = [per_region + (1 if region < remainder else 0) for region in range(3)]
    indices: list[int] = []
    for region, region_count in enumerate(region_counts):
        start = total * region // 3
        end = total * (region + 1) // 3
        indices.extend(_evenly_spaced_indices(start, end, region_count))
    return indices


def _signal_score(segment: dict[str, Any]) -> int:
    text = str(segment.get("text", ""))
    return sum(bool(pattern.search(text)) for pattern in _SIGNAL_PATTERNS)


def select_representative_segments(
    segments: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    """Select ordered, high-signal, broadly distributed transcript segments.

    The sampler deliberately avoids a simple first-N slice. It first reserves
    roughly one third of the budget for early/middle/late coverage, then fills
    the remaining slots with keyword-ranked segments. Ties use original order.
    """
    if budget <= 0 or not segments:
        return []

    total = len(segments)
    target = min(total, budget)
    coverage_count = min(target, max(3, target // 3))
    selected_indices = set(_coverage_indices(total, coverage_count))

    ranked_indices = sorted(
        range(total),
        key=lambda index: (-_signal_score(segments[index]), index),
    )
    for index in ranked_indices:
        if len(selected_indices) >= target:
            break
        selected_indices.add(index)

    selected: list[dict[str, Any]] = []
    for index in sorted(selected_indices):
        segment = dict(segments[index])
        segment["source_index"] = index
        selected.append(segment)
    return selected
