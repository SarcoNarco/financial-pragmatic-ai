"""Utility helpers for better-than-FinBERT evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


SIGNAL_LABELS = ["growth", "neutral", "risk"]

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "pragmatic_intent_dataset_clean.csv"
)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def ensure_results_dir(results_dir: str | Path | None = None) -> Path:
    path = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_evaluation_dataset(
    dataset_path: str | Path | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    resolved = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    df = pd.read_csv(resolved)
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("Dataset must include 'text' and 'intent' columns")

    selected = ["text", "intent"]
    if "speaker" in df.columns:
        selected.append("speaker")

    df = df[selected].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["intent"] = df["intent"].fillna("GENERAL_UPDATE").astype(str).str.upper()
    if "speaker" in df.columns:
        df["speaker"] = df["speaker"].fillna("EXECUTIVE").astype(str).str.upper()

    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples).reset_index(drop=True)

    return df


def intent_to_ground_truth_signal(intent: str) -> str:
    if intent == "COST_PRESSURE":
        return "risk"
    if intent == "EXPANSION":
        return "growth"
    return "neutral"


def build_ground_truth_signals(intents: Iterable[str]) -> List[str]:
    return [intent_to_ground_truth_signal(str(intent).upper()) for intent in intents]


def baseline_sentiment_to_signal(sentiment_label: str) -> str:
    label = sentiment_label.strip().lower()
    if "positive" in label:
        return "growth"
    if "negative" in label:
        return "risk"
    return "neutral"


def normalize_confidence_to_percent(value: float) -> float:
    if value <= 1.0:
        return round(value * 100.0, 4)
    return round(value, 4)


def agreement_rate(a: List[str], b: List[str]) -> float:
    if len(a) == 0:
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


def average_confidence_per_class(
    signals: List[str],
    confidences: List[float],
    labels: List[str] | None = None,
) -> Dict[str, float]:
    target_labels = labels or SIGNAL_LABELS
    output: Dict[str, float] = {}

    for label in target_labels:
        class_conf = [c for s, c in zip(signals, confidences) if s == label]
        output[label] = round(sum(class_conf) / len(class_conf), 4) if class_conf else 0.0

    return output


def snippet(text: str, max_chars: int = 180) -> str:
    cleaned = str(text).replace("\n", " ").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def explain_our_decision(segments: List[dict], signal: str) -> str:
    if not segments:
        return "No parsed segments available; fallback signal used."

    intent_counts: Dict[str, int] = {}
    for item in segments:
        intent = str(item.get("intent", "GENERAL_UPDATE"))
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    dominant_intent = max(intent_counts, key=intent_counts.get)
    first_segment = segments[0]
    first_speaker = first_segment.get("speaker", "UNKNOWN")
    first_text = snippet(first_segment.get("text", ""), max_chars=130)

    return (
        f"Signal={signal}, dominant_intent={dominant_intent}, counts={intent_counts}, "
        f"first_segment=({first_speaker}) {first_text}"
    )
