"""Unified V2 training pipeline for intent + conversation attention models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from financial_pragmatic_ai.models.conversation_attention_model import (
    train_conversation_attention_model,
)
from financial_pragmatic_ai.models.finbert_intent_model import (
    INTENT_LABELS,
    FinBERTIntentModel,
    train_finbert_intent_model,
)
from financial_pragmatic_ai.models.speaker_embedding import SPEAKER_TO_INDEX


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PACKAGE_ROOT / "data" / "pragmatic_intent_dataset_clean.csv"
FINBERT_INTENT_PATH = PACKAGE_ROOT / "models" / "finbert_intent.pt"
CONVERSATION_ATTENTION_PATH = PACKAGE_ROOT / "models" / "conversation_attention.pt"

VALID_INTENTS = set(INTENT_LABELS)


def _clean_row(row: pd.Series) -> Dict[str, str] | None:
    text = str(row.get("text", "")).strip()
    speaker = str(row.get("speaker", "")).strip().upper()
    intent = str(row.get("intent", "")).strip().upper()

    if not text or not speaker or not intent:
        return None
    if intent not in VALID_INTENTS:
        return None
    return {"text": text, "speaker": speaker, "intent": intent}


def _assign_signal(ceo_intent: str, cfo_intent: str, analyst_intent: str) -> str:
    if cfo_intent == "COST_PRESSURE":
        return "risk"
    if ceo_intent == "EXPANSION" and analyst_intent == "STRATEGIC_PROBING":
        return "neutral"
    if ceo_intent == "EXPANSION":
        return "growth"
    return "neutral"


def build_conversation_sequences(df: pd.DataFrame) -> List[Dict[str, object]]:
    """
    Build ordered CEO -> CFO -> ANALYST sequences from pragmatic intent rows.
    Invalid and incomplete sequences are skipped.
    """
    cleaned_rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        cleaned = _clean_row(row)
        if cleaned is not None:
            cleaned_rows.append(cleaned)

    sequences: List[Dict[str, object]] = []
    pending_ceo: Dict[str, str] | None = None
    pending_cfo: Dict[str, str] | None = None

    for row in cleaned_rows:
        speaker = row["speaker"]

        if speaker == "CEO":
            pending_ceo = row
            pending_cfo = None
            continue

        if speaker == "CFO":
            if pending_ceo is not None and pending_cfo is None:
                pending_cfo = row
            continue

        if speaker == "ANALYST":
            if pending_ceo is not None and pending_cfo is not None:
                texts = [pending_ceo["text"], pending_cfo["text"], row["text"]]
                intents = [pending_ceo["intent"], pending_cfo["intent"], row["intent"]]
                speakers = ["CEO", "CFO", "ANALYST"]
                signal = _assign_signal(intents[0], intents[1], intents[2])

                sequences.append(
                    {
                        "texts": texts,
                        "intents": intents,
                        "speakers": speakers,
                        "signal": signal,
                    }
                )

            pending_ceo = None
            pending_cfo = None
            continue

    return sequences


def _speaker_vector_3d(speaker: str) -> torch.Tensor:
    # Reuse existing speaker definitions for normalization/validation.
    _ = SPEAKER_TO_INDEX.get(speaker.upper(), SPEAKER_TO_INDEX["EXECUTIVE"])
    if speaker.upper() == "CEO":
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    if speaker.upper() == "CFO":
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    if speaker.upper() == "ANALYST":
        return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


def build_embedding_sequences(
    sequences: List[Dict[str, object]],
    finbert_wrapper: FinBERTIntentModel,
) -> tuple[List[torch.Tensor], List[str]]:
    embedding_sequences: List[torch.Tensor] = []
    signal_labels: List[str] = []

    for sequence in sequences:
        texts = sequence["texts"]  # type: ignore[assignment]
        speakers = sequence["speakers"]  # type: ignore[assignment]
        signal = str(sequence["signal"])

        if len(texts) != 3 or len(speakers) != 3:
            continue

        segment_embeddings: List[torch.Tensor] = []
        for text, speaker in zip(texts, speakers):
            output = finbert_wrapper.predict(str(text), max_length=128)
            cls_embedding = output["embedding"].float()  # (768,)
            speaker_vec = _speaker_vector_3d(str(speaker))  # (3,)
            final_embedding = torch.cat([cls_embedding, speaker_vec], dim=-1)  # (771,)
            segment_embeddings.append(final_embedding)

        if len(segment_embeddings) != 3:
            continue

        stacked = torch.stack(segment_embeddings)  # (3, 771)
        embedding_sequences.append(stacked)
        signal_labels.append(signal)

    return embedding_sequences, signal_labels


def main() -> None:
    print("=== V2 Training Pipeline ===")
    print(f"Dataset path: {DATA_PATH}")
    print(f"FinBERT intent output: {FINBERT_INTENT_PATH}")
    print(f"Conversation attention output: {CONVERSATION_ATTENTION_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).sample(10000, random_state=42)
    print(f"Loaded pragmatic dataset rows: {len(df)}")

    print("\n[1/4] Training FinBERT intent model...")
    finbert_wrapper = train_finbert_intent_model(
        dataset_path=DATA_PATH,
        output_path=FINBERT_INTENT_PATH,
        max_length=128,
        batch_size=16,
        epochs=4,
        learning_rate=2e-5,
    )
    print(f"Saved FinBERT intent model: {FINBERT_INTENT_PATH}")

    print("\n[2/4] Building conversation sequences (CEO -> CFO -> ANALYST)...")
    sequences = build_conversation_sequences(df)
    print(f"Conversation sequence dataset size: {len(sequences)}")
    if sequences:
        print("Sample sequence:")
        print(sequences[0])
    else:
        print("No valid sequences found in dataset.")

    print("\n[3/4] Building speaker-aware embeddings...")
    embedding_sequences, signal_labels = build_embedding_sequences(sequences, finbert_wrapper)
    print(f"Embedding sequences built: {len(embedding_sequences)}")
    if embedding_sequences:
        print(f"Sample embedding shape: {tuple(embedding_sequences[0].shape)}")
        print(f"Sample signal label: {signal_labels[0]}")

    # Safety fallback for very small/empty datasets.
    if not embedding_sequences:
        print(
            "[WARN] No valid embedding sequences available. "
            "Creating a minimal neutral bootstrap sample to avoid crash."
        )
        embedding_sequences = [torch.zeros(3, 771, dtype=torch.float32)]
        signal_labels = ["neutral"]

    print("\n[4/4] Training conversation attention model...")
    train_conversation_attention_model(
        embedding_sequences=embedding_sequences,
        signals=signal_labels,
        output_path=CONVERSATION_ATTENTION_PATH,
        batch_size=16,
        epochs=8,
        learning_rate=1e-4,
    )
    print(f"Saved conversation attention model: {CONVERSATION_ATTENTION_PATH}")

    print("\nV2 pipeline complete.")


if __name__ == "__main__":
    main()
