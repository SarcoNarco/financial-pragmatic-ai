from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from financial_pragmatic_ai.evaluation.better_than_fin.utils import (
    build_ground_truth_signals,
    load_evaluation_dataset,
)


MODEL_NAME = "yiyanghkust/finbert-tone"
INTENT_LABELS = [
    "EXPANSION",
    "COST_PRESSURE",
    "STRATEGIC_PROBING",
    "GENERAL_UPDATE",
]
INTENT_TO_INDEX = {label: idx for idx, label in enumerate(INTENT_LABELS)}
INDEX_TO_INTENT = {idx: label for label, idx in INTENT_TO_INDEX.items()}
LABEL2ID = {
    "EXPANSION": 0,
    "COST_PRESSURE": 1,
    "STRATEGIC_PROBING": 2,
    "GENERAL_UPDATE": 3,
}
ID2LABEL = {
    0: "EXPANSION",
    1: "COST_PRESSURE",
    2: "STRATEGIC_PROBING",
    3: "GENERAL_UPDATE",
}

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "pragmatic_intent_dataset_clean.csv"
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[0] / "finbert_intent_v2"

_SIGNAL_TO_INTENT = {
    "GROWTH": "EXPANSION",
    "RISK": "COST_PRESSURE",
    "NEUTRAL": "GENERAL_UPDATE",
}

_ANALYST_QUESTION_PATTERN = re.compile(
    r"\?|\bcould you\b|\bcan you\b|\bhow\b|\bwhat\b|\bwhy\b|\bguidance\b",
    re.IGNORECASE,
)
_PROBING_START_PATTERN = re.compile(
    r"^\s*(?:can you|what is|how|could you)\b",
    re.IGNORECASE,
)


def _resolve_output_dir(output_path: Path | str | None) -> Path:
    if output_path is None:
        return DEFAULT_MODEL_DIR

    output = Path(output_path)
    if output.suffix:
        return output.with_suffix("")
    return output


def _resolve_dataset_path(dataset_path: Path | str | None) -> Path:
    if dataset_path is None:
        return DEFAULT_DATASET_PATH
    return Path(dataset_path)


def _build_intent_frame_from_eval(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["text"] = prepared["text"].fillna("").astype(str)
    if "speaker" in prepared.columns:
        prepared["speaker"] = prepared["speaker"].fillna("EXECUTIVE").astype(str).str.upper()
    else:
        prepared["speaker"] = "EXECUTIVE"
    prepared["intent"] = prepared["intent"].fillna("GENERAL_UPDATE").astype(str).str.upper()

    prepared = prepared[prepared["text"].str.strip().str.len() > 0].reset_index(drop=True)
    if prepared.empty:
        raise ValueError("Evaluation dataset is empty after text cleaning.")

    signals = build_ground_truth_signals(prepared["intent"].tolist())
    mapped_intent = [_SIGNAL_TO_INTENT[signal.upper()] for signal in signals]
    prepared["mapped_intent"] = mapped_intent

    question_mask = prepared["text"].str.contains(_ANALYST_QUESTION_PATTERN, regex=True)
    starter_mask = prepared["text"].str.contains(_PROBING_START_PATTERN, regex=True)
    probing_mask = question_mask | starter_mask

    prepared.loc[probing_mask, "mapped_intent"] = "STRATEGIC_PROBING"

    return prepared[["text", "mapped_intent"]].rename(columns={"mapped_intent": "intent"})


def _balance_intent_frame(frame: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    counts = frame["intent"].value_counts().to_dict()
    missing = [label for label in INTENT_LABELS if counts.get(label, 0) == 0]
    if missing:
        raise ValueError(
            f"Cannot train 4-class model; missing labels after mapping/heuristics: {missing}. "
            f"Counts: {counts}"
        )

    target_size = min(counts[label] for label in INTENT_LABELS)
    if target_size == 0:
        raise ValueError(f"Invalid balanced target size 0. Counts: {counts}")

    balanced_parts = []
    for label in INTENT_LABELS:
        subset = frame[frame["intent"] == label]
        balanced_parts.append(subset.sample(n=target_size, random_state=random_state))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced


def _build_hf_train_dataset(frame: pd.DataFrame, tokenizer, max_length: int = 128) -> HFDataset:
    # Required HF fields: text, label
    hf_source = pd.DataFrame(
        {
            "text": frame["text"].astype(str),
            "label": frame["intent"].map(INTENT_TO_INDEX).astype(int),
        }
    )

    dataset = HFDataset.from_pandas(hf_source, preserve_index=False)

    def _tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize_batch, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


class FinBERTIntentModel:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_dir: Path | str | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        candidate_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR

        if candidate_dir.exists() and (candidate_dir / "config.json").exists():
            source = str(candidate_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(source)
            self.model = AutoModelForSequenceClassification.from_pretrained(source)
        else:
            source = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(source)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                source,
                num_labels=4,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
            )

        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded FinBERT intent model num_labels={self.model.config.num_labels}")

    def save_pretrained(self, output_dir: Path | str = DEFAULT_MODEL_DIR):
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)

    def predict(self, text: str, max_length: int = 128):
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits.squeeze(0)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze(0)

        logits_cpu = logits.detach().cpu()
        probs = torch.softmax(logits_cpu, dim=-1)
        pred_class = int(torch.argmax(probs).item())
        label_map = {
            0: "EXPANSION",
            1: "COST_PRESSURE",
            2: "STRATEGIC_PROBING",
            3: "GENERAL_UPDATE",
        }
        intent = label_map.get(pred_class, "GENERAL_UPDATE")
        print("LOGITS:", logits_cpu.tolist())
        print("PRED CLASS:", pred_class)
        print(f"[DEBUG] CLASS → INTENT: {pred_class} → {intent}")

        return {
            "intent": intent,
            "logits": logits_cpu,
            "embedding": cls_embedding.detach().cpu().float(),
            "confidence": float(probs[pred_class].item()),
        }


def train_finbert_intent_model(
    dataset_path: Path | str | None = None,
    output_path: Path | str | None = None,
    max_length: int = 128,
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
):
    output_dir = _resolve_output_dir(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    print(f"Loading evaluation dataset source: {resolved_dataset_path}")
    eval_frame = load_evaluation_dataset(dataset_path=resolved_dataset_path)
    train_frame = _build_intent_frame_from_eval(eval_frame)
    train_frame = _balance_intent_frame(train_frame, random_state=42)

    if train_frame.empty:
        raise ValueError("Training dataset is empty after mapping and balancing.")

    class_counts = train_frame["intent"].value_counts().reindex(INTENT_LABELS, fill_value=0).to_dict()
    print(f"Prepared training rows: {len(train_frame)}")
    print(f"Training class distribution: {class_counts}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = _build_hf_train_dataset(train_frame, tokenizer=tokenizer, max_length=max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_artifacts"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved 4-class FinBERT intent model to: {output_dir}")

    wrapper_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FinBERTIntentModel(model_dir=output_dir, device=wrapper_device)


if __name__ == "__main__":
    train_finbert_intent_model()
