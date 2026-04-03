from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
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


def _to_intent_label(raw_value: str) -> str:
    value = str(raw_value or "").strip().upper()
    if value in INTENT_LABELS:
        return value
    if value in _SIGNAL_TO_INTENT:
        return _SIGNAL_TO_INTENT[value]
    return "GENERAL_UPDATE"


def _prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {column: column.strip().lower() for column in frame.columns}
    frame = frame.rename(columns=columns)

    if "text" not in frame.columns:
        raise ValueError("Dataset must include 'text' column")

    if "intent" not in frame.columns and "signal" not in frame.columns:
        raise ValueError("Dataset must include either 'intent' or 'signal' column")

    if "intent" in frame.columns:
        intent_source = frame["intent"]
    else:
        intent_source = frame["signal"]

    prepared = pd.DataFrame(
        {
            "text": frame["text"].fillna("").astype(str),
            "intent": intent_source.fillna("GENERAL_UPDATE").astype(str),
            "speaker": frame.get("speaker", "EXECUTIVE"),
        }
    )

    prepared["speaker"] = prepared["speaker"].fillna("EXECUTIVE").astype(str).str.upper()
    prepared["intent"] = prepared["intent"].map(_to_intent_label)

    analyst_mask = prepared["speaker"].eq("ANALYST")
    question_mask = prepared["text"].str.contains(_ANALYST_QUESTION_PATTERN, regex=True)
    probing_mask = analyst_mask | question_mask
    prepared.loc[probing_mask, "intent"] = "STRATEGIC_PROBING"

    prepared = prepared[prepared["text"].str.strip().str.len() > 0].reset_index(drop=True)

    missing_labels = [label for label in INTENT_LABELS if label not in set(prepared["intent"])]
    if missing_labels:
        seed_rows = pd.DataFrame(
            {
                "text": [
                    "We are expanding capacity and seeing strong demand.",
                    "Margins face pressure due to rising costs.",
                    "Could you clarify your guidance assumptions?",
                    "We will provide a routine operational update.",
                ],
                "intent": INTENT_LABELS,
                "speaker": ["CEO", "CFO", "ANALYST", "EXECUTIVE"],
            }
        )
        prepared = pd.concat([prepared, seed_rows], ignore_index=True)

    return prepared


class IntentTextDataset(Dataset):

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 128):
        self.labels = [INTENT_TO_INDEX[intent] for intent in frame["intent"].tolist()]
        encoded = tokenizer(
            frame["text"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


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
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_dataset_path}")

    output_dir = _resolve_output_dir(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training dataset: {resolved_dataset_path}")
    frame = pd.read_csv(resolved_dataset_path)
    train_frame = _prepare_training_frame(frame)

    class_counts = train_frame["intent"].value_counts().to_dict()
    print(f"Prepared training rows: {len(train_frame)}")
    print(f"Class distribution: {class_counts}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = IntentTextDataset(train_frame, tokenizer=tokenizer, max_length=max_length)

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
