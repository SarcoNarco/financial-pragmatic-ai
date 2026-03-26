from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "yiyanghkust/finbert-tone"
INTENT_LABELS = [
    "EXPANSION",
    "COST_PRESSURE",
    "STRATEGIC_PROBING",
    "GENERAL_UPDATE",
]
INTENT_TO_INDEX = {label: idx for idx, label in enumerate(INTENT_LABELS)}
INDEX_TO_INTENT = {idx: label for label, idx in INTENT_TO_INDEX.items()}

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "conversation_dataset.csv"
)
FALLBACK_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "pragmatic_intent_dataset_clean.csv"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[0] / "finbert_intent.pt"


class IntentTextDataset(Dataset):

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 128):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(INTENT_TO_INDEX[row["intent"]], dtype=torch.long),
        }


def _resolve_dataset_path(dataset_path: Path | str | None) -> Path:
    if dataset_path is not None:
        return Path(dataset_path)
    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH
    return FALLBACK_DATASET_PATH


class FinBERTIntentModel:

    def __init__(self, model_name: str = MODEL_NAME, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(INTENT_LABELS),
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> bool:
        path = Path(model_path)
        if not path.exists():
            return False
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        return True

    def save_weights(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

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

        logits = outputs.logits.squeeze(0).detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs).item())
        cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze(0).detach().cpu()

        return {
            "intent": INDEX_TO_INTENT[pred_idx],
            "logits": logits,
            "embedding": cls_embedding,
            "confidence": float(probs[pred_idx].item()),
        }


def train_finbert_intent_model(
    dataset_path: Path | str | None = None,
    output_path: Path | str = DEFAULT_MODEL_PATH,
    max_length: int = 128,
    batch_size: int = 16,
    epochs: int = 4,
    learning_rate: float = 2e-5,
):
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_dataset_path}")

    frame = pd.read_csv(resolved_dataset_path)
    frame = frame.rename(columns={column: column.strip().lower() for column in frame.columns})

    if "text" not in frame.columns or "intent" not in frame.columns:
        raise ValueError("Dataset must include 'text' and 'intent' columns")

    frame["text"] = frame["text"].fillna("").astype(str)
    frame["intent"] = frame["intent"].fillna("GENERAL_UPDATE").astype(str).str.upper()
    frame = frame[frame["intent"].isin(INTENT_LABELS)].reset_index(drop=True)

    model_wrapper = FinBERTIntentModel()
    dataset = IntentTextDataset(frame, model_wrapper.tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print("DataLoader initialized. Starting training...")
    model_wrapper.model.train()
    optimizer = AdamW(model_wrapper.model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(model_wrapper.device)
            attention_mask = batch["attention_mask"].to(model_wrapper.device)
            labels = batch["label"].to(model_wrapper.device)

            outputs = model_wrapper.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} Loss {avg_loss:.4f}")

    model_wrapper.save_weights(output_path)
    model_wrapper.model.eval()
    print(f"Saved finetuned intent model to: {output_path}")
    return model_wrapper


if __name__ == "__main__":
    train_finbert_intent_model(epochs=4)
