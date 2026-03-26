import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModel, AutoTokenizer

# Let PyTorch use all available CPU cores for BERT inference
# (torch.set_num_threads(2) was here — removed because it throttled
# CPU-based BERT forward passes to ~30-60s per batch)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Disable unstable SDPA backends; use only safe math kernel
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


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
        self.max_length = max_length

        texts = self.frame["text"].astype(str).tolist()
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor(
            [INTENT_TO_INDEX[intent] for intent in self.frame["intent"].tolist()],
            dtype=torch.long,
        )

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": self.labels[index],
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
        self.encoder_device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",  # Disable SDPA to prevent kernel dispatch hang on Colab
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, len(INTENT_LABELS)),
        )
        self.model = self.model.float()
        self.classifier = self.classifier.float()
        self.model.to(self.encoder_device)
        self.classifier.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def load_weights(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> bool:
        path = Path(model_path)
        if not path.exists():
            return False
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "classifier" in state:
            self.classifier.load_state_dict(state["classifier"])
        elif isinstance(state, dict):
            try:
                self.classifier.load_state_dict(state)
            except RuntimeError:
                return False
        else:
            return False
        self.model.eval()
        self.classifier.eval()
        return True

    def save_weights(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"classifier": self.classifier.state_dict()}, path)

    def predict(self, text: str, max_length: int = 128):
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        encoded_cpu = {key: value.to(self.encoder_device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_cpu)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(cls_embedding.to(self.device)).squeeze(0)

        logits = logits.detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs).item())
        cls_embedding = cls_embedding.squeeze(0).detach().cpu()

        return {
            "intent": INDEX_TO_INTENT[pred_idx],
            "logits": logits,
            "embedding": cls_embedding,
            "confidence": float(probs[pred_idx].item()),
        }


def train_finbert_intent_model(
    dataset_path: Path | str | None = None,
    output_path: Path | str = DEFAULT_MODEL_PATH,
    max_length: int = 64,
    batch_size: int = 64,
    epochs: int = 4,
    learning_rate: float = 2e-5,
):
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_dataset_path}")

    print("Loading dataset...")
    frame = pd.read_csv(resolved_dataset_path)
    frame = frame.rename(columns={column: column.strip().lower() for column in frame.columns})

    if "text" not in frame.columns or "intent" not in frame.columns:
        raise ValueError("Dataset must include 'text' and 'intent' columns")

    frame["text"] = frame["text"].fillna("").astype(str)
    frame["intent"] = frame["intent"].fillna("GENERAL_UPDATE").astype(str).str.upper()
    frame = frame[frame["intent"].isin(INTENT_LABELS)].reset_index(drop=True)

    print("Initializing model...")
    model_wrapper = FinBERTIntentModel()
    print("Creating dataset (THIS IS TOKENIZATION)...")
    dataset = IntentTextDataset(frame, model_wrapper.tokenizer, max_length=max_length)
    print("Dataset ready.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Starting training loop...")
    model_wrapper.model.to(model_wrapper.encoder_device)
    model_wrapper.classifier.to(model_wrapper.device)
    model_wrapper.model.eval()
    model_wrapper.classifier.train()
    optimizer = AdamW(model_wrapper.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_batches = len(loader)
    print(f"Precomputing CLS embeddings... ({total_batches} batches)")
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 20 == 0:
                print(f"  Embedding batch {batch_idx + 1}/{total_batches}")
            input_ids_cpu = batch["input_ids"].to(model_wrapper.encoder_device)
            attention_mask_cpu = batch["attention_mask"].to(model_wrapper.encoder_device)
            labels = batch["label"]
            outputs = model_wrapper.model(
                input_ids=input_ids_cpu,
                attention_mask=attention_mask_cpu,
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embedding)
            all_labels.append(labels)

    X = torch.cat(all_embeddings, dim=0).to(model_wrapper.device)
    y = torch.cat(all_labels, dim=0).to(model_wrapper.device)

    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Embedding cache ready. X={tuple(X.shape)}, y={tuple(y.shape)}")

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if batch_idx == 0:
                print("Running first classifier batch...")

            optimizer.zero_grad()
            logits = model_wrapper.classifier(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} Loss {avg_loss:.4f}")

    model_wrapper.save_weights(output_path)
    model_wrapper.model.eval()
    model_wrapper.classifier.eval()
    print(f"Saved finetuned intent model to: {output_path}")
    return model_wrapper


if __name__ == "__main__":
    train_finbert_intent_model(epochs=4)
