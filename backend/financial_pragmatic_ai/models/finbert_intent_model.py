import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification

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

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 64):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self.frame["text"].astype(str).tolist()
        self.labels = [INTENT_TO_INDEX[intent] for intent in self.frame["intent"].tolist()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
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
        self.encoder_device = self.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(INTENT_LABELS),
            ignore_mismatched_sizes=True,
            attn_implementation="eager",  # Disable SDPA to prevent kernel dispatch hang on Colab
        )
        hidden_size = int(self.model.config.hidden_size)
        self.model.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, len(INTENT_LABELS)),
        )
        self.classifier = self.model.classifier
        self.model = self.model.float()
        self.model.classifier = self.model.classifier.float()
        self.model.to(self.encoder_device)
        self.model.classifier.to(self.device)
        self.model.eval()
        for param in self.model.bert.parameters():
            param.requires_grad = False

    def load_weights(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> bool:
        path = Path(model_path)
        if not path.exists():
            return False

        state = torch.load(path, map_location=self.device)
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []

        if isinstance(state, dict) and "classifier" in state and isinstance(state["classifier"], dict):
            classifier_state = state["classifier"]
            required_classifier_keys = {"0.weight", "0.bias", "2.weight", "2.bias"}
            missing_classifier_keys = sorted(required_classifier_keys - set(classifier_state.keys()))
            if missing_classifier_keys:
                raise RuntimeError(
                    "Classifier weights missing in finbert_intent checkpoint: "
                    f"{missing_classifier_keys}"
                )
            missing_keys, unexpected_keys = self.model.classifier.load_state_dict(
                classifier_state,
                strict=False,
            )
        elif isinstance(state, dict):
            load_info = self.model.load_state_dict(state, strict=False)
            missing_keys = list(load_info.missing_keys)
            unexpected_keys = list(load_info.unexpected_keys)
            missing_classifier = [key for key in missing_keys if key.startswith("classifier")]
            if missing_classifier:
                raise RuntimeError(
                    "Classifier weights were not loaded into model classifier head: "
                    f"{missing_classifier}"
                )
        else:
            return False

        print(f"[FinBERTIntentModel] missing_keys: {missing_keys}")
        print(f"[FinBERTIntentModel] unexpected_keys: {unexpected_keys}")
        self.model.eval()
        self.model.classifier.eval()
        return True

    def save_weights(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"classifier": self.model.classifier.state_dict()}, path)

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
            outputs = self.model(
                **encoded_cpu,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits.squeeze(0)
            cls_embedding = outputs.hidden_states[-1][:, 0, :]

        logits = logits.detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs).item())
        cls_embedding = cls_embedding.squeeze(0).detach().cpu()
        print("LOGITS:", logits.tolist())
        print("PRED CLASS:", pred_idx)

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
    batch_size: int = 8,
    epochs: int = 4,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 4,
    max_samples: int | None = 20000,
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
    if max_samples is not None and len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Safe mode enabled: sampled {len(frame)} rows (max_samples={max_samples})")

    print("Initializing model...")
    model_wrapper = FinBERTIntentModel()
    if hasattr(model_wrapper.model, "encoder"):
        for parameter in model_wrapper.model.encoder.parameters():
            parameter.requires_grad = False

    print("Creating dataset (THIS IS TOKENIZATION)...")
    dataset = IntentTextDataset(frame, model_wrapper.tokenizer, max_length=max_length)
    print("Dataset ready.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print("Starting training loop...")
    model_wrapper.model.to(model_wrapper.encoder_device)
    model_wrapper.model.classifier.to(model_wrapper.device)
    model_wrapper.model.eval()
    model_wrapper.model.classifier.train()
    optimizer = AdamW(model_wrapper.model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = Path(output_path).with_name("finbert_intent_checkpoint.pt")
    total_steps = len(loader)
    optimizer.zero_grad()

    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(model_wrapper.encoder_device)
            attention_mask = batch["attention_mask"].to(model_wrapper.encoder_device)
            labels = batch["label"].to(model_wrapper.device)

            with torch.no_grad():
                outputs = model_wrapper.model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

            logits = model_wrapper.model.classifier(cls_embeddings.to(model_wrapper.device))
            loss = criterion(logits, labels)
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            if step % gradient_accumulation_steps == 0 or step == total_steps:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += float(loss.item())

            if step % 50 == 0 or step == total_steps:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"Step {step}/{total_steps} "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / max(total_steps, 1)
        print(f"Epoch {epoch + 1}/{epochs} Loss {avg_loss:.4f}")
        torch.save({"classifier": model_wrapper.model.classifier.state_dict()}, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    model_wrapper.save_weights(output_path)
    model_wrapper.model.eval()
    model_wrapper.model.classifier.eval()
    print(f"Saved finetuned intent model to: {output_path}")
    return model_wrapper


if __name__ == "__main__":
    train_finbert_intent_model(epochs=4)
