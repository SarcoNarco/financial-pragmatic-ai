"""Train FinancialPragmaticTransformer on pragmatic intent data."""

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from financial_pragmatic_ai.models.financial_pragmatic_transformer_v2 import (
    FinancialPragmaticTransformer,
)
from financial_pragmatic_ai.models.speaker_embedding import get_speaker_embedding

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training device:", device)


INTENT_TO_INDEX = {
    "EXPANSION": 0,
    "COST_PRESSURE": 1,
    "STRATEGIC_PROBING": 2,
    "GENERAL_UPDATE": 3,
}

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "pragmatic_intent_dataset_clean.csv"
)
MODEL_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "pragmatic_transformer_trained.pt"
)

class PragmaticIntentDataset(Dataset):
    """Dataset that tokenizes text and returns speaker + intent labels."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 128) -> None:
        frame = frame.copy()
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["speaker"] = frame["speaker"].fillna("EXECUTIVE").astype(str)
        frame["intent"] = frame["intent"].fillna("GENERAL_UPDATE").astype(str)
        frame = frame[frame["intent"].isin(INTENT_TO_INDEX)]
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict:
        row = self.frame.iloc[idx]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "speaker": row["speaker"],
            "label": torch.tensor(INTENT_TO_INDEX[row["intent"]], dtype=torch.long),
        }


def collate_batch(batch: List[Dict]) -> Dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "speakers": [item["speaker"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch]),
    }


def build_speaker_batch(speakers: List[str], device: torch.device) -> torch.Tensor:
    embeddings = [get_speaker_embedding(speaker).squeeze(0) for speaker in speakers]
    return torch.stack(embeddings).to(device)


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print("Limiting dataset for debugging run...")
    df = df.head(20000)
    print("Dataset size after limit:", len(df))

    print("Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    dataset = PragmaticIntentDataset(df, tokenizer=tokenizer)
    print("Dataset loaded:", len(dataset))

    batch_size = 8
    if batch_size > 2:
        batch_size = 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_batch,
    )
    print("Creating dataloader...")
    print("Total batches:", len(dataloader))

    model = FinancialPragmaticTransformer()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    accumulation_steps = 4

    print("Starting training...")
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            speaker_embeddings = build_speaker_batch(batch["speakers"], device)
            labels = batch["labels"].to(device)

            if epoch == 0 and step == 0:
                print("Model device:", next(model.parameters()).device)
                print("Batch device:", input_ids.device)

            autocast_ctx = (
                torch.autocast(device_type="mps", dtype=torch.float16)
                if device.type == "mps"
                else nullcontext()
            )
            with autocast_ctx:
                logits = model(
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    speaker_embeddings,
                )
                loss = criterion(logits, labels)
            if step % 100 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")

            epoch_loss += loss.item()
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(dataloader) > 0 and len(dataloader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Epoch {epoch + 1}/3 - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"Saved trained model to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
