"""
Colab training script for email phishing classifier.

Usage in Colab:
1. Upload your merged CSV from Files tab (must contain: text, label)
2. Run this script
3. Enter uploaded CSV filename when prompted
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("transformers>=4.38", "accelerate>=0.26", "scikit-learn", "pandas", "numpy")

ID2LABEL = {0: "genuine", 1: "phishing"}
LABEL2ID = {"genuine": 0, "phishing": 1}

MODEL_NAME = "roberta-base"
OUTPUT_DIR = Path("/content/email_model")
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
VALIDATION_SPLIT = 0.10
SEED = 42


class EmailDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text, label")

    cleaned = df.dropna(subset=["text", "label"]).copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"].astype(bool)]
    cleaned["label"] = pd.to_numeric(cleaned["label"], errors="coerce")
    cleaned = cleaned.dropna(subset=["label"])
    cleaned["label"] = cleaned["label"].astype(int)
    cleaned = cleaned[cleaned["label"].isin([0, 1])].reset_index(drop=True)
    return cleaned


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return {
        "loss": float(total_loss / max(1, len(loader))),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, average="binary", zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, average="binary", zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, average="binary", zero_division=0)),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("If needed, upload file via Colab Files panel first.")
    csv_name = input("Enter CSV filename in /content (example: merged_email_dataset.csv): ").strip()
    data_path = Path("/content") / csv_name
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_dataset(data_path)
    print(f"Rows after cleaning: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        stratify=df["label"],
        random_state=SEED,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    train_ds = EmailDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LENGTH)
    val_ds = EmailDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=LEARNING_RATE,
        eps=1e-8,
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | train_loss={train_loss:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} "
            f"| val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    final_metrics = evaluate(model, val_loader, device)
    with (OUTPUT_DIR / "train_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "rows_total": int(len(df)),
                "rows_train": int(len(train_df)),
                "rows_val": int(len(val_df)),
                "label_distribution": {int(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
                "metrics": final_metrics,
            },
            fp,
            indent=2,
        )

    print("Training complete")
    print(f"Saved model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
