"""
Colab retraining script for email phishing classifier with feedback loop.

Flow:
1. Upload an existing model folder at /content/email_model (or zip and extract manually)
2. Upload two CSVs at /content: old training data + new feedback data
3. Script loads existing model/tokenizer, merges both CSVs, retrains, and writes output CSV artifacts

Expected CSV schema (minimum):
- text
- label (0=genuine, 1=phishing)
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass
class ColabConfig:
    model_dir: Path = Path("/content/email_model")
    old_csv: Path = Path("/content/old_email_data.csv")
    feedback_csv: Path = Path("/content/new_feedback_data.csv")
    output_dir: Path = Path("/content/email_model_retrained")

    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    validation_split: float = 0.10
    seed: int = 42


ID2LABEL = {0: "genuine", 1: "phishing"}
LABEL2ID = {"genuine": 0, "phishing": 1}


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


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must have columns text,label. Found: {list(df.columns)}")

    cleaned = df.dropna(subset=["text", "label"]).copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"].astype(bool)]
    cleaned["label"] = pd.to_numeric(cleaned["label"], errors="coerce")
    cleaned = cleaned.dropna(subset=["label"])
    cleaned["label"] = cleaned["label"].astype(int)
    cleaned = cleaned[cleaned["label"].isin([0, 1])].reset_index(drop=True)
    return cleaned


def evaluate(model, loader, device: torch.device) -> tuple[dict[str, float], list[int], list[int]]:
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

    metrics = {
        "loss": float(total_loss / max(1, len(loader))),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, average="binary", zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, average="binary", zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, average="binary", zero_division=0)),
    }
    return metrics, all_preds, all_labels


def run_retraining(config: ColabConfig) -> None:
    pip_install("transformers>=4.38", "accelerate>=0.26", "scikit-learn", "pandas", "numpy")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if not config.model_dir.exists() or not config.model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {config.model_dir}. Upload your existing model folder first."
        )

    old_df = load_dataset(config.old_csv)
    feedback_df = load_dataset(config.feedback_csv)

    merged_df = pd.concat([old_df, feedback_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if merged_df.empty:
        raise RuntimeError("Merged dataset is empty after cleaning")

    print(f"Old rows: {len(old_df)}")
    print(f"Feedback rows: {len(feedback_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged label distribution: {merged_df['label'].value_counts().to_dict()}")

    train_df, val_df = train_test_split(
        merged_df,
        test_size=config.validation_split,
        stratify=merged_df["label"],
        random_state=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_dir,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_ds = EmailDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, config.max_length)
    val_ds = EmailDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, config.max_length)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=config.learning_rate,
        eps=1e-8,
    )

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, config.num_epochs + 1):
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
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics, _, _ = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{config.num_epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Required output CSV: merged dataset used for retraining.
    merged_out_csv = config.output_dir / "merged_old_feedback_email_dataset.csv"
    merged_df.to_csv(merged_out_csv, index=False)

    # Useful output CSV: validation predictions.
    val_metrics, val_preds, val_labels = evaluate(model, val_loader, device)
    val_pred_df = val_df.copy().reset_index(drop=True)
    val_pred_df["pred_label"] = val_preds
    val_pred_df["true_label"] = val_labels
    val_pred_df.to_csv(config.output_dir / "validation_predictions.csv", index=False)

    with (config.output_dir / "retrain_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "rows_old": int(len(old_df)),
                "rows_feedback": int(len(feedback_df)),
                "rows_merged": int(len(merged_df)),
                "rows_train": int(len(train_df)),
                "rows_val": int(len(val_df)),
                "label_distribution": {int(k): int(v) for k, v in merged_df["label"].value_counts().to_dict().items()},
                "metrics": val_metrics,
                "model_loaded_from": str(config.model_dir),
            },
            fp,
            indent=2,
        )

    print("Retraining complete")
    print(f"Saved model folder: {config.output_dir}")
    print(f"Saved merged CSV: {merged_out_csv}")
    print(f"Saved validation CSV: {config.output_dir / 'validation_predictions.csv'}")


def main() -> None:
    print("Upload these paths in Colab before running:")
    print("- /content/email_model (folder)")
    print("- /content/old_email_data.csv")
    print("- /content/new_feedback_data.csv")

    config = ColabConfig()

    # Optional interactive override for CSV names in Colab.
    old_name = input("Old CSV filename in /content [old_email_data.csv]: ").strip()
    if old_name:
        config.old_csv = Path("/content") / old_name

    feedback_name = input("Feedback CSV filename in /content [new_feedback_data.csv]: ").strip()
    if feedback_name:
        config.feedback_csv = Path("/content") / feedback_name

    model_dir_name = input("Model folder in /content [email_model]: ").strip()
    if model_dir_name:
        config.model_dir = Path("/content") / model_dir_name

    run_retraining(config)


if __name__ == "__main__":
    main()
