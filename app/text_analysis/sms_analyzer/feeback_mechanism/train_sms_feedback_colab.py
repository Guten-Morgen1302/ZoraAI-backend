from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
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

LABEL2ID = {"safe": 0, "scam": 1}
ID2LABEL = {0: "safe", 1: "scam"}

TEXT_COLUMN_CANDIDATES = ["text", "message", "sms", "content", "body"]
LABEL_COLUMN_CANDIDATES = ["label", "predicted_label", "target", "class", "is_spam"]

STRING_SAFE_LABELS = {"safe", "ham", "legitimate", "benign", "not scam", "normal"}
STRING_SCAM_LABELS = {"scam", "spam", "phishing", "fraud", "malicious"}


@dataclass
class ColabConfig:
    # Upload these in Colab before running.
    base_model_dir: Path = Path("/content/sms_model")
    old_data_csv: Path = Path("/content/old_training_data.csv")
    feedback_data_csv: Path = Path("/content/new_feedback_data.csv")

    output_dir: Path = Path("/content/sms_model_retrained")
    output_zip_prefix: str = "sms_model_retrained"

    validation_split: float = 0.1
    seed: int = 42
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0

    auto_install: bool = True


class SMSDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
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
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def ensure_packages() -> None:
    required = [
        "transformers>=4.38",
        "accelerate>=0.26",
        "scikit-learn",
        "pandas",
        "numpy",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *required])


def _find_column(candidates: list[str], columns: list[str]) -> str:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"None of columns {candidates} found. Available columns: {columns}")


def _normalize_label(value) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, (int, np.integer)):
        return 1 if int(value) == 1 else 0

    if isinstance(value, float):
        return 1 if int(value) == 1 else 0

    label = str(value).strip().lower()
    if not label:
        return None
    if label.isdigit():
        return 1 if int(label) == 1 else 0
    if label in STRING_SAFE_LABELS:
        return 0
    if label in STRING_SCAM_LABELS:
        return 1
    return 0


def load_and_normalize_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    text_col = _find_column(TEXT_COLUMN_CANDIDATES, list(df.columns))
    label_col = _find_column(LABEL_COLUMN_CANDIDATES, list(df.columns))

    normalized = pd.DataFrame()
    normalized["text"] = df[text_col].astype(str).str.strip()
    normalized["label"] = df[label_col].apply(_normalize_label)

    normalized = normalized.dropna(subset=["text", "label"])
    normalized = normalized[normalized["text"].astype(bool)]
    normalized["label"] = normalized["label"].astype(int)
    normalized = normalized[normalized["label"].isin([0, 1])].reset_index(drop=True)
    return normalized


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

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


def train_colab_feedback_retrain(config: ColabConfig) -> Path:
    if config.auto_install:
        ensure_packages()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    old_df = load_and_normalize_dataset(config.old_data_csv)
    feedback_df = load_and_normalize_dataset(config.feedback_data_csv)

    combined_df = pd.concat([old_df, feedback_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if combined_df.empty:
        raise RuntimeError("Merged dataset is empty after normalization")

    train_df, val_df = train_test_split(
        combined_df,
        test_size=config.validation_split,
        stratify=combined_df["label"],
        random_state=config.seed,
    )

    if not config.base_model_dir.exists() or not config.base_model_dir.is_dir():
        raise FileNotFoundError(
            f"Base model folder not found at {config.base_model_dir}. Upload your sms_model folder in Colab first."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_dir,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_ds = SMSDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, config.max_length)
    val_ds = SMSDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, config.max_length)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = AdamW(
        [
            {
                "params": [p for name, p in model.named_parameters() if not any(nd in name for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for name, p in model.named_parameters() if any(nd in name for nd in no_decay)],
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

    print(f"Train rows: {len(train_ds)} | Val rows: {len(val_ds)}")

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
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{config.num_epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    merged_csv_path = config.output_dir / "merged_old_feedback_dataset.csv"
    combined_df.to_csv(merged_csv_path, index=False)

    stats = {
        "old_rows": int(len(old_df)),
        "feedback_rows": int(len(feedback_df)),
        "merged_rows": int(len(combined_df)),
        "label_distribution": {int(k): int(v) for k, v in combined_df["label"].value_counts().to_dict().items()},
        "best_val_f1": round(float(best_f1), 4),
        "base_model_dir": str(config.base_model_dir),
        "old_data_csv": str(config.old_data_csv),
        "feedback_data_csv": str(config.feedback_data_csv),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    stats_path = config.output_dir / "retrain_stats.json"
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    zip_base = Path("/content") / config.output_zip_prefix
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(config.output_dir))

    print("Retraining complete")
    print(f"Saved model folder: {config.output_dir}")
    print(f"Saved merged CSV: {merged_csv_path}")
    print(f"Saved stats: {stats_path}")
    print(f"Saved zip: {zip_path}")

    return Path(zip_path)


def main() -> None:
    config = ColabConfig()
    train_colab_feedback_retrain(config)


if __name__ == "__main__":
    main()
