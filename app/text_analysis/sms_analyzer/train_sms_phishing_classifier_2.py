from __future__ import annotations

import argparse
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

LABEL2ID = {"safe": 0, "scam": 1}
ID2LABEL = {0: "safe", 1: "scam"}

TEXT_COLUMN_CANDIDATES = ["text", "message", "sms", "content", "body"]
LABEL_COLUMN_CANDIDATES = ["label", "predicted_label", "target", "class", "is_spam"]

STRING_SAFE_LABELS = {"safe", "ham", "legitimate", "benign", "not scam", "normal"}
STRING_SCAM_LABELS = {"scam", "spam", "phishing", "fraud", "malicious"}


@dataclass
class TrainConfig:
    new_data: Path
    old_data: Path
    output_dir: Path
    base_model_dir: Path
    fallback_model_name: str
    old_replay_ratio: float
    max_old_samples: int
    validation_split: float
    seed: int
    max_length: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    auto_install: bool
    upload_first: bool


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
    cmd = [sys.executable, "-m", "pip", "install", "-q", *required]
    subprocess.check_call(cmd)


def upload_files_if_requested(upload_first: bool) -> None:
    if not upload_first:
        return

    try:
        from google.colab import files  # type: ignore
    except Exception:
        print("upload_first was enabled, but google.colab is not available. Skipping upload prompt.")
        return

    print("Upload your files now: sms_model.zip or sms_model folder files, and both CSV datasets.")
    files.upload()


def _find_column(candidates: list[str], columns: list[str]) -> str:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"None of the columns {candidates} found in dataset columns: {columns}")


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
    normalized["text"] = df[text_col].astype(str)
    normalized["label"] = df[label_col].apply(_normalize_label)

    normalized = normalized.dropna(subset=["text", "label"])
    normalized["text"] = normalized["text"].str.strip()
    normalized = normalized[normalized["text"].astype(bool)]
    normalized["label"] = normalized["label"].astype(int)
    normalized = normalized[normalized["label"].isin([0, 1])].reset_index(drop=True)

    return normalized


def stratified_sample(df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if n_samples <= 0 or df.empty:
        return df.iloc[0:0].copy()

    n_samples = min(n_samples, len(df))
    if n_samples == len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    sampled_parts: list[pd.DataFrame] = []
    classes = sorted(df["label"].unique().tolist())

    allocated = 0
    for cls in classes:
        class_df = df[df["label"] == cls]
        ratio = len(class_df) / len(df)
        take = max(1, int(round(n_samples * ratio)))
        take = min(take, len(class_df))
        allocated += take
        sampled_parts.append(class_df.sample(n=take, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    if allocated > n_samples:
        sampled = sampled.sample(n=n_samples, random_state=seed)
    elif allocated < n_samples:
        # Simple top-up from remaining rows.
        extra = n_samples - allocated
        extras = df.sample(n=min(extra, len(df)), random_state=seed)
        sampled = pd.concat([sampled, extras], ignore_index=True)

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def resolve_base_model_source(base_model_dir: Path) -> Path | str:
    if base_model_dir.exists() and base_model_dir.is_dir():
        return base_model_dir

    zip_path = base_model_dir.with_suffix(".zip")
    if zip_path.exists():
        extract_dir = base_model_dir.parent / f"{base_model_dir.name}_unzipped"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        shutil.unpack_archive(str(zip_path), str(extract_dir))

        nested = extract_dir / base_model_dir.name
        if nested.exists() and nested.is_dir():
            return nested
        return extract_dir

    return "roberta-base"


def build_training_dataframe(config: TrainConfig) -> tuple[pd.DataFrame, dict]:
    new_df = load_and_normalize_dataset(config.new_data)
    old_df = load_and_normalize_dataset(config.old_data)

    replay_target = int(round(len(new_df) * config.old_replay_ratio))
    if config.max_old_samples > 0:
        replay_target = min(replay_target, config.max_old_samples)

    old_replay_df = stratified_sample(old_df, replay_target, config.seed)

    train_df = pd.concat([new_df, old_replay_df], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    stats = {
        "new_rows": int(len(new_df)),
        "old_rows_available": int(len(old_df)),
        "old_rows_replayed": int(len(old_replay_df)),
        "final_rows": int(len(train_df)),
        "label_distribution": {int(k): int(v) for k, v in train_df["label"].value_counts().to_dict().items()},
    }

    return train_df, stats


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


def train_incremental(config: TrainConfig) -> None:
    if config.auto_install:
        ensure_packages()

    upload_files_if_requested(config.upload_first)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    combined_df, stats = build_training_dataframe(config)
    if combined_df.empty:
        raise RuntimeError("Combined dataset is empty after preprocessing.")

    print("\nDataset stats:")
    print(json.dumps(stats, indent=2))

    train_df, val_df = train_test_split(
        combined_df,
        test_size=config.validation_split,
        stratify=combined_df["label"],
        random_state=config.seed,
    )

    model_source = resolve_base_model_source(config.base_model_dir)
    print(f"\nLoading model/tokenizer from: {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
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
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if not any(nd in name for nd in no_decay)
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if any(nd in name for nd in no_decay)
                ],
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

    print("\nTraining incremental model...")
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
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    target_output_dir = config.output_dir
    target_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.save_pretrained(target_output_dir)
        tokenizer.save_pretrained(target_output_dir)
    except OSError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_dir = target_output_dir.parent / f"{target_output_dir.name}_incremental_{timestamp}"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(fallback_dir)
        tokenizer.save_pretrained(fallback_dir)
        target_output_dir = fallback_dir

    replay_snapshot = target_output_dir / "incremental_mix_snapshot.csv"
    combined_df.to_csv(replay_snapshot, index=False)

    stats_path = target_output_dir / "incremental_train_stats.json"
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                **stats,
                "best_val_f1": round(best_f1, 4),
                "model_source": str(model_source),
                "output_dir": str(target_output_dir),
                "old_data": str(config.old_data),
                "new_data": str(config.new_data),
            },
            fp,
            indent=2,
        )

    print("\nDone.")
    print(f"Saved model to: {target_output_dir}")
    print(f"Saved training stats to: {stats_path}")
    print(f"Saved replay snapshot to: {replay_snapshot}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Colab one-shot incremental SMS scam classifier training with replay from old data.",
    )
    parser.add_argument("--new-data", type=Path, default=Path("/content/scam_data_new.csv"))
    parser.add_argument("--old-data", type=Path, default=Path("/content/final_merged_sms_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/sms_model_updated"))
    parser.add_argument("--base-model-dir", type=Path, default=Path("/content/sms_model"))
    parser.add_argument("--fallback-model-name", default="roberta-base")
    parser.add_argument("--old-replay-ratio", type=float, default=0.35)
    parser.add_argument("--max-old-samples", type=int, default=4000)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Install required packages inside Colab before training.",
    )
    parser.add_argument(
        "--upload-first",
        action="store_true",
        help="Open Colab upload prompt before starting training.",
    )

    args = parser.parse_args()

    return TrainConfig(
        new_data=args.new_data,
        old_data=args.old_data,
        output_dir=args.output_dir,
        base_model_dir=args.base_model_dir,
        fallback_model_name=args.fallback_model_name,
        old_replay_ratio=args.old_replay_ratio,
        max_old_samples=args.max_old_samples,
        validation_split=args.validation_split,
        seed=args.seed,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        auto_install=args.auto_install,
        upload_first=args.upload_first,
    )


if __name__ == "__main__":
    config = parse_args()
    train_incremental(config)
