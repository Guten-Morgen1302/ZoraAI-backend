from __future__ import annotations

import argparse
import json
import random
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

ID2LABEL = {0: "genuine", 1: "phishing"}
LABEL2ID = {"genuine": 0, "phishing": 1}


@dataclass
class RetrainConfig:
    new_data: Path
    old_data: Path
    base_model_dir: Path
    fallback_model_name: str
    output_dir: Path
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


def load_merged_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must have columns text,label: {path}")

    cleaned = df.dropna(subset=["text", "label"]).copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"].astype(bool)]
    cleaned["label"] = pd.to_numeric(cleaned["label"], errors="coerce")
    cleaned = cleaned.dropna(subset=["label"])
    cleaned["label"] = cleaned["label"].astype(int)
    cleaned = cleaned[cleaned["label"].isin([0, 1])].reset_index(drop=True)
    return cleaned


def sample_old_data(old_df: pd.DataFrame, target_count: int, seed: int) -> pd.DataFrame:
    if target_count <= 0 or old_df.empty:
        return old_df.iloc[0:0].copy()

    target_count = min(target_count, len(old_df))
    grouped = []
    labels = sorted(old_df["label"].unique().tolist())

    allocated = 0
    for label in labels:
        group = old_df[old_df["label"] == label]
        portion = len(group) / len(old_df)
        take = max(1, int(round(target_count * portion)))
        take = min(take, len(group))
        allocated += take
        grouped.append(group.sample(n=take, random_state=seed))

    sampled = pd.concat(grouped, ignore_index=True)

    if allocated > target_count:
        sampled = sampled.sample(n=target_count, random_state=seed)
    elif allocated < target_count:
        extra = min(target_count - allocated, len(old_df))
        sampled = pd.concat([sampled, old_df.sample(n=extra, random_state=seed)], ignore_index=True)

    return sampled.drop_duplicates().sample(frac=1, random_state=seed).reset_index(drop=True)


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


def retrain(config: RetrainConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    new_df = load_merged_csv(config.new_data)
    old_df = load_merged_csv(config.old_data)

    replay_target = int(round(len(new_df) * config.old_replay_ratio))
    replay_target = min(replay_target, config.max_old_samples) if config.max_old_samples > 0 else replay_target
    replay_df = sample_old_data(old_df, replay_target, config.seed)

    combined_df = pd.concat([new_df, replay_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    if combined_df.empty:
        raise RuntimeError("Combined retraining dataset is empty.")

    train_df, val_df = train_test_split(
        combined_df,
        test_size=config.validation_split,
        stratify=combined_df["label"],
        random_state=config.seed,
    )

    model_source = config.base_model_dir if config.base_model_dir.exists() else config.fallback_model_name
    print(f"Loading model/tokenizer from: {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
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
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{config.num_epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} "
            f"| val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / f"email_model_retrained_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    combined_df.to_csv(output_dir / "incremental_mix_snapshot.csv", index=False)
    final_metrics = evaluate(model, val_loader, device)

    with (output_dir / "incremental_train_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "new_rows": int(len(new_df)),
                "old_rows_available": int(len(old_df)),
                "old_rows_replayed": int(len(replay_df)),
                "combined_rows": int(len(combined_df)),
                "label_distribution": {int(k): int(v) for k, v in combined_df["label"].value_counts().to_dict().items()},
                "metrics": final_metrics,
                "model_source": str(model_source),
                "new_data": str(config.new_data),
                "old_data": str(config.old_data),
            },
            fp,
            indent=2,
        )

    print("Retraining complete")
    print(f"Saved retrained model to: {output_dir}")


def parse_args() -> RetrainConfig:
    parser = argparse.ArgumentParser(description="Incrementally retrain email phishing model with old-data replay.")
    parser.add_argument("--new-data", type=Path, default=Path("app/text_analysis/email_analyzer/merged_email_dataset.csv"))
    parser.add_argument("--old-data", type=Path, required=True, help="Historical merged CSV with text,label")
    parser.add_argument("--base-model-dir", type=Path, default=Path("app/text_analysis/email_analyzer/email_model"))
    parser.add_argument("--fallback-model-name", default="roberta-base")
    parser.add_argument("--output-dir", type=Path, default=Path("app/text_analysis/email_analyzer"))
    parser.add_argument("--old-replay-ratio", type=float, default=0.35)
    parser.add_argument("--max-old-samples", type=int, default=5000)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    args = parser.parse_args()
    return RetrainConfig(
        new_data=args.new_data,
        old_data=args.old_data,
        base_model_dir=args.base_model_dir,
        fallback_model_name=args.fallback_model_name,
        output_dir=args.output_dir,
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
    )


if __name__ == "__main__":
    retrain(parse_args())
