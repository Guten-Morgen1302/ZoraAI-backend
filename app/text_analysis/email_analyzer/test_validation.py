"""
Test the email phishing classifier on merged_validation.csv
Loads the model from email_model/ and predicts labels for each email.
Outputs a CSV with the text and predicted label + confidence.
"""
from __future__ import annotations

import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "email_model"
DATA_PATH = BASE_DIR / "validation.csv"
OUTPUT_PATH = BASE_DIR / "validation_results.csv"

# ── Label mapping ─────────────────────────────────────────────
ID2LABEL = {0: "legitimate", 1: "phishing"}

MAX_LENGTH = 256


def _pick_text_column(df: pd.DataFrame) -> str:
    for col in ("tex", "text", "body", "message", "content"):
        if col in df.columns:
            return col
    raise ValueError(f"No supported text column found. Columns available: {list(df.columns)}")


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model & tokenizer
    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    # Load validation data
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} emails")
    print(f"Columns: {list(df.columns)}")

    text_col = _pick_text_column(df)
    texts = df[text_col].astype(str).tolist()

    # Run inference
    predictions = []
    confidences = []
    label_names = []

    print(f"\nRunning inference on {len(texts)} emails...\n")

    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0][pred].item()

            predictions.append(pred)
            confidences.append(conf)
            label_names.append(ID2LABEL[pred])

            # Print each result
            snippet = text[:80].replace("\n", " ") + ("..." if len(text) > 80 else "")
            print(f"  [{i+1:3d}/{len(texts)}] {ID2LABEL[pred]:>8s} ({conf:.2%})  │ {snippet}")

    # Create final output DataFrame with just text and label
    output_df = pd.DataFrame({
        "text": texts,
        "label": label_names
    })

    # Save results
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {OUTPUT_PATH}")

    # Summary
    genuine_count = sum(1 for p in predictions if p == 0)
    phishing_count = sum(1 for p in predictions if p == 1)
    print(f"\n  Total emails     : {len(texts)}")
    print(f"  Legitimate (0)   : {genuine_count}")
    print(f"  Phishing (1)     : {phishing_count}")
    print(f"  Avg confidence   : {sum(confidences)/len(confidences):.2%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
