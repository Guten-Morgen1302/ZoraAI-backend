"""
=============================================================================
SMS Phishing Classifier  ·  RoBERTa-base Fine-Tuning  (Pure PyTorch)
=============================================================================
Colab instructions:
  1. Upload final_merged_sms_dataset.csv to /content/
  2. Runtime → Change runtime type → GPU
  3. Runtime → Run all

Why RoBERTa instead of DeBERTa-v3-small?
  DeBERTa-v3 uses "StableDropout" in training mode which produces NaN
  gradients on T4 GPUs in float32. RoBERTa is equally strong for text
  classification and has zero numerical issues.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0 · INSTALL
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys
def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("transformers>=4.38", "accelerate>=0.26")
pip("scikit-learn", "pandas", "numpy", "tqdm")

# ─────────────────────────────────────────────────────────────────────────────
# 1 · IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, math, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 2 · CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH        = "/content/final_merged_sms_dataset.csv"
MODEL_NAME       = "roberta-base"        # stable, no NaN, great at classification
OUTPUT_DIR       = "sms_phishing_model"
MAX_LENGTH       = 128
BATCH_SIZE       = 16
LEARNING_RATE    = 2e-5
NUM_EPOCHS       = 3
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.06
MAX_GRAD_NORM    = 1.0
SEED             = 42
VALIDATION_SPLIT = 0.10

ID2LABEL = {0: "safe", 1: "scam"}
LABEL2ID = {"safe": 0, "scam": 1}

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("  SMS Phishing Classifier  ·  RoBERTa-base")
print("=" * 60)
print(f"  Device  : {device}")
if device.type == "cuda":
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3 · LOAD & CLEAN  (verified: data itself is healthy)
# ─────────────────────────────────────────────────────────────────────────────
print("\n📥  Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"    Raw rows : {len(df)}")

df = df.dropna(subset=["text", "label"])
df["text"]  = df["text"].astype(str)
df          = df[df["text"].str.strip().astype(bool)]
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df          = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df          = df[df["label"].isin([0, 1])].reset_index(drop=True)

print(f"    Clean    : {len(df)}")
print(f"    Labels   : {dict(df['label'].value_counts())}")
assert len(df) > 0, "Dataset empty after cleaning!"

# ─────────────────────────────────────────────────────────────────────────────
# 4 · SPLIT
# ─────────────────────────────────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=VALIDATION_SPLIT, stratify=df["label"], random_state=SEED
)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
print(f"\n📊  Train: {len(train_df)}   Val: {len(val_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5 · TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n🤗  Loading tokenizer ({MODEL_NAME}) …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# 6 · DATASET  (plain PyTorch, no HF Datasets / DataCollator magic)
# ─────────────────────────────────────────────────────────────────────────────
class SMSDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            # RoBERTa doesn't use token_type_ids — leave them out entirely
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_ds = SMSDataset(train_df["text"], train_df["label"])
val_ds   = SMSDataset(val_df["text"],   val_df["label"])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Batch sanity check
batch0 = next(iter(train_loader))
print(f"\n✅  Batch keys      : {list(batch0.keys())}")
print(f"    input_ids      : {batch0['input_ids'].shape}")
print(f"    attention_mask : {batch0['attention_mask'].shape}")
print(f"    labels (first8): {batch0['labels'][:8].tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 7 · MODEL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n🧠  Loading model ({MODEL_NAME}) …")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model.to(device)
print(f"    Parameters : {sum(p.numel() for p in model.parameters()):,}")

# Pre-train forward sanity check (eval mode)
model.eval()
with torch.no_grad():
    out = model(
        input_ids      = batch0["input_ids"][:2].to(device),
        attention_mask = batch0["attention_mask"][:2].to(device),
    )
nan_check = torch.isnan(out.logits).any() or torch.isinf(out.logits).any()
print(f"    Pre-train logits: {out.logits.cpu().tolist()}  → {'❌ NaN!' if nan_check else '✅ OK'}")
assert not nan_check, "NaN before training — check installation."

# Also check in TRAIN mode on same batch
model.train()
with torch.no_grad():   # no grad just for check
    out_train = model(
        input_ids      = batch0["input_ids"][:2].to(device),
        attention_mask = batch0["attention_mask"][:2].to(device),
    )
nan_train = torch.isnan(out_train.logits).any() or torch.isinf(out_train.logits).any()
print(f"    Train-mode logits: {out_train.logits.cpu().tolist()}  → {'❌ NaN!' if nan_train else '✅ OK'}")
assert not nan_train, "NaN in train mode before gradient step — model issue."

# ─────────────────────────────────────────────────────────────────────────────
# 8 · OPTIMIZER & SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
no_decay = ["bias", "LayerNorm.weight"]
optimizer = AdamW(
    [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n,p in model.named_parameters() if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ],
    lr=LEARNING_RATE, eps=1e-8,
)
total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
print(f"\n    Total steps  : {total_steps}")
print(f"    Warmup steps : {warmup_steps}")

# ─────────────────────────────────────────────────────────────────────────────
# 9 · EVAL HELPER
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    loss_fn    = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labs  = batch["labels"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            total_loss += loss_fn(logits, labs).item()
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(labs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "loss"     : total_loss / len(loader),
        "accuracy" : accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
        "recall"   : recall_score(all_labels, all_preds, average="binary", zero_division=0),
        "f1"       : f1_score(all_labels, all_preds, average="binary", zero_division=0),
        "preds"    : all_preds,
        "labels"   : all_labels,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 10 · TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏋️  Training …\n")
print(f"{'Ep':>3}  {'Train Loss':>10}  {'Val Loss':>9}  {'Acc':>6}  {'F1':>6}")
print("─" * 45)

loss_fn        = nn.CrossEntropyLoss()
best_f1        = 0.0
best_wts       = None
history        = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Ep {epoch}/{NUM_EPOCHS}", leave=False)
    for step, batch in enumerate(pbar, 1):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        labs  = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss   = loss_fn(logits, labs)

        if torch.isnan(loss):
            print(f"\n⚠️  NaN loss ep{epoch} step{step}  logits={logits[:2].tolist()}")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

    train_loss  = running_loss / len(train_loader)
    val_metrics = evaluate(model, val_loader)

    print(f"{epoch:>3}  {train_loss:>10.4f}  {val_metrics['loss']:>9.4f}  "
          f"{val_metrics['accuracy']:>6.4f}  {val_metrics['f1']:>6.4f}")

    history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k,v in val_metrics.items() if k not in ("preds","labels")}})

    if val_metrics["f1"] > best_f1:
        best_f1  = val_metrics["f1"]
        best_wts = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        print(f"     ↑ New best F1 = {best_f1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 11 · FINAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n📥  Restoring best weights …")
model.load_state_dict(best_wts)
model.to(device)

final = evaluate(model, val_loader)
preds, true_labels = final["preds"], final["labels"]

print("\n" + "=" * 55)
print("  📊  FINAL METRICS")
print("=" * 55)
for k in ("loss","accuracy","precision","recall","f1"):
    print(f"  {k:12s}: {final[k]:.4f}")

unique_cls, counts = np.unique(preds, return_counts=True)
pred_dist = dict(zip(unique_cls.tolist(), counts.tolist()))
print(f"\n  Pred dist : {pred_dist}")
if len(unique_cls) < 2:
    print("  ❌  Predicting only one class!")
else:
    print("  ✅  Both classes predicted.")

cm = confusion_matrix(true_labels, preds)
print(f"\n  Confusion Matrix:")
print(f"              Pred=safe  Pred=scam")
print(f"  True=safe   {cm[0][0]:>9}  {cm[0][1]:>9}")
print(f"  True=scam   {cm[1][0]:>9}  {cm[1][1]:>9}")
print(f"\n{classification_report(true_labels, preds, target_names=['safe','scam'], zero_division=0)}")

if final["f1"] < 0.75:
    print(f"  ⚠️   F1={final['f1']:.4f} < 0.75 — try 1 more epoch.")
else:
    print(f"  ✅  F1={final['f1']:.4f} — above 0.75 threshold.")

# ─────────────────────────────────────────────────────────────────────────────
# 12 · SAVE
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n💾  Saved to ./{OUTPUT_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# 13 · INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    cls   = torch.argmax(probs).item()
    return {"label": ID2LABEL[cls], "confidence": round(probs[cls].item(), 4)}

# ─────────────────────────────────────────────────────────────────────────────
# 14 · EXAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  🔮  EXAMPLE PREDICTIONS")
print("=" * 60)
test_msgs = [
    ("URGENT: Your SBI account is blocked. Update KYC at http://sbi-kyc-verify.xyz/login",        "scam"),
    ("Congratulations! You won Rs 10 Lakh in the RBI Draw. Pay Rs 2500 to claim.",                "scam"),
    ("Your PhonePe account flagged. Reverify at http://phonepe-reverify.in/secure in 12 hrs.",    "scam"),
    ("Earn Rs 1200/day online. No experience. Pay Rs 999 registration to start.",                  "scam"),
    ("WINNER!! Selected to receive $900 prize. Call 09061701461 now!",                             "scam"),
    ("Hey, are you coming to the party tonight? Let me know!",                                     "safe"),
    ("Your OTP for login is 482910. Valid 5 minutes. Do not share.",                               "safe"),
    ("Reminder: dentist appointment tomorrow at 3 PM.",                                            "safe"),
    ("Mom said dinner is ready. Come home soon.",                                                  "safe"),
    ("Meeting rescheduled to 4 PM. Please update your calendar.",                                  "safe"),
    ("Thanks for your order! Package arrives Thursday.",                                            "safe"),
    ("Happy birthday! Wishing you a great year ahead 🎂",                                          "safe"),
]
correct = 0
print(f"\n  {'OK':>2}  {'PRED':>5}  {'CONF':>6}  {'EXP':>5}  TEXT")
print("  " + "─" * 80)
for text, expected in test_msgs:
    r  = predict(text)
    ok = "✅" if r["label"] == expected else "❌"
    correct += (r["label"] == expected)
    print(f"  {ok}  {r['label'].upper():>5s}  {r['confidence']:.4f}  {expected.upper():>5s}  \"{text[:55]}{'…' if len(text)>55 else ''}\"")
print(f"\n  Test accuracy: {correct}/{len(test_msgs)}")

# ─────────────────────────────────────────────────────────────────────────────
# 15 · SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅  COMPLETE")
print("=" * 60)
print(f"  Model    : {MODEL_NAME}")
print(f"  Train    : {len(train_ds)} samples")
print(f"  Val      : {len(val_ds)} samples")
print(f"  Best F1  : {best_f1:.4f}")
print(f"  Saved    : ./{OUTPUT_DIR}/")
print("=" * 60)
print()
print("📌  To reload:")
print(f"    tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")
print(f"    model     = AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')")