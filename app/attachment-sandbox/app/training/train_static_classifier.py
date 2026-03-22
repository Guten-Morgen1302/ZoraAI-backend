"""
train_static_classifier.py — Train the XGBoost static malware classifier.

Usage:
    python -m app.training.train_static_classifier
    — OR —
    cd app/attachment-sandbox && python -m app.training.train_static_classifier

Reads EMBER dataset from EMBER_DATA_DIR, trains an XGBClassifier, saves to
STATIC_MODEL_PATH, and runs end-to-end verification via classifier.predict().
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── Make imports work from any CWD ──────────────────────────────────────────
_SANDBOX_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _SANDBOX_ROOT not in sys.path:
    sys.path.insert(0, _SANDBOX_ROOT)

from app.static_analysis.classifier import FEATURE_COLS, build_feature_vector
from app.training.prepare_dataset import (
    build_dataset,
    save_processed_dataset,
)

# ── Paths from environment ──────────────────────────────────────────────────
_MODEL_PATH = os.environ.get("STATIC_MODEL_PATH", "models/static_classifier.pkl")
_FEATURES_PATH = os.environ.get(
    "STATIC_FEATURES_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "static_features.parquet"),
)
_VAL_SPLIT = float(os.environ.get("TRAINING_VAL_SPLIT", "0.15"))
_TEST_SPLIT = float(os.environ.get("TRAINING_TEST_SPLIT", "0.15"))
_MIN_SAMPLES = int(os.environ.get("MIN_TRAINING_SAMPLES", "500"))


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load training features and labels; build dataset if not cached."""
    if os.path.isfile(_FEATURES_PATH):
        print(f"Loading cached features from {_FEATURES_PATH}")
        df = pd.read_parquet(_FEATURES_PATH)
    else:
        print("No cached features found — building dataset from EMBER...")
        df = build_dataset()
        save_processed_dataset(df, _FEATURES_PATH)

    if len(df) < _MIN_SAMPLES:
        raise RuntimeError(
            f"Dataset has only {len(df)} samples (minimum: {_MIN_SAMPLES}). "
            f"Ensure EMBER data is available at {os.environ.get('EMBER_DATA_DIR', 'data/emberdataset')}"
        )

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> XGBClassifier:
    """Train an XGBClassifier with early stopping."""
    n_benign = int((y_train == 0).sum())
    n_malicious = int((y_train == 1).sum())
    scale_pos = n_benign / max(n_malicious, 1)

    print(f"\nTraining set: {len(y_train)} samples "
          f"(benign={n_benign}, malicious={n_malicious}, scale_pos_weight={scale_pos:.3f})")

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print(f"Best iteration: {model.best_iteration}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evaluate model and print detailed metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_test, y_pred,
        target_names=["Benign (0)", "Malicious (1)"],
        digits=4,
    )
    print(report)

    # ROC-AUC
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc:.6f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted 0   Predicted 1")
    print(f"  Actual 0       {cm[0][0]:>10}   {cm[0][1]:>10}")
    print(f"  Actual 1       {cm[1][0]:>10}   {cm[1][1]:>10}")

    # FP / FN rates
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    print(f"\nFalse Positive Rate (at 0.5): {fpr:.6f}")
    print(f"False Negative Rate (at 0.5): {fnr:.6f}")

    return {
        "roc_auc": auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Feature importance
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(model: XGBClassifier, feature_names: list[str]) -> None:
    """Print a text bar chart of top feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n" + "=" * 60)
    print("TOP FEATURE IMPORTANCES")
    print("=" * 60)

    max_bars = 20  # width of full bar
    top_n = min(len(feature_names), 15)

    for rank in range(top_n):
        idx = indices[rank]
        name = feature_names[idx]
        imp = importances[idx]
        bar_len = int(imp / max(importances) * max_bars)
        bar = "█" * bar_len
        print(f"  {name:<25s} {bar:<{max_bars}s} {imp:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Model saving
# ═══════════════════════════════════════════════════════════════════════════

def save_model(model: XGBClassifier, path: str | None = None) -> str:
    """Save the trained model to disk using pickle."""
    path = path or _MODEL_PATH
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    with open(path, "wb") as fh:
        pickle.dump(model, fh)

    size_kb = os.path.getsize(path) / 1024
    print(f"\nModel saved → {path} ({size_kb:.1f} KB)")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_end_to_end() -> None:
    """Verify the saved model works through classifier.predict()."""
    print("\n" + "=" * 60)
    print("END-TO-END VERIFICATION")
    print("=" * 60)

    # Force reload of the cached model in classifier module
    import app.static_analysis.classifier as clf_module
    clf_module._MODEL_LOADED = False
    clf_module._MODEL = None

    loaded = clf_module.load_model()
    assert loaded is not None, "load_model() returned None — model file missing!"
    print("  ✓ Model loaded successfully")

    # Create a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"This is a benign test file for verification purposes.\n" * 10)
        tmp_path = tmp.name

    try:
        # Reset singleton so predict() picks up the newly loaded model
        clf_module._MODEL = loaded
        clf_module._MODEL_LOADED = True

        prob, feat_dict = clf_module.predict(tmp_path)

        assert isinstance(prob, float), f"prob is {type(prob)}, expected float"
        assert 0.0 <= prob <= 1.0, f"prob={prob} out of [0, 1] range"
        assert isinstance(feat_dict, dict), f"feat_dict is {type(feat_dict)}"
        assert len(feat_dict) > 0, "feat_dict is empty"

        print(f"  ✓ predict() returned prob={prob:.6f}, {len(feat_dict)} features")
        print(f"  ✓ Probability in valid range [0, 1]")
        print(f"\n  ✅ END-TO-END VERIFICATION PASSED")

    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Full training pipeline."""
    print("=" * 60)
    print("STATIC MALWARE CLASSIFIER — TRAINING")
    print("=" * 60)

    # 1. Load data
    X, y = load_training_data()
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Split: 70% train, 15% val, 15% test (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=_TEST_SPLIT,
        stratify=y,
        random_state=42,
    )
    val_fraction = _VAL_SPLIT / (1.0 - _TEST_SPLIT)  # fraction of remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=42,
    )

    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # 3. Train
    model = train_model(X_train, y_train, X_val, y_val)

    # 4. Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # 5. Feature importance
    plot_feature_importance(model, FEATURE_COLS)

    # 6. Save
    save_model(model)

    # 7. End-to-end verification
    verify_end_to_end()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  ROC-AUC:  {metrics['roc_auc']:.6f}")
    print(f"  FPR:      {metrics['false_positive_rate']:.6f}")
    print(f"  FNR:      {metrics['false_negative_rate']:.6f}")
    print(f"  Model:    {_MODEL_PATH}")


if __name__ == "__main__":
    main()
