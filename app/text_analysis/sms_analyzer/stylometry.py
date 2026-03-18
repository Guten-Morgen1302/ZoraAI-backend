from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
URGENCY_KEYWORDS = (
    "urgent",
    "immediately",
    "verify now",
    "account suspended",
    "click link",
)

MODEL_FILENAME = "stylometry_model.joblib"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def extract_stylometry_features(text: str) -> dict[str, float | int]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized_text = text.strip()
    words = WORD_PATTERN.findall(normalized_text)
    lower_text = normalized_text.lower()

    uppercase_chars = sum(1 for ch in normalized_text if ch.isupper())
    alphabetic_chars = sum(1 for ch in normalized_text if ch.isalpha())

    exclamation_count = normalized_text.count("!")
    urgency_hits = sum(1 for keyword in URGENCY_KEYWORDS if keyword in lower_text)

    avg_word_length = _safe_ratio(sum(len(word) for word in words), len(words))
    caps_ratio = _safe_ratio(uppercase_chars, alphabetic_chars)
    punctuation_score = _safe_ratio(exclamation_count, max(1, len(normalized_text)))
    urgency_score = _safe_ratio(urgency_hits, len(URGENCY_KEYWORDS))

    return {
        "avg_word_length": round(avg_word_length, 4),
        "caps_ratio": round(caps_ratio, 4),
        "exclamation_count": int(exclamation_count),
        "urgency_keywords_count": int(urgency_hits),
        "message_length": int(len(normalized_text)),
        "urgency_score": round(urgency_score, 4),
        "punctuation_score": round(punctuation_score, 4),
    }


def _feature_frame(texts: list[str]) -> pd.DataFrame:
    rows = [extract_stylometry_features(text) for text in texts]
    return pd.DataFrame(rows)


def _normalize_binary_label(label_value: Any) -> int:
    value = str(label_value).strip().lower()
    if value in {"1", "spam", "scam", "phishing", "fraud", "malicious"}:
        return 1
    if value in {"0", "ham", "safe", "benign", "legitimate"}:
        return 0
    return 0


def train_stylometry_model(
    dataset_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    model_output_path: str | Path | None = None,
) -> dict[str, float | int | str]:
    """Train a RandomForest stylometry model and persist it to disk."""
    csv_path = Path(dataset_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")
    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")

    data = df[[text_column, label_column]].dropna().copy()
    data[text_column] = data[text_column].astype(str)
    data = data[data[text_column].str.strip().astype(bool)]
    data["binary_label"] = data[label_column].map(_normalize_binary_label)

    x = _feature_frame(data[text_column].tolist())
    y = data["binary_label"].astype(int)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_val)
    acc = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions, zero_division=0)

    output_path = Path(model_output_path) if model_output_path else csv_path.parent / MODEL_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "feature_order": list(x.columns),
    }
    joblib.dump(payload, output_path)

    return {
        "model_path": str(output_path),
        "samples": int(len(data)),
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
    }


def load_stylometry_model(model_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(model_path) if model_path else Path(__file__).resolve().parent / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Stylometry model not found: {path}")
    return joblib.load(path)


def predict_stylometry_score(text: str, model_path: str | Path | None = None) -> dict[str, float]:
    """Return stylometry score in the requested output shape."""
    artifact = load_stylometry_model(model_path)
    model: RandomForestClassifier = artifact["model"]
    feature_order: list[str] = artifact["feature_order"]

    feature_map = extract_stylometry_features(text)
    row = pd.DataFrame([{name: feature_map[name] for name in feature_order}])

    probabilities = model.predict_proba(row)[0]
    positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
    stylometry_score = float(probabilities[positive_index])

    return {
        "stylometry_score": round(stylometry_score, 4),
    }


if __name__ == "__main__":
    default_dataset = Path(__file__).resolve().parent / "merged_sms_dataset.csv"
    if default_dataset.exists():
        result = train_stylometry_model(default_dataset, text_column="text", label_column="label")
        print(f"Trained stylometry model: {result}")
    else:
        print(
            "No default dataset found for training. "
            f"Expected: {default_dataset}"
        )
