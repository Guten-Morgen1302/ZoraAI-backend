from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("zora.email_analyzer.stylometry")

WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

URGENCY_TERMS = (
    "urgent",
    "immediately",
    "action required",
    "asap",
    "within 24 hours",
    "suspended",
)

FINANCIAL_TERMS = (
    "account",
    "bank",
    "payment",
    "invoice",
    "password",
    "verify",
    "otp",
)

SOCIAL_ENGINEERING_TERMS = (
    "dear customer",
    "click here",
    "confirm now",
    "security alert",
    "update your account",
    "limited time",
)

SIGNOFF_TERMS = (
    "regards",
    "best regards",
    "thanks",
    "sincerely",
)

MODEL_FILENAME = "email_stylometry_model.joblib"
DEFAULT_DATASET_PATH = Path("app/fraud_memory/data/data_email_scam/merged_email_dataset.csv")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _normalize_binary_label(label_value: Any) -> int:
    value = str(label_value).strip().lower()
    if value in {"1", "spam", "scam", "phishing", "fraud", "malicious"}:
        return 1
    if value in {"0", "ham", "safe", "benign", "legitimate", "genuine"}:
        return 0
    return 0


def _count_term_hits(text_lower: str, terms: tuple[str, ...]) -> int:
    return sum(text_lower.count(term) for term in terms)


def extract_stylometry_features(text: str) -> dict[str, float | int]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized_text = text.strip()
    lower_text = normalized_text.lower()

    words = WORD_PATTERN.findall(normalized_text)
    sentence_parts = [s for s in SENTENCE_SPLIT_PATTERN.split(normalized_text) if s.strip()]

    alpha_chars = sum(1 for ch in normalized_text if ch.isalpha())
    upper_chars = sum(1 for ch in normalized_text if ch.isupper())
    digit_chars = sum(1 for ch in normalized_text if ch.isdigit())
    special_chars = sum(1 for ch in normalized_text if not ch.isalnum() and not ch.isspace())

    exclamation_count = normalized_text.count("!")
    question_count = normalized_text.count("?")
    currency_count = sum(normalized_text.count(symbol) for symbol in ("$", "€", "£", "₹"))

    urls = URL_PATTERN.findall(normalized_text)
    emails = EMAIL_PATTERN.findall(normalized_text)
    domains = DOMAIN_PATTERN.findall(normalized_text)
    html_tags = HTML_TAG_PATTERN.findall(normalized_text)

    urgency_hits = _count_term_hits(lower_text, URGENCY_TERMS)
    financial_hits = _count_term_hits(lower_text, FINANCIAL_TERMS)
    social_hits = _count_term_hits(lower_text, SOCIAL_ENGINEERING_TERMS)
    signoff_hits = _count_term_hits(lower_text, SIGNOFF_TERMS)

    avg_word_length = _safe_ratio(sum(len(w) for w in words), len(words))
    lexical_diversity = _safe_ratio(len(set(w.lower() for w in words)), len(words))
    avg_sentence_length = _safe_ratio(len(words), len(sentence_parts))

    caps_ratio = _safe_ratio(upper_chars, alpha_chars)
    digit_ratio = _safe_ratio(digit_chars, max(1, len(normalized_text)))
    special_ratio = _safe_ratio(special_chars, max(1, len(normalized_text)))

    repeated_punctuation = len(re.findall(r"[!?]{2,}", normalized_text))

    return {
        "text_length": int(len(normalized_text)),
        "word_count": int(len(words)),
        "sentence_count": int(len(sentence_parts)),
        "avg_word_length": round(avg_word_length, 4),
        "avg_sentence_length": round(avg_sentence_length, 4),
        "lexical_diversity": round(lexical_diversity, 4),
        "caps_ratio": round(caps_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "special_char_ratio": round(special_ratio, 4),
        "exclamation_count": int(exclamation_count),
        "question_count": int(question_count),
        "currency_symbol_count": int(currency_count),
        "repeated_punctuation_count": int(repeated_punctuation),
        "url_count": int(len(urls)),
        "email_count": int(len(emails)),
        "domain_count": int(len(domains)),
        "html_tag_count": int(len(html_tags)),
        "urgency_hits": int(urgency_hits),
        "financial_hits": int(financial_hits),
        "social_engineering_hits": int(social_hits),
        "signoff_hits": int(signoff_hits),
        "urgency_score": round(_safe_ratio(urgency_hits, len(URGENCY_TERMS)), 4),
    }


def _feature_frame(texts: list[str]) -> pd.DataFrame:
    rows = [extract_stylometry_features(text) for text in texts]
    return pd.DataFrame(rows)


def train_stylometry_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    text_column: str = "text",
    label_column: str = "label",
    model_output_path: str | Path | None = None,
) -> dict[str, float | int | str]:
    """Train an email stylometry model and persist it to disk."""
    csv_path = Path(dataset_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    logger.info("Loading email stylometry dataset from %s", csv_path)
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
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_val)
    acc = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    output_path = Path(model_output_path) if model_output_path else Path(__file__).resolve().parent / MODEL_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "feature_order": list(x.columns),
        "text_column": text_column,
        "label_column": label_column,
    }
    joblib.dump(payload, output_path)

    logger.info(
        "Email stylometry model trained. samples=%s accuracy=%.4f f1=%.4f output=%s",
        len(data),
        acc,
        f1,
        output_path,
    )
    print(
        "[email-stylometry] Training complete "
        f"samples={len(data)} accuracy={acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
    )

    return {
        "model_path": str(output_path),
        "samples": int(len(data)),
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def load_stylometry_model(model_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(model_path) if model_path else Path(__file__).resolve().parent / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Stylometry model not found: {path}")
    return joblib.load(path)


def predict_stylometry_score(text: str, model_path: str | Path | None = None) -> dict[str, float]:
    """Return stylometry score output compatible with the existing SMS flow."""
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train email stylometry model from merged_email_dataset.csv")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser


def _main() -> None:
    args = _build_arg_parser().parse_args()
    result = train_stylometry_model(
        dataset_path=args.dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
        model_output_path=args.output_path,
    )
    print(f"[email-stylometry] Model saved: {result}")


if __name__ == "__main__":
    _main()
