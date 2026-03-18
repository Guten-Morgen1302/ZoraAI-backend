from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(raw_label: Any) -> str:
    value = normalize_text(raw_label).lower()
    if value in {"spam", "scam", "fraud", "phishing", "1", "true", "yes"}:
        return "scam"
    if value in {"ham", "safe", "legitimate", "0", "false", "no"}:
        return "safe"
    return "unknown"


def extract_text_from_row(row: dict[str, Any]) -> str:
    columns = ("text", "message", "msg", "Message", "Msg", "v2", "sms", "body")
    for column in columns:
        value = normalize_text(row.get(column))
        if value:
            return value

    for value in row.values():
        text = normalize_text(value)
        if text:
            return text
    return ""


def extract_label_from_row(row: dict[str, Any]) -> str:
    columns = ("label", "Label", "v1", "class", "category")
    for column in columns:
        label = normalize_text(row.get(column))
        if label:
            return normalize_label(label)
    return "unknown"
