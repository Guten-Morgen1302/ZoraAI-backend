from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_email_text(row: dict[str, Any]) -> str:
    for column in ("message", "Message", "text", "body", "content"):
        value = normalize_text(row.get(column))
        if value:
            return value
    return ""


def infer_email_label(row: dict[str, Any]) -> str:
    value = normalize_text(row.get("label") or row.get("Label")).lower()
    if value in {"spam", "scam", "fraud", "phishing", "1", "true"}:
        return "scam"
    if value in {"ham", "safe", "0", "false"}:
        return "safe"
    return "unknown"
