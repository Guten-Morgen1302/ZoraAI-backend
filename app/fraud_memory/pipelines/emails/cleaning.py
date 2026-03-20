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


def _first_non_empty(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = normalize_text(row.get(column))
        if value:
            return value
    return ""


def extract_email_body(row: dict[str, Any]) -> str:
    return _first_non_empty(row, ("body", "message", "Message", "text", "content"))


def extract_email_subject(row: dict[str, Any]) -> str:
    return _first_non_empty(row, ("subject", "Subject", "title"))


def extract_email_sender(row: dict[str, Any]) -> str:
    return _first_non_empty(row, ("sender", "from", "From", "sender_address"))


def extract_email_urls(row: dict[str, Any]) -> str:
    return _first_non_empty(row, ("urls", "url", "links", "link"))


def infer_email_label(row: dict[str, Any]) -> str:
    value = normalize_text(row.get("label") or row.get("Label") or row.get("is_spam")).lower()

    if value in {"phishing", "1", "1.0", "true", "yes", "spam", "scam", "fraud"}:
        return "phishing"
    if value in {"genuine", "0", "0.0", "false", "no", "ham", "safe", "legitimate"}:
        return "genuine"

    try:
        numeric = float(value)
    except ValueError:
        return "unknown"

    if numeric == 1.0:
        return "phishing"
    if numeric == 0.0:
        return "genuine"

    return "unknown"
