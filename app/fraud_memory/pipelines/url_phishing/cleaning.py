from __future__ import annotations

from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_label(value: Any) -> str:
    text = normalize_text(value).lower()
    if text in {"0", "false", "scam", "phishing", "spam"}:
        return "scam"
    if text in {"1", "true", "safe", "legitimate", "ham"}:
        return "safe"
    return "unknown"


def row_to_text(row: dict[str, Any]) -> str:
    url = normalize_text(row.get("URL"))
    if not url:
        return ""
    return f"URL risk profile: {url}"
