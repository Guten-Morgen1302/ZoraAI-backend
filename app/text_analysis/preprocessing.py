from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

WHITESPACE_PATTERN = re.compile(r"\s+")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_OR_DOMAIN_PATTERN = re.compile(
    r"(?i)\b((?:https?://|www\.)[^\s<>'\"]+|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>'\"]*)?)"
)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{4}(?!\w)"
)


def _deduplicate_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _normalize_text(raw_text: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw_text)
    without_zero_width = normalized.replace("\u200b", " ").replace("\xa0", " ")
    without_extra_whitespace = WHITESPACE_PATTERN.sub(" ", without_zero_width).strip()
    return without_extra_whitespace.lower()


def _extract_urls(clean_text: str) -> list[str]:
    raw_urls = [match.group(1) for match in URL_OR_DOMAIN_PATTERN.finditer(clean_text)]
    urls: list[str] = []
    for raw_url in raw_urls:
        cleaned_url = raw_url.rstrip(".,;!?)")
        if cleaned_url.startswith("www."):
            cleaned_url = f"https://{cleaned_url}"
        elif not cleaned_url.startswith(("http://", "https://")):
            cleaned_url = f"https://{cleaned_url}"
        urls.append(cleaned_url)
    return _deduplicate_preserve_order(urls)


def _extract_emails(clean_text: str) -> list[str]:
    return _deduplicate_preserve_order([match.group(0).lower() for match in EMAIL_PATTERN.finditer(clean_text)])


def _extract_phones(raw_text: str) -> list[str]:
    candidates = [match.group(0).strip() for match in PHONE_PATTERN.finditer(raw_text)]
    normalized: list[str] = []
    for phone in candidates:
        digits = "".join(ch for ch in phone if ch.isdigit())
        if 10 <= len(digits) <= 15:
            normalized.append(phone)
    return _deduplicate_preserve_order(normalized)


def preprocess_text(raw_text: str) -> dict[str, Any]:
    """Shared preprocessing output for all downstream models."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    clean_text = _normalize_text(raw_text)
    urls = _extract_urls(clean_text)
    phones = _extract_phones(raw_text)
    emails = _extract_emails(clean_text)

    return {
        "clean_text": clean_text,
        "urls": urls,
        "phones": phones,
        "emails": emails,
    }
