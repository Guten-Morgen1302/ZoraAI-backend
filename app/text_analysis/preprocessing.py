from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

try:
    from confusable_homoglyphs import confusables
except ImportError:  # pragma: no cover - optional dependency in some environments
    confusables = None

WHITESPACE_PATTERN = re.compile(r"\s+")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_OR_DOMAIN_PATTERN = re.compile(
    r"(?i)\b((?:https?://|www\.)[^\s<>'\"]+|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>'\"]*)?)"
)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{4}(?!\w)"
)

HOMOGLYPH_FALLBACK_MAP = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "у": "y",
        "х": "x",
        "і": "i",
        "ј": "j",
        "Α": "A",
        "Β": "B",
        "Ε": "E",
        "Ζ": "Z",
        "Η": "H",
        "Ι": "I",
        "Κ": "K",
        "Μ": "M",
        "Ν": "N",
        "Ο": "O",
        "Ρ": "P",
        "Τ": "T",
        "Υ": "Y",
        "Χ": "X",
    }
)

MAX_SMS_TEXT_CHARS = 4096
MIN_MEANINGFUL_SMS_CHARS = 12
MIN_MEANINGFUL_SMS_WORDS = 3


def _deduplicate_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _normalize_text(raw_text: str) -> str:
    normalized = normalize_homoglyphs(unicodedata.normalize("NFKC", raw_text))
    without_zero_width = normalized.replace("\u200b", " ").replace("\xa0", " ")
    without_extra_whitespace = WHITESPACE_PATTERN.sub(" ", without_zero_width).strip()
    return without_extra_whitespace.lower()


def normalize_homoglyphs(text: str) -> str:
    """Normalizes lookalike Unicode characters to safer representations."""
    if not text:
        return text

    if confusables is not None:
        normalize_fn = getattr(confusables, "normalize", None)
        if callable(normalize_fn):
            try:
                normalized = normalize_fn(text)
                if isinstance(normalized, list):
                    normalized = "".join(str(part) for part in normalized)
                if isinstance(normalized, str) and normalized:
                    return normalized
            except Exception:
                pass

    return text.translate(HOMOGLYPH_FALLBACK_MAP)


def validate_sms_text_quality(
    text: str,
    *,
    max_chars: int = MAX_SMS_TEXT_CHARS,
    min_chars: int = MIN_MEANINGFUL_SMS_CHARS,
    min_words: int = MIN_MEANINGFUL_SMS_WORDS,
) -> str:
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    normalized = WHITESPACE_PATTERN.sub(" ", text).strip()
    if not normalized:
        raise ValueError("Input text must not be empty")

    if len(normalized) > max_chars:
        raise ValueError(f"Input text is too large. Maximum allowed is {max_chars} characters")

    alnum_chars = sum(1 for char in normalized if char.isalnum())
    word_count = len([word for word in normalized.split(" ") if word])
    if alnum_chars < min_chars or word_count < min_words:
        raise ValueError(
            "Input text is too small for reliable fraud judgment. "
            "Please provide a longer message with meaningful context"
        )

    return normalized


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
