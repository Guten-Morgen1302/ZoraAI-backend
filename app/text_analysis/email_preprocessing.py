from __future__ import annotations

import copy
import html
import re
import unicodedata
from collections.abc import Sequence
from email.utils import parseaddr
from functools import lru_cache
from ipaddress import ip_address
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional runtime dependency
    BeautifulSoup = None

from app.text_analysis.preprocessing import normalize_homoglyphs

ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")
NON_PRINTABLE_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")

SIGNATURE_PATTERNS = [
    re.compile(r"(?im)^sent from my .*$"),
    re.compile(r"(?im)^best regards,?.*$"),
    re.compile(r"(?im)^kind regards,?.*$"),
    re.compile(r"(?im)^thanks,?.*$"),
    re.compile(r"(?im)^thank you,?.*$"),
]

DISCLAIMER_MARKERS = (
    "this message and any attachments",
    "intended only for the recipient",
    "if you are not the intended recipient",
    "confidentiality notice",
)

URGENCY_TERMS = (
    "urgent",
    "immediately",
    "action required",
    "verify now",
    "account suspended",
    "limited time",
)

FINANCIAL_TERMS = (
    "account",
    "bank",
    "verify",
    "password",
    "payment",
    "invoice",
)

TRUSTED_BRANDS = (
    "paypal",
    "microsoft",
    "google",
    "amazon",
    "apple",
    "bankofamerica",
    "chase",
)


def _deduplicate_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _strip_signatures_and_disclaimers(text: str) -> str:
    stripped = text
    for pattern in SIGNATURE_PATTERNS:
        stripped = pattern.sub("", stripped)

    lowered = stripped.lower()
    for marker in DISCLAIMER_MARKERS:
        index = lowered.find(marker)
        if index >= 0:
            stripped = stripped[:index]
            lowered = stripped.lower()

    return stripped.strip()


def _remove_invisible_and_control_chars(text: str) -> str:
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = NON_PRINTABLE_PATTERN.sub(" ", text)
    return text


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _normalize_for_nlp(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalize_homoglyphs(normalized)
    normalized = _remove_invisible_and_control_chars(normalized)
    normalized = _normalize_whitespace(normalized)
    return normalized.lower()


def _extract_urls_and_link_signals(body: str) -> tuple[list[str], list[dict[str, Any]], str]:
    html_text_for_model = body
    anchors: list[dict[str, Any]] = []
    collected_urls: list[str] = []

    if BeautifulSoup is not None:
        soup = BeautifulSoup(body, "html.parser")

        for anchor in soup.find_all("a"):
            href = (anchor.get("href") or "").strip()
            anchor_text = _normalize_whitespace(anchor.get_text(" ", strip=True))
            if href:
                collected_urls.append(href)
            if href or anchor_text:
                anchors.append(
                    {
                        "anchor_text": anchor_text,
                        "href": href,
                        "display_href_mismatch": bool(anchor_text and href and anchor_text.lower() not in href.lower()),
                    }
                )

        html_text_for_model = soup.get_text(" ")
    else:
        html_text_for_model = re.sub(r"<[^>]+>", " ", body)

    regex_urls = [match.group(0).rstrip(".,;!?)\"") for match in URL_PATTERN.finditer(body)]
    collected_urls.extend(regex_urls)

    canonical_urls: list[str] = []
    for url in collected_urls:
        url_candidate = url.strip()
        if not url_candidate:
            continue
        if url_candidate.startswith("www."):
            url_candidate = f"https://{url_candidate}"
        if not url_candidate.startswith(("http://", "https://")):
            continue
        canonical_urls.append(url_candidate)

    return _deduplicate_preserve_order(canonical_urls), anchors, html_text_for_model


def _extract_domain_parts(hostname: str) -> tuple[str, str]:
    host = hostname.lower().strip(".")
    host = host[4:] if host.startswith("www.") else host
    chunks = [chunk for chunk in host.split(".") if chunk]
    if len(chunks) >= 2:
        domain = ".".join(chunks[-2:])
        subdomain = ".".join(chunks[:-2])
        return domain, subdomain
    return host, ""


def _is_ip_host(hostname: str) -> bool:
    if not hostname:
        return False
    candidate = hostname.strip("[]")
    try:
        ip_address(candidate)
        return True
    except ValueError:
        return False


def _brand_impersonation_signal(hostname: str) -> bool:
    ascii_host = normalize_homoglyphs(hostname.lower())
    for brand in TRUSTED_BRANDS:
        if brand in ascii_host and ascii_host != brand and not ascii_host.endswith(f"{brand}.com"):
            return True
    return False


def _analyze_urls(urls: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    url_analysis: list[dict[str, Any]] = []
    has_ip_url = False
    suspicious_long_domain = False
    possible_typosquat = False

    for url in urls:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower().strip()
        domain, subdomain = _extract_domain_parts(hostname)
        query_dict = parse_qs(parsed.query)
        is_ip = _is_ip_host(hostname)
        long_domain = len(hostname) > 35
        looks_like_typosquat = _brand_impersonation_signal(hostname)

        has_ip_url = has_ip_url or is_ip
        suspicious_long_domain = suspicious_long_domain or long_domain
        possible_typosquat = possible_typosquat or looks_like_typosquat

        url_analysis.append(
            {
                "url": url,
                "scheme": parsed.scheme,
                "domain": domain,
                "subdomain": subdomain,
                "hostname": hostname,
                "path": parsed.path,
                "query_params": {key: values[:5] for key, values in query_dict.items()},
                "is_ip_url": is_ip,
                "is_long_domain": long_domain,
                "possible_typosquat": looks_like_typosquat,
            }
        )

    risk_flags = {
        "has_ip_url": has_ip_url,
        "has_long_domain": suspicious_long_domain,
        "possible_typosquat": possible_typosquat,
    }
    return url_analysis, risk_flags


def _sender_metadata(sender: str) -> dict[str, Any]:
    display_name, email_address = parseaddr(sender)
    normalized_email = email_address.strip().lower()
    sender_domain = ""
    sender_tld = ""

    if "@" in normalized_email:
        sender_domain = normalized_email.split("@", 1)[1]
        parts = sender_domain.rsplit(".", 1)
        sender_tld = f".{parts[1]}" if len(parts) == 2 else ""

    display_lower = display_name.lower().strip()
    mismatch = bool(display_lower and sender_domain and sender_domain.split(".")[0] not in display_lower)

    return {
        "sender_raw": sender,
        "sender_name": display_name.strip(),
        "sender_email": normalized_email,
        "sender_domain": sender_domain,
        "sender_tld": sender_tld,
        "display_domain_mismatch": mismatch,
    }


def _count_term_hits(text: str, terms: Sequence[str]) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(term) for term in terms)


def _stylometry_features(original_text: str, normalized_text: str) -> dict[str, float | int]:
    words = WORD_PATTERN.findall(normalized_text)
    sentence_count = len([segment for segment in SENTENCE_SPLIT_PATTERN.split(normalized_text) if segment.strip()])

    alpha_chars = [ch for ch in original_text if ch.isalpha()]
    uppercase_chars = [ch for ch in alpha_chars if ch.isupper()]
    punctuation_chars = [ch for ch in original_text if ch in "!?.,;:"]
    special_chars = [ch for ch in original_text if not ch.isalnum() and not ch.isspace()]

    avg_word_length = (sum(len(word) for word in words) / len(words)) if words else 0.0
    uppercase_ratio = (len(uppercase_chars) / len(alpha_chars)) if alpha_chars else 0.0
    punctuation_ratio = (len(punctuation_chars) / len(original_text)) if original_text else 0.0
    special_char_ratio = (len(special_chars) / len(original_text)) if original_text else 0.0

    return {
        "text_length": len(original_text),
        "word_count": len(words),
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 4),
        "uppercase_ratio": round(uppercase_ratio, 4),
        "punctuation_ratio": round(punctuation_ratio, 4),
        "special_char_ratio": round(special_char_ratio, 4),
    }


def _safe_text_for_llm(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()} ...[truncated]"


def _build_combined_text(sender: str, subject: str, body: str) -> str:
    return f"Sender: {sender}\nSubject: {subject}\nBody: {body}".strip()


@lru_cache(maxsize=512)
def _preprocess_email_cached(sender: str, subject: str, body: str, max_safe_chars: int) -> dict[str, Any]:
    raw_sender = sender or ""
    raw_subject = html.unescape(subject or "")
    raw_body = html.unescape(body or "")

    urls, anchors, body_text = _extract_urls_and_link_signals(raw_body)
    sender_meta = _sender_metadata(raw_sender)

    cleaned_body = _strip_signatures_and_disclaimers(body_text)
    combined_original = _build_combined_text(raw_sender, raw_subject, cleaned_body)

    normalized_text = _normalize_for_nlp(combined_original)
    clean_text = _normalize_whitespace(combined_original)

    url_analysis, url_risk = _analyze_urls(urls)

    urgency_hits = _count_term_hits(normalized_text, URGENCY_TERMS)
    financial_hits = _count_term_hits(normalized_text, FINANCIAL_TERMS)

    stylometry = _stylometry_features(clean_text, normalized_text)

    anchor_mismatch_count = sum(1 for a in anchors if a.get("display_href_mismatch"))

    features: dict[str, Any] = {
        **stylometry,
        "num_urls": len(urls),
        "num_anchor_mismatch": anchor_mismatch_count,
        "urgency_term_hits": urgency_hits,
        "financial_term_hits": financial_hits,
        "urgency_score": round(min(1.0, urgency_hits / max(len(URGENCY_TERMS), 1)), 4),
        "has_ip_url": url_risk["has_ip_url"],
        "has_long_domain": url_risk["has_long_domain"],
        "possible_typosquat": url_risk["possible_typosquat"],
        "display_domain_mismatch": sender_meta["display_domain_mismatch"],
    }

    metadata: dict[str, Any] = {
        **sender_meta,
        "subject": raw_subject,
        "anchor_links": anchors,
        "url_analysis": url_analysis,
        "safe_text": _safe_text_for_llm(normalized_text, max_chars=max_safe_chars),
        "truncated_for_llm": len(normalized_text) > max_safe_chars,
        "preprocessing_version": "email-v1",
    }

    return {
        "clean_text": clean_text,
        "normalized_text": normalized_text,
        "original_text": combined_original,
        "urls": urls,
        "sender_domain": sender_meta["sender_domain"],
        "features": features,
        "metadata": metadata,
    }


def preprocess_email_message(
    sender: str,
    subject: str,
    body: str,
    *,
    max_safe_chars: int = 4000,
) -> dict[str, Any]:
    """Preprocesses an email message for NLP, stylometry, and feature models."""
    if not isinstance(sender, str) or not isinstance(subject, str) or not isinstance(body, str):
        raise TypeError("sender, subject, and body must all be strings")

    output = _preprocess_email_cached(sender, subject, body, max_safe_chars)
    return copy.deepcopy(output)
