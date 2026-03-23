"""Production-grade URL intelligence feature extraction utilities."""

from __future__ import annotations

import ipaddress
import math
from collections import Counter
from typing import Any
from urllib.parse import parse_qsl, urlparse

try:
    import tldextract
except ImportError:  # pragma: no cover - optional dependency fallback
    tldextract = None

SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "bank",
    "confirm",
    "password",
)

# Disable live suffix list fetch for deterministic/offline-safe behavior.
_TLD_EXTRACTOR = (
    tldextract.TLDExtract(suffix_list_urls=None) if tldextract is not None else None
)


def _default_feature_payload() -> dict[str, Any]:
    """Return the default payload for invalid or empty URL input."""
    return {
        "url_length": 0,
        "num_dots": 0,
        "num_hyphens": 0,
        "num_underscores": 0,
        "num_slashes": 0,
        "num_question_marks": 0,
        "num_equal_signs": 0,
        "num_at_symbols": 0,
        "num_percent_symbols": 0,
        "has_ip": False,
        "domain": "",
        "subdomain": "",
        "tld": "",
        "path_length": 0,
        "num_query_params": 0,
        "has_https": False,
        "suspicious_keywords": [],
        "num_suspicious_keywords": 0,
        "entropy": 0.0,
    }


def _normalize_url(url: str) -> str:
    """Normalize URL text by trimming and ensuring a parseable scheme."""
    value = (url or "").strip()
    if not value:
        return ""

    parsed = urlparse(value)
    if not parsed.scheme and not parsed.netloc:
        # Common raw input form: example.com/login
        value = f"http://{value}"
    return value


def _safe_hostname(parsed_url) -> str:
    """Return normalized lowercase hostname, handling IDN where possible."""
    host = (parsed_url.hostname or "").strip().rstrip(".")
    if not host:
        return ""

    try:
        # Normalize internationalized domains to a stable ASCII form.
        host = host.encode("idna").decode("ascii")
    except Exception:
        host = host.lower()
    return host.lower()


def _is_ip_address(host: str) -> bool:
    """Detect whether a host value is an IPv4/IPv6 address."""
    if not host:
        return False
    candidate = host.strip().strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def _shannon_entropy(value: str) -> float:
    """Compute Shannon entropy of an input string."""
    if not value:
        return 0.0

    counts = Counter(value)
    length = len(value)
    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


def _extract_suspicious_keywords(host: str, path: str) -> list[str]:
    """Return suspicious keywords found in host/path (case-insensitive)."""
    haystack = f"{host} {path}".lower()
    return [keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in haystack]


def _extract_domain_parts(host: str) -> tuple[str, str, str]:
    """Extract (domain, subdomain, tld) with tldextract or fallback logic."""
    if not host or _is_ip_address(host):
        return "", "", ""

    if _TLD_EXTRACTOR is not None:
        extracted = _TLD_EXTRACTOR(host)
        domain = extracted.domain.lower() if extracted.domain else ""
        subdomain = extracted.subdomain.lower() if extracted.subdomain else ""
        tld = extracted.suffix.lower() if extracted.suffix else ""
        return domain, subdomain, tld

    # Fallback heuristic if tldextract is not installed.
    labels = [part for part in host.split(".") if part]
    if len(labels) < 2:
        if not labels:
            return "", "", ""
        return labels[0].lower(), "", ""

    domain = labels[-2].lower()
    tld = labels[-1].lower()
    subdomain = ".".join(labels[:-2]).lower()
    return domain, subdomain, tld


def extract_url_features(url: str) -> dict[str, Any]:
    """Extract phishing-intelligence features from a URL.

    Args:
        url: A raw URL string (with or without scheme).

    Returns:
        Dictionary containing length, structure, keyword, and entropy features.
    """
    normalized_url = _normalize_url(url)
    if not normalized_url:
        return _default_feature_payload()

    parsed = urlparse(normalized_url)
    host = _safe_hostname(parsed)

    domain, subdomain, tld = _extract_domain_parts(host)

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    suspicious = _extract_suspicious_keywords(host=host, path=parsed.path or "")

    features: dict[str, Any] = {
        "url_length": len(normalized_url),
        "num_dots": normalized_url.count("."),
        "num_hyphens": normalized_url.count("-"),
        "num_underscores": normalized_url.count("_"),
        "num_slashes": normalized_url.count("/"),
        "num_question_marks": normalized_url.count("?"),
        "num_equal_signs": normalized_url.count("="),
        "num_at_symbols": normalized_url.count("@"),
        "num_percent_symbols": normalized_url.count("%"),
        "has_ip": _is_ip_address(host),
        "domain": domain,
        "subdomain": subdomain,
        "tld": tld,
        "path_length": len(parsed.path or ""),
        "num_query_params": len(query_pairs),
        "has_https": (parsed.scheme or "").lower() == "https",
        "suspicious_keywords": suspicious,
        "num_suspicious_keywords": len(suspicious),
        "entropy": round(_shannon_entropy(normalized_url), 6),
    }

    return features
