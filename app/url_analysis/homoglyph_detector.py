"""Homoglyph and punycode phishing domain detection utilities."""

from __future__ import annotations

import unicodedata
from typing import Any
from urllib.parse import urlparse

import idna

try:
    import tldextract
except ImportError:  # pragma: no cover - optional dependency fallback
    tldextract = None

try:
    from rapidfuzz.distance import Levenshtein as RapidLevenshtein
except ImportError:  # pragma: no cover - optional dependency fallback
    RapidLevenshtein = None

_TLD_EXTRACTOR = (
    tldextract.TLDExtract(suffix_list_urls=None) if tldextract is not None else None
)

TOP_BRANDS: tuple[str, ...] = (
    "paypal",
    "google",
    "amazon",
    "facebook",
    "microsoft",
    "apple",
)

# Common Cyrillic and Greek confusables used in phishing domains.
HOMOGLYPH_MAP: dict[str, str] = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    "к": "k",
    "м": "m",
    "т": "t",
    "в": "b",
    "н": "h",
    "і": "i",
    "ӏ": "l",
    "ԁ": "d",
    "գ": "g",
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
    "α": "a",
    "β": "b",
    "γ": "y",
    "δ": "d",
    "ε": "e",
    "ι": "i",
    "κ": "k",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "u",
    "χ": "x",
}


def _safe_default(domain: str = "") -> dict[str, Any]:
    """Return safe defaults for extraction failures."""
    return {
        "original_domain": domain,
        "is_punycode": False,
        "decoded_domain": domain,
        "mixed_scripts": False,
        "normalized_domain": domain,
        "brand_similarity_score": 0.0,
        "matched_brand": "",
        "is_homoglyph_attack": False,
    }


def _normalize_host(host: str) -> str:
    """Normalize host by stripping wrappers and forcing lowercase."""
    value = (host or "").strip().strip("[]").rstrip(".").lower()
    if not value:
        return ""
    return value


def extract_domain(url_or_domain: str) -> str:
    """Extract the registered domain (eTLD+1) from URL/domain input."""
    raw = (url_or_domain or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    host = parsed.hostname if parsed.scheme else None
    if not host:
        parsed = urlparse(f"http://{raw}")
        host = parsed.hostname or raw.split("/")[0]

    host = _normalize_host(host)
    if not host:
        return ""

    if _TLD_EXTRACTOR is None:
        # Fallback if tldextract is not available.
        labels = [part for part in host.split(".") if part]
        if len(labels) < 2:
            return host
        return f"{labels[-2]}.{labels[-1]}"

    extracted = _TLD_EXTRACTOR(host)
    registered = getattr(extracted, "top_domain_under_public_suffix", "") or getattr(
        extracted, "registered_domain", ""
    )
    return (registered or host).lower()


def decode_punycode(domain: str) -> tuple[bool, str]:
    """Detect punycode labels and decode to Unicode where possible."""
    value = _normalize_host(domain)
    if not value:
        return False, ""

    labels = value.split(".")
    is_punycode = any(label.startswith("xn--") for label in labels)
    if not is_punycode:
        return False, value

    try:
        return True, idna.decode(value)
    except Exception:
        decoded_labels: list[str] = []
        for label in labels:
            if label.startswith("xn--"):
                try:
                    decoded_labels.append(idna.decode(label))
                except Exception:
                    decoded_labels.append(label)
            else:
                decoded_labels.append(label)
        return True, ".".join(decoded_labels)


def _char_script(char: str) -> str:
    """Return coarse script group for a single character."""
    if not char.isalpha():
        return ""

    try:
        name = unicodedata.name(char)
    except ValueError:
        return "UNKNOWN"

    if "LATIN" in name:
        return "LATIN"
    if "CYRILLIC" in name:
        return "CYRILLIC"
    if "GREEK" in name:
        return "GREEK"
    if "ARABIC" in name:
        return "ARABIC"
    if "HEBREW" in name:
        return "HEBREW"
    if "DEVANAGARI" in name:
        return "DEVANAGARI"
    if "HIRAGANA" in name:
        return "HIRAGANA"
    if "KATAKANA" in name:
        return "KATAKANA"
    if "HANGUL" in name:
        return "HANGUL"
    if "CJK" in name or "IDEOGRAPH" in name:
        return "CJK"
    return "OTHER"


def detect_mixed_scripts(domain: str) -> bool:
    """Detect whether alphabetic chars in domain belong to multiple scripts."""
    scripts = {script for script in (_char_script(char) for char in domain) if script}
    return len(scripts) > 1


def normalize_domain(domain: str) -> str:
    """Normalize domain by replacing known homoglyphs with Latin lookalikes."""
    return "".join(HOMOGLYPH_MAP.get(char, char) for char in domain)


def _extract_root_label(domain: str) -> str:
    """Extract root label from registered domain (without TLD)."""
    value = _normalize_host(domain)
    if not value:
        return ""

    if _TLD_EXTRACTOR is not None:
        extracted = _TLD_EXTRACTOR(value)
        if extracted.domain:
            return extracted.domain.lower()

    labels = [part for part in value.split(".") if part]
    if len(labels) >= 2:
        return labels[-2].lower()
    return labels[0].lower() if labels else ""


def _levenshtein_similarity(a: str, b: str) -> float:
    """Compute normalized Levenshtein similarity score in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    if RapidLevenshtein is not None:
        return float(RapidLevenshtein.normalized_similarity(a, b))

    # Fallback dynamic-programming implementation.
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    distance = previous[-1]
    max_len = max(len(a), len(b))
    return max(0.0, 1.0 - (distance / max_len))


def compute_similarity(domain: str, brands: tuple[str, ...] = TOP_BRANDS) -> tuple[float, str]:
    """Compute best similarity score against known brands."""
    root = _extract_root_label(domain)
    if not root:
        return 0.0, ""

    best_score = 0.0
    best_brand = ""
    for brand in brands:
        score = _levenshtein_similarity(root, brand)
        if score > best_score:
            best_score = score
            best_brand = brand
    return round(best_score, 6), best_brand


def detect_attack(
    *,
    is_punycode: bool,
    decoded_domain: str,
    mixed_scripts: bool,
    similarity_score: float,
) -> bool:
    """Apply homoglyph attack decision logic."""
    base_rule = mixed_scripts and similarity_score > 0.8

    # Additional hardening for punycode domains carrying non-ASCII Unicode.
    has_non_ascii = any(ord(char) > 127 for char in decoded_domain)
    punycode_rule = is_punycode and has_non_ascii
    return bool(base_rule or punycode_rule)


def extract_homoglyph_features(url_or_domain: str) -> dict[str, Any]:
    """Extract homoglyph and punycode phishing indicators from URL/domain input."""
    original_domain = extract_domain(url_or_domain)
    payload = _safe_default(original_domain)
    if not original_domain:
        return payload

    is_punycode, decoded_domain = decode_punycode(original_domain)
    mixed_scripts = detect_mixed_scripts(decoded_domain)
    normalized_domain = normalize_domain(decoded_domain)
    similarity_score, matched_brand = compute_similarity(normalized_domain)
    is_attack = detect_attack(
        is_punycode=is_punycode,
        decoded_domain=decoded_domain,
        mixed_scripts=mixed_scripts,
        similarity_score=similarity_score,
    )

    payload.update(
        {
            "original_domain": original_domain,
            "is_punycode": is_punycode,
            "decoded_domain": decoded_domain,
            "mixed_scripts": mixed_scripts,
            "normalized_domain": normalized_domain,
            "brand_similarity_score": similarity_score,
            "matched_brand": matched_brand,
            "is_homoglyph_attack": is_attack,
        }
    )
    return payload


def _self_test_cases() -> None:
    """Run required baseline test cases with simple assertions."""
    cases = {
        "paypal.com": False,
        "раypal.com": True,
        "xn--80a2a.com": True,
        "google.com": False,
        "gооgle.com": True,
    }
    for value, expected in cases.items():
        result = extract_homoglyph_features(value)
        observed = bool(result["is_homoglyph_attack"])
        assert observed == expected, (
            f"Expected {expected} for '{value}', got {observed}. Result={result}"
        )


if __name__ == "__main__":
    _self_test_cases()
