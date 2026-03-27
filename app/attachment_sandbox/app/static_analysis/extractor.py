from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

from app.static_analysis.mime_detector import detect_mime

logger = logging.getLogger(__name__)

# ── Heuristic regex patterns ───────────────────────────────────────────────
_IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)
_REGISTRY_PATTERN = re.compile(
    r"(?:HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER|HKLM|HKCU)\\", re.IGNORECASE
)
_POWERSHELL_PATTERN = re.compile(
    r"(?:powershell|Invoke-Expression|IEX|Set-ExecutionPolicy|"
    r"Invoke-WebRequest|New-Object\s+System\.Net\.WebClient)",
    re.IGNORECASE,
)
_BASE64_PATTERN = re.compile(
    r"[A-Za-z0-9+/]{40,}={0,2}"
)
_HTTP_URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>()[\]{}]+",
    re.IGNORECASE,
)


def compute_entropy(data: bytes) -> float:
    """Compute Shannon entropy of *data* in bits per byte."""
    if not data:
        return 0.0
    length = len(data)
    freq: dict[int, int] = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def extract_strings(data: bytes, min_len: int = 4) -> list[str]:
    """Extract printable ASCII strings of at least *min_len* characters."""
    pattern = re.compile(rb"[\x20-\x7E]{%d,}" % min_len)
    return [match.decode("ascii", errors="ignore") for match in pattern.findall(data)]


def extract_base_features(file_path: str) -> dict[str, Any]:
    """Extract base static features from any file."""
    result: dict[str, Any] = {
        "file_size": 0,
        "entropy": 0.0,
        "mime_type": "application/octet-stream",
        "strings_count": 0,
        "has_ip_pattern": False,
        "has_registry_keys": False,
        "has_powershell": False,
        "has_base64_blob": False,
        "extracted_urls": [],
        "extracted_url_count": 0,
    }

    try:
        result["file_size"] = os.path.getsize(file_path)
        result["mime_type"] = detect_mime(file_path)

        with open(file_path, "rb") as fh:
            data = fh.read()

        result["entropy"] = compute_entropy(data)
        strings = extract_strings(data)
        result["strings_count"] = len(strings)

        joined_strings = "\n".join(strings)
        result["has_ip_pattern"] = bool(_IP_PATTERN.search(joined_strings))
        result["has_registry_keys"] = bool(_REGISTRY_PATTERN.search(joined_strings))
        result["has_powershell"] = bool(_POWERSHELL_PATTERN.search(joined_strings))
        result["has_base64_blob"] = bool(_BASE64_PATTERN.search(joined_strings))

        # Keep a compact list for downstream URL pipeline fan-out in the portal.
        extracted_urls: list[str] = []
        seen_urls: set[str] = set()
        for match in _HTTP_URL_PATTERN.findall(joined_strings):
            cleaned = match.rstrip("),.;\"'")
            if cleaned and cleaned not in seen_urls:
                seen_urls.add(cleaned)
                extracted_urls.append(cleaned)
            if len(extracted_urls) >= 20:
                break

        result["extracted_urls"] = extracted_urls
        result["extracted_url_count"] = len(extracted_urls)

    except Exception:
        logger.exception("Failed to extract base features from %s", file_path)

    return result
