from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {
    "xyz",
    "top",
    "click",
    "gq",
    "work",
    "fit",
    "buzz",
    "rest",
    "country",
    "stream",
}

IP_HOST_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
RANDOM_SEGMENT_PATTERN = re.compile(r"[a-z0-9]{12,}")


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _extract_tld(host: str) -> str:
    parts = host.split(".")
    if not parts:
        return ""
    return parts[-1].lower()


def _url_flags(url: str) -> set[str]:
    flags: set[str] = set()
    parsed = urlparse(url)
    host = (parsed.netloc or "").split(":")[0].lower()

    if not host and parsed.path:
        host = parsed.path.split("/")[0].lower()

    tld = _extract_tld(host)
    if tld in SUSPICIOUS_TLDS:
        flags.add("suspicious_tld")

    if len(url) > 100:
        flags.add("long_url")

    if host and IP_HOST_PATTERN.match(host):
        flags.add("ip_based_url")

    normalized = re.sub(r"[^a-z0-9]", "", url.lower())
    entropy = _shannon_entropy(normalized)
    if entropy >= 4.0 or RANDOM_SEGMENT_PATTERN.search(normalized):
        flags.add("random_string")

    return flags


def analyze_urls(urls: list[str]) -> dict[str, object]:
    if not urls:
        return {"url_risk_score": 0.0, "flags": []}

    all_flags: set[str] = set()
    for url in urls:
        all_flags.update(_url_flags(url))

    score = min(1.0, round(len(all_flags) / 4, 4))
    return {
        "url_risk_score": score,
        "flags": sorted(all_flags),
    }
