from __future__ import annotations

import ipaddress
import math
import re
import socket
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    import dns.resolver
except ImportError:  # pragma: no cover - optional dependency
    dns = None
else:  # pragma: no cover - import style guard
    dns = dns.resolver

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
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "msclkid",
    "dclid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref",
    "ref_src",
    "source",
}

FAST_FLUX_IP_THRESHOLD = 3


def _is_internal_ip(ip_value: str) -> bool:
    try:
        parsed_ip = ipaddress.ip_address(ip_value)
    except ValueError:
        return False
    return (
        parsed_ip.is_private
        or parsed_ip.is_loopback
        or parsed_ip.is_link_local
        or parsed_ip.is_reserved
        or parsed_ip.is_multicast
    )


def _strip_tracking_parameters(parsed_url) -> tuple[str, bool]:
    query_pairs = parse_qsl(parsed_url.query, keep_blank_values=True)
    filtered_pairs: list[tuple[str, str]] = []
    removed = False

    for key, value in query_pairs:
        normalized_key = key.strip().lower()
        if normalized_key.startswith("utm_") or normalized_key in TRACKING_QUERY_KEYS:
            removed = True
            continue
        filtered_pairs.append((key, value))

    clean_query = urlencode(filtered_pairs, doseq=True)
    return clean_query, removed


def _resolve_dns_ips(host: str) -> set[str]:
    host = host.strip().lower()
    if not host:
        return set()

    ips: set[str] = set()
    if dns is not None:
        try:
            resolver = dns.Resolver()  # type: ignore[attr-defined]
            resolver.lifetime = 1.5
            resolver.timeout = 1.5
            answers = resolver.resolve(host, "A")
            for answer in answers:
                ips.add(str(answer))
        except Exception:
            pass

    if not ips:
        try:
            resolved = socket.getaddrinfo(host, None)
            for item in resolved:
                if item and len(item) >= 5 and item[4]:
                    ip_value = item[4][0]
                    if isinstance(ip_value, str):
                        ips.add(ip_value)
        except Exception:
            return set()

    return ips


def _sandbox_url(url: str) -> tuple[str, set[str]]:
    flags: set[str] = set()
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        flags.add("non_http_scheme")

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        flags.add("invalid_url")
        return url, flags

    if _is_internal_ip(host):
        flags.add("internal_ip_blocked")

    try:
        host_ascii = host.encode("idna").decode("ascii")
    except Exception:
        host_ascii = host

    clean_query, tracking_removed = _strip_tracking_parameters(parsed)
    if tracking_removed:
        flags.add("tracking_params_stripped")

    normalized_netloc = host_ascii
    if parsed.port:
        normalized_netloc = f"{host_ascii}:{parsed.port}"

    canonical_url = urlunparse(
        (
            parsed.scheme or "https",
            normalized_netloc,
            parsed.path or "",
            "",
            clean_query,
            "",
        )
    )

    resolved_ips = _resolve_dns_ips(host_ascii)
    if resolved_ips:
        public_ips = {ip for ip in resolved_ips if not _is_internal_ip(ip)}
        private_ips = {ip for ip in resolved_ips if _is_internal_ip(ip)}
        if private_ips:
            flags.add("internal_ip_blocked")
        if len(public_ips) >= FAST_FLUX_IP_THRESHOLD:
            flags.add("fast_flux_suspected")
    else:
        flags.add("dns_unresolved")

    return canonical_url, flags


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
        return {"url_risk_score": 0.0, "flags": [], "sanitized_urls": []}

    all_flags: set[str] = set()
    sanitized_urls: list[str] = []

    for url in urls:
        sanitized_url, sandbox_flags = _sandbox_url(url)
        sanitized_urls.append(sanitized_url)

        all_flags.update(sandbox_flags)
        all_flags.update(_url_flags(sanitized_url))

    score = min(1.0, round(len(all_flags) / 4, 4))
    return {
        "url_risk_score": score,
        "flags": sorted(all_flags),
        "sanitized_urls": sanitized_urls,
    }
