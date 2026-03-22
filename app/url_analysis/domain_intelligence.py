"""DNS and WHOIS infrastructure intelligence feature extraction."""

from __future__ import annotations

import re
import socket
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlparse

try:
    import dns.resolver
except ImportError:  # pragma: no cover - optional dependency
    dns = None
else:  # pragma: no cover - import style guard
    dns = dns.resolver

try:
    import whois
except ImportError:  # pragma: no cover - optional dependency
    whois = None

PRIVACY_PATTERN = re.compile(r"privacy|protected|whoisguard", re.IGNORECASE)
DATE_CANDIDATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


def _default_payload(domain: str = "") -> dict[str, Any]:
    """Return a safe default payload for extraction failures."""
    return {
        "domain": domain,
        "registrar": "",
        "is_whois_private": False,
        "num_a_records": 0,
        "a_records": [],
        "ttl": 0,
        "has_mx_records": False,
        "mx_records": [],
        "num_ns_records": 0,
        "fast_flux_detected": False,
        "suspicious_ttl": False,
    }


def _normalize_domain_input(value: str) -> str:
    """Normalize raw URL/domain input into a lowercase ASCII hostname."""
    raw = (value or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    if parsed.scheme:
        host = parsed.hostname or ""
    else:
        parsed_with_scheme = urlparse(f"http://{raw}")
        host = parsed_with_scheme.hostname or raw.split("/")[0]

    host = host.strip().strip("[]").rstrip(".").lower()
    if not host:
        return ""

    try:
        return host.encode("idna").decode("ascii").lower()
    except Exception:
        return host


def _extract_datetime(candidate: Any) -> datetime | None:
    """Parse datetime values returned by WHOIS providers."""
    if candidate is None:
        return None

    if isinstance(candidate, datetime):
        return candidate.astimezone(timezone.utc) if candidate.tzinfo else candidate.replace(tzinfo=timezone.utc)

    if isinstance(candidate, date):
        return datetime(candidate.year, candidate.month, candidate.day, tzinfo=timezone.utc)

    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            return None

        iso_text = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso_text)
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

        for fmt in (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
            "%d-%b-%Y",
            "%d-%b-%Y %H:%M:%S %Z",
            "%Y.%m.%d",
            "%Y/%m/%d",
        ):
            try:
                parsed = datetime.strptime(text, fmt)
                return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        match = DATE_CANDIDATE_PATTERN.search(text)
        if match:
            try:
                return datetime.strptime(match.group(0), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                return None

    return None


def _pick_creation_date(value: Any) -> datetime | None:
    """Return earliest creation date from WHOIS values."""
    values = value if isinstance(value, list) else [value]
    parsed = [dt for dt in (_extract_datetime(item) for item in values) if dt is not None]
    return min(parsed) if parsed else None


def _pick_expiration_date(value: Any) -> datetime | None:
    """Return latest expiration date from WHOIS values."""
    values = value if isinstance(value, list) else [value]
    parsed = [dt for dt in (_extract_datetime(item) for item in values) if dt is not None]
    return max(parsed) if parsed else None


def _fetch_whois_data(domain: str) -> dict[str, Any]:
    """Fetch WHOIS data and normalize key fields for feature extraction."""
    result = {
        "creation_dt": None,
        "expiration_dt": None,
        "registrar": "",
        "is_private": False,
    }

    if not domain or whois is None:
        return result

    try:
        record = whois.whois(domain)
    except Exception:
        return result

    creation_raw = getattr(record, "creation_date", None)
    expiration_raw = getattr(record, "expiration_date", None)
    registrar_raw = getattr(record, "registrar", "")
    text_parts = [
        str(getattr(record, "name", "") or ""),
        str(getattr(record, "org", "") or ""),
        str(getattr(record, "registrant_name", "") or ""),
        str(getattr(record, "registrant_organization", "") or ""),
        str(getattr(record, "emails", "") or ""),
        str(getattr(record, "text", "") or ""),
    ]

    result["creation_dt"] = _pick_creation_date(creation_raw)
    result["expiration_dt"] = _pick_expiration_date(expiration_raw)
    result["registrar"] = str(registrar_raw).strip() if registrar_raw else ""
    result["is_private"] = bool(PRIVACY_PATTERN.search(" ".join(text_parts)))
    return result


def _resolve_a_records(domain: str) -> tuple[list[str], int | None]:
    """Resolve A records and return unique IPs with TTL when available."""
    if not domain:
        return [], None

    ips: list[str] = []
    ttl_value: int | None = None

    if dns is not None:
        try:
            resolver = dns.Resolver()  # type: ignore[attr-defined]
            resolver.timeout = 2.0
            resolver.lifetime = 3.0
            answer = resolver.resolve(domain, "A")
            for item in answer:
                ip = item.to_text().strip()
                if ip and ip not in ips:
                    ips.append(ip)

            if hasattr(answer, "rrset") and answer.rrset is not None:
                ttl_value = int(answer.rrset.ttl)
        except Exception:
            ips = []
            ttl_value = None

    if not ips:
        try:
            for row in socket.getaddrinfo(domain, None):
                ip = row[4][0]
                if ip and ip not in ips:
                    ips.append(ip)
        except Exception:
            return [], ttl_value

    return ips, ttl_value


def _resolve_mx_records(domain: str) -> list[str]:
    """Resolve MX records as hostnames."""
    if not domain or dns is None:
        return []

    records: list[str] = []
    try:
        resolver = dns.Resolver()  # type: ignore[attr-defined]
        resolver.timeout = 2.0
        resolver.lifetime = 3.0
        answer = resolver.resolve(domain, "MX")
        for item in answer:
            exchange = getattr(item, "exchange", None)
            target = str(exchange).rstrip(".") if exchange is not None else item.to_text().strip()
            if target and target not in records:
                records.append(target)
    except Exception:
        return []

    return records


def _resolve_ns_records(domain: str) -> list[str]:
    """Resolve NS records as hostnames."""
    if not domain or dns is None:
        return []

    records: list[str] = []
    try:
        resolver = dns.Resolver()  # type: ignore[attr-defined]
        resolver.timeout = 2.0
        resolver.lifetime = 3.0
        answer = resolver.resolve(domain, "NS")
        for item in answer:
            target = item.to_text().rstrip(".").strip()
            if target and target not in records:
                records.append(target)
    except Exception:
        return []

    return records


def extract_domain_features(input_value: str) -> dict[str, Any]:
    """Extract DNS and WHOIS infrastructure features from a URL or domain."""
    domain = _normalize_domain_input(input_value)
    payload = _default_payload(domain)
    if not domain:
        return payload

    whois_data = _fetch_whois_data(domain)
    a_records, ttl_value = _resolve_a_records(domain)
    mx_records = _resolve_mx_records(domain)
    ns_records = _resolve_ns_records(domain)

    suspicious_ttl = ttl_value is not None and ttl_value < 300
    fast_flux = len(a_records) > 5 or suspicious_ttl

    payload.update(
        {
            "domain": domain,
            "registrar": str(whois_data.get("registrar", "") or ""),
            "is_whois_private": bool(whois_data.get("is_private", False)),
            "num_a_records": len(a_records),
            "a_records": a_records,
            "ttl": int(ttl_value) if ttl_value is not None else 0,
            "has_mx_records": len(mx_records) > 0,
            "mx_records": mx_records,
            "num_ns_records": len(ns_records),
            "fast_flux_detected": fast_flux,
            "suspicious_ttl": suspicious_ttl,
        }
    )
    return payload
