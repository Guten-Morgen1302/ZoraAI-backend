"""TLS/SSL security intelligence feature extraction for phishing detection."""

from __future__ import annotations

import http.client
import socket
import ssl
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


def _default_payload(domain: str = "", has_https: bool = False) -> dict[str, Any]:
    """Return safe defaults for TLS extraction failures."""
    return {
        "domain": domain,
        "has_https": has_https,
        "certificate_issuer": "",
        "is_self_signed": False,
        "certificate_valid": False,
        "certificate_expiry_date": "",
        "days_to_expiry": 0,
        "tls_version": "",
        "hsts_enabled": False,
        "hsts_header": "",
    }


def _normalize_input(value: str) -> tuple[str, str]:
    """Normalize input and return (domain, scheme)."""
    raw = (value or "").strip()
    if not raw:
        return "", ""

    parsed = urlparse(raw)
    if parsed.scheme:
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
    else:
        scheme = ""
        parsed_with_scheme = urlparse(f"https://{raw}")
        host = parsed_with_scheme.hostname or raw.split("/")[0]

    host = host.strip().strip("[]").rstrip(".").lower()
    if not host:
        return "", scheme

    try:
        host = host.encode("idna").decode("ascii").lower()
    except Exception:
        host = host.lower()

    return host, scheme


def _extract_name_field(name_tuples: Any) -> str:
    """Flatten OpenSSL name tuple structure into a stable string."""
    if not isinstance(name_tuples, (list, tuple)):
        return ""

    values: list[str] = []
    for rdn in name_tuples:
        if not isinstance(rdn, (list, tuple)):
            continue
        for entry in rdn:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            key, value = entry
            key_s = str(key).strip()
            value_s = str(value).strip()
            if key_s and value_s:
                values.append(f"{key_s}={value_s}")

    return ", ".join(values)


def _extract_issuer_org(issuer_tuples: Any) -> str:
    """Extract issuer organization/common name from issuer tuples."""
    if not isinstance(issuer_tuples, (list, tuple)):
        return ""

    fallback = ""
    for rdn in issuer_tuples:
        if not isinstance(rdn, (list, tuple)):
            continue
        for entry in rdn:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            key, value = str(entry[0]).lower(), str(entry[1]).strip()
            if key == "organizationname" and value:
                return value
            if key == "commonname" and value and not fallback:
                fallback = value
    return fallback


def _parse_cert_expiry(not_after: str) -> tuple[str, int, bool]:
    """Parse cert expiry date and compute days to expiry and validity."""
    if not not_after:
        return "", 0, False

    try:
        # Example: "May 12 23:59:59 2027 GMT"
        expiry_dt = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
    except ValueError:
        return "", 0, False

    now = datetime.now(timezone.utc)
    delta = expiry_dt - now
    days_to_expiry = max(delta.days, 0)
    certificate_valid = delta.total_seconds() > 0
    return expiry_dt.date().isoformat(), days_to_expiry, certificate_valid


def _get_ssl_certificate(domain: str, timeout: float = 6.0) -> dict[str, Any]:
    """Retrieve certificate and TLS metadata; tolerate verification failures."""
    result = {
        "certificate": {},
        "tls_version": "",
        "connected": False,
        "verified": False,
    }
    if not domain:
        return result

    try:
        verified_ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=timeout) as sock:
            with verified_ctx.wrap_socket(sock, server_hostname=domain) as tls_sock:
                result["certificate"] = tls_sock.getpeercert() or {}
                result["tls_version"] = tls_sock.version() or ""
                result["connected"] = True
                result["verified"] = True
                return result
    except Exception:
        pass

    # Fallback: still gather certificate metadata even if verification fails.
    try:
        unverified_ctx = ssl.create_default_context()
        unverified_ctx.check_hostname = False
        unverified_ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((domain, 443), timeout=timeout) as sock:
            with unverified_ctx.wrap_socket(sock, server_hostname=domain) as tls_sock:
                result["certificate"] = tls_sock.getpeercert() or {}
                result["tls_version"] = tls_sock.version() or ""
                result["connected"] = True
                result["verified"] = False
    except Exception:
        return result

    return result


def _is_self_signed_certificate(cert: dict[str, Any]) -> bool:
    """Determine if certificate appears self-signed via subject/issuer match."""
    issuer_name = _extract_name_field(cert.get("issuer", ()))
    subject_name = _extract_name_field(cert.get("subject", ()))
    if not issuer_name or not subject_name:
        return False
    return issuer_name == subject_name


def _fetch_hsts_header(domain: str, timeout: float = 6.0) -> tuple[bool, str]:
    """Fetch HSTS header via HTTPS HEAD request."""
    if not domain:
        return False, ""

    conn: http.client.HTTPSConnection | None = None
    try:
        context = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection(domain, 443, timeout=timeout, context=context)
        conn.request("HEAD", "/")
        response = conn.getresponse()
        hsts_header = response.getheader("Strict-Transport-Security") or ""
        return bool(hsts_header.strip()), hsts_header.strip()
    except Exception:
        return False, ""
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def extract_tls_features(input_value: str) -> dict[str, Any]:
    """Extract TLS/SSL security features from a URL or domain input."""
    domain, scheme = _normalize_input(input_value)

    explicit_http = scheme == "http"
    wants_https = scheme == "https" or scheme == ""
    has_https = False if explicit_http else wants_https

    payload = _default_payload(domain=domain, has_https=has_https)
    if not domain:
        return payload

    ssl_data = _get_ssl_certificate(domain)
    cert = ssl_data.get("certificate", {}) if isinstance(ssl_data, dict) else {}
    connected = bool(ssl_data.get("connected", False)) if isinstance(ssl_data, dict) else False

    if scheme == "":
        # No scheme provided: infer HTTPS from successful TLS connection.
        has_https = connected
    elif scheme == "https":
        has_https = connected

    issuer = _extract_issuer_org(cert.get("issuer", ())) if isinstance(cert, dict) else ""
    expiry_raw = cert.get("notAfter", "") if isinstance(cert, dict) else ""
    expiry_date, days_to_expiry, certificate_valid = _parse_cert_expiry(str(expiry_raw or ""))
    is_self_signed = _is_self_signed_certificate(cert if isinstance(cert, dict) else {})
    tls_version = str(ssl_data.get("tls_version", "") or "") if isinstance(ssl_data, dict) else ""

    hsts_enabled, hsts_header = _fetch_hsts_header(domain) if has_https else (False, "")

    payload.update(
        {
            "domain": domain,
            "has_https": has_https,
            "certificate_issuer": issuer,
            "is_self_signed": is_self_signed,
            "certificate_valid": bool(certificate_valid and has_https),
            "certificate_expiry_date": expiry_date,
            "days_to_expiry": int(days_to_expiry) if has_https else 0,
            "tls_version": tls_version,
            "hsts_enabled": hsts_enabled,
            "hsts_header": hsts_header,
        }
    )
    return payload
