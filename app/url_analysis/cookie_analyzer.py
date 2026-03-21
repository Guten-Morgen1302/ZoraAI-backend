"""Cookie security analyzer for web application security scanning.

This module analyzes cookies captured from browser automation tools (such as
Playwright) and detects common cookie security weaknesses, misconfigurations,
and session fixation vulnerabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Risk weights from the specification.
RISK_WEIGHTS: dict[str, int] = {
    "missing_http_only": 2,
    "missing_secure": 2,
    "same_site_none": 2,
    "broad_domain_scope": 1,
}

MAX_COOKIE_RISK_POINTS = sum(RISK_WEIGHTS.values())

SESSION_COOKIE_HINTS: tuple[str, ...] = (
    "session",
    "sess",
    "sid",
    "jsessionid",
    "phpsessid",
    "auth",
    "token",
    "jwt",
    "access_token",
    "refresh_token",
)


@dataclass(frozen=True)
class NormalizedCookie:
    """Internal normalized representation of a browser cookie."""

    name: str
    value: str
    domain: str
    path: str
    secure: bool
    http_only: bool
    same_site: str


def _to_bool(value: Any) -> bool:
    """Best-effort bool conversion for cookie fields coming from various sources."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _normalize_same_site(value: Any) -> str:
    """Normalize SameSite value to Strict/Lax/None/Unknown."""
    if value is None:
        return "Unknown"

    raw = str(value).strip()
    if not raw:
        return "Unknown"

    lowered = raw.lower()
    if lowered == "strict":
        return "Strict"
    if lowered == "lax":
        return "Lax"
    if lowered == "none":
        return "None"
    return "Unknown"


def _normalize_domain(value: Any) -> str:
    """Normalize cookie domain for consistent analysis."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_path(value: Any) -> str:
    """Normalize cookie path to a stable non-empty value."""
    if value is None:
        return "/"

    text = str(value).strip()
    if not text:
        return "/"
    if not text.startswith("/"):
        return f"/{text}"
    return text


def _normalize_cookie(cookie: dict[str, Any] | None) -> NormalizedCookie:
    """Build a normalized cookie object from raw Playwright cookie dict."""
    payload = cookie if isinstance(cookie, dict) else {}

    name = str(payload.get("name") or "").strip()
    value = str(payload.get("value") or "")
    domain = _normalize_domain(payload.get("domain"))
    path = _normalize_path(payload.get("path"))
    secure = _to_bool(payload.get("secure", False))
    http_only = _to_bool(
        payload.get("httpOnly", payload.get("http_only", payload.get("httponly", False)))
    )
    same_site = _normalize_same_site(
        payload.get("sameSite", payload.get("same_site", payload.get("samesite")))
    )

    return NormalizedCookie(
        name=name,
        value=value,
        domain=domain,
        path=path,
        secure=secure,
        http_only=http_only,
        same_site=same_site,
    )


def _is_broad_domain_scope(domain: str) -> bool:
    """Heuristic for subdomain exposure risk in cookie domain scoping."""
    if not domain:
        return False

    d = domain.strip().lower()

    # Domain cookies prefixed with '.' are intentionally shared with subdomains.
    if d.startswith("."):
        return True

    # Wildcard/invalid broad forms should be treated as high-risk configuration.
    if "*" in d:
        return True

    labels = [label for label in d.split(".") if label]
    # Very short domains like "com" / "co.uk" indicate very broad scope.
    if len(labels) <= 1:
        return True
    if len(labels) == 2 and len(labels[-1]) <= 2:
        return True

    return False


def analyze_cookie_attributes(cookie: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize cookie attributes used for security evaluation."""
    normalized = _normalize_cookie(cookie)
    return {
        "name": normalized.name,
        "value": normalized.value,
        "domain": normalized.domain,
        "path": normalized.path,
        "secure": normalized.secure,
        "httpOnly": normalized.http_only,
        "sameSite": normalized.same_site,
    }


def detect_cookie_issues(cookie: dict[str, Any]) -> list[str]:
    """Return human-readable security issues for a single cookie."""
    attributes = analyze_cookie_attributes(cookie)
    issues: list[str] = []

    if not attributes["httpOnly"]:
        issues.append("Missing HttpOnly flag (XSS risk)")

    if not attributes["secure"]:
        issues.append("Missing Secure flag (MITM risk)")

    if attributes["sameSite"] == "None":
        issues.append("SameSite=None allows potential CSRF risk")

    if _is_broad_domain_scope(attributes["domain"]):
        issues.append("Broad domain scope allows subdomain access risk")

    if not attributes["name"]:
        issues.append("Cookie name is missing or empty")

    return issues


def _cookie_risk_points(issues: list[str]) -> int:
    """Convert issue strings into weighted risk points for one cookie."""
    points = 0

    if any("HttpOnly" in issue for issue in issues):
        points += RISK_WEIGHTS["missing_http_only"]
    if any("Secure" in issue for issue in issues):
        points += RISK_WEIGHTS["missing_secure"]
    if any("SameSite=None" in issue for issue in issues):
        points += RISK_WEIGHTS["same_site_none"]
    if any("Broad domain scope" in issue for issue in issues):
        points += RISK_WEIGHTS["broad_domain_scope"]

    return points


def compute_cookie_score(cookies: list[dict[str, Any]] | None) -> float:
    """Compute normalized cookie risk score in [0, 1]."""
    cookie_list = cookies if isinstance(cookies, list) else []
    if not cookie_list:
        return 0.0

    total_points = 0
    max_points = len(cookie_list) * MAX_COOKIE_RISK_POINTS

    for cookie in cookie_list:
        issues = detect_cookie_issues(cookie if isinstance(cookie, dict) else {})
        total_points += _cookie_risk_points(issues)

    if max_points <= 0:
        return 0.0

    score = total_points / max_points
    return max(0.0, min(round(score, 6), 1.0))


def _looks_like_session_cookie(name: str) -> bool:
    """Identify likely session/auth cookie names."""
    if not name:
        return False
    lowered = name.lower()
    return any(hint in lowered for hint in SESSION_COOKIE_HINTS)


def _index_session_cookies(cookies: list[dict[str, Any]] | None) -> dict[tuple[str, str, str], str]:
    """Index candidate session cookies by (name, domain, path) => value."""
    indexed: dict[tuple[str, str, str], str] = {}
    if not isinstance(cookies, list):
        return indexed

    for raw_cookie in cookies:
        cookie = _normalize_cookie(raw_cookie if isinstance(raw_cookie, dict) else {})
        if not _looks_like_session_cookie(cookie.name):
            continue
        if not cookie.name:
            continue

        key = (cookie.name.lower(), cookie.domain, cookie.path)
        indexed[key] = cookie.value

    return indexed


def detect_session_fixation(
    cookies_before_login: list[dict[str, Any]] | None,
    cookies_after_login: list[dict[str, Any]] | None,
) -> bool:
    """Detect session fixation by checking unchanged session cookie values post-login."""
    before_index = _index_session_cookies(cookies_before_login)
    after_index = _index_session_cookies(cookies_after_login)

    if not before_index or not after_index:
        return False

    for key, old_value in before_index.items():
        new_value = after_index.get(key)
        if new_value is None:
            continue

        if old_value and old_value == new_value:
            return True

    return False


def analyze_cookies(
    cookies: list[dict[str, Any]] | None,
    cookies_before_login: list[dict[str, Any]] | None = None,
    cookies_after_login: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze cookie set and return consolidated cookie security report."""
    cookie_list = cookies if isinstance(cookies, list) else []
    insecure_cookies: list[dict[str, Any]] = []

    for raw_cookie in cookie_list:
        cookie = raw_cookie if isinstance(raw_cookie, dict) else {}
        attributes = analyze_cookie_attributes(cookie)
        issues = detect_cookie_issues(cookie)
        if issues:
            insecure_cookies.append(
                {
                    "name": attributes["name"] or "<unnamed>",
                    "issues": issues,
                }
            )

    return {
        "total_cookies": len(cookie_list),
        "insecure_cookies": insecure_cookies,
        "cookie_risk_score": compute_cookie_score(cookie_list),
        "session_fixation_vulnerable": detect_session_fixation(
            cookies_before_login=cookies_before_login,
            cookies_after_login=cookies_after_login,
        ),
    }


if __name__ == "__main__":
    import unittest

    class TestCookieAnalyzer(unittest.TestCase):
        def test_secure_cookies_safe(self) -> None:
            cookies = [
                {
                    "name": "session_id",
                    "value": "rotated-456",
                    "domain": "app.example.com",
                    "path": "/",
                    "secure": True,
                    "httpOnly": True,
                    "sameSite": "Strict",
                }
            ]

            result = analyze_cookies(cookies)
            self.assertEqual(result["total_cookies"], 1)
            self.assertEqual(result["insecure_cookies"], [])
            self.assertEqual(result["cookie_risk_score"], 0.0)
            self.assertFalse(result["session_fixation_vulnerable"])

        def test_missing_flags_insecure(self) -> None:
            cookies = [
                {
                    "name": "session",
                    "value": "abc123",
                    "domain": ".example.com",
                    "path": "/",
                    "secure": False,
                    "httpOnly": False,
                    "sameSite": "None",
                }
            ]

            result = analyze_cookies(cookies)
            self.assertEqual(result["total_cookies"], 1)
            self.assertEqual(len(result["insecure_cookies"]), 1)
            self.assertGreater(result["cookie_risk_score"], 0.9)

        def test_session_fixation_detected(self) -> None:
            before_login = [
                {
                    "name": "session",
                    "value": "fixed-token-123",
                    "domain": "example.com",
                    "path": "/",
                    "secure": True,
                    "httpOnly": True,
                    "sameSite": "Lax",
                }
            ]
            after_login = [
                {
                    "name": "session",
                    "value": "fixed-token-123",
                    "domain": "example.com",
                    "path": "/",
                    "secure": True,
                    "httpOnly": True,
                    "sameSite": "Lax",
                }
            ]

            result = analyze_cookies(
                cookies=after_login,
                cookies_before_login=before_login,
                cookies_after_login=after_login,
            )
            self.assertTrue(result["session_fixation_vulnerable"])

    unittest.main()
