"""Playwright-based phishing behavior analysis for redirect, iframe, and CSP risks."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

try:
    import tldextract
except ImportError:  # pragma: no cover - optional dependency fallback
    tldextract = None

_TLD_EXTRACTOR = (
    tldextract.TLDExtract(suffix_list_urls=None) if tldextract is not None else None
)

LOGIN_KEYWORDS: tuple[str, ...] = ("login", "auth", "verify")
MAX_NETWORK_LOGS = 500


def _normalize_input_url(url: str) -> str:
    """Normalize URL input and default to HTTPS if scheme is missing."""
    value = (url or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if not parsed.scheme:
        return f"https://{value}"
    return value


def _safe_output() -> dict[str, Any]:
    """Default output shape for robust error handling."""
    return {
        "redirect_analysis": {
            "count": 0,
            "domains": [],
            "suspicious": False,
        },
        "iframe_analysis": {
            "count": 0,
            "external_iframes": False,
            "suspicious_iframe": False,
            "iframe_sources": [],
        },
        "csp_analysis": {
            "present": False,
            "issues": ["Missing Content-Security-Policy header"],
            "risk_level": "high",
        },
        "network_requests": [],
        "error": "",
    }


def _registered_domain(hostname: str) -> str:
    """Return a stable registered-domain-like value for cross-domain checks."""
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return ""

    if _TLD_EXTRACTOR is not None:
        extracted = _TLD_EXTRACTOR(host)
        registered = getattr(extracted, "top_domain_under_public_suffix", "") or getattr(
            extracted, "registered_domain", ""
        )
        return (registered or host).lower()

    labels = [part for part in host.split(".") if part]
    if len(labels) >= 2:
        return f"{labels[-2]}.{labels[-1]}"
    return host


def _url_domain(value: str) -> str:
    """Extract registered domain from URL; return empty on failure."""
    raw = (value or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return ""
    return _registered_domain(host)


def analyze_redirect_chain(initial_url: str, final_url: str, redirect_chain: list[str] | None) -> dict[str, Any]:
    """Analyze redirect chain depth and domain spread for phishing indicators."""
    chain: list[str] = []
    for item in [initial_url] + (redirect_chain or []) + [final_url]:
        value = (item or "").strip()
        if value and value not in chain:
            chain.append(value)

    domains = [domain for domain in (_url_domain(url) for url in chain) if domain]
    unique_domains: list[str] = []
    for domain in domains:
        if domain not in unique_domains:
            unique_domains.append(domain)

    redirect_count = max(len(chain) - 1, 0)
    suspicious = redirect_count > 3 or len(unique_domains) > 1

    return {
        "count": redirect_count,
        "domains": unique_domains,
        "suspicious": suspicious,
    }


async def analyze_iframes(page: Any, main_page_url: str) -> dict[str, Any]:
    """Analyze iframe usage and detect cross-domain or login-like iframe patterns."""
    main_domain = _url_domain(main_page_url)

    try:
        iframe_sources_raw = await page.eval_on_selector_all(
            "iframe",
            "elements => elements.map(el => el.getAttribute('src') || '').filter(Boolean)",
        )
    except Exception:
        iframe_sources_raw = []

    iframe_sources = [str(item).strip() for item in iframe_sources_raw if str(item).strip()]
    count = len(iframe_sources)

    external_iframes = False
    suspicious_iframe = False

    for src in iframe_sources:
        src_lower = src.lower()
        src_domain = _url_domain(src)
        if src_domain and main_domain and src_domain != main_domain:
            external_iframes = True

        if any(keyword in src_lower for keyword in LOGIN_KEYWORDS):
            suspicious_iframe = True

    return {
        "count": count,
        "external_iframes": external_iframes,
        "suspicious_iframe": suspicious_iframe,
        "iframe_sources": iframe_sources,
    }


def analyze_csp_headers(csp_header: str) -> dict[str, Any]:
    """Analyze CSP configuration and assign a risk level."""
    value = (csp_header or "").strip()
    issues: list[str] = []

    if not value:
        issues.append("Missing Content-Security-Policy header")
    else:
        lowered = value.lower()
        if "unsafe-inline" in lowered:
            issues.append("CSP allows unsafe-inline")
        if "frame-ancestors *" in lowered:
            issues.append("CSP uses frame-ancestors *")

    if not value:
        risk_level = "high"
    elif not issues:
        risk_level = "low"
    elif len(issues) == 1:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "present": bool(value),
        "issues": issues,
        "risk_level": risk_level,
    }


def _extract_navigation_csp_header(responses: list[Any], final_url: str) -> str:
    """Find the CSP header from final navigation response with safe fallbacks."""
    final = (final_url or "").strip()
    for response in reversed(responses):
        try:
            request = response.request
            if request and request.is_navigation_request():
                response_url = str(response.url or "").strip()
                if final and response_url != final:
                    continue
                headers = response.headers
                if isinstance(headers, dict):
                    for key, value in headers.items():
                        if str(key).lower() == "content-security-policy":
                            return str(value or "")
        except Exception:
            continue

    # Fallback: first response carrying CSP.
    for response in responses:
        try:
            headers = response.headers
            if not isinstance(headers, dict):
                continue
            for key, value in headers.items():
                if str(key).lower() == "content-security-policy":
                    return str(value or "")
        except Exception:
            continue

    return ""


async def analyze_page_phishing_behavior(
    page: Any,
    initial_url: str,
    final_url: str,
    redirect_chain: list[str] | None,
    responses: list[Any] | None,
    network_requests: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Run phishing behavior analysis on an already-loaded Playwright page."""
    output = _safe_output()
    response_items = responses if isinstance(responses, list) else []
    network_items = network_requests if isinstance(network_requests, list) else []

    try:
        redirect_analysis = analyze_redirect_chain(initial_url, final_url, redirect_chain)
        iframe_analysis = await analyze_iframes(page=page, main_page_url=final_url or initial_url)
        csp_header = _extract_navigation_csp_header(response_items, final_url)
        csp_analysis = analyze_csp_headers(csp_header)

        output.update(
            {
                "redirect_analysis": redirect_analysis,
                "iframe_analysis": iframe_analysis,
                "csp_analysis": csp_analysis,
                "network_requests": network_items,
                "error": "",
            }
        )
        return output
    except Exception as exc:
        output["error"] = f"phishing_behavior_analysis_failed:{exc.__class__.__name__}"
        output["network_requests"] = network_items
        return output


async def analyze_url_phishing_behavior(url: str, timeout_ms: int = 15_000) -> dict[str, Any]:
    """Analyze phishing behavior for a URL using async Playwright in one call."""
    normalized_url = _normalize_input_url(url)
    output = _safe_output()

    if not normalized_url:
        output["error"] = "empty_url"
        return output

    playwright = None
    browser = None
    context = None
    page = None

    navigation_urls: list[str] = []
    responses: list[Any] = []
    network_requests: list[dict[str, Any]] = []

    try:
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-notifications",
                "--disable-popup-blocking",
                "--disable-background-networking",
                "--disable-dev-shm-usage",
                "--no-first-run",
            ],
        )
        context = await browser.new_context(ignore_https_errors=True, java_script_enabled=True)
        page = await context.new_page()
        page.set_default_timeout(timeout_ms)

        def _on_request(request: Any) -> None:
            if len(network_requests) >= MAX_NETWORK_LOGS:
                return
            network_requests.append({"url": request.url, "method": request.method})

        def _on_response(response: Any) -> None:
            responses.append(response)
            try:
                request = response.request
                if request and request.is_navigation_request() and request.frame == page.main_frame:
                    navigation_urls.append(response.url)
            except Exception:
                return

        page.on("request", _on_request)
        page.on("response", _on_response)

        try:
            await page.goto(normalized_url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 7_000))
        except Exception:
            # Keep partial telemetry for resilience.
            pass

        final_url = page.url or normalized_url
        redirect_chain: list[str] = []
        for item in [normalized_url] + navigation_urls + [final_url]:
            value = (item or "").strip()
            if value and value not in redirect_chain:
                redirect_chain.append(value)

        result = await analyze_page_phishing_behavior(
            page=page,
            initial_url=normalized_url,
            final_url=final_url,
            redirect_chain=redirect_chain,
            responses=responses,
            network_requests=network_requests,
        )
        return result

    except ImportError:
        output["error"] = "playwright_not_installed"
        output["network_requests"] = network_requests
        return output
    except Exception as exc:
        output["error"] = f"playwright_analysis_failed:{exc.__class__.__name__}"
        output["network_requests"] = network_requests
        return output
    finally:
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass
