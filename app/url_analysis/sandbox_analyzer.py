"""Headless browser sandbox analyzer for untrusted URL behavior inspection."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from typing import Any
from urllib.parse import urlparse

try:
    import tldextract
except ImportError:  # pragma: no cover - optional dependency fallback
    tldextract = None

_TLD_EXTRACTOR = (
    tldextract.TLDExtract(suffix_list_urls=None) if tldextract is not None else None
)

from app.url_analysis.phishing_behavior_analyzer import analyze_page_phishing_behavior
from app.url_analysis.fingerprint_beacon_analyzer import (
    FINGERPRINT_BEACON_INIT_SCRIPT,
    analyze_page_fingerprint_and_beaconing,
)

DEFAULT_TIMEOUT_MS = 18_000
MAX_NETWORK_LOGS = 500

logger = logging.getLogger("zora.url_analysis.sandbox_analyzer")

SUSPICIOUS_ENDPOINT_KEYWORDS: tuple[str, ...] = (
    "login",
    "signin",
    "verify",
    "password",
    "token",
    "auth",
    "account",
    "session",
    "wallet",
    "bank",
    "invoice",
    "payment",
)


def _safe_output(initial_url: str = "") -> dict[str, Any]:
    """Return default structured output for failures and edge-cases."""
    return {
        "initial_url": initial_url,
        "final_url": "",
        "redirect_chain": [],
        "dom_length": 0,
        "raw_html": "",
        "num_scripts": 0,
        "external_js": [],
        "network_requests": [],
        "external_domains": [],
        "suspicious_endpoints": [],
        "set_cookie_headers": [],
        "cookies": [],
        "phishing_behavior_analysis": {},
        "fingerprint_beacon_analysis": {},
        "error": "",
    }


def _format_error(exc: Exception) -> str:
    """Return concise exception text for structured error output."""
    detail = str(exc).strip()
    if not detail:
        return f"sandbox_failure:{exc.__class__.__name__}"
    if len(detail) > 220:
        detail = f"{detail[:220]}..."
    return f"sandbox_failure:{exc.__class__.__name__}:{detail}"


def _normalize_input_url(url: str) -> str:
    """Normalize user URL input and default to HTTPS if scheme is missing."""
    value = (url or "").strip()
    if not value:
        return ""

    parsed = urlparse(value)
    if not parsed.scheme:
        value = f"https://{value}"

    return value


def _registered_domain(hostname: str) -> str:
    """Return registered domain from hostname for external-domain checks."""
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


def _is_suspicious_endpoint(url: str, method: str) -> bool:
    """Simple heuristic for suspicious endpoint detection in network logs."""
    lowered_url = (url or "").lower()
    lowered_method = (method or "").upper()
    if any(keyword in lowered_url for keyword in SUSPICIOUS_ENDPOINT_KEYWORDS):
        return True
    if lowered_method == "POST" and ("api" in lowered_url or "submit" in lowered_url):
        return True
    return False


def _debug(run_id: str, message: str, level: str = "info") -> None:
    """Emit both print and logger output for fast local debugging."""
    line = f"[sandbox-debug][{run_id}] {message}"
    print(line)

    if level == "warning":
        logger.warning(line)
    elif level == "error":
        logger.error(line)
    else:
        logger.info(line)


def _loop_supports_subprocess(loop: asyncio.AbstractEventLoop) -> bool:
    """Return whether current loop can spawn subprocesses (required by Playwright)."""
    if sys.platform != "win32":
        return True

    return "ProactorEventLoop" in loop.__class__.__name__


async def launch_browser() -> tuple[Any, Any]:
    """Launch async Playwright Chromium browser in hardened headless mode."""
    from playwright.async_api import async_playwright

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-notifications",
            "--disable-popup-blocking",
            "--disable-background-networking",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-dev-shm-usage",
            "--mute-audio",
            "--no-first-run",
        ],
    )
    return playwright, browser


def capture_network(page: Any, initial_registered_domain: str) -> dict[str, Any]:
    """Attach network listeners and return mutable state collector."""
    state: dict[str, Any] = {
        "network_requests": [],
        "external_domains": set(),
        "suspicious_endpoints": [],
        "navigation_urls": [],
        "responses": [],
    }

    def _on_request(request: Any) -> None:
        if len(state["network_requests"]) >= MAX_NETWORK_LOGS:
            return

        request_url = request.url
        method = request.method
        state["network_requests"].append({"url": request_url, "method": method})

        parsed = urlparse(request_url)
        host = (parsed.hostname or "").lower()
        host_registered = _registered_domain(host)
        if host_registered and host_registered != initial_registered_domain:
            state["external_domains"].add(host_registered)

        if _is_suspicious_endpoint(request_url, method):
            state["suspicious_endpoints"].append({"url": request_url, "method": method})

    def _on_response(response: Any) -> None:
        state["responses"].append(response)
        request = response.request
        if request and request.is_navigation_request():
            frame = request.frame
            if frame and frame == page.main_frame:
                state["navigation_urls"].append(response.url)

    page.on("request", _on_request)
    page.on("response", _on_response)
    return state


async def extract_dom(page: Any) -> tuple[int, str]:
    """Return DOM length and raw HTML snapshot."""
    try:
        html = await page.content()
    except Exception:
        return 0, ""
    return len(html), html


async def extract_scripts(page: Any) -> tuple[int, list[str]]:
    """Extract total script count and external JS URLs from DOM."""
    try:
        script_data = await page.evaluate(
            """
            () => {
                const scripts = Array.from(document.querySelectorAll('script'));
                const external = scripts
                    .map(s => s.src)
                    .filter(src => typeof src === 'string' && src.length > 0);
                return {
                    total: scripts.length,
                    external
                };
            }
            """
        )
    except Exception:
        return 0, []

    total = int(script_data.get("total", 0)) if isinstance(script_data, dict) else 0
    external = script_data.get("external", []) if isinstance(script_data, dict) else []
    if not isinstance(external, list):
        external = []
    return total, [str(item) for item in external]


async def extract_cookies(context: Any, responses: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract Set-Cookie headers and browser cookies after navigation."""
    set_cookie_headers: list[str] = []

    for response in responses:
        try:
            headers = await response.all_headers()
        except Exception:
            continue

        for key, value in headers.items():
            if key.lower() == "set-cookie" and value:
                set_cookie_headers.append(str(value))

    try:
        browser_cookies = await context.cookies()
    except Exception:
        browser_cookies = []

    cookies: list[dict[str, Any]] = []
    for cookie in browser_cookies:
        cookies.append(
            {
                "name": cookie.get("name", ""),
                "domain": cookie.get("domain", ""),
                "path": cookie.get("path", ""),
                "secure": bool(cookie.get("secure", False)),
                "httponly": bool(cookie.get("httpOnly", False)),
                "samesite": cookie.get("sameSite", ""),
            }
        )

    return set_cookie_headers, cookies


async def _analyze_url_impl(url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict[str, Any]:
    """Internal analyzer implementation that executes Playwright sandbox flow."""
    run_id = str(uuid.uuid4())[:8]
    current_stage = "normalize_input"

    normalized_url = _normalize_input_url(url)
    output = _safe_output(normalized_url)
    _debug(run_id, f"start analyze_url raw={url!r} normalized={normalized_url!r}")

    if not normalized_url:
        output["error"] = "empty_url"
        _debug(run_id, "aborting: empty normalized URL", level="warning")
        return output

    try:
        current_stage = "parse_initial_domain"
        parsed = urlparse(normalized_url)
        initial_host = (parsed.hostname or "").lower()
        initial_registered_domain = _registered_domain(initial_host)
        _debug(
            run_id,
            f"initial_host={initial_host!r} initial_registered_domain={initial_registered_domain!r}",
        )
    except Exception:
        output["error"] = "invalid_url"
        _debug(run_id, "aborting: invalid URL after parsing", level="warning")
        return output

    playwright = None
    browser = None
    context = None
    page = None

    try:
        current_stage = "launch_browser"
        playwright, browser = await launch_browser()
        _debug(run_id, "browser launched")

        current_stage = "create_context"
        context = await browser.new_context(
            accept_downloads=False,
            ignore_https_errors=True,
            java_script_enabled=True,
        )
        _debug(run_id, "browser context created")

        # Explicitly deny sensitive APIs in page runtime.
        current_stage = "add_security_init_script"
        await context.add_init_script(
            """
            (() => {
                try {
                    if (typeof Notification !== 'undefined') {
                        Notification.requestPermission = () => Promise.resolve('denied');
                    }
                } catch (e) {}

                try {
                    if (navigator && navigator.mediaDevices) {
                        navigator.mediaDevices.getUserMedia = () => Promise.reject(new Error('Blocked by sandbox'));
                    }
                } catch (e) {}
            })();
            """
        )

        current_stage = "add_fingerprint_init_script"
        await context.add_init_script(FINGERPRINT_BEACON_INIT_SCRIPT)
        _debug(run_id, "init scripts injected")

        current_stage = "create_page"
        page = await context.new_page()
        await page.set_viewport_size({"width": 1366, "height": 768})
        page.set_default_timeout(timeout_ms)
        _debug(run_id, f"page ready timeout_ms={timeout_ms}")

        current_stage = "attach_network_listeners"
        network_state = capture_network(page, initial_registered_domain)
        _debug(run_id, "network listeners attached")

        try:
            current_stage = "navigate"
            _debug(run_id, f"navigating to {normalized_url}")
            await page.goto(normalized_url, wait_until="domcontentloaded", timeout=timeout_ms)

            current_stage = "wait_networkidle"
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 7_000))
            _debug(run_id, "navigation completed with networkidle")
        except Exception:
            # Continue with partial data for resilient output.
            _debug(
                run_id,
                "navigation/load-state failed; continuing with partial telemetry",
                level="warning",
            )
            pass

        current_stage = "extract_page_artifacts"
        final_url = page.url or normalized_url
        dom_length, raw_html = await extract_dom(page)
        num_scripts, external_js = await extract_scripts(page)
        set_cookie_headers, cookies = await extract_cookies(context, network_state["responses"])
        _debug(
            run_id,
            (
                f"final_url={final_url!r} dom_length={dom_length} scripts={num_scripts} "
                f"network_requests={len(network_state['network_requests'])}"
            ),
        )

        current_stage = "build_redirect_chain"
        redirect_chain: list[str] = []
        for nav_url in [normalized_url] + network_state["navigation_urls"] + [final_url]:
            if nav_url and nav_url not in redirect_chain:
                redirect_chain.append(nav_url)
        _debug(run_id, f"redirect_chain_length={len(redirect_chain)}")

        current_stage = "analyze_phishing_behavior"
        phishing_behavior_analysis = await analyze_page_phishing_behavior(
            page=page,
            initial_url=normalized_url,
            final_url=final_url,
            redirect_chain=redirect_chain,
            responses=network_state["responses"],
            network_requests=network_state["network_requests"],
        )
        _debug(
            run_id,
            f"phishing_behavior_keys={list(phishing_behavior_analysis.keys()) if isinstance(phishing_behavior_analysis, dict) else []}",
        )

        current_stage = "analyze_fingerprint_beacon"
        fingerprint_beacon_analysis = await analyze_page_fingerprint_and_beaconing(
            page=page,
            main_page_url=final_url,
            network_requests=network_state["network_requests"],
        )
        _debug(
            run_id,
            f"fingerprint_beacon_keys={list(fingerprint_beacon_analysis.keys()) if isinstance(fingerprint_beacon_analysis, dict) else []}",
        )

        current_stage = "compose_output"
        output.update(
            {
                "initial_url": normalized_url,
                "final_url": final_url,
                "redirect_chain": redirect_chain,
                "dom_length": dom_length,
                "raw_html": raw_html,
                "num_scripts": num_scripts,
                "external_js": external_js,
                "network_requests": network_state["network_requests"],
                "external_domains": sorted(network_state["external_domains"]),
                "suspicious_endpoints": network_state["suspicious_endpoints"],
                "set_cookie_headers": set_cookie_headers,
                "cookies": cookies,
                "phishing_behavior_analysis": phishing_behavior_analysis,
                "fingerprint_beacon_analysis": fingerprint_beacon_analysis,
                "error": "",
            }
        )
        _debug(run_id, "analysis completed successfully")
        return output

    except ImportError:
        output["error"] = "playwright_not_installed"
        _debug(run_id, "playwright is not installed", level="error")
        return output
    except Exception as exc:
        output["error"] = f"{_format_error(exc)}|stage={current_stage}|run_id={run_id}"
        logger.exception(
            "Sandbox analyze_url failed run_id=%s stage=%s normalized_url=%s",
            run_id,
            current_stage,
            normalized_url,
        )
        _debug(
            run_id,
            f"failure stage={current_stage} exc={exc.__class__.__name__}: {exc}",
            level="error",
        )
        return output
    finally:
        current_stage = "cleanup"
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
        _debug(run_id, "cleanup complete")


async def analyze_url(url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict[str, Any]:
    """Analyze a URL in a hardened headless browser sandbox."""
    run_id = str(uuid.uuid4())[:8]

    # On Windows, Playwright requires a loop with subprocess support (Proactor).
    # Some ASGI loop configurations end up using SelectorEventLoop, which causes
    # asyncio.create_subprocess_exec to raise NotImplementedError.
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and not _loop_supports_subprocess(running_loop):
        _debug(
            run_id,
            (
                "detected loop without subprocess support; "
                "delegating sandbox execution to proactor thread"
            ),
            level="warning",
        )
        return await asyncio.to_thread(analyze_url_sync, url, timeout_ms)

    return await _analyze_url_impl(url, timeout_ms)


def analyze_url_sync(url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict[str, Any]:
    """Synchronous wrapper around async sandbox analyzer for non-async callers."""
    normalized_url = _normalize_input_url(url)
    run_id = str(uuid.uuid4())[:8]
    _debug(run_id, f"start analyze_url_sync normalized={normalized_url!r} timeout_ms={timeout_ms}")

    # Keep Windows policy explicit to avoid external overrides in long-lived apps.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        _debug(run_id, "set WindowsProactorEventLoopPolicy")

    try:
        with asyncio.Runner() as runner:
            return runner.run(_analyze_url_impl(normalized_url, timeout_ms=timeout_ms))
    except RuntimeError:
        # If already in an event loop, do not crash the caller.
        result = _safe_output(normalized_url)
        result["error"] = "event_loop_running_use_async_api"
        _debug(run_id, "runtime error: event loop already running", level="error")
        return result


async def run_sandbox_test_cases() -> list[dict[str, Any]]:
    """Run baseline sandbox test cases requested for validation."""
    test_urls = [
        "https://google.com",  # normal site
        "http://github.com",  # redirect-heavy
        "https://www.youtube.com",  # JS-heavy
    ]

    results: list[dict[str, Any]] = []
    for test_url in test_urls:
        results.append(await analyze_url(test_url))
    return results


if __name__ == "__main__":
    summary = asyncio.run(run_sandbox_test_cases())
    print({"test_runs": len(summary), "errors": [item.get("error", "") for item in summary]})
