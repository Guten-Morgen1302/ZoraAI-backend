"""Headless browser sandbox analyzer for untrusted URL behavior inspection."""

from __future__ import annotations

import asyncio
import logging
import os
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
from app.url_analysis.docker_sandbox_executor import analyze_url_via_docker

DEFAULT_TIMEOUT_MS = 18_000
MAX_NETWORK_LOGS = 500
PLAYWRIGHT_LAUNCH_RETRIES = 2
PLAYWRIGHT_LAUNCH_RETRY_DELAY_SEC = 0.35

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

VALID_SANDBOX_MODES: set[str] = {"local", "docker", "auto"}


def _safe_output(initial_url: str = "") -> dict[str, Any]:
    """Return default structured output for failures and edge-cases."""
    return {
        "status": "error",
        "error_stage": "",
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


def _sandbox_mode() -> str:
    """Return sandbox execution mode from environment with safe fallback."""
    configured = (os.getenv("URL_SANDBOX_MODE", "local") or "local").strip().lower()
    return configured if configured in VALID_SANDBOX_MODES else "local"


def _normalize_docker_result(payload: dict[str, Any], normalized_url: str) -> dict[str, Any]:
    """Map Docker sandbox payload into the URL-analysis sandbox output schema."""
    output = _safe_output(normalized_url)

    if not isinstance(payload, dict):
        output["error"] = "docker_invalid_payload"
        output["error_stage"] = "docker_payload_parse"
        return output

    network_requests = payload.get("network_requests")
    if not isinstance(network_requests, list):
        requests_alias = payload.get("requests")
        network_requests = requests_alias if isinstance(requests_alias, list) else []

    status = str(payload.get("status") or "error").strip().lower()
    if status not in {"success", "timeout", "error"}:
        status = "error"

    output.update(
        {
            "status": status,
            "error_stage": str(payload.get("error_stage") or ""),
            "initial_url": str(payload.get("initial_url") or normalized_url),
            "final_url": str(payload.get("final_url") or payload.get("initial_url") or normalized_url),
            "redirect_chain": payload.get("redirect_chain") if isinstance(payload.get("redirect_chain"), list) else [],
            "dom_length": int(payload.get("dom_length") or 0),
            "raw_html": str(payload.get("raw_html") or ""),
            "num_scripts": int(payload.get("num_scripts") or 0),
            "external_js": payload.get("external_js") if isinstance(payload.get("external_js"), list) else [],
            "network_requests": network_requests,
            "external_domains": payload.get("external_domains") if isinstance(payload.get("external_domains"), list) else [],
            "suspicious_endpoints": payload.get("suspicious_endpoints") if isinstance(payload.get("suspicious_endpoints"), list) else [],
            "set_cookie_headers": payload.get("set_cookie_headers") if isinstance(payload.get("set_cookie_headers"), list) else [],
            "cookies": payload.get("cookies") if isinstance(payload.get("cookies"), list) else [],
            "error": str(payload.get("error") or ""),
            "phishing_behavior_analysis": payload.get("phishing_behavior_analysis") if isinstance(payload.get("phishing_behavior_analysis"), dict) else {},
            "fingerprint_beacon_analysis": payload.get("fingerprint_beacon_analysis") if isinstance(payload.get("fingerprint_beacon_analysis"), dict) else {},
        }
    )

    if output["status"] == "success":
        output["error"] = ""
        output["error_stage"] = ""
    elif not output["error"]:
        output["error"] = f"docker_sandbox_{output['status']}"
        output["error_stage"] = output["error_stage"] or "docker_execution"

    return output


def _analyze_url_via_docker(normalized_url: str, timeout_ms: int) -> dict[str, Any]:
    """Execute disposable Docker sandbox for a URL and normalize output."""
    docker_result = analyze_url_via_docker(normalized_url, timeout_ms=timeout_ms)
    return _normalize_docker_result(docker_result, normalized_url=normalized_url)


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


def _is_playwright_driver_disconnect(exc: Exception) -> bool:
    """Detect transient Playwright driver disconnection failure signature."""
    message = str(exc or "").lower()
    return "connection closed while reading from the driver" in message


async def launch_browser(run_id: str | None = None) -> tuple[Any, Any]:
    """Launch async Playwright Chromium browser in hardened headless mode."""
    from playwright.async_api import async_playwright

    last_exc: Exception | None = None
    for attempt in range(1, PLAYWRIGHT_LAUNCH_RETRIES + 1):
        playwright = None
        try:
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
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if playwright is not None:
                try:
                    await playwright.stop()
                except Exception:
                    pass

            should_retry = _is_playwright_driver_disconnect(exc) and attempt < PLAYWRIGHT_LAUNCH_RETRIES
            if should_retry:
                _debug(
                    run_id or "no-run-id",
                    f"playwright launch retry={attempt + 1} after transient driver disconnect",
                    level="warning",
                )
                await asyncio.sleep(PLAYWRIGHT_LAUNCH_RETRY_DELAY_SEC)
                continue

            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("playwright_launch_failed")


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
        resource_type = request.resource_type
        state["network_requests"].append(
            {
                "url": request_url,
                "method": method,
                "resource_type": resource_type,
            }
        )

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

    def _on_frame_navigated(frame: Any) -> None:
        if frame == page.main_frame:
            current_url = str(frame.url or "").strip()
            if current_url:
                state["navigation_urls"].append(current_url)

    page.on("request", _on_request)
    page.on("response", _on_response)
    page.on("framenavigated", _on_frame_navigated)
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
        output["status"] = "error"
        output["error"] = "empty_url"
        output["error_stage"] = "normalize_input"
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
        output["status"] = "error"
        output["error"] = "invalid_url"
        output["error_stage"] = "parse_initial_domain"
        _debug(run_id, "aborting: invalid URL after parsing", level="warning")
        return output

    playwright = None
    browser = None
    context = None
    page = None

    try:
        current_stage = "launch_browser"
        playwright, browser = await launch_browser(run_id=run_id)
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
                "status": "success",
                "error_stage": "",
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
        output["status"] = "error"
        output["error"] = "playwright_not_installed"
        output["error_stage"] = current_stage
        _debug(run_id, "playwright is not installed", level="error")
        return output
    except Exception as exc:
        output["status"] = "error"
        output["error_stage"] = current_stage
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
    mode = _sandbox_mode()
    normalized_url = _normalize_input_url(url)

    if mode in {"docker", "auto"} and normalized_url:
        docker_result = await asyncio.to_thread(_analyze_url_via_docker, normalized_url, timeout_ms)
        docker_status = str(docker_result.get("status") or "error").strip().lower()

        if docker_status == "success" or mode == "docker":
            _debug(run_id, f"sandbox mode={mode} returning docker status={docker_status}")
            return docker_result

        _debug(
            run_id,
            "docker sandbox failed in auto mode; falling back to local Playwright sandbox",
            level="warning",
        )

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
            loop = runner.get_loop()
            previous_handler = loop.get_exception_handler()

            def _loop_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
                exc = context.get("exception")
                message = str(exc or context.get("message") or "")

                # Playwright sometimes leaves an init task unresolved when driver bootstrap dies.
                # We already surface the primary error path from _analyze_url_impl.
                if "Connection.init: Connection closed while reading from the driver" in message:
                    _debug(run_id, "suppressed orphaned Playwright init task exception", level="warning")
                    return

                if previous_handler is not None:
                    previous_handler(loop, context)
                else:
                    loop.default_exception_handler(context)

            loop.set_exception_handler(_loop_exception_handler)
            return runner.run(_analyze_url_impl(normalized_url, timeout_ms=timeout_ms))
    except RuntimeError:
        # If already in an event loop, do not crash the caller.
        result = _safe_output(normalized_url)
        result["status"] = "error"
        result["error"] = "event_loop_running_use_async_api"
        result["error_stage"] = "analyze_url_sync"
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
