from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

try:
    import tldextract
except ImportError:  # pragma: no cover
    tldextract = None

_TLD_EXTRACTOR = (
    tldextract.TLDExtract(suffix_list_urls=None) if tldextract is not None else None
)

DEFAULT_TIMEOUT_MS = 12_000
MAX_NETWORK_EVENTS = 700
MAX_CONSOLE_EVENTS = 100
EXTERNAL_DOMAIN_THRESHOLD = 15
SCRIPT_THRESHOLD = 80
DEBUG_ENABLED = (os.getenv("URL_SANDBOX_DEBUG", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}

logger = logging.getLogger("zora.url_sandbox.runner")
logging.basicConfig(level=logging.DEBUG if DEBUG_ENABLED else logging.INFO)

FINGERPRINT_BEACON_INIT_SCRIPT = r"""
(() => {
    try {
        if (!window.__securitySignals) {
            window.__securitySignals = {
                fingerprintingMethods: [],
                beaconCalls: []
            };
        }

        const addMethod = (name) => {
            try {
                if (!window.__securitySignals.fingerprintingMethods.includes(name)) {
                    window.__securitySignals.fingerprintingMethods.push(name);
                }
            } catch (e) {}
        };

        const logBeacon = (url, data) => {
            try {
                let dataSize = 0;
                if (typeof data === 'string') {
                    dataSize = data.length;
                } else if (data && typeof data.byteLength === 'number') {
                    dataSize = data.byteLength;
                } else if (data && typeof data.size === 'number') {
                    dataSize = data.size;
                }
                window.__securitySignals.beaconCalls.push({
                    url: String(url || ''),
                    data_size: Number(dataSize) || 0,
                    timestamp: Date.now(),
                });
            } catch (e) {}
        };

        try {
            if (typeof HTMLCanvasElement !== 'undefined' && HTMLCanvasElement.prototype) {
                const originalGetContext = HTMLCanvasElement.prototype.getContext;
                if (typeof originalGetContext === 'function') {
                    HTMLCanvasElement.prototype.getContext = function(...args) {
                        try {
                            addMethod('canvas.getContext');
                            const contextName = String((args && args[0]) || '').toLowerCase();
                            if (contextName.includes('webgl')) {
                                addMethod('webgl.getContext');
                            }
                        } catch (e) {}
                        return originalGetContext.apply(this, args);
                    };
                }

                const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
                if (typeof originalToDataURL === 'function') {
                    HTMLCanvasElement.prototype.toDataURL = function(...args) {
                        try {
                            addMethod('canvas.toDataURL');
                        } catch (e) {}
                        return originalToDataURL.apply(this, args);
                    };
                }
            }
        } catch (e) {}

        try {
            const AudioCtor = window.AudioContext || window.webkitAudioContext;
            if (typeof AudioCtor === 'function') {
                const WrappedAudioContext = function(...args) {
                    addMethod('AudioContext');
                    return Reflect.construct(AudioCtor, args, WrappedAudioContext);
                };
                WrappedAudioContext.prototype = AudioCtor.prototype;
                Object.setPrototypeOf(WrappedAudioContext, AudioCtor);
                if (window.AudioContext) {
                    window.AudioContext = WrappedAudioContext;
                }
                if (window.webkitAudioContext) {
                    window.webkitAudioContext = WrappedAudioContext;
                }
            }
        } catch (e) {}

        try {
            const originalSendBeacon = navigator.sendBeacon ? navigator.sendBeacon.bind(navigator) : null;
            navigator.sendBeacon = function(url, data) {
                logBeacon(url, data);
                if (originalSendBeacon) {
                    return originalSendBeacon(url, data);
                }
                return true;
            };
        } catch (e) {}
    } catch (e) {}
})();
"""

SUSPICIOUS_ENDPOINT_KEYWORDS: tuple[str, ...] = (
    "login",
    "signin",
    "verify",
    "password",
    "token",
    "auth",
    "wallet",
    "bank",
    "invoice",
    "payment",
    "otp",
)


@dataclass
class RunState:
    initial_url: str
    initial_registered_domain: str
    timeout_ms: int
    requests: list[dict[str, Any]]
    responses: list[Any]
    redirect_chain: list[str]
    external_domains: set[str]
    blocked_internal_requests: list[str]
    console_messages: list[str]
    request_errors: list[dict[str, Any]]
    csp_headers: list[str]


def _debug(message: str) -> None:
    line = f"[url-sandbox] {message}"
    print(line)
    if DEBUG_ENABLED:
        logger.debug(line)


def _safe_output(initial_url: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error": "",
        "error_stage": "",
        "initial_url": initial_url,
        "final_url": "",
        "redirect_chain": [],
        "dom_length": 0,
        "dom_metrics": {
            "node_count": 0,
            "dom_length": 0,
        },
        "raw_html": "",
        "num_scripts": 0,
        "external_js": [],
        "requests": [],
        "network_requests": [],
        "external_domains": [],
        "third_party_domains": [],
        "suspicious_endpoints": [],
        "set_cookie_headers": [],
        "set_cookie_details": [],
        "cookies": [],
        "console_messages": [],
        "blocked_internal_requests": [],
        "request_failures": [],
        "suspicious_activity_flags": [],
        "phishing_behavior_analysis": {},
        "fingerprint_beacon_analysis": {},
    }


def _normalize_input_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""

    parsed = urlparse(value)
    if not parsed.scheme:
        value = f"https://{value}"

    return value


def _registered_domain(hostname: str) -> str:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return ""

    if _TLD_EXTRACTOR is not None:
        extracted = _TLD_EXTRACTOR(host)
        registered = getattr(extracted, "top_domain_under_public_suffix", "") or getattr(
            extracted,
            "registered_domain",
            "",
        )
        return (registered or host).lower()

    labels = [part for part in host.split(".") if part]
    if len(labels) >= 2:
        return f"{labels[-2]}.{labels[-1]}"
    return host


def _is_private_or_loopback_host(hostname: str) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return False

    if host in {"localhost", "localhost.localdomain"}:
        return True

    try:
        ip = ipaddress.ip_address(host)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        pass

    if host.endswith(".localhost") or host.endswith(".local"):
        return True

    return False


def _is_suspicious_endpoint(url: str, method: str) -> bool:
    lowered_url = (url or "").lower()
    lowered_method = (method or "").upper()
    if any(keyword in lowered_url for keyword in SUSPICIOUS_ENDPOINT_KEYWORDS):
        return True
    return lowered_method == "POST" and ("submit" in lowered_url or "api" in lowered_url)


def _suspicious_flags(payload: dict[str, Any]) -> list[str]:
    flags: list[str] = []

    if len(payload.get("external_domains", [])) > EXTERNAL_DOMAIN_THRESHOLD:
        flags.append("too_many_external_domains")

    if len(payload.get("redirect_chain", [])) > 5:
        flags.append("deep_redirect_chain")

    if int(payload.get("num_scripts", 0) or 0) > SCRIPT_THRESHOLD:
        flags.append("excessive_script_count")

    if len(payload.get("suspicious_endpoints", [])) > 0:
        flags.append("suspicious_endpoints_detected")

    if len(payload.get("blocked_internal_requests", [])) > 0:
        flags.append("attempted_internal_network_access")

    return flags


def _url_domain(value: str) -> str:
    parsed = urlparse((value or "").strip())
    return _registered_domain((parsed.hostname or "").strip().lower())


def _analyze_redirect_chain(initial_url: str, final_url: str, redirect_chain: list[str] | None) -> dict[str, Any]:
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


def _analyze_csp_headers(csp_header: str) -> dict[str, Any]:
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


async def _analyze_iframes(page: Any, main_page_url: str) -> dict[str, Any]:
    main_domain = _url_domain(main_page_url)
    try:
        iframe_sources_raw = await page.eval_on_selector_all(
            "iframe",
            "elements => elements.map(el => el.getAttribute('src') || '').filter(Boolean)",
        )
    except Exception:
        iframe_sources_raw = []

    iframe_sources = [str(item).strip() for item in iframe_sources_raw if str(item).strip()]
    external_iframes = False
    suspicious_iframe = False

    for src in iframe_sources:
        src_domain = _url_domain(src)
        if src_domain and main_domain and src_domain != main_domain:
            external_iframes = True
        src_lower = src.lower()
        if "login" in src_lower or "auth" in src_lower or "verify" in src_lower:
            suspicious_iframe = True

    return {
        "count": len(iframe_sources),
        "external_iframes": external_iframes,
        "suspicious_iframe": suspicious_iframe,
        "iframe_sources": iframe_sources,
    }


def _fingerprinting_risk(methods: list[str]) -> tuple[int, str]:
    unique_methods = sorted({str(item).strip() for item in methods if str(item).strip()})
    score = len(unique_methods)
    if score >= 2:
        return score, "high"
    if score == 1:
        return score, "medium"
    return score, "low"


def _beaconing_risk(beacon_calls: list[dict[str, Any]], suspicious_requests: int, external_requests: int) -> str:
    if suspicious_requests > 5 or len(beacon_calls) > 0:
        return "high"
    if suspicious_requests > 0 or external_requests > 0:
        return "medium"
    return "low"


def _analyze_network_for_beaconing(network_requests: list[dict[str, Any]], main_page_url: str) -> tuple[int, int]:
    main_domain = _url_domain(main_page_url)
    suspicious_count = 0
    external_count = 0
    for request in network_requests:
        if not isinstance(request, dict):
            continue
        request_url = str(request.get("url", "")).strip()
        if not request_url:
            continue
        lowered = request_url.lower()
        if any(token in lowered for token in ("track", "collect", "beacon", "fingerprint", "analytics")):
            suspicious_count += 1
        request_domain = _url_domain(request_url)
        if request_domain and main_domain and request_domain != main_domain:
            external_count += 1
    return suspicious_count, external_count


async def _extract_runtime_signals(page: Any) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        payload = await page.evaluate(
            """
            () => {
              const state = window.__securitySignals || {};
              const methods = Array.isArray(state.fingerprintingMethods) ? state.fingerprintingMethods : [];
              const beaconCalls = Array.isArray(state.beaconCalls) ? state.beaconCalls : [];
              return { methods, beaconCalls };
            }
            """
        )
    except Exception:
        return [], []

    if not isinstance(payload, dict):
        return [], []

    methods_raw = payload.get("methods", [])
    calls_raw = payload.get("beaconCalls", [])
    methods = [str(item).strip() for item in methods_raw if str(item).strip()] if isinstance(methods_raw, list) else []

    beacon_calls: list[dict[str, Any]] = []
    if isinstance(calls_raw, list):
        for item in calls_raw:
            if not isinstance(item, dict):
                continue
            beacon_calls.append(
                {
                    "url": str(item.get("url", "")),
                    "data_size": int(item.get("data_size", 0) or 0),
                    "timestamp": int(item.get("timestamp", 0) or 0),
                }
            )
    return methods, beacon_calls


def _parse_set_cookie_value(raw: str) -> dict[str, Any]:
    value = (raw or "").strip()
    if not value:
        return {
            "raw": "",
            "name": "",
            "value": "",
            "domain": "",
            "path": "",
            "secure": False,
            "httponly": False,
            "samesite": "",
            "max_age": "",
            "expires": "",
        }

    parts = [part.strip() for part in value.split(";") if part.strip()]
    name = ""
    cookie_value = ""
    domain = ""
    path = ""
    secure = False
    httponly = False
    samesite = ""
    max_age = ""
    expires = ""

    if parts and "=" in parts[0]:
        name, cookie_value = parts[0].split("=", 1)

    for part in parts[1:]:
        if "=" in part:
            key, attr_value = part.split("=", 1)
            lowered = key.strip().lower()
            attr = attr_value.strip()
            if lowered == "domain":
                domain = attr
            elif lowered == "path":
                path = attr
            elif lowered == "samesite":
                samesite = attr
            elif lowered == "max-age":
                max_age = attr
            elif lowered == "expires":
                expires = attr
        else:
            lowered = part.strip().lower()
            if lowered == "secure":
                secure = True
            elif lowered == "httponly":
                httponly = True

    return {
        "raw": value,
        "name": name,
        "value": cookie_value,
        "domain": domain,
        "path": path,
        "secure": secure,
        "httponly": httponly,
        "samesite": samesite,
        "max_age": max_age,
        "expires": expires,
    }


def _append_unique(items: list[str], value: str) -> None:
    text = (value or "").strip()
    if text and text not in items:
        items.append(text)


async def _extract_dom_metrics(page: Any) -> tuple[int, int, str]:
    try:
        html = await page.content()
    except Exception:
        return 0, 0, ""

    dom_length = len(html)
    try:
        node_count = await page.evaluate("() => document.getElementsByTagName('*').length")
        node_count = int(node_count or 0)
    except Exception:
        node_count = 0

    return dom_length, node_count, html


async def _extract_scripts(page: Any) -> tuple[int, list[str]]:
    try:
        script_data = await page.evaluate(
            """
            () => {
                const scripts = Array.from(document.querySelectorAll('script'));
                const external = scripts
                  .map((s) => s.src)
                  .filter((src) => typeof src === 'string' && src.length > 0);
                return { total: scripts.length, external };
            }
            """
        )
    except Exception:
        return 0, []

    if not isinstance(script_data, dict):
        return 0, []

    total = int(script_data.get("total", 0) or 0)
    external = script_data.get("external", [])
    if not isinstance(external, list):
        external = []

    return total, [str(item) for item in external if str(item).strip()]


async def _extract_cookies(context: Any, responses: list[Any]) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    set_cookie_headers: list[str] = []

    for response in responses:
        try:
            headers_array = await response.headers_array()
        except Exception:
            headers_array = []

        if isinstance(headers_array, list) and headers_array:
            for item in headers_array:
                if not isinstance(item, dict):
                    continue
                if str(item.get("name", "")).lower() == "set-cookie":
                    header_value = str(item.get("value", "")).strip()
                    if header_value:
                        set_cookie_headers.append(header_value)
            continue

        try:
            headers = await response.all_headers()
        except Exception:
            headers = {}

        if isinstance(headers, dict):
            for key, value in headers.items():
                if str(key).lower() == "set-cookie" and value:
                    set_cookie_headers.append(str(value))

    try:
        browser_cookies = await context.cookies()
    except Exception:
        browser_cookies = []

    cookies: list[dict[str, Any]] = []
    for cookie in browser_cookies:
        if not isinstance(cookie, dict):
            continue
        cookies.append(
            {
                "name": cookie.get("name", ""),
                "domain": cookie.get("domain", ""),
                "path": cookie.get("path", ""),
                "secure": bool(cookie.get("secure", False)),
                "httponly": bool(cookie.get("httpOnly", False)),
                "samesite": cookie.get("sameSite", ""),
                "expires": cookie.get("expires", ""),
            }
        )

    parsed_set_cookie = [_parse_set_cookie_value(value) for value in set_cookie_headers]

    return set_cookie_headers, parsed_set_cookie, cookies


def _attach_page_event_hooks(page: Any, state: RunState) -> None:
    def _on_request(request: Any) -> None:
        if len(state.requests) >= MAX_NETWORK_EVENTS:
            return

        request_url = str(request.url)
        method = str(request.method)
        resource_type = str(request.resource_type)

        state.requests.append(
            {
                "url": request_url,
                "method": method,
                "resource_type": resource_type,
            }
        )

        host = (urlparse(request_url).hostname or "").lower()
        request_registered = _registered_domain(host)
        if request_registered and request_registered != state.initial_registered_domain:
            state.external_domains.add(request_registered)

    def _on_response(response: Any) -> None:
        state.responses.append(response)
        try:
            headers = response.headers
            if isinstance(headers, dict):
                csp_value = str(headers.get("content-security-policy") or "").strip()
                if csp_value:
                    state.csp_headers.append(csp_value)
        except Exception:
            pass

    def _on_console(message: Any) -> None:
        if len(state.console_messages) >= MAX_CONSOLE_EVENTS:
            return
        text = str(message.text or "").strip()
        if text:
            state.console_messages.append(text)

    def _on_request_failed(request: Any) -> None:
        if len(state.request_errors) >= MAX_NETWORK_EVENTS:
            return
        failure = request.failure
        state.request_errors.append(
            {
                "url": str(request.url),
                "method": str(request.method),
                "resource_type": str(request.resource_type),
                "failure": str(failure or ""),
            }
        )

    def _on_frame_navigated(frame: Any) -> None:
        try:
            if frame == page.main_frame:
                _append_unique(state.redirect_chain, str(frame.url or ""))
        except Exception:
            return

    page.on("request", _on_request)
    page.on("response", _on_response)
    page.on("console", _on_console)
    page.on("requestfailed", _on_request_failed)
    page.on("framenavigated", _on_frame_navigated)


async def _install_internal_network_guard(page: Any, state: RunState) -> None:
    async def _guard(route: Any, request: Any) -> None:
        host = (urlparse(str(request.url)).hostname or "").strip().lower()
        if _is_private_or_loopback_host(host):
            state.blocked_internal_requests.append(str(request.url))
            await route.abort(error_code="blockedbyclient")
            return
        await route.continue_()

    await page.route("**/*", _guard)


async def analyze_url(target_url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict[str, Any]:
    normalized_url = _normalize_input_url(target_url)
    output = _safe_output(normalized_url)

    if not normalized_url:
        _debug("empty URL received")
        output["error"] = "empty_url"
        output["error_stage"] = "normalize_input"
        return output

    parsed = urlparse(normalized_url)
    initial_host = (parsed.hostname or "").lower()
    initial_registered_domain = _registered_domain(initial_host)

    state = RunState(
        initial_url=normalized_url,
        initial_registered_domain=initial_registered_domain,
        timeout_ms=timeout_ms,
        requests=[],
        responses=[],
        redirect_chain=[normalized_url],
        external_domains=set(),
        blocked_internal_requests=[],
        console_messages=[],
        request_errors=[],
        csp_headers=[],
    )

    playwright = None
    browser = None
    context = None
    page = None

    error_stage = "launch_browser"

    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright

        _debug(f"starting Playwright for url={normalized_url} timeout_ms={timeout_ms}")
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--disable-extensions",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--mute-audio",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )

        error_stage = "create_context"
        context = await browser.new_context(
            accept_downloads=False,
            ignore_https_errors=True,
            java_script_enabled=True,
            bypass_csp=False,
        )
        await context.add_init_script(FINGERPRINT_BEACON_INIT_SCRIPT)
        _debug("context created and fingerprint/beacon init script injected")

        error_stage = "create_page"
        page = await context.new_page()
        page.set_default_timeout(timeout_ms)
        page.set_default_navigation_timeout(timeout_ms)
        await page.set_viewport_size({"width": 1366, "height": 768})

        _attach_page_event_hooks(page, state)

        error_stage = "install_network_guard"
        await _install_internal_network_guard(page, state)

        try:
            error_stage = "navigate"
            await page.goto(normalized_url, wait_until="domcontentloaded", timeout=timeout_ms)

            error_stage = "wait_for_network_idle"
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 7_000))
        except PlaywrightTimeoutError:
            _debug(f"navigation timeout at stage={error_stage}")
            output["status"] = "timeout"
            output["error"] = "navigation_timeout"
            output["error_stage"] = error_stage
        except Exception as exc:  # noqa: BLE001
            _debug(f"navigation error at stage={error_stage}: {exc.__class__.__name__}:{exc}")
            output["status"] = "error"
            output["error"] = f"navigation_failed:{exc.__class__.__name__}:{exc}"
            output["error_stage"] = error_stage

        error_stage = "extract_page_artifacts"
        final_url = page.url or normalized_url
        _append_unique(state.redirect_chain, final_url)

        dom_length, node_count, raw_html = await _extract_dom_metrics(page)
        num_scripts, external_js = await _extract_scripts(page)
        set_cookie_headers, set_cookie_details, cookies = await _extract_cookies(context, state.responses)

        suspicious_endpoints = [
            request
            for request in state.requests
            if _is_suspicious_endpoint(request.get("url", ""), request.get("method", ""))
        ]

        redirect_analysis = _analyze_redirect_chain(normalized_url, final_url, state.redirect_chain)
        iframe_analysis = await _analyze_iframes(page=page, main_page_url=final_url or normalized_url)
        csp_analysis = _analyze_csp_headers(state.csp_headers[-1] if state.csp_headers else "")

        methods, beacon_calls = await _extract_runtime_signals(page)
        fp_score, fp_risk = _fingerprinting_risk(methods)
        suspicious_count, external_count = _analyze_network_for_beaconing(state.requests, final_url)
        beacon_risk = _beaconing_risk(beacon_calls, suspicious_count, external_count)

        payload = {
            "initial_url": normalized_url,
            "final_url": final_url,
            "redirect_chain": state.redirect_chain,
            "dom_length": dom_length,
            "dom_metrics": {
                "node_count": node_count,
                "dom_length": dom_length,
            },
            "raw_html": raw_html,
            "num_scripts": num_scripts,
            "external_js": external_js,
            "requests": state.requests,
            "network_requests": state.requests,
            "external_domains": sorted(state.external_domains),
            "third_party_domains": sorted(state.external_domains),
            "suspicious_endpoints": suspicious_endpoints,
            "set_cookie_headers": set_cookie_headers,
            "set_cookie_details": set_cookie_details,
            "cookies": cookies,
            "console_messages": state.console_messages,
            "blocked_internal_requests": state.blocked_internal_requests,
            "request_failures": state.request_errors,
            "phishing_behavior_analysis": {
                "redirect_analysis": redirect_analysis,
                "iframe_analysis": iframe_analysis,
                "csp_analysis": csp_analysis,
                "network_requests": state.requests,
                "error": "",
            },
            "fingerprint_beacon_analysis": {
                "fingerprinting": {
                    "methods": sorted({method for method in methods}),
                    "score": fp_score,
                    "risk": fp_risk,
                },
                "beaconing": {
                    "beacon_calls": beacon_calls,
                    "suspicious_requests": suspicious_count,
                    "external_requests": external_count,
                    "risk": beacon_risk,
                },
                "network_requests": state.requests,
                "error": "",
            },
        }
        payload["suspicious_activity_flags"] = _suspicious_flags(payload)

        output.update(payload)

        if output["status"] not in {"timeout", "error"}:
            output["status"] = "success"
            output["error"] = ""
            output["error_stage"] = ""
        _debug(
            (
                f"completed status={output.get('status')} final_url={output.get('final_url')} "
                f"requests={len(output.get('network_requests', []))} redirects={len(output.get('redirect_chain', []))}"
            )
        )

        return output

    except ImportError:
        _debug("playwright_not_installed")
        output["status"] = "error"
        output["error"] = "playwright_not_installed"
        output["error_stage"] = error_stage
        return output
    except Exception as exc:  # noqa: BLE001
        _debug(f"sandbox failure stage={error_stage} exc={exc.__class__.__name__}:{exc}")
        output["status"] = "error"
        output["error"] = f"sandbox_failure:{exc.__class__.__name__}:{exc}"
        output["error_stage"] = error_stage
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run URL sandbox telemetry capture")
    parser.add_argument("url", help="Target URL")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="Navigation and runtime timeout in milliseconds",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timeout_ms = max(1_000, min(int(args.timeout_ms), 20_000))

    result = asyncio.run(analyze_url(args.url, timeout_ms=timeout_ms))
    print(json.dumps(result, separators=(",", ":"), default=str))


if __name__ == "__main__":
    main()
