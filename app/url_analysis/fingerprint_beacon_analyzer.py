"""Browser fingerprinting and beaconing behavior analyzer using Playwright."""

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

SUSPICIOUS_BEACON_KEYWORDS: tuple[str, ...] = (
    "track",
    "collect",
    "beacon",
    "fingerprint",
    "analytics",
)

MAX_NETWORK_LOGS = 500


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


def _safe_output() -> dict[str, Any]:
    """Return default structured output for resilient callers."""
    return {
        "fingerprinting": {
            "methods": [],
            "score": 0,
            "risk": "low",
        },
        "beaconing": {
            "beacon_calls": [],
            "suspicious_requests": 0,
            "external_requests": 0,
            "risk": "low",
        },
        "network_requests": [],
        "error": "",
    }


def _normalize_input_url(url: str) -> str:
    """Normalize URL and default missing scheme to HTTPS."""
    value = (url or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if not parsed.scheme:
        return f"https://{value}"
    return value


def _registered_domain(hostname: str) -> str:
    """Return registered domain from hostname for same-site checks."""
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


def _extract_domain_from_url(value: str) -> str:
    """Extract normalized registered domain from URL text."""
    raw = (value or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return ""
    return _registered_domain(host)


def fingerprinting_risk(methods: list[str]) -> tuple[int, str]:
    """Calculate fingerprinting score and risk bucket."""
    unique_methods = sorted({str(item).strip() for item in methods if str(item).strip()})
    score = len(unique_methods)

    if score >= 2:
        risk = "high"
    elif score == 1:
        risk = "medium"
    else:
        risk = "low"

    return score, risk


def beaconing_risk(
    beacon_calls: list[dict[str, Any]],
    suspicious_requests: int,
    external_requests: int,
) -> str:
    """Calculate beaconing risk bucket from request and beacon telemetry."""
    if suspicious_requests > 5 or len(beacon_calls) > 0:
        return "high"
    if suspicious_requests > 0 or external_requests > 0:
        return "medium"
    return "low"


async def extract_runtime_signals(page: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract fingerprinting methods and sendBeacon calls from injected globals."""
    try:
        payload = await page.evaluate(
            """
            () => {
              const state = window.__securitySignals || {};
              const methods = Array.isArray(state.fingerprintingMethods)
                ? state.fingerprintingMethods
                : [];
              const beaconCalls = Array.isArray(state.beaconCalls)
                ? state.beaconCalls
                : [];
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


def analyze_network_for_beaconing(
    network_requests: list[dict[str, Any]],
    main_page_url: str,
) -> tuple[int, int]:
    """Return counts of suspicious and cross-domain requests."""
    main_domain = _extract_domain_from_url(main_page_url)

    suspicious_count = 0
    external_count = 0

    for request in network_requests:
        if not isinstance(request, dict):
            continue
        request_url = str(request.get("url", "")).strip()
        if not request_url:
            continue

        lowered = request_url.lower()
        if any(keyword in lowered for keyword in SUSPICIOUS_BEACON_KEYWORDS):
            suspicious_count += 1

        request_domain = _extract_domain_from_url(request_url)
        if request_domain and main_domain and request_domain != main_domain:
            external_count += 1

    return suspicious_count, external_count


async def analyze_page_fingerprint_and_beaconing(
    page: Any,
    main_page_url: str,
    network_requests: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Analyze an already-loaded Playwright page for fingerprinting and beaconing."""
    output = _safe_output()
    request_items = network_requests if isinstance(network_requests, list) else []

    try:
        methods, beacon_calls = await extract_runtime_signals(page)
        fp_score, fp_risk = fingerprinting_risk(methods)
        suspicious_count, external_count = analyze_network_for_beaconing(
            network_requests=request_items,
            main_page_url=main_page_url,
        )
        b_risk = beaconing_risk(
            beacon_calls=beacon_calls,
            suspicious_requests=suspicious_count,
            external_requests=external_count,
        )

        output.update(
            {
                "fingerprinting": {
                    "methods": sorted({method for method in methods}),
                    "score": fp_score,
                    "risk": fp_risk,
                },
                "beaconing": {
                    "beacon_calls": beacon_calls,
                    "suspicious_requests": suspicious_count,
                    "external_requests": external_count,
                    "risk": b_risk,
                },
                "network_requests": request_items,
                "error": "",
            }
        )
        return output
    except Exception as exc:
        output["error"] = f"fingerprint_beacon_analysis_failed:{exc.__class__.__name__}"
        output["network_requests"] = request_items
        return output


async def analyze_url_fingerprint_and_beaconing(
    url: str,
    timeout_ms: int = 15_000,
) -> dict[str, Any]:
    """Standalone async Playwright URL analyzer for fingerprinting and beaconing."""
    normalized_url = _normalize_input_url(url)
    output = _safe_output()

    if not normalized_url:
        output["error"] = "empty_url"
        return output

    playwright = None
    browser = None
    context = None
    page = None

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
        await context.add_init_script(FINGERPRINT_BEACON_INIT_SCRIPT)

        page = await context.new_page()
        page.set_default_timeout(timeout_ms)

        def _on_request(request: Any) -> None:
            if len(network_requests) >= MAX_NETWORK_LOGS:
                return
            network_requests.append({"url": request.url, "method": request.method})

        page.on("request", _on_request)

        try:
            await page.goto(normalized_url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 7_000))
        except Exception:
            pass

        final_url = page.url or normalized_url
        result = await analyze_page_fingerprint_and_beaconing(
            page=page,
            main_page_url=final_url,
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
