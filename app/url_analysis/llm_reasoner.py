from __future__ import annotations

import copy
import json
import logging
import os
import time
from threading import Lock
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger("zora.url_analysis.llm_reasoner")

# =========================
# CONFIG
# =========================
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
DEFAULT_CONNECT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "5"))
DEFAULT_MAX_TOKENS = int(os.getenv("OLLAMA_URL_MAX_TOKENS", "400"))
DEFAULT_MIN_NUM_PREDICT = int(os.getenv("OLLAMA_URL_MIN_NUM_PREDICT", "140"))
DEFAULT_URL_FEATURE_MAX_CHARS = int(os.getenv("OLLAMA_URL_FEATURE_MAX_CHARS", "320"))
DEFAULT_URL_PROMPT_MAX_CHARS = int(os.getenv("OLLAMA_URL_PROMPT_MAX_CHARS", "3600"))
DEFAULT_URL_MAX_INPUT_TOKENS = int(os.getenv("OLLAMA_URL_MAX_INPUT_TOKENS", "900"))
DEFAULT_TIMEOUT_MAX_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_MAX_SECONDS", "180"))
DEFAULT_BACKUP_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_BACKUP_TIMEOUT_SECONDS", "60"))
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
DEFAULT_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))

# =========================
# GLOBAL STATE (OPTIMIZED)
# =========================
_session = requests.Session()
_session.headers.update({"Connection": "keep-alive"})

_executor = ThreadPoolExecutor(max_workers=2)

_cache_lock = Lock()
_response_cache: dict[tuple[str, str], dict[str, Any]] = {}

_model_warmed_up: set[str] = set()
_available_models: set[str] | None = None

# Prebuilt payload template
_BASE_PAYLOAD: dict[str, Any] = {
    "stream": False,
    "format": "json",
    "keep_alive": "10m",
    "options": {
        "temperature": DEFAULT_TEMPERATURE,
        "num_predict": DEFAULT_MAX_TOKENS,
    },
}


# =========================
# HELPERS
# =========================
def _normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"phishing", "malicious", "fraud", "suspicious", "unsafe"}:
        return "phishing"
    if label in {"safe", "genuine", "legitimate", "benign"}:
        return "genuine"
    return "unknown"


def _compact_features(features: Any, max_chars: int = 600) -> str:
    """Produce a compact JSON snippet; truncate to max_chars."""
    if not isinstance(features, dict) or not features:
        return "{}"
    try:
        raw = json.dumps(features, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "..."


def _estimate_tokens(text: str) -> int:
    # Rough approximation for LLM token estimation.
    return max(1, len(text) // 4)


def _enforce_prompt_budget(prompt: str) -> str:
    limited = prompt

    if len(limited) > DEFAULT_URL_PROMPT_MAX_CHARS:
        limited = limited[:DEFAULT_URL_PROMPT_MAX_CHARS]

    max_chars_by_tokens = max(400, DEFAULT_URL_MAX_INPUT_TOKENS * 4)
    if len(limited) > max_chars_by_tokens:
        limited = limited[:max_chars_by_tokens]

    return limited


def _shrink_prompt_for_retry(prompt: str) -> str:
    if len(prompt) <= 600:
        return prompt
    return prompt[: int(len(prompt) * 0.75)]


# =========================
# PROMPT
# =========================
def _build_prompt(data: dict[str, Any]) -> str:
    url = str(data.get("url") or "")
    final_url = str(data.get("final_url") or "")

    probability = float(data.get("phishing_probability") or 0.0)
    risk_score = float(data.get("risk_score") or 0.0)
    risk_level = str(data.get("risk_level") or "unknown")

    url_features = data.get("url_features") or {}
    domain_features = data.get("domain_features") or {}
    tls_features = data.get("tls_features") or {}
    homoglyph_features = data.get("homoglyph_features") or {}
    cookie_features = data.get("cookie_features") or {}
    behavior_features = data.get("phishing_behavior_features") or {}
    fp_beacon_features = data.get("fingerprint_beacon_features") or {}

    return (
        "You are a senior URL threat analyst. "
        "Analyze whether this URL is phishing or genuine using ONLY the provided evidence. "
        "Provide a concise explanation and practical recommendations. "
        "Return STRICT JSON only. Do not include markdown, code blocks, or extra commentary.\n\n"
        "Required JSON schema:\n"
        '{"final_label":"phishing|genuine","confidence":0.0,"explanation":"clear concise reason","key_indicators":["..."],"recommendations":["..."]}\n\n'
        f"URL: {url}\n"
        f"Final URL after redirects: {final_url}\n"
        f"Model phishing_probability: {probability:.4f}\n"
        f"Composite risk_score: {risk_score:.4f}\n"
        f"Composite risk_level: {risk_level}\n\n"
        f"URL Features: {_compact_features(url_features, max_chars=DEFAULT_URL_FEATURE_MAX_CHARS)}\n"
        f"Domain Features: {_compact_features(domain_features, max_chars=DEFAULT_URL_FEATURE_MAX_CHARS)}\n"
        f"TLS Features: {_compact_features(tls_features, max_chars=DEFAULT_URL_FEATURE_MAX_CHARS)}\n"
        f"Homoglyph Features: {_compact_features(homoglyph_features, max_chars=DEFAULT_URL_FEATURE_MAX_CHARS)}\n"
        f"Cookie Features: {_compact_features(cookie_features, max_chars=DEFAULT_URL_FEATURE_MAX_CHARS)}\n"
        f"Behavior Features: {_compact_features(behavior_features, max_chars=DEFAULT_URL_FEATURE_MAX_CHARS)}\n"
        f"Fingerprint/Beacon Features: {_compact_features(fp_beacon_features, max_chars=DEFAULT_URL_FEATURE_MAX_CHARS)}\n"
    )


# =========================
# PARSER
# =========================
def _parse_response(raw_text: str) -> dict[str, Any]:
    fallback = {
        "final_label": "unknown",
        "confidence": 0.0,
        "explanation": "LLM explanation unavailable",
        "key_indicators": [],
        "recommendations": [],
    }

    text = (raw_text or "").strip()
    if not text:
        return fallback

    candidates = [text]

    # Fast JSON extraction (no regex)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.insert(0, text[start : end + 1])

    for i, candidate in enumerate(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            if i == 0:
                continue
            break

        final_label = _normalize_label(parsed.get("final_label"))
        explanation = str(parsed.get("explanation") or "").strip()

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        key_indicators_raw = parsed.get("key_indicators", [])
        recommendations_raw = parsed.get("recommendations", [])

        key_indicators = []
        if isinstance(key_indicators_raw, list):
            key_indicators = [str(item).strip() for item in key_indicators_raw if str(item).strip()][:8]

        recommendations = []
        if isinstance(recommendations_raw, list):
            recommendations = [str(item).strip() for item in recommendations_raw if str(item).strip()][:8]

        if final_label == "unknown" or not explanation:
            continue

        return {
            "final_label": final_label,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "key_indicators": key_indicators,
            "recommendations": recommendations,
        }

    return fallback


# =========================
# MODEL PREP (PARALLEL)
# =========================
def _ensure_ollama_available(model_name: str) -> None:
    global _available_models

    if _available_models is not None:
        if model_name not in _available_models:
            raise RuntimeError(f"Model '{model_name}' not found")
        return

    response = _session.get(
        DEFAULT_OLLAMA_TAGS_URL,
        timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, 10),
    )
    response.raise_for_status()

    payload = response.json()
    models = payload.get("models") or []
    _available_models = {str(item.get("name") or "") for item in models}

    if model_name not in _available_models:
        raise RuntimeError(
            f"Model '{model_name}' not found in Ollama. Available models: {sorted(_available_models)}"
        )


def _warmup_model_once(model_name: str) -> None:
    with _cache_lock:
        if model_name in _model_warmed_up:
            return

    try:
        _session.post(
            DEFAULT_OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": "Return JSON: {\"ok\": true}",
                "stream": False,
                "format": "json",
                "keep_alive": "10m",
                "options": {"temperature": 0, "num_predict": 8},
            },
            timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, min(DEFAULT_TIMEOUT_SECONDS, 30)),
        )
    except requests.RequestException as exc:
        logger.warning("Ollama warm-up failed; continuing anyway: %s", exc)

    with _cache_lock:
        _model_warmed_up.add(model_name)


def _prepare_model(model_name: str) -> None:
    futures = [
        _executor.submit(_ensure_ollama_available, model_name),
        _executor.submit(_warmup_model_once, model_name),
    ]
    for f in futures:
        f.result()


# =========================
# LLM CALL (OPTIMIZED)
# =========================
def _call_llm(prompt: str) -> str:
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    bounded_prompt = _enforce_prompt_budget(prompt)
    prompt_tokens = _estimate_tokens(bounded_prompt)
    adaptive_read_timeout = min(
        DEFAULT_TIMEOUT_MAX_SECONDS,
        max(DEFAULT_TIMEOUT_SECONDS, 40 + (prompt_tokens * 0.08)),
    )

    # Deep copy to avoid mutating the template
    payload = copy.deepcopy(_BASE_PAYLOAD)
    payload["model"] = model_name
    payload["prompt"] = bounded_prompt

    _prepare_model(model_name)

    logger.info("Calling local Ollama model=%s for URL analysis", model_name)
    print(f"[url-llm] Calling Ollama model={model_name}")
    timeout = (DEFAULT_CONNECT_TIMEOUT_SECONDS, adaptive_read_timeout)

    raw_text = ""
    last_error: Exception | None = None

    for attempt in range(1, DEFAULT_RETRIES + 2):
        try:
            response = _session.post(
                DEFAULT_OLLAMA_URL,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()

            raw_text = body.get("response") or ""
            if raw_text:
                raw_text = raw_text.strip()
                print(f"[url-llm] Raw response ({len(raw_text)} chars): {raw_text[:200]}")
                break

        except requests.exceptions.ReadTimeout as exc:
            last_error = exc
            current_prompt = str(payload.get("prompt") or "")
            shrunken_prompt = _shrink_prompt_for_retry(current_prompt)
            if shrunken_prompt != current_prompt:
                payload["prompt"] = shrunken_prompt

            options = payload.get("options") or {}
            current_num_predict = int(options.get("num_predict") or DEFAULT_MAX_TOKENS)
            reduced_num_predict = max(DEFAULT_MIN_NUM_PREDICT, int(current_num_predict * 0.75))
            options["num_predict"] = reduced_num_predict
            payload["options"] = options

            logger.warning(
                "Ollama read timeout (attempt %s). prompt_tokens=%s retry_prompt_tokens=%s num_predict=%s",
                attempt,
                _estimate_tokens(current_prompt),
                _estimate_tokens(str(payload.get("prompt") or "")),
                reduced_num_predict,
            )
            if attempt <= DEFAULT_RETRIES:
                time.sleep(0.4 * attempt)

        except requests.RequestException as exc:
            last_error = exc
            logger.warning("Ollama call failed (attempt %s): %s", attempt, exc)
            if attempt <= DEFAULT_RETRIES:
                time.sleep(0.3 * attempt)

    if not raw_text:
        if last_error:
            raise last_error
        raise RuntimeError("Ollama returned an empty response")

    return raw_text


# =========================
# PIPELINE
# =========================
def explain_url_with_llm(data: dict[str, Any]) -> dict[str, Any]:
    prompt = _build_prompt(data)
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    cache_key = (model_name, prompt)
    started_at = time.monotonic()

    # Fast path (no lock)
    cached = _response_cache.get(cache_key)
    if cached is not None:
        logger.info("Using cached URL LLM explanation")
        print("[url-llm] Cache hit")
        return cached

    try:
        raw_output = _call_llm(prompt)
        parsed = _parse_response(raw_output)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - started_at
        is_timeout = isinstance(exc, requests.exceptions.ReadTimeout) or "Read timed out" in str(exc)
        if is_timeout and elapsed >= DEFAULT_BACKUP_TIMEOUT_SECONDS:
            with _cache_lock:
                cached_backup = _response_cache.get(cache_key)
            if cached_backup is not None:
                logger.warning(
                    "URL LLM timed out after %.2fs; using cached backup response",
                    elapsed,
                )
                return cached_backup

        logger.warning("URL LLM call failed: %s", exc)
        print(f"[url-llm] Failed: {exc}")
        parsed = {
            "final_label": "unknown",
            "confidence": 0.0,
            "explanation": "LLM explanation unavailable",
            "key_indicators": [],
            "recommendations": [],
        }

    with _cache_lock:
        _response_cache[cache_key] = parsed

    return parsed
