from __future__ import annotations

import json
import logging
import os
from threading import Lock
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("zora.url_analysis.llm_reasoner")

DEFAULT_OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30"))
DEFAULT_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "320"))
DEFAULT_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))

_cache_lock = Lock()
_response_cache: dict[tuple[str, str], dict[str, Any]] = {}


def _normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"phishing", "malicious", "fraud", "suspicious", "unsafe"}:
        return "phishing"
    if label in {"safe", "genuine", "legitimate", "benign"}:
        return "genuine"
    return "unknown"


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
        "You are a senior URL threat analyst. Analyze whether this URL is phishing or genuine using ONLY provided evidence. Explain everything in short"
        "Provide a concise explanation and practical recommendations. Return STRICT JSON only.\n"
        "Required JSON schema:\n"
        '{"final_label":"phishing|genuine","confidence":0.0,"explanation":"clear concise reason","key_indicators":["..."],"recommendations":["..."]}\n\n'
        f"URL: {url}\n"
        f"Final URL after redirects: {final_url}\n"
        f"Model phishing_probability: {probability:.4f}\n"
        f"Composite risk_score: {risk_score:.4f}\n"
        f"Composite risk_level: {risk_level}\n\n"
        f"URL Features: {json.dumps(url_features, ensure_ascii=False)[:1400]}\n"
        f"Domain Features: {json.dumps(domain_features, ensure_ascii=False)[:1400]}\n"
        f"TLS Features: {json.dumps(tls_features, ensure_ascii=False)[:1400]}\n"
        f"Homoglyph Features: {json.dumps(homoglyph_features, ensure_ascii=False)[:1400]}\n"
        f"Cookie Features: {json.dumps(cookie_features, ensure_ascii=False)[:1400]}\n"
        f"Redirect/IFrame/CSP Features: {json.dumps(behavior_features, ensure_ascii=False)[:1600]}\n"
        f"Fingerprint/Beacon Features: {json.dumps(fp_beacon_features, ensure_ascii=False)[:1600]}\n"
    )


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

    candidate = text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return fallback

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
        return fallback

    return {
        "final_label": final_label,
        "confidence": round(confidence, 4),
        "explanation": explanation,
        "key_indicators": key_indicators,
        "recommendations": recommendations,
    }


def _call_openrouter(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing in environment")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required. Install with 'pip install openai'") from exc

    model_name = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    client = OpenAI(
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )

    logger.info("Calling OpenRouter URL model=%s", model_name)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a concise cybersecurity URL analysis assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )

    return str(response.choices[0].message.content or "")


def explain_url_with_llm(data: dict[str, Any]) -> dict[str, Any]:
    prompt = _build_prompt(data)
    model_name = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    cache_key = (model_name, prompt)

    cached = _response_cache.get(cache_key)
    if cached is not None:
        logger.info("Using cached URL LLM explanation")
        return cached

    try:
        raw_output = _call_openrouter(prompt)
        parsed = _parse_response(raw_output)
    except Exception as exc:  # noqa: BLE001
        logger.warning("URL LLM call failed: %s", exc)
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
