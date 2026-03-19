from __future__ import annotations

import json
import logging
import os
import time
from threading import Lock
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger("zora.text_analysis.llm")

# =========================
# CONFIG
# =========================
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "50"))
DEFAULT_CONNECT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "5"))
DEFAULT_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "120"))
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
DEFAULT_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))

MALICIOUS_LABELS = {"spam", "phishing", "scam"}
ALLOWED_LABELS = MALICIOUS_LABELS | {"safe"}

# =========================
# GLOBAL STATE (OPTIMIZED)
# =========================
_session = requests.Session()
_session.headers.update({"Connection": "keep-alive"})

_executor = ThreadPoolExecutor(max_workers=2)

_cache_lock = Lock()
_response_cache: dict[tuple[str, str], str] = {}

_model_warmed_up: set[str] = set()
_available_models: set[str] | None = None

# Prebuilt payload template
_BASE_PAYLOAD = {
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
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ALLOWED_LABELS else "unknown"


# =========================
# PROMPT
# =========================
def build_prompt(data: dict[str, Any]) -> str:
    sms_text = str(data.get("sms_text") or "")
    nlp_prediction = str(data.get("nlp_prediction") or "unknown")
    nlp_confidence = float(data.get("nlp_confidence") or 0.0)
    url_flags = data.get("url_flags") or []
    stylometry_score = float(data.get("stylometry_score") or 0.0)
    similarity_score = float(data.get("similarity_score") or 0.0)
    rule_flags = data.get("rule_flags") or []

    return (
        "You are a fraud detection expert specializing in SMS phishing, social engineering, try to explain the sms_text"
        "and scam pattern analysis.\n"
        "You must reason from the given evidence and sms_text and return STRICT JSON only.\n"
        "Do not include markdown, code blocks, or extra commentary.\n\n"
        "Evidence:\n"
        f"- SMS text: {sms_text}\n"
        f"- NLP prediction: {nlp_prediction}\n"
        f"- NLP confidence: {nlp_confidence:.4f}\n"
        f"- URL flags: {json.dumps(url_flags)}\n"
        f"- Stylometry score: {stylometry_score:.4f}\n"
        f"- Similarity score: {similarity_score:.4f}\n"
        f"- Rule flags: {json.dumps(rule_flags)}\n\n"
        "Output format (strict JSON object):\n"
        '{"final_label":"safe|spam|phishing|scam","confidence":0.0,"explanation":"clear human explanation"}'
    )


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
def call_llm(prompt: str) -> str:
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    cache_key = (model_name, prompt)

    # Fast path (no lock)
    cached = _response_cache.get(cache_key)
    if cached:
        logger.info("Using cached LLM response")
        return cached

    payload = _BASE_PAYLOAD.copy()
    payload["model"] = model_name
    payload["prompt"] = prompt

    _prepare_model(model_name)

    logger.info("Calling local Ollama model: %s", model_name)
    timeout = (DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS)

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
                break

        except requests.RequestException as exc:
            last_error = exc
            logger.warning("Ollama call failed (attempt %s): %s", attempt, exc)
            if attempt <= DEFAULT_RETRIES:
                time.sleep(0.3 * attempt)

    if not raw_text:
        if last_error:
            raise last_error
        raise RuntimeError("Ollama returned an empty response")

    # Cache write (locked)
    with _cache_lock:
        _response_cache[cache_key] = raw_text

    return raw_text


# =========================
# PARSER (FAST)
# =========================
def parse_llm_output(response: str) -> dict[str, Any]:
    fallback = {
        "final_label": "unknown",
        "confidence": 0.0,
        "explanation": "LLM parsing failed",
    }

    if not response:
        return fallback

    text = response.strip()
    candidates = [text]

    # Fast JSON extraction (no regex)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.insert(0, text[start:end + 1])

    for i, candidate in enumerate(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            if i == 0:
                continue
            break

        label = _normalize_label(parsed.get("final_label"))
        confidence = parsed.get("confidence", 0.0)
        explanation = str(parsed.get("explanation") or "").strip()

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        if label == "unknown" or not explanation:
            continue

        return {
            "final_label": label,
            "confidence": round(confidence, 4),
            "explanation": explanation,
        }

    return fallback


# =========================
# PIPELINE
# =========================
def analyze_with_llm(data: dict[str, Any]) -> dict[str, Any]:
    prompt = build_prompt(data)

    try:
        raw_output = call_llm(prompt)
    except requests.RequestException as exc:
        logger.warning("LLM call failed: %s", exc)
        return {
            "final_label": "unknown",
            "confidence": 0.0,
            "explanation": "LLM request failed",
        }
    except Exception as exc:
        logger.warning("Unexpected LLM error: %s", exc)
        return {
            "final_label": "unknown",
            "confidence": 0.0,
            "explanation": "LLM request failed",
        }

    return parse_llm_output(raw_output)


# =========================
# TEST
# =========================
def test_llm_reasoner() -> dict[str, Any]:
    sample = {
        "sms_text": "Urgent: Your account is suspended. Verify now at http://secure-bank-update.xyz",
        "nlp_prediction": "phishing",
        "nlp_confidence": 0.62,
        "url_flags": ["suspicious_tld", "long_url"],
        "stylometry_score": 0.71,
        "similarity_score": 0.84,
        "rule_flags": ["urgent_language", "verify_now", "click_link"],
    }

    logger.info("Running LLM reasoner smoke test")
    result = analyze_with_llm(sample)
    print("LLM_TEST_RESULT:", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    test_llm_reasoner()