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

logger = logging.getLogger("zora.email_analyzer.llm_reasoner")

# =========================
# CONFIG
# =========================
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
DEFAULT_CONNECT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "5"))
DEFAULT_MAX_TOKENS = int(os.getenv("OLLAMA_EMAIL_MAX_TOKENS", "300"))
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
_response_backup_cache: dict[tuple[str, str], dict[str, Any]] = {}

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
    if label in {"phishing", "spam", "scam", "fraud", "malicious"}:
        return "phishing"
    if label in {"safe", "genuine", "legitimate", "benign"}:
        return "genuine"
    return "unknown"


# =========================
# PROMPT
# =========================
def _build_prompt(data: dict[str, Any]) -> str:
    sender = str(data.get("sender") or "")
    subject = str(data.get("subject") or "")
    body = str(data.get("body") or "")
    nlp_label = str(data.get("nlp_label") or "unknown")
    nlp_score = float(data.get("nlp_score") or 0.0)
    similarity_score = float(data.get("similarity_score") or 0.0)
    stylometry_score = float(data.get("stylometry_score") or 0.0)
    final_score = float(data.get("risk_score") or 0.0)

    body_short = body[:800]

    return (
        "You are an email fraud analyst. "
        "Analyze if this email is phishing or genuine. "
        "Summarize the email body and explain why the model gave the following scores. "
        "Use only evidence in the email and scores below.\n"
        "Return STRICT JSON only. Do not include markdown, code blocks, or extra commentary.\n\n"
        "Required JSON schema:\n"
        '{"final_label":"phishing|genuine","confidence":0.0,"explanation":"short clear reason"}\n\n'
        f"Email Sender: {sender}\n"
        f"Email Subject: {subject}\n"
        f"Email Body: {body_short}\n\n"
        "Model Signals:\n"
        f"- nlp_label: {nlp_label}\n"
        f"- nlp_score: {nlp_score:.4f}\n"
        f"- similarity_score: {similarity_score:.4f}\n"
        f"- stylometry_score: {stylometry_score:.4f}\n"
        f"- final_risk_score: {final_score:.4f}\n"
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
def _call_llm(prompt: str) -> str:
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    # Deep copy to avoid mutating the template
    payload = copy.deepcopy(_BASE_PAYLOAD)
    payload["model"] = model_name
    payload["prompt"] = prompt

    _prepare_model(model_name)

    logger.info("Calling local Ollama model=%s", model_name)
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

    return raw_text

# =========================
# PARSER (FAST)
# =========================
def _parse_response(raw_text: str) -> dict[str, Any]:
    fallback = {
        "final_label": "unknown",
        "confidence": 0.0,
        "explanation": "LLM explanation unavailable",
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
        confidence_raw = parsed.get("confidence", 0.0)
        explanation = str(parsed.get("explanation") or "").strip()

        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        if final_label == "unknown" or not explanation:
            continue

        return {
            "final_label": final_label,
            "confidence": round(confidence, 4),
            "explanation": explanation,
        }

    return fallback

# =========================
# PIPELINE
# =========================
def explain_email_with_llm(data: dict[str, Any]) -> dict[str, Any]:
    prompt = _build_prompt(data)
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    cache_key = (model_name, prompt)
    started_at = time.monotonic()

    try:
        raw_output = _call_llm(prompt)
        parsed = _parse_response(raw_output)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - started_at
        is_timeout = isinstance(exc, requests.exceptions.ReadTimeout) or "Read timed out" in str(exc)
        if is_timeout and elapsed >= DEFAULT_BACKUP_TIMEOUT_SECONDS:
            with _cache_lock:
                cached_backup = _response_backup_cache.get(cache_key)
            if cached_backup is not None:
                logger.warning(
                    "Email LLM timed out after %.2fs; using cached backup response",
                    elapsed,
                )
                return cached_backup

        logger.warning("Email LLM call failed: %s", exc)
        parsed = {
            "final_label": "unknown",
            "confidence": 0.0,
            "explanation": "LLM explanation unavailable",
        }

    if parsed.get("final_label") != "unknown" and str(parsed.get("explanation") or "").strip():
        with _cache_lock:
            _response_backup_cache[cache_key] = parsed

    return parsed
