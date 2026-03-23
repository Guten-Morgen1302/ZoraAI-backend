from __future__ import annotations

import json
import logging
import os
from threading import Lock
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("zora.email_analyzer.llm_reasoner")

DEFAULT_OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30"))
DEFAULT_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "180"))
DEFAULT_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))

_cache_lock = Lock()
_response_cache: dict[tuple[str, str], dict[str, Any]] = {}


def _normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"phishing", "spam", "scam", "fraud", "malicious"}:
        return "phishing"
    if label in {"safe", "genuine", "legitimate", "benign"}:
        return "genuine"
    return "unknown"


def _build_prompt(data: dict[str, Any]) -> str:
    sender = str(data.get("sender") or "")
    subject = str(data.get("subject") or "")
    body = str(data.get("body") or "")
    nlp_label = str(data.get("nlp_label") or "unknown")
    nlp_score = float(data.get("nlp_score") or 0.0)
    similarity_score = float(data.get("similarity_score") or 0.0)
    stylometry_score = float(data.get("stylometry_score") or 0.0)
    final_score = float(data.get("risk_score") or 0.0)

    body_short = body[:1200]

    return (
        "You are an email fraud analyst. Analyze if this email is phishing or genuine. Explain the body in summarized form and tell why the my model gave the following output"
        "Use only evidence in the email and scores below. Return STRICT JSON only.\n"
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


def _parse_response(raw_text: str) -> dict[str, Any]:
    fallback = {
        "final_label": "unknown",
        "confidence": 0.0,
        "explanation": "LLM explanation unavailable",
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
    confidence_raw = parsed.get("confidence", 0.0)
    explanation = str(parsed.get("explanation") or "").strip()

    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if final_label == "unknown" or not explanation:
        return fallback

    return {
        "final_label": final_label,
        "confidence": round(confidence, 4),
        "explanation": explanation,
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

    print(f"[email-llm] Calling OpenRouter model={model_name}")
    logger.info("Calling OpenRouter model=%s", model_name)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a concise phishing intelligence assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )

    return str(response.choices[0].message.content or "")


def explain_email_with_llm(data: dict[str, Any]) -> dict[str, Any]:
    prompt = _build_prompt(data)
    model_name = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    cache_key = (model_name, prompt)

    cached = _response_cache.get(cache_key)
    if cached is not None:
        logger.info("Using cached email LLM explanation")
        print("[email-llm] Cache hit")
        return cached

    try:
        raw_output = _call_openrouter(prompt)
        parsed = _parse_response(raw_output)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Email LLM call failed: %s", exc)
        print(f"[email-llm] Failed: {exc}")
        parsed = {
            "final_label": "unknown",
            "confidence": 0.0,
            "explanation": "LLM explanation unavailable",
        }

    with _cache_lock:
        _response_cache[cache_key] = parsed

    return parsed
