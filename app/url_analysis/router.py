from __future__ import annotations

import json
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import URLAnalysisRequest, URLThreatResult
from app.schemas import URLAnalyzeRequest, URLAnalyzeResponse
from app.url_analysis.llm_reasoner import explain_url_with_llm
from app.url_analysis.ml_risk_engine import URLMLRiskEngine
from app.url_analysis.url_analysis import extract_phase_4_features_async

router = APIRouter(prefix="/url", tags=["url-analysis"])

URL_RISK_ENGINE = URLMLRiskEngine()
URL_RISK_ENGINE_READY = False
URL_RISK_ENGINE_ERROR: str | None = None


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_url_input(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("URL is required")

    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Invalid URL format")
    return raw


def _get_url_risk_engine() -> tuple[URLMLRiskEngine | None, str | None]:
    global URL_RISK_ENGINE_READY, URL_RISK_ENGINE_ERROR

    if URL_RISK_ENGINE_READY:
        return URL_RISK_ENGINE, None

    if URL_RISK_ENGINE_ERROR is not None:
        return None, URL_RISK_ENGINE_ERROR

    try:
        URL_RISK_ENGINE.load()
        URL_RISK_ENGINE_READY = True
        return URL_RISK_ENGINE, None
    except Exception as exc:  # noqa: BLE001
        URL_RISK_ENGINE_ERROR = str(exc)
        return None, URL_RISK_ENGINE_ERROR


def _fallback_risk_from_payload(phase_payload: dict[str, object]) -> dict[str, object]:
    fused = phase_payload.get("fused_features") if isinstance(phase_payload, dict) else {}
    if not isinstance(fused, dict):
        fused = {}

    sub_scores = fused.get("sub_scores")
    if not isinstance(sub_scores, dict):
        sub_scores = {}

    url_score = _to_float(sub_scores.get("url"), 0.0)
    content_score = _to_float(sub_scores.get("content"), 0.0)
    infra_score = _to_float(sub_scores.get("infra"), 0.0)
    behavior_score = _to_float(sub_scores.get("behavior"), 0.0)

    cookie_features = phase_payload.get("cookie_features") if isinstance(phase_payload, dict) else {}
    if not isinstance(cookie_features, dict):
        cookie_features = {}
    cookie_score = _to_float(cookie_features.get("cookie_risk_score"), 0.0)

    probability = max(
        0.0,
        min(
            round(
                (0.35 * url_score + 0.25 * content_score + 0.2 * infra_score + 0.2 * behavior_score),
                6,
            ),
            1.0,
        ),
    )

    risk = URL_RISK_ENGINE.score_risk(
        phishing_probability=probability,
        sub_scores={
            "url": url_score,
            "content": content_score,
            "infra": infra_score,
            "behavior": behavior_score,
        },
        cookie_score=cookie_score,
    )

    return {
        "phishing_probability": probability,
        "model": "heuristic_fallback",
        "risk": risk,
    }


def _build_pipeline_checks(phase_payload: dict[str, object]) -> dict[str, object]:
    url_features = phase_payload.get("url_features") if isinstance(phase_payload, dict) else {}
    domain_features = phase_payload.get("domain_features") if isinstance(phase_payload, dict) else {}
    tls_features = phase_payload.get("tls_features") if isinstance(phase_payload, dict) else {}
    homoglyph_features = phase_payload.get("homoglyph_features") if isinstance(phase_payload, dict) else {}
    sandbox_features = phase_payload.get("sandbox_features") if isinstance(phase_payload, dict) else {}
    cookie_features = phase_payload.get("cookie_features") if isinstance(phase_payload, dict) else {}
    behavior_features = phase_payload.get("phishing_behavior_features") if isinstance(phase_payload, dict) else {}
    fpb_features = phase_payload.get("fingerprint_beacon_features") if isinstance(phase_payload, dict) else {}
    fused_features = phase_payload.get("fused_features") if isinstance(phase_payload, dict) else {}

    if not isinstance(url_features, dict):
        url_features = {}
    if not isinstance(domain_features, dict):
        domain_features = {}
    if not isinstance(tls_features, dict):
        tls_features = {}
    if not isinstance(homoglyph_features, dict):
        homoglyph_features = {}
    if not isinstance(sandbox_features, dict):
        sandbox_features = {}
    if not isinstance(cookie_features, dict):
        cookie_features = {}
    if not isinstance(behavior_features, dict):
        behavior_features = {}
    if not isinstance(fpb_features, dict):
        fpb_features = {}
    if not isinstance(fused_features, dict):
        fused_features = {}

    sandbox_error = str(sandbox_features.get("error") or "").strip()
    whois_age = _to_float(domain_features.get("domain_age_days"), 0.0)
    whois_registrar = str(domain_features.get("registrar") or "").strip()
    whois_creation = str(domain_features.get("domain_creation_date") or "").strip()

    return {
        "phase_1_static_extracted": bool(url_features) and bool(domain_features),
        "whois_extractor_invoked": "domain_age_days" in domain_features and "registrar" in domain_features,
        "whois_has_live_data": bool(whois_registrar or whois_creation or whois_age > 0),
        "phase_2_tls_extracted": bool(tls_features),
        "phase_3_homoglyph_extracted": bool(homoglyph_features),
        "playwright_sandbox_invoked": bool(sandbox_features),
        "playwright_sandbox_success": sandbox_error == "",
        "playwright_network_events": len(sandbox_features.get("network_requests", [])) if isinstance(sandbox_features.get("network_requests"), list) else 0,
        "playwright_redirect_events": len(sandbox_features.get("redirect_chain", [])) if isinstance(sandbox_features.get("redirect_chain"), list) else 0,
        "cookie_analyzer_invoked": "cookie_risk_score" in cookie_features,
        "phishing_behavior_invoked": bool(behavior_features),
        "fingerprint_beacon_invoked": bool(fpb_features),
        "feature_fusion_invoked": isinstance(fused_features.get("feature_vector"), list),
        "sandbox_error": sandbox_error,
    }


def _safe_user_uuid(request: Request) -> uuid.UUID | None:
    state_user = request.state.user_id if hasattr(request.state, "user_id") else None
    if state_user is None:
        return None

    if isinstance(state_user, uuid.UUID):
        return state_user

    try:
        return uuid.UUID(str(state_user))
    except (ValueError, TypeError):
        return None


@router.post("/analyze", response_model=URLAnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_url(payload: URLAnalyzeRequest, request: Request, db: Session = Depends(get_db)):
    try:
        normalized_url = _normalize_url_input(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    request_row = URLAnalysisRequest(
        user_id=_safe_user_uuid(request),
        source_url=payload.url,
        normalized_url=normalized_url,
        status="processing",
    )
    db.add(request_row)
    db.commit()
    db.refresh(request_row)

    try:
        phase_payload = await extract_phase_4_features_async(normalized_url)
    except Exception as exc:  # noqa: BLE001
        request_row.status = "failed"
        db.add(request_row)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL analysis failed: {exc}",
        ) from exc

    risk_engine, _ = _get_url_risk_engine()
    if risk_engine is not None:
        try:
            prediction = risk_engine.predict_from_phase_payload(phase_payload)
        except Exception:  # noqa: BLE001
            prediction = _fallback_risk_from_payload(phase_payload)
    else:
        prediction = _fallback_risk_from_payload(phase_payload)

    risk = prediction.get("risk") if isinstance(prediction, dict) else {}
    if not isinstance(risk, dict):
        risk = {}

    llm_enhanced = False
    llm_label: str | None = None
    llm_confidence: float | None = None
    llm_explanation: str | None = None
    llm_key_indicators: list[str] = []
    llm_recommendations: list[str] = []

    if payload.with_llm_explanation:
        llm_result = explain_url_with_llm(
            {
                "url": normalized_url,
                "final_url": (phase_payload.get("sandbox_features") or {}).get("final_url", normalized_url),
                "phishing_probability": prediction.get("phishing_probability", 0.0),
                "risk_score": risk.get("risk_score", 0.0),
                "risk_level": risk.get("risk_level", "Low"),
                "url_features": phase_payload.get("url_features", {}),
                "domain_features": phase_payload.get("domain_features", {}),
                "tls_features": phase_payload.get("tls_features", {}),
                "homoglyph_features": phase_payload.get("homoglyph_features", {}),
                "cookie_features": phase_payload.get("cookie_features", {}),
                "phishing_behavior_features": phase_payload.get("phishing_behavior_features", {}),
                "fingerprint_beacon_features": phase_payload.get("fingerprint_beacon_features", {}),
            }
        )

        label_value = str(llm_result.get("final_label") or "").strip()
        llm_label = label_value if label_value else None
        llm_confidence = _to_float(llm_result.get("confidence"), 0.0)
        explanation_value = str(llm_result.get("explanation") or "").strip()
        llm_explanation = explanation_value or None

        indicators = llm_result.get("key_indicators")
        if isinstance(indicators, list):
            llm_key_indicators = [str(item) for item in indicators if str(item).strip()]

        recommendations = llm_result.get("recommendations")
        if isinstance(recommendations, list):
            llm_recommendations = [str(item) for item in recommendations if str(item).strip()]

        llm_enhanced = llm_explanation is not None

    components = risk.get("components") if isinstance(risk, dict) else {}
    if not isinstance(components, dict):
        components = {}

    pipeline_checks = _build_pipeline_checks(phase_payload)

    response_payload = {
        "request_id": str(request_row.id),
        "url": normalized_url,
        "phishing_probability": _to_float(prediction.get("phishing_probability"), 0.0),
        "risk_score": _to_float(risk.get("risk_score"), 0.0),
        "risk_level": str(risk.get("risk_level") or "Low"),
        "model": str(prediction.get("model") or "unknown"),
        "persisted": True,
        "pipeline_checks": pipeline_checks,
        "risk_components": {
            "url_score": _to_float(components.get("url_score"), 0.0),
            "content_score": _to_float(components.get("content_score"), 0.0),
            "cookie_score": _to_float(components.get("cookie_score"), 0.0),
            "infra_score": _to_float(components.get("infra_score"), 0.0),
            "behavior_score": _to_float(components.get("behavior_score"), 0.0),
        },
        "llm_enhanced": llm_enhanced,
        "llm_label": llm_label,
        "llm_confidence": llm_confidence,
        "llm_explanation": llm_explanation,
        "llm_key_indicators": llm_key_indicators,
        "llm_recommendations": llm_recommendations,
        "url_features": phase_payload.get("url_features", {}),
        "domain_features": phase_payload.get("domain_features", {}),
        "tls_features": phase_payload.get("tls_features", {}),
        "homoglyph_features": phase_payload.get("homoglyph_features", {}),
        "sandbox_features": phase_payload.get("sandbox_features", {}),
        "cookie_features": phase_payload.get("cookie_features", {}),
        "phishing_behavior_features": phase_payload.get("phishing_behavior_features", {}),
        "fingerprint_beacon_features": phase_payload.get("fingerprint_beacon_features", {}),
        "fused_features": phase_payload.get("fused_features", {}),
    }

    result_row = URLThreatResult(
        request_id=request_row.id,
        result=json.dumps(response_payload, default=str),
        prediction=json.dumps(prediction, default=str),
        explanation=llm_explanation or str(risk.get("risk_level") or "unknown"),
    )

    request_row.status = "completed"
    db.add(result_row)
    db.add(request_row)
    db.commit()

    return URLAnalyzeResponse(**response_payload)
