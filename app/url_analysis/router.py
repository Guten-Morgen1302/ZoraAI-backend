from __future__ import annotations

import json
import logging
import uuid
from urllib.parse import urlparse

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import URLAnalysisRequest, URLFeedback, URLThreatResult
from app.schemas import URLAnalyzeRequest, URLAnalyzeResponse, URLFeedbackRequest, URLFeedbackResponse
from app.url_analysis.llm_reasoner import explain_url_with_llm
from app.url_analysis.ml_risk_engine import URLMLRiskEngine
from app.url_analysis.url_analysis import extract_phase_4_features_async

router = APIRouter(prefix="/url", tags=["url-analysis"])

logger = logging.getLogger("zora.url_analysis.router")

SHORT_URL_TIMEOUT_SEC = 6.0
SHORT_URL_MAX_REDIRECTS = 8
SHORT_URL_PROVIDERS: dict[str, str] = {
    "tinyurl.com": "tinyurl",
    "bit.ly": "bitly",
    "t.co": "twitter",
    "ow.ly": "hootsuite",
    "is.gd": "isgd",
    "tiny.cc": "tinycc",
    "cutt.ly": "cuttly",
    "buff.ly": "buffer",
    "rebrand.ly": "rebrandly",
    "shorturl.at": "shorturlat",
    "soo.gd": "soogd",
    "s2r.co": "s2r",
}

URL_RISK_ENGINE = URLMLRiskEngine()
URL_RISK_ENGINE_READY = False
URL_RISK_ENGINE_ERROR: str | None = None


def _safe_json_loads(raw: str | None) -> dict[str, object]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


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


def _normalized_hostname(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").strip().lower().rstrip(".")


def _short_url_provider(url: str) -> str | None:
    host = _normalized_hostname(url)
    if not host:
        return None

    if host in SHORT_URL_PROVIDERS:
        return SHORT_URL_PROVIDERS[host]

    for provider_host, provider_name in SHORT_URL_PROVIDERS.items():
        if host.endswith(f".{provider_host}"):
            return provider_name
    return None


def _dedupe_urls(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        candidate = str(value or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        output.append(candidate)
    return output


def _clean_resolution_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    if len(message) > 180:
        return f"{message[:180]}..."
    return message


async def _resolve_short_url(url: str) -> dict[str, object]:
    provider = _short_url_provider(url)
    result: dict[str, object] = {
        "input_url": url,
        "analysis_url": url,
        "provider": provider,
        "is_short_url": provider is not None,
        "expanded": False,
        "redirect_chain": [url],
        "error": "",
    }

    if provider is None:
        return result

    timeout = aiohttp.ClientTimeout(total=SHORT_URL_TIMEOUT_SEC, connect=2.5)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }

    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers, trust_env=True) as session:
            async with session.get(url, allow_redirects=True, max_redirects=SHORT_URL_MAX_REDIRECTS) as response:
                history_chain = [str(item.url) for item in response.history]
                final_url = str(response.url or "").strip()

        redirect_chain = _dedupe_urls([url] + history_chain + ([final_url] if final_url else []))
        result["redirect_chain"] = redirect_chain

        if not final_url or final_url == url:
            return result

        parsed_final = urlparse(final_url)
        if parsed_final.scheme not in {"http", "https"} or not parsed_final.netloc:
            result["error"] = "Resolved short URL is not a valid HTTP(S) destination"
            return result

        result["analysis_url"] = final_url
        result["expanded"] = True
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = _clean_resolution_error(exc)
        logger.warning("short_url_resolution_failed provider=%s url=%s error=%s", provider, url, result["error"])
        return result


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


def _build_pipeline_checks(
    phase_payload: dict[str, object], url_resolution: dict[str, object] | None = None
) -> dict[str, object]:
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
    whois_registrar = str(domain_features.get("registrar") or "").strip()
    whois_private_present = "is_whois_private" in domain_features
    resolution = url_resolution if isinstance(url_resolution, dict) else {}

    resolution_redirect_chain = resolution.get("redirect_chain")
    if not isinstance(resolution_redirect_chain, list):
        resolution_redirect_chain = []

    short_url_provider = str(resolution.get("provider") or "").strip()
    short_url_error = str(resolution.get("error") or "").strip()

    return {
        "phase_1_static_extracted": bool(url_features) and bool(domain_features),
        "whois_extractor_invoked": "registrar" in domain_features and whois_private_present,
        "whois_has_live_data": bool(whois_registrar),
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
        "short_url_detected": bool(resolution.get("is_short_url")),
        "short_url_provider": short_url_provider,
        "short_url_expanded": bool(resolution.get("expanded")),
        "short_url_redirect_hops": max(0, len(resolution_redirect_chain) - 1),
        "short_url_resolution_error": short_url_error,
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


def _sandbox_features_for_response(sandbox_features: object) -> dict[str, object]:
    """Return sandbox features with heavy HTML removed from API response payload."""
    if not isinstance(sandbox_features, dict):
        return {}

    sanitized = dict(sandbox_features)
    sanitized.pop("raw_html", None)
    return sanitized


@router.post("/feedback", response_model=URLFeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_url_feedback(payload: URLFeedbackRequest, request: Request, db: Session = Depends(get_db)):
    try:
        request_uuid = uuid.UUID(payload.analysis_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid analysis_id") from exc

    analysis_exists = db.query(URLAnalysisRequest.id).filter(URLAnalysisRequest.id == request_uuid).first()
    if not analysis_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="URL analysis not found")

    feedback_row = URLFeedback(
        analysis_id=str(request_uuid),
        user_id=_safe_user_uuid(request),
        normalized_url=payload.normalized_url,
        model_prediction=payload.model_prediction,
        model_risk_score=payload.model_risk_score,
        model_phishing_probability=payload.model_phishing_probability,
        human_label=payload.human_label.value,
        prediction_type=payload.prediction_type.value,
        notes=payload.notes.strip() if payload.notes else None,
    )

    db.add(feedback_row)
    db.commit()
    db.refresh(feedback_row)

    return URLFeedbackResponse(
        id=feedback_row.id,
        analysis_id=feedback_row.analysis_id,
        status="stored",
        created_at=feedback_row.created_at.isoformat(),
    )


@router.get("/history", status_code=status.HTTP_200_OK)
def get_url_history(request: Request, db: Session = Depends(get_db)):
    user_id = _safe_user_uuid(request)

    query = (
        db.query(URLAnalysisRequest, URLThreatResult)
        .outerjoin(URLThreatResult, URLThreatResult.request_id == URLAnalysisRequest.id)
    )
    if user_id is not None:
        query = query.filter(URLAnalysisRequest.user_id == user_id)

    rows = query.order_by(URLAnalysisRequest.created_at.desc()).limit(20).all()

    history: list[dict[str, object]] = []
    for req_row, result_row in rows:
        parsed_result = _safe_json_loads(result_row.result) if result_row and result_row.result else {}
        history.append(
            {
                "request_id": str(req_row.id),
                "url": req_row.normalized_url,
                "created_at": req_row.created_at.isoformat() if req_row.created_at else None,
                "status": req_row.status,
                "risk_score": _to_float(parsed_result.get("risk_score"), 0.0) if parsed_result else None,
                "risk_level": str(parsed_result.get("risk_level") or "") if parsed_result else None,
                "phishing_probability": _to_float(parsed_result.get("phishing_probability"), 0.0) if parsed_result else None,
            }
        )
    return history


@router.get("/history/{request_id}", response_model=URLAnalyzeResponse, status_code=status.HTTP_200_OK)
def get_url_history_detail(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = _safe_user_uuid(request)

    try:
        request_uuid = uuid.UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = (
        db.query(URLAnalysisRequest, URLThreatResult)
        .join(URLThreatResult, URLThreatResult.request_id == URLAnalysisRequest.id)
        .filter(URLAnalysisRequest.id == request_uuid)
    )
    if user_id is not None:
        query = query.filter(URLAnalysisRequest.user_id == user_id)

    row = query.first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="URL analysis not found")

    _, result_row = row
    parsed_result = _safe_json_loads(result_row.result)
    if not parsed_result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="URL analysis payload missing")

    return URLAnalyzeResponse(**parsed_result)


@router.delete("/history/{request_id}", status_code=status.HTTP_200_OK)
def delete_url_history_item(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = _safe_user_uuid(request)

    try:
        request_uuid = uuid.UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = db.query(URLAnalysisRequest).filter(URLAnalysisRequest.id == request_uuid)
    if user_id is not None:
        query = query.filter(URLAnalysisRequest.user_id == user_id)

    request_row = query.first()
    if not request_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="URL analysis not found")

    db.query(URLThreatResult).filter(URLThreatResult.request_id == request_uuid).delete(synchronize_session=False)
    db.query(URLAnalysisRequest).filter(URLAnalysisRequest.id == request_uuid).delete(synchronize_session=False)
    db.commit()
    return {"status": "deleted", "request_id": request_id}


@router.delete("/history", status_code=status.HTTP_200_OK)
def clear_url_history(request: Request, db: Session = Depends(get_db)):
    user_id = _safe_user_uuid(request)

    base_query = db.query(URLAnalysisRequest.id)
    if user_id is not None:
        base_query = base_query.filter(URLAnalysisRequest.user_id == user_id)

    request_ids = [row[0] for row in base_query.all()]
    if not request_ids:
        return {"status": "cleared", "deleted": 0}

    db.query(URLThreatResult).filter(URLThreatResult.request_id.in_(request_ids)).delete(synchronize_session=False)
    deleted_count = db.query(URLAnalysisRequest).filter(URLAnalysisRequest.id.in_(request_ids)).delete(synchronize_session=False)
    db.commit()
    return {"status": "cleared", "deleted": int(deleted_count or 0)}


@router.post("/analyze", response_model=URLAnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_url(payload: URLAnalyzeRequest, request: Request, db: Session = Depends(get_db)):
    try:
        normalized_input_url = _normalize_url_input(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    url_resolution = await _resolve_short_url(normalized_input_url)
    analysis_url = str(url_resolution.get("analysis_url") or normalized_input_url)

    request_row = URLAnalysisRequest(
        user_id=_safe_user_uuid(request),
        source_url=payload.url,
        normalized_url=analysis_url,
        status="processing",
    )
    db.add(request_row)
    db.commit()
    db.refresh(request_row)

    try:
        phase_payload = await extract_phase_4_features_async(analysis_url)
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
                "url": analysis_url,
                "source_url": normalized_input_url,
                "final_url": (phase_payload.get("sandbox_features") or {}).get("final_url", analysis_url),
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

    pipeline_checks = _build_pipeline_checks(phase_payload, url_resolution=url_resolution)
    sandbox_features_response = _sandbox_features_for_response(phase_payload.get("sandbox_features", {}))

    response_payload = {
        "request_id": str(request_row.id),
        "url": analysis_url,
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
        "sandbox_features": sandbox_features_response,
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
