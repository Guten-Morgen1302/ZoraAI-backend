from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    EmailFeedbackRequest,
    EmailFeedbackResponse,
    EmailFeedbackRetrainRequest,
    EmailFeedbackRetrainResponse,
    EmailAnalyzeByIdRequest,
    EmailAnalyzeManualRequest,
    LatestEmailFetchRequest,
    LatestEmailFetchResponse,
    LatestEmailAnalyzeRequest,
    LatestEmailAnalyzeResponse,
    SMSAnalyzeRequest,
    SMSAnalyzeResponse,
    SMSFeedbackRequest,
    SMSFeedbackResponse,
    SMSFeedbackRetrainRequest,
    SMSFeedbackRetrainResponse,
    SMSModelPredictRequest,
    SMSModelPredictResponse,
    SMSVectorSearchRequest,
    SMSVectorSearchResponse,
    TextAnalyzeRequest,
    TextAnalyzeResponse,
)
from app.text_analysis.email_analyzer.gmail_client import (
    GmailClientError,
    fetch_email_by_message_id,
    fetch_latest_email,
)
from app.text_analysis.email_analyzer.model_inference import predict_email_text
from app.text_analysis.email_analyzer.similarity import find_similar_email_messages
from app.text_analysis.email_analyzer.stylometry import predict_stylometry_score
from app.text_analysis.email_analyzer.threat_scoring import score_email_threat
from app.text_analysis.email_analyzer.llm_reasoner import explain_email_with_llm
from app.text_analysis.email_preprocessing import preprocess_email_message
from app.text_analysis.embedding_service import find_similar_sms_messages
from app.text_analysis.model_inference import predict_sms_text
from app.text_analysis.repository import PhishingRepository
from app.text_analysis.service import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_TOP_K,
    SMSFraudAnalysisService,
    TextAnalysisService,
)
from app.text_analysis.sms_analyzer.feeback_mechanism.sms_feedback_service import SMSFeedbackService
from app.text_analysis.email_analyzer.feeback_mechanism.email_feedback_service import EmailFeedbackService
from app.auth.security import get_token_subject
from app.models import ApiKey, SmsThreatResult, EmailThreatResult, PhishingRequest as PhishingRequestModel

router = APIRouter(prefix="/text", tags=["text-analysis"])
GMAIL_CLIENT_SECRETS_FILE = Path(__file__).resolve().parent / "email_analyzer" / "gmail_client_secrets.json"
EMAIL_SIMILARITY_TOP_K = 3
EMAIL_SIMILARITY_THRESHOLD = 0.85


def _safe_json_loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _normalize_sms_history_payload(req: PhishingRequestModel, threat: SmsThreatResult) -> SMSAnalyzeResponse:
    parsed_result = _safe_json_loads(threat.result)
    parsed_prediction = _safe_json_loads(threat.prediction)
    parsed_similarity = parsed_result.get("similarity") if isinstance(parsed_result.get("similarity"), dict) else {}

    similarity_payload = {
        "similarity_score": float(parsed_similarity.get("similarity_score") or 0.0),
        "matched_label": parsed_similarity.get("matched_label"),
        "high_risk": bool(parsed_similarity.get("high_risk")),
        "threshold": float(parsed_similarity.get("threshold") or DEFAULT_SIMILARITY_THRESHOLD),
        "top_k": int(parsed_similarity.get("top_k") or DEFAULT_SIMILARITY_TOP_K),
        "matched_text": parsed_similarity.get("matched_text"),
        "matched_source": parsed_similarity.get("matched_source"),
        "top_k_matches": parsed_similarity.get("top_k_matches") if isinstance(parsed_similarity.get("top_k_matches"), list) else [],
    }

    return SMSAnalyzeResponse(
        request_id=req.id,
        risk_score=float(parsed_result.get("risk_score") or 0.0),
        fraud_type=str(parsed_result.get("fraud_type") or "unknown"),
        confidence=float(parsed_result.get("confidence") or 0.0),
        flags=parsed_result.get("flags") if isinstance(parsed_result.get("flags"), list) else [],
        explanation=str(parsed_result.get("explanation") or threat.explanation or "No explanation available."),
        llm_enhanced=bool(parsed_result.get("llm_enhanced")),
        llm_explanation=parsed_result.get("llm_explanation"),
        nlp_score=float(parsed_result.get("nlp_score") or 0.0),
        similarity_score=float(parsed_result.get("similarity_score") or 0.0),
        stylometry_score=float(parsed_result.get("stylometry_score") or 0.0),
        prediction=parsed_prediction,
        similarity=SMSVectorSearchResponse(**similarity_payload),
        url_risk_score=float(parsed_result.get("url_risk_score") or 0.0),
        urgency_score=float(parsed_result.get("urgency_score") or 0.0),
    )


def _normalize_email_history_payload(req: PhishingRequestModel, threat: EmailThreatResult) -> LatestEmailAnalyzeResponse:
    parsed_result = _safe_json_loads(threat.result)
    parsed_prediction = _safe_json_loads(threat.prediction)
    parsed_similarity = parsed_result.get("similarity") if isinstance(parsed_result.get("similarity"), dict) else {}

    sender = str(parsed_result.get("sender") or "")
    subject = str(parsed_result.get("subject") or "")
    body = str(parsed_result.get("body") or "")
    if not sender and req.text:
        # request text format: From: <sender>\nSubject: <subject>\n\n<body>
        lines = req.text.splitlines()
        if lines and lines[0].lower().startswith("from:"):
            sender = lines[0].split(":", 1)[1].strip()
        if len(lines) > 1 and lines[1].lower().startswith("subject:"):
            subject = lines[1].split(":", 1)[1].strip()
        if not body:
            parts = req.text.split("\n\n", 1)
            body = parts[1].strip() if len(parts) > 1 else ""

    return LatestEmailAnalyzeResponse(
        request_id=req.id,
        message_id=str(parsed_result.get("message_id") or str(req.id)),
        thread_id=parsed_result.get("thread_id"),
        sender=sender,
        subject=subject,
        body=body,
        risk_score=float(parsed_result.get("risk_score") or 0.0),
        nlp_score=float(parsed_result.get("nlp_score") or 0.0),
        similarity_score=float(parsed_result.get("similarity_score") or 0.0),
        stylometry_score=float(parsed_result.get("stylometry_score") or 0.0),
        confidence=float(parsed_result.get("confidence") or 0.0),
        fraud_type=str(parsed_result.get("fraud_type") or "unknown"),
        nlp_prediction=parsed_prediction,
        similarity=parsed_similarity,
        llm_enhanced=bool(parsed_result.get("llm_enhanced")),
        llm_explanation=parsed_result.get("llm_explanation"),
        llm_label=parsed_result.get("llm_label"),
        llm_confidence=float(parsed_result.get("llm_confidence")) if parsed_result.get("llm_confidence") is not None else None,
    )


@router.get("/sms/history", status_code=status.HTTP_200_OK)
def get_sms_history(request: Request, db: Session = Depends(get_db)):
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

    query = (
        db.query(PhishingRequestModel, SmsThreatResult)
        .outerjoin(SmsThreatResult, SmsThreatResult.request_id == PhishingRequestModel.id)
        .filter(PhishingRequestModel.source == "sms")
    )
    if user_id:
        query = query.filter(PhishingRequestModel.user_id == user_id)

    rows = query.order_by(PhishingRequestModel.created_at.desc()).limit(20).all()

    history = []
    for req, threat in rows:
        risk_score = None
        fraud_type = None
        if threat and threat.result:
            try:
                parsed = json.loads(threat.result)
                risk_score = parsed.get("risk_score")
                fraud_type = parsed.get("fraud_type")
            except Exception:
                pass
        history.append({
            "request_id": str(req.id),
            "text": (req.text or "")[:120],
            "created_at": req.created_at.isoformat() if req.created_at else None,
            "risk_score": risk_score,
            "fraud_type": fraud_type,
        })
    return history


@router.get("/sms/history/{request_id}", response_model=SMSAnalyzeResponse, status_code=status.HTTP_200_OK)
def get_sms_history_detail(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

    try:
        request_uuid = UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = (
        db.query(PhishingRequestModel, SmsThreatResult)
        .join(SmsThreatResult, SmsThreatResult.request_id == PhishingRequestModel.id)
        .filter(PhishingRequestModel.id == request_uuid)
        .filter(PhishingRequestModel.source == "sms")
    )
    if user_id:
        query = query.filter(PhishingRequestModel.user_id == user_id)

    row = query.first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SMS analysis not found")

    req, threat = row
    return _normalize_sms_history_payload(req, threat)


@router.delete("/sms/history/{request_id}", status_code=status.HTTP_200_OK)
def delete_sms_history_item(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

    try:
        request_uuid = UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = db.query(PhishingRequestModel).filter(
        PhishingRequestModel.id == request_uuid,
        PhishingRequestModel.source == "sms",
    )
    if user_id:
        query = query.filter(PhishingRequestModel.user_id == user_id)

    request_row = query.first()
    if not request_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SMS analysis not found")

    db.query(SmsThreatResult).filter(SmsThreatResult.request_id == request_uuid).delete(synchronize_session=False)
    db.query(PhishingRequestModel).filter(PhishingRequestModel.id == request_uuid).delete(synchronize_session=False)
    db.commit()
    return {"status": "deleted", "request_id": request_id}


@router.delete("/sms/history", status_code=status.HTTP_200_OK)
def clear_sms_history(request: Request, db: Session = Depends(get_db)):
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

    base_query = db.query(PhishingRequestModel.id).filter(PhishingRequestModel.source == "sms")
    if user_id:
        base_query = base_query.filter(PhishingRequestModel.user_id == user_id)

    request_ids = [row[0] for row in base_query.all()]
    if not request_ids:
        return {"status": "cleared", "deleted": 0}

    db.query(SmsThreatResult).filter(SmsThreatResult.request_id.in_(request_ids)).delete(synchronize_session=False)
    deleted_count = db.query(PhishingRequestModel).filter(PhishingRequestModel.id.in_(request_ids)).delete(synchronize_session=False)
    db.commit()
    return {"status": "cleared", "deleted": int(deleted_count or 0)}


@router.get("/email/history", status_code=status.HTTP_200_OK)
def get_email_history(request: Request, db: Session = Depends(get_db)):
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

    query = (
        db.query(PhishingRequestModel, EmailThreatResult)
        .outerjoin(EmailThreatResult, EmailThreatResult.request_id == PhishingRequestModel.id)
        .filter(PhishingRequestModel.source == "email")
    )
    if user_id:
        query = query.filter(PhishingRequestModel.user_id == user_id)

    rows = query.order_by(PhishingRequestModel.created_at.desc()).limit(20).all()

    history = []
    for req, threat in rows:
        risk_score = None
        fraud_type = None
        subject = None
        if threat and threat.result:
            try:
                parsed = json.loads(threat.result)
                risk_score = parsed.get("risk_score")
                fraud_type = parsed.get("fraud_type")
                subject = parsed.get("subject")
            except Exception:
                pass
        history.append({
            "request_id": str(req.id),
            "text": (req.text or "")[:120],
            "subject": subject,
            "created_at": req.created_at.isoformat() if req.created_at else None,
            "risk_score": risk_score,
            "fraud_type": fraud_type,
        })
    return history


@router.get("/email/history/{request_id}", response_model=LatestEmailAnalyzeResponse, status_code=status.HTTP_200_OK)
def get_email_history_detail(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

    try:
        request_uuid = UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = (
        db.query(PhishingRequestModel, EmailThreatResult)
        .join(EmailThreatResult, EmailThreatResult.request_id == PhishingRequestModel.id)
        .filter(PhishingRequestModel.id == request_uuid)
        .filter(PhishingRequestModel.source == "email")
    )
    if user_id:
        query = query.filter(PhishingRequestModel.user_id == user_id)

    row = query.first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Email analysis not found")

    req, threat = row
    return _normalize_email_history_payload(req, threat)


@router.delete("/email/history/{request_id}", status_code=status.HTTP_200_OK)
def delete_email_history_item(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

    try:
        request_uuid = UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = db.query(PhishingRequestModel).filter(
        PhishingRequestModel.id == request_uuid,
        PhishingRequestModel.source == "email",
    )
    if user_id:
        query = query.filter(PhishingRequestModel.user_id == user_id)

    request_row = query.first()
    if not request_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Email analysis not found")

    db.query(EmailThreatResult).filter(EmailThreatResult.request_id == request_uuid).delete(synchronize_session=False)
    db.query(PhishingRequestModel).filter(PhishingRequestModel.id == request_uuid).delete(synchronize_session=False)
    db.commit()
    return {"status": "deleted", "request_id": request_id}


@router.delete("/email/history", status_code=status.HTTP_200_OK)
def clear_email_history(request: Request, db: Session = Depends(get_db)):
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

    base_query = db.query(PhishingRequestModel.id).filter(PhishingRequestModel.source == "email")
    if user_id:
        base_query = base_query.filter(PhishingRequestModel.user_id == user_id)

    request_ids = [row[0] for row in base_query.all()]
    if not request_ids:
        return {"status": "cleared", "deleted": 0}

    db.query(EmailThreatResult).filter(EmailThreatResult.request_id.in_(request_ids)).delete(synchronize_session=False)
    deleted_count = db.query(PhishingRequestModel).filter(PhishingRequestModel.id.in_(request_ids)).delete(synchronize_session=False)
    db.commit()
    return {"status": "cleared", "deleted": int(deleted_count or 0)}


def _truncate_body_preview(body: str, max_chars: int = 180) -> str:
    snippet = (body or "").strip()
    if not snippet:
        return "..."
    return f"{snippet[:max_chars]}..."


def _run_email_full_analysis(
    *,
    db: Session,
    message_id: str,
    thread_id: str | None,
    sender: str,
    subject: str,
    body: str,
    with_llm_explanation: bool = False,
    user_id: str | None = None,
):
    repository = PhishingRepository(db)

    preprocessing = preprocess_email_message(sender=sender, subject=subject, body=body)
    model_input = str(preprocessing.get("normalized_text") or "")

    nlp_prediction = predict_email_text(model_input)
    similarity = find_similar_email_messages(
        text=model_input,
        top_k=EMAIL_SIMILARITY_TOP_K,
        threshold=EMAIL_SIMILARITY_THRESHOLD,
    )
    stylometry = predict_stylometry_score(model_input)

    scoring = score_email_threat(
        nlp_label=nlp_prediction.get("label"),
        nlp_confidence=float(nlp_prediction.get("confidence") or 0.0),
        similarity_score=float(similarity.get("similarity_score") or 0.0),
        stylometry_score=float(stylometry.get("stylometry_score") or 0.0),
    )

    llm_enhanced = False
    llm_explanation: str | None = None
    llm_label: str | None = None
    llm_confidence: float | None = None

    if with_llm_explanation:
        llm_result = explain_email_with_llm(
            {
                "sender": sender,
                "subject": subject,
                "body": body,
                "nlp_label": nlp_prediction.get("label"),
                "nlp_score": scoring.nlp_score,
                "similarity_score": scoring.similarity_score,
                "stylometry_score": scoring.stylometry_score,
                "risk_score": scoring.final_score,
            }
        )
        llm_label = str(llm_result.get("final_label") or "unknown")
        llm_confidence = float(llm_result.get("confidence") or 0.0)
        llm_explanation = str(llm_result.get("explanation") or "").strip() or None
        llm_enhanced = llm_explanation is not None

    request_text = f"From: {sender}\nSubject: {subject}\n\n{body}".strip()
    phishing_request = repository.create_request(
        text=request_text,
        source="email",
        user_id=user_id,
    )

    feature_map = preprocessing.get("features") or {}
    link_count = int(feature_map.get("url_count") or len(preprocessing.get("urls") or []))
    urgency_score = float(feature_map.get("urgency_score") or 0.0)

    repository.create_analysis(
        request_id=phishing_request.id,
        link_count=link_count,
        urgency_score=urgency_score,
        status="completed",
    )

    response_payload = {
        "request_id": str(phishing_request.id),
        "message_id": message_id,
        "thread_id": thread_id,
        "sender": sender,
        "subject": subject,
        "body": _truncate_body_preview(body),
        "risk_score": scoring.final_score,
        "nlp_score": scoring.nlp_score,
        "similarity_score": scoring.similarity_score,
        "stylometry_score": scoring.stylometry_score,
        "confidence": scoring.confidence,
        "fraud_type": scoring.fraud_type,
        "llm_enhanced": llm_enhanced,
        "llm_explanation": llm_explanation,
        "llm_label": llm_label,
        "llm_confidence": llm_confidence,
    }

    repository.create_email_threat_result(
        request_id=phishing_request.id,
        result=json.dumps(response_payload),
        prediction=json.dumps(nlp_prediction),
        explanation=llm_explanation or scoring.fraud_type,
    )
    db.commit()

    return LatestEmailAnalyzeResponse(
        request_id=phishing_request.id,
        message_id=message_id,
        thread_id=thread_id,
        sender=sender,
        subject=subject,
        body=_truncate_body_preview(body),
        risk_score=scoring.final_score,
        nlp_score=scoring.nlp_score,
        similarity_score=scoring.similarity_score,
        stylometry_score=scoring.stylometry_score,
        confidence=scoring.confidence,
        fraud_type=scoring.fraud_type,
        nlp_prediction=nlp_prediction,
        similarity=similarity,
        llm_enhanced=llm_enhanced,
        llm_explanation=llm_explanation,
        llm_label=llm_label,
        llm_confidence=llm_confidence,
    )


def _resolve_user_from_cookies(request: Request) -> str:
    existing_user_id = request.state.user_id if hasattr(request.state, "user_id") else None
    if existing_user_id:
        return str(existing_user_id)

    access_token = request.cookies.get("access_token")
    if not access_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication cookie missing")

    user_id = get_token_subject(access_token, expected_type="access")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")

    request.state.user_id = user_id
    return str(user_id)


def _resolve_user_id_for_api_or_cookie(request: Request, db: Session) -> str | None:
    existing_user_id = request.state.user_id if hasattr(request.state, "user_id") else None
    if existing_user_id:
        return str(existing_user_id)

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")

        key_record = (
            db.query(ApiKey)
            .filter(ApiKey.api_key == token, ApiKey.is_active.is_(True), ApiKey.revoked_at.is_(None))
            .first()
        )
        if not key_record or key_record.expires_at <= datetime.utcnow():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired API key")

        request.state.user_id = str(key_record.user_id)
        return str(key_record.user_id)

    access_token = request.cookies.get("access_token")
    if access_token:
        user_id = get_token_subject(access_token, expected_type="access")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
        request.state.user_id = user_id
        return str(user_id)

    return None

## API route for analyzing text

@router.post("/analyze", response_model=TextAnalyzeResponse, status_code=status.HTTP_202_ACCEPTED)
def analyze_text(payload: TextAnalyzeRequest, request: Request, db: Session = Depends(get_db)):
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

    service = TextAnalysisService(db)
    result = service.analyze(text=payload.text, source=payload.source.value, user_id=user_id)

    return TextAnalyzeResponse(
        request_id=result.request_id,
        links_detected=result.links_detected,
        urgent_language=result.urgent_language,
        status=result.status,
    )


@router.post("/model/predict", response_model=SMSModelPredictResponse, status_code=status.HTTP_200_OK)
def predict_sms_model(payload: SMSModelPredictRequest):
    try:
        prediction = predict_sms_text(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return SMSModelPredictResponse(prediction=prediction)


@router.post("/sms/similarity", response_model=SMSVectorSearchResponse, status_code=status.HTTP_200_OK)
def similarity_search_sms(payload: SMSVectorSearchRequest):
    try:
        result = find_similar_sms_messages(
            text=payload.text,
            top_k=DEFAULT_SIMILARITY_TOP_K,
            threshold=DEFAULT_SIMILARITY_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return SMSVectorSearchResponse(**result)


@router.post("/sms/analyze", response_model=SMSAnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze_sms(payload: SMSAnalyzeRequest, request: Request, db: Session = Depends(get_db)):
    user_id = _resolve_user_id_for_api_or_cookie(request, db)

    service = SMSFraudAnalysisService(db)
    try:
        result = service.analyze_sms(
            text=payload.text,
            include_llm_explanation=payload.include_llm_explanation,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return SMSAnalyzeResponse(
        request_id=result.request_id,
        risk_score=result.risk_score,
        fraud_type=result.fraud_type,
        confidence=result.confidence,
        flags=result.flags,
        explanation=result.explanation,
        llm_enhanced=result.llm_enhanced,
        llm_explanation=result.llm_explanation,
        nlp_score=result.nlp_score,
        similarity_score=result.similarity_score,
        stylometry_score=result.stylometry_score,
        prediction=result.prediction,
        similarity=SMSVectorSearchResponse(**result.similarity),
        url_risk_score=result.url_risk_score,
        urgency_score=result.urgency_score,
    )


@router.post("/sms/feedback", response_model=SMSFeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_sms_feedback(payload: SMSFeedbackRequest, db: Session = Depends(get_db)):
    service = SMSFeedbackService(db)
    try:
        result = service.submit_feedback(
            analysis_id=payload.analysis_id,
            source=payload.source,
            human_label=payload.human_label,
            model_prediction=payload.model_prediction,
            model_confidence=payload.model_confidence,
            feedback_type=payload.feedback_type,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return SMSFeedbackResponse(
        id=result.id,
        analysis_id=result.analysis_id,
        input_hash=result.input_hash,
        status="stored",
        created_at=result.created_at,
    )


@router.post("/sms/feedback/retrain", response_model=SMSFeedbackRetrainResponse, status_code=status.HTTP_200_OK)
def retrain_from_sms_feedback(payload: SMSFeedbackRetrainRequest, db: Session = Depends(get_db)):
    service = SMSFeedbackService(db)
    try:
        result = service.export_retraining_dataset_and_upsert(
            max_records=payload.max_records,
            namespace=payload.namespace,
            batch_size=payload.batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return SMSFeedbackRetrainResponse(
        status="completed",
        candidate_feedback=result.candidate_feedback,
        exported_rows=result.exported_rows,
        csv_path=result.csv_path,
        namespace=result.namespace,
        vectors_inserted=result.vectors_inserted,
        vectors_skipped=result.vectors_skipped,
    )


@router.post("/email/feeback", response_model=EmailFeedbackResponse, status_code=status.HTTP_201_CREATED)
@router.post("/email/feedback", response_model=EmailFeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_email_feedback(payload: EmailFeedbackRequest, db: Session = Depends(get_db)):
    service = EmailFeedbackService(db)
    try:
        result = service.submit_feedback(
            analysis_id=payload.analysis_id,
            source=payload.source,
            human_label=payload.human_label,
            model_prediction=payload.model_prediction,
            model_confidence=payload.model_confidence,
            feedback_type=payload.feedback_type,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return EmailFeedbackResponse(
        id=result.id,
        analysis_id=result.analysis_id,
        input_hash=result.input_hash,
        status="stored",
        created_at=result.created_at,
    )


@router.post("/email/retrain/feedback", response_model=EmailFeedbackRetrainResponse, status_code=status.HTTP_200_OK)
def retrain_from_email_feedback(payload: EmailFeedbackRetrainRequest, db: Session = Depends(get_db)):
    service = EmailFeedbackService(db)
    try:
        result = service.export_retraining_dataset_and_upsert(
            max_records=payload.max_records,
            namespace=payload.namespace,
            batch_size=payload.batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return EmailFeedbackRetrainResponse(
        status="completed",
        candidate_feedback=result.candidate_feedback,
        exported_rows=result.exported_rows,
        csv_path=result.csv_path,
        namespace=result.namespace,
        vectors_inserted=result.vectors_inserted,
        vectors_skipped=result.vectors_skipped,
    )


@router.post("/email/latest", response_model=LatestEmailFetchResponse, status_code=status.HTTP_200_OK)
def fetch_and_preprocess_latest_email(payload: LatestEmailFetchRequest | None = None):
    try:
        latest_email = fetch_latest_email(
            client_secrets_path=GMAIL_CLIENT_SECRETS_FILE,
            query=(payload.query if payload else None),
        )
    except GmailClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch latest email: {exc}",
        ) from exc

    sender = str(latest_email.get("sender") or "")
    subject = str(latest_email.get("subject") or "")
    body = str(latest_email.get("body") or "")

    preprocessing = preprocess_email_message(sender=sender, subject=subject, body=body)

    return LatestEmailFetchResponse(
        message_id=str(latest_email.get("message_id") or ""),
        thread_id=latest_email.get("thread_id"),
        sender=sender,
        subject=subject,
        body=body,
        preprocessing=preprocessing,
    )


@router.post("/email/analyze/latest", response_model=LatestEmailAnalyzeResponse, status_code=status.HTTP_200_OK)
def fetch_latest_email_and_analyze(payload: LatestEmailAnalyzeRequest, request: Request, db: Session = Depends(get_db)):
    try:
        latest_email = fetch_latest_email(
            client_secrets_path=GMAIL_CLIENT_SECRETS_FILE,
            query=payload.query,
            force_reauth=payload.force_reauth,
        )
    except GmailClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch latest email for analysis: {exc}",
        ) from exc

    sender = str(latest_email.get("sender") or "")
    subject = str(latest_email.get("subject") or "")
    body = str(latest_email.get("body") or "")

    user_id = request.state.user_id if hasattr(request.state, "user_id") else None
    try:
        return _run_email_full_analysis(
            db=db,
            message_id=str(latest_email.get("message_id") or ""),
            thread_id=latest_email.get("thread_id"),
            sender=sender,
            subject=subject,
            body=body,
            with_llm_explanation=payload.with_llm_explanation,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, GmailClientError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/email/analyze/by-id", response_model=LatestEmailAnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze_email_by_ids(payload: EmailAnalyzeByIdRequest, request: Request, db: Session = Depends(get_db)):
    try:
        email_data = fetch_email_by_message_id(
            client_secrets_path=GMAIL_CLIENT_SECRETS_FILE,
            message_id=payload.message_id,
            thread_id=payload.thread_id,
            force_reauth=payload.force_reauth,
        )
    except GmailClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch email by message/thread id: {exc}",
        ) from exc

    try:
        user_id = request.state.user_id if hasattr(request.state, "user_id") else None
        return _run_email_full_analysis(
            db=db,
            message_id=str(email_data.get("message_id") or ""),
            thread_id=email_data.get("thread_id"),
            sender=str(email_data.get("sender") or ""),
            subject=str(email_data.get("subject") or ""),
            body=str(email_data.get("body") or ""),
            with_llm_explanation=payload.with_llm_explanation,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, GmailClientError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/email/analyze/extension", response_model=LatestEmailAnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze_email_manual(payload: EmailAnalyzeManualRequest, request: Request, db: Session = Depends(get_db)):
    try:
        user_id = request.state.user_id if hasattr(request.state, "user_id") else None
        return _run_email_full_analysis(
            db=db,
            message_id="manual-input",
            thread_id="manual-input",
            sender=payload.sender,
            subject=payload.subject,
            body=payload.body,
            with_llm_explanation=payload.with_llm_explanation,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, GmailClientError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/email/analyze", response_model=LatestEmailAnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze_email_manual_with_user_cookie(
    payload: EmailAnalyzeManualRequest,
    request: Request,
    user_id: str = Depends(_resolve_user_from_cookies),
    db: Session = Depends(get_db),
):
    try:
        return _run_email_full_analysis(
            db=db,
            message_id="manual-input",
            thread_id="manual-input",
            sender=payload.sender,
            subject=payload.subject,
            body=payload.body,
            with_llm_explanation=payload.with_llm_explanation,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, GmailClientError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
