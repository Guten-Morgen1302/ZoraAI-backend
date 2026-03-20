from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    EmailAnalyzeByIdRequest,
    EmailAnalyzeManualRequest,
    LatestEmailFetchRequest,
    LatestEmailFetchResponse,
    LatestEmailAnalyzeRequest,
    LatestEmailAnalyzeResponse,
    SMSAnalyzeRequest,
    SMSAnalyzeResponse,
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
from app.text_analysis.service import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_TOP_K,
    SMSFraudAnalysisService,
    TextAnalysisService,
)

router = APIRouter(prefix="/text", tags=["text-analysis"])
GMAIL_CLIENT_SECRETS_FILE = Path(__file__).resolve().parent / "email_analyzer" / "gmail_client_secrets.json"
EMAIL_SIMILARITY_TOP_K = 3
EMAIL_SIMILARITY_THRESHOLD = 0.85


def _truncate_body_preview(body: str, max_chars: int = 180) -> str:
    snippet = (body or "").strip()
    if not snippet:
        return "..."
    return f"{snippet[:max_chars]}..."


def _run_email_full_analysis(
    *,
    message_id: str,
    thread_id: str | None,
    sender: str,
    subject: str,
    body: str,
    with_llm_explanation: bool = False,
):
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
        print("[email-debug] LLM explanation requested; calling OpenRouter")
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

    return LatestEmailAnalyzeResponse(
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
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None

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
def fetch_latest_email_and_analyze(payload: LatestEmailAnalyzeRequest):
    print(
        "[email-debug] Starting /text/email/analyze/latest "
        f"query={payload.query!r} force_reauth={payload.force_reauth}"
    )
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

    print(f"[email-debug] Running NLP + similarity + stylometry for message_id={latest_email.get('message_id')}")
    try:
        return _run_email_full_analysis(
            message_id=str(latest_email.get("message_id") or ""),
            thread_id=latest_email.get("thread_id"),
            sender=sender,
            subject=subject,
            body=body,
            with_llm_explanation=payload.with_llm_explanation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, GmailClientError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/email/analyze/by-id", response_model=LatestEmailAnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze_email_by_ids(payload: EmailAnalyzeByIdRequest):
    print(
        "[email-debug] Starting /text/email/analyze/by-id "
        f"message_id={payload.message_id!r} thread_id={payload.thread_id!r}"
    )
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
        return _run_email_full_analysis(
            message_id=str(email_data.get("message_id") or ""),
            thread_id=email_data.get("thread_id"),
            sender=str(email_data.get("sender") or ""),
            subject=str(email_data.get("subject") or ""),
            body=str(email_data.get("body") or ""),
            with_llm_explanation=payload.with_llm_explanation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, GmailClientError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/email/analyze/extension", response_model=LatestEmailAnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze_email_manual(payload: EmailAnalyzeManualRequest):
    print("[email-debug] Starting /text/email/analyze/extension")
    try:
        return _run_email_full_analysis(
            message_id="manual-input",
            thread_id="manual-input",
            sender=payload.sender,
            subject=payload.subject,
            body=payload.body,
            with_llm_explanation=payload.with_llm_explanation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, GmailClientError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
