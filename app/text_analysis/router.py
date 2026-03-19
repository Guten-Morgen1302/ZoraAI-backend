from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    SMSAnalyzeRequest,
    SMSAnalyzeResponse,
    SMSModelPredictRequest,
    SMSModelPredictResponse,
    SMSVectorSearchRequest,
    SMSVectorSearchResponse,
    TextAnalyzeRequest,
    TextAnalyzeResponse,
)
from app.text_analysis.embedding_service import find_similar_sms_messages
from app.text_analysis.model_inference import predict_sms_text
from app.text_analysis.service import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_TOP_K,
    SMSFraudAnalysisService,
    TextAnalysisService,
)

router = APIRouter(prefix="/text", tags=["text-analysis"])

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
