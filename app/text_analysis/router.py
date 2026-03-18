from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    SMSModelPredictRequest,
    SMSModelPredictResponse,
    TextAnalyzeRequest,
    TextAnalyzeResponse,
)
from app.text_analysis.model_inference import predict_sms_text
from app.text_analysis.service import TextAnalysisService

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
