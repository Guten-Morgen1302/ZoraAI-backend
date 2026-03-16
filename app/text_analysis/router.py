from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import TextAnalyzeRequest, TextAnalyzeResponse
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
