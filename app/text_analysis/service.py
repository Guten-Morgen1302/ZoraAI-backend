from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.text_analysis.pipeline import TextPreprocessingPipeline
from app.text_analysis.repository import PhishingRepository


@dataclass
class AnalyzeTextResult:
    request_id: str
    links_detected: int
    urgent_language: bool
    status: str


class TextAnalysisService:
    """Application service orchestrating request persistence and preprocessing."""

    def __init__(self, db: Session, pipeline: TextPreprocessingPipeline | None = None):
        self.repository = PhishingRepository(db)
        self.db = db
        self.pipeline = pipeline or TextPreprocessingPipeline()

    def analyze(self, text: str, source: str, user_id: str | None = None) -> AnalyzeTextResult:
        phishing_request = self.repository.create_request(text=text, source=source, user_id=user_id)

        preprocess_result = self.pipeline.run(text)

        status = "processing"
        self.repository.create_analysis(
            request_id=phishing_request.id,
            link_count=preprocess_result.link_count,
            urgency_score=preprocess_result.urgency_score,
            status=status,
        )
        self.db.commit()

        return AnalyzeTextResult(
            request_id=str(phishing_request.id),
            links_detected=preprocess_result.link_count,
            urgent_language=preprocess_result.urgency_score >= 0.5,
            status=status,
        )
