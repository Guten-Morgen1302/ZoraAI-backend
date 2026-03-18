from __future__ import annotations

from dataclasses import dataclass
import json

from sqlalchemy.orm import Session

from app.text_analysis.embedding_service import find_similar_sms_messages
from app.text_analysis.model_inference import predict_sms_text
from app.text_analysis.pipeline import TextPreprocessingPipeline
from app.text_analysis.repository import PhishingRepository
from app.text_analysis.sms_analyzer.stylometry import predict_stylometry_score
from app.text_analysis.threat_scoring import score_sms_threat


@dataclass
class AnalyzeTextResult:
    request_id: str
    links_detected: int
    urgent_language: bool
    status: str


@dataclass
class SMSAnalyzeResult:
    request_id: str
    risk_score: float
    fraud_type: str
    confidence: float
    flags: list[str]
    explanation: str
    nlp_score: float
    similarity_score: float
    stylometry_score: float
    prediction: dict
    similarity: dict
    url_risk_score: float
    urgency_score: float


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


class SMSFraudAnalysisService:
    """Unified SMS fraud analysis orchestrator for preprocessing, models, and persistence."""

    def __init__(self, db: Session, pipeline: TextPreprocessingPipeline | None = None):
        self.repository = PhishingRepository(db)
        self.db = db
        self.pipeline = pipeline or TextPreprocessingPipeline()

    @staticmethod
    def _fallback_stylometry_score(urgency_score: float, url_risk_score: float) -> dict[str, float]:
        # Fallback if the RandomForest artifact is not present yet.
        fallback = min(1.0, (0.7 * urgency_score) + (0.3 * url_risk_score))
        return {"stylometry_score": round(fallback, 4)}

    def analyze_sms(
        self,
        text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.85,
        user_id: str | None = None,
    ) -> SMSAnalyzeResult:
        phishing_request = self.repository.create_request(text=text, source="sms", user_id=user_id)

        preprocess_result = self.pipeline.run(text)
        features = preprocess_result.features
        cleaned_text = str(features.get("clean_text") or text)
        url_risk_score = float(features.get("url_risk_score") or 0.0)
        urgency_score = float(features.get("urgency_score") or 0.0)

        prediction = predict_sms_text(cleaned_text)

        try:
            stylometry = predict_stylometry_score(text)
        except (FileNotFoundError, RuntimeError, ValueError):
            stylometry = self._fallback_stylometry_score(
                urgency_score=urgency_score,
                url_risk_score=url_risk_score,
            )

        similarity = find_similar_sms_messages(
            text=cleaned_text,
            top_k=top_k,
            threshold=similarity_threshold,
        )

        scoring = score_sms_threat(
            nlp_label=prediction.get("label"),
            nlp_confidence=float(prediction.get("confidence") or 0.0),
            similarity_score=float(similarity.get("similarity_score") or 0.0),
            stylometry_score=float(stylometry.get("stylometry_score") or 0.0),
            url_risk_score=url_risk_score,
            urgency_score=urgency_score,
            matched_label=similarity.get("matched_label"),
            similarity_high_risk=bool(similarity.get("high_risk")),
        )

        self.repository.create_analysis(
            request_id=phishing_request.id,
            link_count=preprocess_result.link_count,
            urgency_score=preprocess_result.urgency_score,
            status="completed",
        )

        result_payload = {
            "risk_score": scoring.risk_score,
            "fraud_type": scoring.fraud_type,
            "confidence": scoring.confidence,
            "flags": scoring.flags,
            "explanation": scoring.explanation,
            "nlp_score": scoring.nlp_score,
            "similarity_score": scoring.similarity_score,
            "stylometry_score": scoring.stylometry_score,
            "url_risk_score": round(url_risk_score, 4),
            "urgency_score": round(urgency_score, 4),
            "similarity": similarity,
        }

        self.repository.create_sms_threat_result(
            request_id=phishing_request.id,
            result=json.dumps(result_payload),
            prediction=json.dumps(prediction),
            explanation=scoring.explanation,
        )
        self.db.commit()

        return SMSAnalyzeResult(
            request_id=str(phishing_request.id),
            risk_score=scoring.risk_score,
            fraud_type=scoring.fraud_type,
            confidence=scoring.confidence,
            flags=scoring.flags,
            explanation=scoring.explanation,
            nlp_score=scoring.nlp_score,
            similarity_score=scoring.similarity_score,
            stylometry_score=scoring.stylometry_score,
            prediction=prediction,
            similarity=similarity,
            url_risk_score=round(url_risk_score, 4),
            urgency_score=round(urgency_score, 4),
        )
