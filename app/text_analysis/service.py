from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.fraud_memory.embedding_service import get_embedding_service
from app.text_analysis.embedding_service import find_similar_sms_messages
from app.text_analysis.model_inference import predict_sms_text
from app.text_analysis.pipeline import TextPreprocessingPipeline
from app.text_analysis.preprocessing import validate_sms_text_quality
from app.text_analysis.repository import PhishingRepository
from app.text_analysis.sms_analyzer.stylometry import predict_stylometry_score
from app.text_analysis.threat_scoring import score_sms_threat

DEFAULT_SIMILARITY_TOP_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.82

logger = logging.getLogger("zora.text_analysis.service")


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
    llm_enhanced: bool
    llm_explanation: str | None
    nlp_score: float
    similarity_score: float
    stylometry_score: float
    prediction: dict
    similarity: dict
    url_risk_score: float
    urgency_score: float


@dataclass
class SMSFraudFeedbackResult:
    feedback_id: str
    request_id: str | None
    vector_id: str
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

    @staticmethod
    def _derive_rule_flags(cleaned_text: str) -> list[str]:
        text = cleaned_text.lower()
        flags: list[str] = []
        phrase_map = {
            "urgent": "urgent_language",
            "verify now": "verify_now",
            "account suspended": "account_suspended",
            "click": "click_link",
        }
        for phrase, flag in phrase_map.items():
            if phrase in text:
                flags.append(flag)
        return flags

    def analyze_sms(
        self,
        text: str,
        top_k: int = DEFAULT_SIMILARITY_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        include_llm_explanation: bool = False,
        user_id: str | None = None,
    ) -> SMSAnalyzeResult:
        validated_text = validate_sms_text_quality(text)
        phishing_request = self.repository.create_request(text=validated_text, source="sms", user_id=user_id)

        preprocess_result = self.pipeline.run(validated_text)
        features = preprocess_result.features
        cleaned_text = str(features.get("clean_text") or validated_text)
        url_risk_score = float(features.get("url_risk_score") or 0.0)
        url_risk_flags = list(features.get("url_risk_flags") or [])
        urgency_score = float(features.get("urgency_score") or 0.0)
        rule_flags = self._derive_rule_flags(cleaned_text)

        prediction = predict_sms_text(cleaned_text)

        try:
            stylometry = predict_stylometry_score(validated_text)
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
            sms_text=validated_text,
            nlp_label=prediction.get("label"),
            nlp_confidence=float(prediction.get("confidence") or 0.0),
            similarity_score=float(similarity.get("similarity_score") or 0.0),
            stylometry_score=float(stylometry.get("stylometry_score") or 0.0),
            url_risk_score=url_risk_score,
            url_flags=url_risk_flags,
            rule_flags=rule_flags,
            urgency_score=urgency_score,
            matched_label=similarity.get("matched_label"),
            similarity_high_risk=bool(similarity.get("high_risk")),
            force_llm_explanation=include_llm_explanation,
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
            "llm_enhanced": scoring.llm_enhanced,
            "llm_explanation": scoring.llm_explanation,
            "nlp_score": scoring.nlp_score,
            "similarity_score": scoring.similarity_score,
            "stylometry_score": scoring.stylometry_score,
            "url_risk_score": round(url_risk_score, 4),
            "url_risk_flags": url_risk_flags,
            "rule_flags": rule_flags,
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
            llm_enhanced=scoring.llm_enhanced,
            llm_explanation=scoring.llm_explanation,
            nlp_score=scoring.nlp_score,
            similarity_score=scoring.similarity_score,
            stylometry_score=scoring.stylometry_score,
            prediction=prediction,
            similarity=similarity,
            url_risk_score=round(url_risk_score, 4),
            urgency_score=round(urgency_score, 4),
        )


class SMSContinuousLearningService:
    """Stores confirmed fraud labels in PostgreSQL and Pinecone to improve future detection."""

    def __init__(self, db: Session):
        self.repository = PhishingRepository(db)
        self.db = db
        self.embedding_service = get_embedding_service()

    def mark_confirmed_fraud(
        self,
        *,
        request_id: UUID | None,
        text: str | None,
        fraud_label: str,
        source: str,
        user_id,
    ) -> SMSFraudFeedbackResult:
        resolved_request = None
        resolved_text = (text or "").strip()

        if request_id is not None:
            resolved_request = self.repository.get_request_by_id(request_id)
            if resolved_request is None:
                raise ValueError(f"request_id not found: {request_id}")

            if not resolved_text:
                resolved_text = resolved_request.text

        if not resolved_text:
            raise ValueError("Provide either request_id or text for confirmed fraud feedback")

        normalized_label = fraud_label.strip().lower()
        if not normalized_label:
            raise ValueError("fraud_label must not be empty")

        vector_result = self.embedding_service.store_embedding(
            text=resolved_text,
            fraud_label=normalized_label,
        )
        vector_id = vector_result["id"]

        fraud_case = self.repository.create_confirmed_fraud_case(
            request_id=resolved_request.id if resolved_request is not None else None,
            user_id=user_id,
            text=resolved_text,
            fraud_label=normalized_label,
            source=source,
            vector_id=vector_id,
        )
        self.db.commit()

        return SMSFraudFeedbackResult(
            feedback_id=str(fraud_case.id),
            request_id=str(resolved_request.id) if resolved_request is not None else None,
            vector_id=vector_id,
            status="stored",
        )
