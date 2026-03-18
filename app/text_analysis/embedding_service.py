from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from app.fraud_memory.embedding_service import FraudMemoryEmbeddingService, get_embedding_service


@dataclass
class SMSVectorSimilarityResult:
    similarity_score: float
    matched_label: str | None
    high_risk: bool
    threshold: float
    top_k: int
    matched_text: str | None
    matched_source: str | None
    top_k_matches: list[dict[str, Any]]


class SMSVectorSimilarityService:
    """Service for SMS embedding generation and Qdrant similarity search."""

    def __init__(self, embedding_service: FraudMemoryEmbeddingService | None = None):
        self.embedding_service = embedding_service or get_embedding_service()

    def find_similar_messages(self, text: str, top_k: int = 5, threshold: float = 0.85) -> SMSVectorSimilarityResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be between 0 and 1")

        matches = self.embedding_service.search_similar(text=text.strip(), limit=top_k)

        best_match = matches[0] if matches else {}
        best_similarity = float(best_match.get("similarity") or 0.0)
        matched_label = best_match.get("label") or best_match.get("fraud_label")

        return SMSVectorSimilarityResult(
            similarity_score=round(best_similarity, 4),
            matched_label=matched_label if isinstance(matched_label, str) else None,
            high_risk=best_similarity >= threshold,
            threshold=threshold,
            top_k=top_k,
            matched_text=best_match.get("text") if isinstance(best_match.get("text"), str) else None,
            matched_source=best_match.get("source") if isinstance(best_match.get("source"), str) else None,
            top_k_matches=matches,
        )


_service_lock = Lock()
_cached_sms_similarity_service: SMSVectorSimilarityService | None = None


def _get_sms_similarity_service() -> SMSVectorSimilarityService:
    global _cached_sms_similarity_service
    if _cached_sms_similarity_service is not None:
        return _cached_sms_similarity_service

    with _service_lock:
        if _cached_sms_similarity_service is None:
            _cached_sms_similarity_service = SMSVectorSimilarityService()

    return _cached_sms_similarity_service


def find_similar_sms_messages(text: str, top_k: int = 5, threshold: float = 0.85) -> dict[str, Any]:
    """Public helper used by the API layer for SMS similarity lookups in Qdrant."""
    service = _get_sms_similarity_service()
    result = service.find_similar_messages(text=text, top_k=top_k, threshold=threshold)
    return {
        "similarity_score": result.similarity_score,
        "matched_label": result.matched_label,
        "high_risk": result.high_risk,
        "threshold": result.threshold,
        "top_k": result.top_k,
        "matched_text": result.matched_text,
        "matched_source": result.matched_source,
        "top_k_matches": result.top_k_matches,
    }
