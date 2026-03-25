from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import Lock
from typing import Any

from app.fraud_memory.embedding_service import FraudMemoryEmbeddingService, get_embedding_service


logger = logging.getLogger("zora.text_analysis.email_similarity")


@dataclass
class EmailVectorSimilarityResult:
    similarity_score: float
    matched_label: str | None
    high_risk: bool
    threshold: float
    top_k: int
    matched_text: str | None
    matched_source: str | None
    top_k_matches: list[dict[str, Any]]


class EmailVectorSimilarityService:
    """Service for email embedding generation and Pinecone similarity search."""

    def __init__(self, embedding_service: FraudMemoryEmbeddingService | None = None):
        self.embedding_service = embedding_service or get_embedding_service(namespace="fraud_emails")

    def find_similar_messages(self, text: str, top_k: int = 5, threshold: float = 0.85) -> EmailVectorSimilarityResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be between 0 and 1")

        logger.info("Running email vector similarity in Pinecone", extra={"namespace": "fraud_emails", "top_k": top_k})
        matches = self.embedding_service.search_similar(text=text.strip(), limit=top_k)

        best_match = matches[0] if matches else {}
        best_similarity = float(best_match.get("similarity") or 0.0)
        matched_label = best_match.get("label") or best_match.get("fraud_label")

        return EmailVectorSimilarityResult(
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
_cached_email_similarity_service: EmailVectorSimilarityService | None = None


def _get_email_similarity_service() -> EmailVectorSimilarityService:
    global _cached_email_similarity_service
    if _cached_email_similarity_service is not None:
        return _cached_email_similarity_service

    with _service_lock:
        if _cached_email_similarity_service is None:
            _cached_email_similarity_service = EmailVectorSimilarityService()

    return _cached_email_similarity_service


def find_similar_email_messages(text: str, top_k: int = 5, threshold: float = 0.85) -> dict[str, Any]:
    """Public helper used by API layer for email similarity lookups in Pinecone."""
    try:
        service = _get_email_similarity_service()
        result = service.find_similar_messages(text=text, top_k=top_k, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Email vector similarity unavailable; returning safe fallback: %s", exc)
        result = EmailVectorSimilarityResult(
            similarity_score=0.0,
            matched_label=None,
            high_risk=False,
            threshold=threshold,
            top_k=top_k,
            matched_text=None,
            matched_source=None,
            top_k_matches=[],
        )

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
