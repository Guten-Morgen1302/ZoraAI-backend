from __future__ import annotations

import logging
from threading import Lock

from app.fraud_memory.pinecone_client import (
    DEFAULT_NAMESPACE,
    PineconeVectorStore,
    get_pinecone_vector_store,
)

logger = logging.getLogger("zora.fraud_memory.embedding")

MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_VECTOR_SIZE = 384

_model_lock = Lock()
_cached_model = None


def _load_model_once(model_name: str = MODEL_NAME):
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    with _model_lock:
        if _cached_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for FraudMemoryEmbeddingService. "
                    "Install it with 'pip install sentence-transformers'."
                ) from exc

            logger.info("Loading sentence-transformer model '%s'", model_name)
            _cached_model = SentenceTransformer(model_name)
    return _cached_model


class FraudMemoryEmbeddingService:
    """Service for generating embeddings and interacting with Pinecone."""

    def __init__(self, vector_store: PineconeVectorStore):
        self.vector_store = vector_store
        self.model = _load_model_once()

    def generate_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text must not be empty")

        embedding = self.model.encode(text.strip(), normalize_embeddings=True).tolist()

        if len(embedding) != EXPECTED_VECTOR_SIZE:
            raise ValueError(
                f"Unexpected embedding dimension: {len(embedding)}. Expected {EXPECTED_VECTOR_SIZE}."
            )

        return embedding

    def store_embedding(self, text: str, fraud_label: str) -> dict[str, str]:
        if not fraud_label or not fraud_label.strip():
            raise ValueError("fraud_label must not be empty")

        embedding = self.generate_embedding(text)
        point_id = self.vector_store.upsert_embedding(
            embedding=embedding,
            text=text,
            fraud_label=fraud_label.strip(),
        )
        return {"id": point_id, "status": "stored"}

    def search_similar(self, text: str, limit: int = 5) -> list[dict[str, str | float | None]]:
        embedding = self.generate_embedding(text)
        return self.vector_store.search(embedding=embedding, limit=limit)


def get_embedding_service(namespace: str = DEFAULT_NAMESPACE) -> FraudMemoryEmbeddingService:
    """Factory function designed for future FastAPI dependency injection."""
    vector_store = get_pinecone_vector_store(namespace=namespace)
    return FraudMemoryEmbeddingService(vector_store=vector_store)
