from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("zora.fraud_memory.qdrant")

DEFAULT_QDRANT_URL = (
    "https://2b00db0a-2c04-4acc-8aa1-ed063c56dcd4.eu-west-1-0.aws.cloud.qdrant.io:6333"
)
DEFAULT_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_COLLECTION_NAME = "fraud_vectors"
DEFAULT_VECTOR_SIZE = 384


def _normalize_qdrant_url(raw_url: str | None) -> str:
    """Normalize URL input from env vars to avoid malformed host parsing."""
    url = (raw_url or "").strip().strip('"').strip("'").rstrip("/")
    if not url:
        return DEFAULT_QDRANT_URL
    if "http://" not in url and "https://" not in url:
        return f"https://{url}"
    return url


class QdrantVectorStore:
    """Wrapper around Qdrant operations used by the fraud memory service."""

    def __init__(
        self,
        client: Any,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.ensure_collection()

    def ensure_collection(self) -> None:
        try:
            from qdrant_client.http.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for Fraud Memory Service. Install it with 'pip install qdrant-client'."
            ) from exc

        existing_collections = self.client.get_collections().collections
        collection_names = {collection.name for collection in existing_collections}

        if self.collection_name in collection_names:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", self.collection_name)

    def upsert_embedding(self, embedding: list[float], text: str, fraud_label: str) -> str:
        point_id = str(uuid4())
        payload = {
            "text": text,
            "fraud_label": fraud_label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.upsert_point(point_id=point_id, vector=embedding, payload=payload, wait=True)
        return point_id

    def upsert_point(self, point_id: str, vector: list[float], payload: dict[str, Any], wait: bool = True) -> str:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{"id": point_id, "vector": vector, "payload": payload}],
            wait=wait,
        )
        return point_id

    def upsert_points(self, points: list[dict[str, Any]], wait: bool = True) -> None:
        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=wait,
        )

    def search(self, embedding: list[float], limit: int = 5) -> list[dict[str, str | float | None]]:
        points = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
        )

        results: list[dict[str, str | float | None]] = []
        for point in points:
            payload = point.payload or {}
            matched_label = payload.get("fraud_label") or payload.get("label")
            results.append(
                {
                    "text": payload.get("text"),
                    "similarity": round(float(point.score), 4),
                    "fraud_label": matched_label,
                    "label": matched_label,
                    "source": payload.get("source"),
                    "source_file": payload.get("source_file"),
                    "timestamp": payload.get("timestamp"),
                }
            )
        return results


def build_qdrant_client():
    """Builds a cloud Qdrant client from env vars, defaulting to provided cloud endpoint."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required for Fraud Memory Service. Install it with 'pip install qdrant-client'."
        ) from exc

    url = _normalize_qdrant_url(os.getenv("QDRANT_URL") or os.getenv("QDRANT_CONNECTION_URL"))
    api_key = os.getenv("QDRANT_API_KEY", DEFAULT_QDRANT_API_KEY)

    if not api_key:
        logger.warning("QDRANT_API_KEY is not set; authenticated Qdrant cloud requests may fail")

    return QdrantClient(url=url, api_key=api_key)


def _build_in_memory_qdrant_store() -> QdrantVectorStore:
    """Fallback store used when remote Qdrant is unreachable."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required for Fraud Memory Service. Install it with 'pip install qdrant-client'."
        ) from exc

    local_client = QdrantClient(path=":memory:")
    return QdrantVectorStore(client=local_client)


def get_qdrant_vector_store() -> QdrantVectorStore:
    client = build_qdrant_client()
    try:
        return QdrantVectorStore(client=client)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Remote Qdrant is unreachable. Falling back to in-memory vector store for this process: %s",
            exc,
        )
        return _build_in_memory_qdrant_store()
