from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger("zora.fraud_memory.qdrant")

DEFAULT_QDRANT_URL = (
    "https://2b00db0a-2c04-4acc-8aa1-ed063c56dcd4.eu-west-1-0.aws.cloud.qdrant.io:6333"
)
DEFAULT_QDRANT_API_KEY = "INSERT_API_KEY"
DEFAULT_COLLECTION_NAME = "fraud_vectors"
DEFAULT_VECTOR_SIZE = 384


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
            results.append(
                {
                    "text": payload.get("text"),
                    "similarity": round(float(point.score), 4),
                    "fraud_label": payload.get("fraud_label"),
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

    url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_CONNECTION_URL") or DEFAULT_QDRANT_URL
    api_key = os.getenv("QDRANT_API_KEY", DEFAULT_QDRANT_API_KEY)

    if api_key == DEFAULT_QDRANT_API_KEY:
        logger.warning("QDRANT_API_KEY is set to placeholder value; replace it for production use")

    return QdrantClient(url=url, api_key=api_key)


def get_qdrant_vector_store() -> QdrantVectorStore:
    client = build_qdrant_client()
    return QdrantVectorStore(client=client)
