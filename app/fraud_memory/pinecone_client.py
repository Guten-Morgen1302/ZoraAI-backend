from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PINECONE_INDEX_NAME = "zora-ai"
DEFAULT_PINECONE_HOST = "https://zora-ai-ks36vlc.svc.aped-4627-b74a.pinecone.io"
DEFAULT_NAMESPACE = "fraud_vectors"

logger = logging.getLogger("zora.fraud_memory.pinecone")


def _sanitize_metadata_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return value
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _sanitize_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        clean_value = _sanitize_metadata_value(value)
        if clean_value is not None:
            sanitized[str(key)] = clean_value
    return sanitized


def _build_pinecone_index():
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise RuntimeError("pinecone package is required. Install it with 'pip install pinecone'.") from exc

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in environment or .env")

    index_name = (os.getenv("PINECONE_INDEX_NAME") or DEFAULT_PINECONE_INDEX_NAME).strip()
    host = (os.getenv("PINECONE_HOST") or DEFAULT_PINECONE_HOST).strip()

    client = Pinecone(api_key=api_key)
    logger.info("Initializing Pinecone index client", extra={"host": host or None, "index_name": index_name})
    if host:
        return client.Index(host=host)
    return client.Index(index_name)


class PineconeVectorStore:
    """Wrapper around Pinecone operations used by fraud memory services."""

    def __init__(self, index: Any, namespace: str = DEFAULT_NAMESPACE):
        self.index = index
        self.namespace = (namespace or DEFAULT_NAMESPACE).strip()
        logger.info("Pinecone vector store ready", extra={"namespace": self.namespace})

    def upsert_embedding(self, embedding: list[float], text: str, fraud_label: str) -> str:
        point_id = str(uuid4())
        payload = {
            "text": text,
            "fraud_label": fraud_label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.upsert_point(point_id=point_id, vector=embedding, payload=payload)
        return point_id

    def upsert_point(self, point_id: str, vector: list[float], payload: dict[str, Any], wait: bool = True) -> str:
        _ = wait
        logger.info("Upserting single vector to Pinecone", extra={"namespace": self.namespace, "vector_id": str(point_id)})
        self.index.upsert(
            vectors=[
                {
                    "id": str(point_id),
                    "values": [float(v) for v in vector],
                    "metadata": _sanitize_metadata(payload),
                }
            ],
            namespace=self.namespace,
        )
        return str(point_id)

    def upsert_points(self, points: list[dict[str, Any]], wait: bool = True) -> None:
        _ = wait
        if not points:
            return

        vectors = []
        for point in points:
            vectors.append(
                {
                    "id": str(point.get("id") or uuid4()),
                    "values": [float(v) for v in (point.get("vector") or [])],
                    "metadata": _sanitize_metadata(point.get("payload") or {}),
                }
            )

        logger.info(
            "Upserting vector batch to Pinecone",
            extra={"namespace": self.namespace, "batch_size": len(vectors)},
        )
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def search(self, embedding: list[float], limit: int = 5) -> list[dict[str, str | float | None]]:
        logger.info("Running Pinecone similarity query", extra={"namespace": self.namespace, "top_k": limit})
        response = self.index.query(
            namespace=self.namespace,
            vector=[float(v) for v in embedding],
            top_k=limit,
            include_metadata=True,
        )

        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches") or []
        matches = matches or []

        results: list[dict[str, str | float | None]] = []
        for match in matches:
            metadata = getattr(match, "metadata", None)
            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata")
            metadata = metadata or {}

            score = getattr(match, "score", None)
            if score is None and isinstance(match, dict):
                score = match.get("score")

            matched_label = metadata.get("fraud_label") or metadata.get("label")
            results.append(
                {
                    "text": metadata.get("text"),
                    "similarity": round(float(score or 0.0), 4),
                    "fraud_label": matched_label,
                    "label": matched_label,
                    "source": metadata.get("source"),
                    "source_file": metadata.get("source_file"),
                    "timestamp": metadata.get("timestamp"),
                }
            )
        logger.info(
            "Pinecone similarity query completed",
            extra={"namespace": self.namespace, "match_count": len(results)},
        )
        return results


def get_pinecone_vector_store(namespace: str = DEFAULT_NAMESPACE) -> PineconeVectorStore:
    index = _build_pinecone_index()
    return PineconeVectorStore(index=index, namespace=namespace)
