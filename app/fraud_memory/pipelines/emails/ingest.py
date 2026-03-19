from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from app.fraud_memory.qdrant_client import QdrantVectorStore, build_qdrant_client
except ModuleNotFoundError:
    from fraud_memory.qdrant_client import QdrantVectorStore, build_qdrant_client

from .cleaning import extract_email_text, infer_email_label
from .models import EmailFraudRecord

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "data_email_scam" / "emails.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH_SIZE = 128
COLLECTION_NAME = "fraud_vectors"

_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. Install it with 'pip install sentence-transformers'."
        ) from exc

    print(f"[emails] Loading embedding model: {MODEL_NAME}")
    _cached_model = SentenceTransformer(MODEL_NAME)
    return _cached_model


def generate_embedding(text: str) -> list[float]:
    vector = _load_model().encode(text, normalize_embeddings=True).tolist()
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"Unexpected embedding size {len(vector)}")
    return vector


class EmailIngestionPipeline:
    def __init__(self):
        self.vector_store = QdrantVectorStore(
            client=build_qdrant_client(),
            collection_name=COLLECTION_NAME,
            vector_size=VECTOR_SIZE,
        )

    def run(self) -> dict[str, int]:
        if not DATA_FILE.exists():
            print(f"[emails] Dataset not found: {DATA_FILE}")
            return {"processed": 0, "inserted": 0, "skipped": 0}

        stats = {"processed": 0, "inserted": 0, "skipped": 0}
        batch_points: list[dict[str, Any]] = []

        print(f"[emails] Reading file: {DATA_FILE.name}")
        with DATA_FILE.open("r", encoding="utf-8-sig", errors="replace", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                stats["processed"] += 1
                record = EmailFraudRecord(
                    text=extract_email_text(row),
                    label=infer_email_label(row),
                    source_file=DATA_FILE.name,
                )
                if not record.text:
                    stats["skipped"] += 1
                    continue

                try:
                    vector = generate_embedding(record.text)
                except Exception as exc:  # noqa: BLE001
                    print(f"[emails] Skipped record due to embedding error: {exc}")
                    stats["skipped"] += 1
                    continue

                payload = {
                    "text": record.text,
                    "label": record.label,
                    "source_file": record.source_file,
                    "source": "emails",
                }
                batch_points.append({"id": str(uuid4()), "vector": vector, "payload": payload})

                if len(batch_points) >= BATCH_SIZE:
                    print(f"[emails] Upserting batch of {len(batch_points)} from {DATA_FILE.name}")
                    self.vector_store.upsert_points(batch_points, wait=True)
                    stats["inserted"] += len(batch_points)
                    batch_points.clear()

        if batch_points:
            print(f"[emails] Upserting final batch of {len(batch_points)} from {DATA_FILE.name}")
            self.vector_store.upsert_points(batch_points, wait=True)
            stats["inserted"] += len(batch_points)

        print(
            f"[emails] Ingestion completed for {DATA_FILE.name}: processed={stats['processed']} inserted={stats['inserted']} skipped={stats['skipped']}"
        )
        return stats


def run_email_ingestion() -> dict[str, int]:
    return EmailIngestionPipeline().run()
