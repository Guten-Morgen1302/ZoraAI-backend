from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from app.fraud_memory.pinecone_client import get_pinecone_vector_store
except ModuleNotFoundError:
    from fraud_memory.pinecone_client import get_pinecone_vector_store

from .cleaning import normalize_label, row_to_text
from .models import UrlFraudRecord

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "data_url_phishing" / "PhiUSIIL_Phishing_URL_Dataset.csv"
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

    print(f"[url_phishing] Loading embedding model: {MODEL_NAME}")
    _cached_model = SentenceTransformer(MODEL_NAME)
    return _cached_model


def generate_embedding(text: str) -> list[float]:
    vector = _load_model().encode(text, normalize_embeddings=True).tolist()
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"Unexpected embedding size {len(vector)}")
    return vector


class UrlPhishingIngestionPipeline:
    def __init__(self):
        self.vector_store = get_pinecone_vector_store(namespace=COLLECTION_NAME)

    def run(self) -> dict[str, int]:
        if not DATA_FILE.exists():
            print(f"[url_phishing] Dataset not found: {DATA_FILE}")
            return {"processed": 0, "inserted": 0, "skipped": 0}

        stats = {"processed": 0, "inserted": 0, "skipped": 0}
        batch_points: list[dict[str, Any]] = []

        print(f"[url_phishing] Reading file: {DATA_FILE.name}")
        with DATA_FILE.open("r", encoding="utf-8-sig", errors="replace", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                stats["processed"] += 1
                record = UrlFraudRecord(
                    text=row_to_text(row),
                    label=normalize_label(row.get("label")),
                    source_file=DATA_FILE.name,
                )
                if not record.text:
                    stats["skipped"] += 1
                    continue

                try:
                    vector = generate_embedding(record.text)
                except Exception as exc:  # noqa: BLE001
                    print(f"[url_phishing] Skipped record due to embedding error: {exc}")
                    stats["skipped"] += 1
                    continue

                payload = {
                    "text": record.text,
                    "label": record.label,
                    "source_file": record.source_file,
                    "source": "url_phishing",
                    "url": row.get("URL"),
                }
                batch_points.append({"id": str(uuid4()), "vector": vector, "payload": payload})

                if len(batch_points) >= BATCH_SIZE:
                    print(f"[url_phishing] Upserting batch of {len(batch_points)} from {DATA_FILE.name}")
                    self.vector_store.upsert_points(batch_points, wait=True)
                    stats["inserted"] += len(batch_points)
                    batch_points.clear()

        if batch_points:
            print(f"[url_phishing] Upserting final batch of {len(batch_points)} from {DATA_FILE.name}")
            self.vector_store.upsert_points(batch_points, wait=True)
            stats["inserted"] += len(batch_points)

        print(
            f"[url_phishing] Ingestion completed for {DATA_FILE.name}: processed={stats['processed']} inserted={stats['inserted']} skipped={stats['skipped']}"
        )
        return stats


def run_url_phishing_ingestion() -> dict[str, int]:
    return UrlPhishingIngestionPipeline().run()
