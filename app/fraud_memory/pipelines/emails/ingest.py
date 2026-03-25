from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from app.fraud_memory.pinecone_client import get_pinecone_vector_store
except ModuleNotFoundError:
    from fraud_memory.pinecone_client import get_pinecone_vector_store

from .cleaning import (
    extract_email_body,
    extract_email_sender,
    extract_email_subject,
    extract_email_urls,
    infer_email_label,
)
from .models import EmailFraudRecord

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "data_email_scam"
TARGET_DATASET_STEMS = {"cleaned_ceas", "nazario", "nigerian_fraud", "spamassasin"}
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH_SIZE = 128
COLLECTION_NAME = "fraud_emails"

_cached_model = None


def _set_max_csv_field_size() -> None:
    # Some datasets include extremely large fields; raise parser limit as much as runtime allows.
    max_size = sys.maxsize
    while max_size > 0:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


_set_max_csv_field_size()


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
        self.vector_store = get_pinecone_vector_store(namespace=COLLECTION_NAME)

    def run(self) -> dict[str, int]:
        if not DATA_DIR.exists():
            print(f"[emails] Dataset folder not found: {DATA_DIR}")
            return {"processed": 0, "inserted": 0, "skipped": 0}

        files = sorted(
            [
                p
                for p in DATA_DIR.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv" and p.stem.lower() in TARGET_DATASET_STEMS
            ]
        )
        if not files:
            print(f"[emails] No target CSV files found in {DATA_DIR}")
            return {"processed": 0, "inserted": 0, "skipped": 0}

        summary = {"processed": 0, "inserted": 0, "skipped": 0}
        for file_path in files:
            file_stats = self._ingest_file(file_path)
            summary["processed"] += file_stats["processed"]
            summary["inserted"] += file_stats["inserted"]
            summary["skipped"] += file_stats["skipped"]

        print(
            f"[emails] Ingestion completed: processed={summary['processed']} inserted={summary['inserted']} skipped={summary['skipped']}"
        )
        return summary

    def _ingest_file(self, file_path: Path) -> dict[str, int]:
        stats = {"processed": 0, "inserted": 0, "skipped": 0}
        batch_points: list[dict[str, Any]] = []

        print(f"[emails] Reading file: {file_path.name}")
        with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fp:
            reader = csv.DictReader(fp)
            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    break
                except csv.Error as exc:
                    print(f"[emails] Skipped malformed CSV row in {file_path.name}: {exc}")
                    stats["skipped"] += 1
                    continue

                stats["processed"] += 1
                try:
                    record = EmailFraudRecord(
                        text=extract_email_body(row),
                        subject=extract_email_subject(row),
                        sender=extract_email_sender(row),
                        urls=extract_email_urls(row),
                        label=infer_email_label(row),
                        source_file=file_path.name,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[emails] Skipped row due to preprocessing error in {file_path.name}: {exc}")
                    stats["skipped"] += 1
                    continue

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
                    "subject": record.subject,
                    "sender": record.sender,
                    "urls": record.urls,
                    "label": record.label,
                    "type": "email",
                    "source": "dataset",
                }
                batch_points.append({"id": str(uuid4()), "vector": vector, "payload": payload})

                if len(batch_points) >= BATCH_SIZE:
                    print(f"[emails] Upserting batch of {len(batch_points)} from {file_path.name}")
                    inserted, skipped = self._safe_upsert_batch(batch_points, file_path.name)
                    stats["inserted"] += inserted
                    stats["skipped"] += skipped
                    batch_points.clear()

        if batch_points:
            print(f"[emails] Upserting final batch of {len(batch_points)} from {file_path.name}")
            inserted, skipped = self._safe_upsert_batch(batch_points, file_path.name)
            stats["inserted"] += inserted
            stats["skipped"] += skipped

        print(
            f"[emails] Ingestion completed for {file_path.name}: processed={stats['processed']} inserted={stats['inserted']} skipped={stats['skipped']}"
        )
        return stats

    def _safe_upsert_batch(self, points: list[dict[str, Any]], source_file: str) -> tuple[int, int]:
        try:
            self.vector_store.upsert_points(points, wait=True)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[emails] Batch upsert failed for {source_file} with {len(points)} points. "
                f"Skipping this batch. Error: {exc}"
            )
            return (0, len(points))
        return (len(points), 0)


def run_email_ingestion() -> dict[str, int]:
    return EmailIngestionPipeline().run()
