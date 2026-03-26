from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from app.fraud_memory.pinecone_client import get_pinecone_vector_store
except ModuleNotFoundError:
    from fraud_memory.pinecone_client import get_pinecone_vector_store

from .cleaning import extract_label_from_row, extract_text_from_row
from .models import TextFraudRecord

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH_SIZE = 128
COLLECTION_NAME = "fraud_vectors"

DATA_DIR = Path(__file__).resolve().parents[2] / "data" /"data_text_scams"

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

    print(f"[text] Loading embedding model: {MODEL_NAME}")
    _cached_model = SentenceTransformer(MODEL_NAME)
    return _cached_model


def generate_embedding(text: str) -> list[float]:
    vector = _load_model().encode(text, normalize_embeddings=True).tolist()
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"Unexpected embedding size {len(vector)}")
    return vector


class TextScamIngestionPipeline:
    def __init__(self):
        self.vector_store = get_pinecone_vector_store(namespace=COLLECTION_NAME)

    def run(self) -> dict[str, int]:
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Text dataset folder not found: {DATA_DIR}")

        files = sorted([p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".csv"}])
        if not files:
            print(f"[text] No JSON/CSV files found in {DATA_DIR}")
            return {"processed": 0, "inserted": 0, "skipped": 0}

        summary = {"processed": 0, "inserted": 0, "skipped": 0}
        for file_path in files:
            print(f"[text] Reading file: {file_path.name}")
            records = self._load_records(file_path)
            file_stats = self._upsert_records(records=records, source_file=file_path.name)
            summary["processed"] += file_stats["processed"]
            summary["inserted"] += file_stats["inserted"]
            summary["skipped"] += file_stats["skipped"]

        return summary

    def _load_records(self, file_path: Path) -> list[TextFraudRecord]:
        if file_path.suffix.lower() == ".json":
            return self._load_json_records(file_path)
        return self._load_csv_records(file_path)

    def _load_json_records(self, file_path: Path) -> list[TextFraudRecord]:
        with file_path.open("r", encoding="utf-8", errors="replace") as fp:
            payload = json.load(fp)

        rows: list[dict[str, Any]]
        if isinstance(payload, list):
            rows = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            data = payload.get("data")
            rows = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
        else:
            rows = []

        records: list[TextFraudRecord] = []
        for row in rows:
            text = extract_text_from_row(row)
            if not text:
                continue
            records.append(
                TextFraudRecord(
                    text=text,
                    label=extract_label_from_row(row),
                    source_file=file_path.name,
                    dataset_type="text_json",
                )
            )
        return records

    def _load_csv_records(self, file_path: Path) -> list[TextFraudRecord]:
        records: list[TextFraudRecord] = []
        with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                text = extract_text_from_row(row)
                if not text:
                    continue
                records.append(
                    TextFraudRecord(
                        text=text,
                        label=extract_label_from_row(row),
                        source_file=file_path.name,
                        dataset_type="text_csv",
                    )
                )
        return records

    def _upsert_records(self, records: list[TextFraudRecord], source_file: str) -> dict[str, int]:
        stats = {"processed": 0, "inserted": 0, "skipped": 0}
        batch_points: list[dict[str, Any]] = []

        print(f"[text] Preparing {len(records)} records from {source_file}")
        for record in records:
            stats["processed"] += 1
            try:
                vector = generate_embedding(record.text)
            except Exception as exc:  # noqa: BLE001
                print(f"[text] Skipped record from {source_file}: {exc}")
                stats["skipped"] += 1
                continue

            payload = {
                "text": record.text,
                "label": record.label,
                "dataset_type": record.dataset_type,
                "source_file": record.source_file,
                "source": "text_scams",
            }
            batch_points.append({"id": str(uuid4()), "vector": vector, "payload": payload})

            if len(batch_points) >= BATCH_SIZE:
                print(f"[text] Upserting batch of {len(batch_points)} from {source_file}")
                self.vector_store.upsert_points(batch_points, wait=True)
                stats["inserted"] += len(batch_points)
                batch_points.clear()

        if batch_points:
            print(f"[text] Upserting final batch of {len(batch_points)} from {source_file}")
            self.vector_store.upsert_points(batch_points, wait=True)
            stats["inserted"] += len(batch_points)

        print(
            f"[text] Finished {source_file}: processed={stats['processed']} inserted={stats['inserted']} skipped={stats['skipped']}"
        )
        return stats


def run_text_ingestion() -> dict[str, int]:
    pipeline = TextScamIngestionPipeline()
    summary = pipeline.run()
    print(
        f"[text] Ingestion completed: processed={summary['processed']} inserted={summary['inserted']} skipped={summary['skipped']}"
    )
    return summary


def run_text_ingestion_for_file(file_name: str) -> dict[str, int]:
    
    pipeline = TextScamIngestionPipeline()

    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[text] Processing ONLY file: {file_name}")

    records = pipeline._load_records(file_path)
    stats = pipeline._upsert_records(records=records, source_file=file_path.name)

    print(
        f"[text] Single file ingestion completed: "
        f"processed={stats['processed']} inserted={stats['inserted']} skipped={stats['skipped']}"
    )

    return stats

