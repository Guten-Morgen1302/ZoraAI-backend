from __future__ import annotations

import argparse
import csv
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

try:
    from app.fraud_memory.pinecone_client import get_pinecone_vector_store
except ModuleNotFoundError:
    from fraud_memory.pinecone_client import get_pinecone_vector_store

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
DEFAULT_BATCH_SIZE = 100

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

    print(f"[csv-upsert] Loading embedding model: {MODEL_NAME}")
    _cached_model = SentenceTransformer(MODEL_NAME)
    return _cached_model


def _generate_embedding(text: str) -> list[float]:
    vector = _load_model().encode(text, normalize_embeddings=True).tolist()
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"Unexpected embedding size {len(vector)}")
    return vector


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upsert CSV rows into Pinecone as vectors.")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV file")
    parser.add_argument("--text-column", required=True, help="Column name that contains text to embed")
    parser.add_argument("--label-column", default="", help="Optional label column name")
    parser.add_argument("--namespace", default="fraud_vectors", help="Pinecone namespace")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Upsert batch size")
    parser.add_argument(
        "--source",
        default="csv_manual",
        help="Metadata source tag stored with vectors",
    )
    return parser


def _safe_get(row: dict[str, str], key: str) -> str:
    if not key:
        return ""
    value = row.get(key)
    return value.strip() if isinstance(value, str) else ""


def upsert_csv_to_pinecone(
    *,
    csv_path: Path,
    text_column: str,
    label_column: str,
    namespace: str,
    batch_size: int,
    source: str,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    vector_store = get_pinecone_vector_store(namespace=namespace)

    stats = {"processed": 0, "inserted": 0, "skipped": 0}
    batch_points: list[dict] = []

    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fp:
        reader = csv.DictReader(fp)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(
                f"text_column '{text_column}' not found. Available columns: {(reader.fieldnames or [])}"
            )

        for row in reader:
            stats["processed"] += 1
            text = _safe_get(row, text_column)
            if not text:
                stats["skipped"] += 1
                continue

            try:
                vector = _generate_embedding(text)
            except Exception as exc:  # noqa: BLE001
                print(f"[csv-upsert] Skipping row due to embedding error: {exc}")
                stats["skipped"] += 1
                continue

            label = _safe_get(row, label_column)
            payload = {
                "text": text,
                "label": label or None,
                "fraud_label": label or None,
                "source": source,
                "source_file": csv_path.name,
            }
            batch_points.append({"id": str(uuid4()), "vector": vector, "payload": payload})

            if len(batch_points) >= batch_size:
                vector_store.upsert_points(batch_points, wait=True)
                stats["inserted"] += len(batch_points)
                print(f"[csv-upsert] Upserted batch of {len(batch_points)}")
                batch_points.clear()

    if batch_points:
        vector_store.upsert_points(batch_points, wait=True)
        stats["inserted"] += len(batch_points)
        print(f"[csv-upsert] Upserted final batch of {len(batch_points)}")

    return stats


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()

    result = upsert_csv_to_pinecone(
        csv_path=Path(args.csv_path).resolve(),
        text_column=args.text_column,
        label_column=args.label_column,
        namespace=args.namespace,
        batch_size=args.batch_size,
        source=args.source,
    )
    print(
        "[csv-upsert] Completed: "
        f"processed={result['processed']} inserted={result['inserted']} skipped={result['skipped']}"
    )


if __name__ == "__main__":
    main()
