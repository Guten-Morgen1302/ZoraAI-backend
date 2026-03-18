from __future__ import annotations

try:
    from app.fraud_memory.pipelines.text.ingest import run_text_ingestion
except ModuleNotFoundError:
    from fraud_memory.pipelines.text.ingest import run_text_ingestion


if __name__ == "__main__":
    run_text_ingestion()
