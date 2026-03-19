from __future__ import annotations

try:
    from app.fraud_memory.pipelines.emails.ingest import run_email_ingestion
except ModuleNotFoundError:
    from fraud_memory.pipelines.emails.ingest import run_email_ingestion


if __name__ == "__main__":
    run_email_ingestion()
