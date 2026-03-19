from __future__ import annotations

try:
    from app.fraud_memory.pipelines.url_phishing.ingest import run_url_phishing_ingestion
except ModuleNotFoundError:
    from fraud_memory.pipelines.url_phishing.ingest import run_url_phishing_ingestion


if __name__ == "__main__":
    run_url_phishing_ingestion()
