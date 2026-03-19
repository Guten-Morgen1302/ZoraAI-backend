from .pipelines.text.ingest_one import run_text_ingestion_for_file

if __name__ == "__main__":
    run_text_ingestion_for_file("scam_data_new.csv")