from __future__ import annotations

import json

from app.database import SessionLocal
from app.text_analysis.sms_analyzer.feeback_mechanism.sms_feedback_service import SMSFeedbackService

MAX_RECORDS = None
NAMESPACE = "fraud_vectors"
BATCH_SIZE = 128


def main() -> None:
    db = SessionLocal()
    try:
        service = SMSFeedbackService(db)
        result = service.export_retraining_dataset_and_upsert(
            max_records=MAX_RECORDS,
            namespace=NAMESPACE,
            batch_size=BATCH_SIZE,
        )
    finally:
        db.close()

    print(
        json.dumps(
            {
                "status": "completed",
                "candidate_feedback": result.candidate_feedback,
                "exported_rows": result.exported_rows,
                "csv_path": result.csv_path,
                "namespace": result.namespace,
                "vectors_inserted": result.vectors_inserted,
                "vectors_skipped": result.vectors_skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
