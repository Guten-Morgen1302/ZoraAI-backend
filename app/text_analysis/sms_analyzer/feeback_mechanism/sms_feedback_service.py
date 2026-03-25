from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.fraud_memory.upsert_csv_to_pinecone import upsert_csv_to_pinecone
from app.text_analysis.repository import PhishingRepository

logger = logging.getLogger("zora.text_analysis.sms_feedback")

SMS_DATASET_DIR = Path(__file__).resolve().parents[3] / "fraud_memory" / "data" / "data_text_scams"


@dataclass
class SMSFeedbackStoreResult:
    id: int
    analysis_id: str
    input_hash: str
    created_at: str


@dataclass
class SMSFeedbackRetrainResult:
    candidate_feedback: int
    exported_rows: int
    csv_path: str
    namespace: str
    vectors_inserted: int
    vectors_skipped: int


class SMSFeedbackService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = PhishingRepository(db)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _label_to_numeric(label: str) -> int:
        normalized = (label or "").strip().lower()
        return 1 if normalized == "scam" else 0

    def submit_feedback(
        self,
        *,
        analysis_id: str,
        source: str,
        human_label: str,
        model_prediction: str,
        model_confidence: float,
        feedback_type: str,
        notes: str | None,
    ) -> SMSFeedbackStoreResult:
        if source != "sms":
            raise ValueError("source must be 'sms'")

        try:
            request_uuid = UUID(analysis_id)
        except ValueError as exc:
            raise ValueError("analysis_id must be a valid UUID string") from exc

        request_row = self.repository.get_request_by_id(request_uuid)
        if request_row is None:
            raise ValueError("analysis_id not found")
        if str(request_row.source).lower() != "sms":
            raise ValueError("analysis_id does not belong to an SMS analysis")

        input_hash = self._hash_text((request_row.text or "").strip())

        feedback = self.repository.create_sms_feedback(
            analysis_id=analysis_id,
            input_hash=input_hash,
            model_prediction=model_prediction.strip().lower(),
            human_label=human_label.strip().lower(),
            model_confidence=float(model_confidence),
            feedback_type=feedback_type.strip().lower(),
            notes=notes.strip() if notes else None,
        )
        self.db.commit()

        logger.info(
            "SMS feedback stored",
            extra={
                "feedback_id": feedback.id,
                "analysis_id": analysis_id,
                "feedback_type": feedback.feedback_type,
                "human_label": feedback.human_label,
            },
        )

        return SMSFeedbackStoreResult(
            id=int(feedback.id),
            analysis_id=feedback.analysis_id,
            input_hash=feedback.input_hash,
            created_at=feedback.created_at.isoformat() if feedback.created_at else datetime.now(timezone.utc).isoformat(),
        )

    def export_retraining_dataset_and_upsert(
        self,
        *,
        max_records: int | None,
        namespace: str,
        batch_size: int,
    ) -> SMSFeedbackRetrainResult:
        rows = self.repository.get_sms_feedback_for_retraining(limit=max_records)
        candidate_feedback = len(rows)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        SMS_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = SMS_DATASET_DIR / f"sms_feedback_retrain_{timestamp}.csv"

        exported_rows = 0
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["text", "label"])
            writer.writeheader()

            for feedback, request_row in rows:
                text = (request_row.text or "").strip()
                if not text:
                    continue
                writer.writerow({"text": text, "label": self._label_to_numeric(feedback.human_label)})
                exported_rows += 1

        vectors_inserted = 0
        vectors_skipped = 0
        if exported_rows > 0:
            upsert_result = upsert_csv_to_pinecone(
                csv_path=csv_path,
                text_column="text",
                label_column="label",
                namespace=namespace,
                batch_size=batch_size,
                source="sms_feedback_retraining",
            )
            vectors_inserted = int(upsert_result.get("inserted") or 0)
            vectors_skipped = int(upsert_result.get("skipped") or 0)

        logger.info(
            "SMS feedback retraining dataset generated",
            extra={
                "candidate_feedback": candidate_feedback,
                "exported_rows": exported_rows,
                "csv_path": str(csv_path),
                "namespace": namespace,
                "vectors_inserted": vectors_inserted,
                "vectors_skipped": vectors_skipped,
            },
        )

        return SMSFeedbackRetrainResult(
            candidate_feedback=candidate_feedback,
            exported_rows=exported_rows,
            csv_path=str(csv_path),
            namespace=namespace,
            vectors_inserted=vectors_inserted,
            vectors_skipped=vectors_skipped,
        )
