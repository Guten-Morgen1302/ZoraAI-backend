## Adds request into my PostgreSQL DB..

from __future__ import annotations

from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models import ConfirmedFraudCase, EmailFeedback, EmailThreatResult, PhishingAnalysis, PhishingRequest, SmsFeedback, SmsThreatResult


class PhishingRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_request(self, text: str, source: str, user_id: str | None = None) -> PhishingRequest:
        phishing_request = PhishingRequest(text=text, source=source, user_id=user_id)
        self.db.add(phishing_request)
        self.db.flush()
        return phishing_request

    def get_request_by_id(self, request_id: UUID) -> PhishingRequest | None:
        return self.db.query(PhishingRequest).filter(PhishingRequest.id == request_id).first()

    def create_analysis(
        self,
        request_id,
        link_count: int,
        urgency_score: float,
        status: str,
    ) -> PhishingAnalysis:
        analysis = PhishingAnalysis(
            request_id=request_id,
            link_count=link_count,
            urgency_score=urgency_score,
            status=status,
        )
        self.db.add(analysis)
        self.db.flush()
        return analysis

    def create_sms_threat_result(
        self,
        request_id,
        result: str,
        prediction: str,
        explanation: str,
    ) -> SmsThreatResult:
        sms_result = SmsThreatResult(
            request_id=request_id,
            result=result,
            prediction=prediction,
            explanation=explanation,
        )
        self.db.add(sms_result)
        self.db.flush()
        return sms_result

    def create_email_threat_result(
        self,
        request_id,
        result: str,
        prediction: str,
        explanation: str,
    ) -> EmailThreatResult:
        email_result = EmailThreatResult(
            request_id=request_id,
            result=result,
            prediction=prediction,
            explanation=explanation,
        )
        self.db.add(email_result)
        self.db.flush()
        return email_result

    def create_confirmed_fraud_case(
        self,
        *,
        request_id,
        user_id,
        text: str,
        fraud_label: str,
        source: str,
        vector_id: str,
    ) -> ConfirmedFraudCase:
        fraud_case = ConfirmedFraudCase(
            request_id=request_id,
            user_id=user_id,
            text=text,
            fraud_label=fraud_label,
            source=source,
            vector_id=vector_id,
        )
        self.db.add(fraud_case)
        self.db.flush()
        return fraud_case

    def create_sms_feedback(
        self,
        *,
        analysis_id: str,
        input_hash: str,
        model_prediction: str,
        human_label: str,
        model_confidence: float,
        feedback_type: str,
        notes: str | None,
    ) -> SmsFeedback:
        feedback = SmsFeedback(
            analysis_id=analysis_id,
            input_hash=input_hash,
            model_prediction=model_prediction,
            human_label=human_label,
            model_confidence=model_confidence,
            feedback_type=feedback_type,
            notes=notes,
        )
        self.db.add(feedback)
        self.db.flush()
        return feedback

    def get_sms_feedback_for_retraining(self, limit: int | None = None):
        query = (
            self.db.query(SmsFeedback)
            .filter(or_(SmsFeedback.feedback_type == "incorrect", SmsFeedback.feedback_type == "modified"))
            .order_by(SmsFeedback.created_at.desc())
        )
        if limit is not None:
            query = query.limit(limit)

        feedback_rows = query.all()
        if not feedback_rows:
            return []

        request_ids: list[UUID] = []
        for row in feedback_rows:
            try:
                request_ids.append(UUID(str(row.analysis_id)))
            except (TypeError, ValueError):
                continue

        if not request_ids:
            return []

        request_rows = (
            self.db.query(PhishingRequest)
            .filter(PhishingRequest.id.in_(request_ids))
            .filter(PhishingRequest.source == "sms")
            .all()
        )
        request_map = {str(req.id): req for req in request_rows}

        return [(feedback, request_map[feedback.analysis_id]) for feedback in feedback_rows if feedback.analysis_id in request_map]

    def create_email_feedback(
        self,
        *,
        analysis_id: str,
        input_hash: str,
        model_prediction: str,
        human_label: str,
        model_confidence: float,
        feedback_type: str,
        notes: str | None,
    ) -> EmailFeedback:
        feedback = EmailFeedback(
            analysis_id=analysis_id,
            input_hash=input_hash,
            model_prediction=model_prediction,
            human_label=human_label,
            model_confidence=model_confidence,
            feedback_type=feedback_type,
            notes=notes,
        )
        self.db.add(feedback)
        self.db.flush()
        return feedback

    def get_email_feedback_for_retraining(self, limit: int | None = None):
        query = (
            self.db.query(EmailFeedback)
            .filter(or_(EmailFeedback.feedback_type == "incorrect", EmailFeedback.feedback_type == "modified"))
            .order_by(EmailFeedback.created_at.desc())
        )
        if limit is not None:
            query = query.limit(limit)

        feedback_rows = query.all()
        if not feedback_rows:
            return []

        request_ids: list[UUID] = []
        for row in feedback_rows:
            try:
                request_ids.append(UUID(str(row.analysis_id)))
            except (TypeError, ValueError):
                continue

        if not request_ids:
            return []

        request_rows = (
            self.db.query(PhishingRequest)
            .filter(PhishingRequest.id.in_(request_ids))
            .filter(PhishingRequest.source == "email")
            .all()
        )
        request_map = {str(req.id): req for req in request_rows}

        return [(feedback, request_map[feedback.analysis_id]) for feedback in feedback_rows if feedback.analysis_id in request_map]
