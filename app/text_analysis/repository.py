## Adds request into my PostgreSQL DB..

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.models import ConfirmedFraudCase, PhishingAnalysis, PhishingRequest, SmsThreatResult


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
