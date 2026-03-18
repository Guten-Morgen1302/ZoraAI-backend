## Adds request into my PostgreSQL DB..

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import PhishingAnalysis, PhishingRequest, SmsThreatResult


class PhishingRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_request(self, text: str, source: str, user_id: str | None = None) -> PhishingRequest:
        phishing_request = PhishingRequest(text=text, source=source, user_id=user_id)
        self.db.add(phishing_request)
        self.db.flush()
        return phishing_request

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
