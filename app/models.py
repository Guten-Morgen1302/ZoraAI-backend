import uuid
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    email = Column(String, unique=True, index=True, nullable=False)

    password_hash = Column(String, nullable=False)

    full_name = Column(String)

    role = Column(String, default="analyst")

    organization_name = Column(String, nullable=True)

    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    updated_at = Column(DateTime, default=datetime.utcnow)

    voice_requests = relationship("VoiceRequest", back_populates="user")
    attachment_requests = relationship("AttachmentRequest", back_populates="user")
    api_keys = relationship("ApiKey", back_populates="user")


class ApiKey(Base):

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    api_key = Column(String, unique=True, nullable=False, index=True)

    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    expires_at = Column(DateTime, nullable=False)

    revoked_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="api_keys")


class PhishingRequest(Base):

    __tablename__ = "phishing_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), nullable=True)

    text = Column(String, nullable=False)

    source = Column(String, nullable=False) #should be email/sms/chat

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis = relationship("PhishingAnalysis", back_populates="request", uselist=False)
    sms_threat_result = relationship("SmsThreatResult", back_populates="request", uselist=False)
    email_threat_result = relationship("EmailThreatResult", back_populates="request", uselist=False)


class PhishingAnalysis(Base):

    __tablename__ = "phishing_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_id = Column(UUID(as_uuid=True), ForeignKey("phishing_requests.id"), nullable=False, unique=True)

    link_count = Column(Integer, nullable=False, default=0) # No. of urls detected in each req

    urgency_score = Column(Float, nullable=False, default=0.0) #checks in urgency from the request..

    status = Column(String, nullable=False, default="processing")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    request = relationship("PhishingRequest", back_populates="analysis")


class SmsThreatResult(Base):

    __tablename__ = "sms_threat_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_id = Column(UUID(as_uuid=True), ForeignKey("phishing_requests.id"), nullable=False, unique=True)

    result = Column(Text, nullable=False)

    prediction = Column(Text, nullable=False)

    explanation = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    request = relationship("PhishingRequest", back_populates="sms_threat_result")


class EmailThreatResult(Base):

    __tablename__ = "email_threat_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_id = Column(UUID(as_uuid=True), ForeignKey("phishing_requests.id"), nullable=False, unique=True)

    result = Column(Text, nullable=False)

    prediction = Column(Text, nullable=False)

    explanation = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    request = relationship("PhishingRequest", back_populates="email_threat_result")


class ConfirmedFraudCase(Base):

    __tablename__ = "confirmed_fraud_cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_id = Column(UUID(as_uuid=True), ForeignKey("phishing_requests.id"), nullable=True)

    user_id = Column(UUID(as_uuid=True), nullable=True)

    text = Column(Text, nullable=False)

    fraud_label = Column(String, nullable=False, default="phishing")

    source = Column(String, nullable=False, default="sms")

    vector_id = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class SmsFeedback(Base):

    __tablename__ = "sms_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)

    analysis_id = Column(String, nullable=False, index=True)

    input_hash = Column(String(64), nullable=False, index=True)

    model_prediction = Column(String, nullable=False)

    human_label = Column(String, nullable=False)

    model_confidence = Column(Float, nullable=False)

    feedback_type = Column(String, nullable=False)

    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class EmailFeedback(Base):

    __tablename__ = "email_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)

    analysis_id = Column(String, nullable=False, index=True)

    input_hash = Column(String(64), nullable=False, index=True)

    model_prediction = Column(String, nullable=False)

    human_label = Column(String, nullable=False)

    model_confidence = Column(Float, nullable=False)

    feedback_type = Column(String, nullable=False)

    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class URLAnalysisRequest(Base):

    __tablename__ = "url_analysis_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), nullable=True)

    source_url = Column(Text, nullable=False)

    normalized_url = Column(Text, nullable=False)

    status = Column(String, nullable=False, default="processing")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    result = relationship("URLThreatResult", back_populates="request", uselist=False)


class URLThreatResult(Base):

    __tablename__ = "url_threat_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_id = Column(UUID(as_uuid=True), ForeignKey("url_analysis_requests.id"), nullable=False, unique=True)

    result = Column(Text, nullable=False)

    prediction = Column(Text, nullable=False)

    explanation = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    request = relationship("URLAnalysisRequest", back_populates="result")


class URLFeedback(Base):

    __tablename__ = "url_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)

    analysis_id = Column(String, nullable=False, index=True)

    user_id = Column(UUID(as_uuid=True), nullable=True)

    normalized_url = Column(Text, nullable=False)

    model_prediction = Column(String, nullable=False)

    model_risk_score = Column(Float, nullable=False)

    model_phishing_probability = Column(Float, nullable=False)

    human_label = Column(String, nullable=False)

    prediction_type = Column(String, nullable=False)

    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class VoiceRequest(Base):

    __tablename__ = "voice_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    filename = Column(String, nullable=False)

    mime_type = Column(String, nullable=True)

    file_size = Column(Integer, nullable=False, default=0)

    transcript = Column(Text, nullable=False)

    status = Column(String, nullable=False, default="transcribed")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="voice_requests")
    analysis = relationship("VoiceAnalysis", back_populates="request", uselist=False)


class VoiceAnalysis(Base):

    __tablename__ = "voice_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_id = Column(UUID(as_uuid=True), ForeignKey("voice_requests.id"), nullable=False, unique=True)

    voice_result = Column(Text, nullable=False)

    fraud_report = Column(Text, nullable=False)

    status = Column(String, nullable=False, default="completed")

    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    request = relationship("VoiceRequest", back_populates="analysis")


class AttachmentRequest(Base):

    __tablename__ = "attachment_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    filename = Column(String, nullable=False)

    mime_type = Column(String, nullable=True)

    file_size = Column(Integer, nullable=False, default=0)

    s3_url = Column(Text, nullable=True)

    status = Column(String, nullable=False, default="uploaded")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="attachment_requests")
    analysis = relationship("AttachmentAnalysis", back_populates="request", uselist=False)


class AttachmentAnalysis(Base):

    __tablename__ = "attachment_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_id = Column(UUID(as_uuid=True), ForeignKey("attachment_requests.id"), nullable=False, unique=True)

    final_verdict = Column(String, nullable=False, default="unknown")

    engines = Column(Text, nullable=False)

    features = Column(Text, nullable=False)

    status = Column(String, nullable=False, default="completed")

    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    request = relationship("AttachmentRequest", back_populates="analysis")


class PortalChat(Base):

    __tablename__ = "portal_chats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)

    title = Column(String, nullable=False, default="New Chat")

    messages = Column(Text, nullable=False, default="[]")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

