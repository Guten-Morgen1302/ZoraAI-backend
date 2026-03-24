from typing import Any

from pydantic import BaseModel, EmailStr
from enum import Enum
from uuid import UUID
from pydantic import Field


class UserCreate(BaseModel):
	email: EmailStr
	password: str
	full_name: str | None = None
	role: str | None = None
	organization_name: str | None = None


class UserLogin(BaseModel):
	email: EmailStr
	password: str


class TokenResponse(BaseModel):
	access_token: str
	refresh_token: str
	token_type: str


class MessageSource(str, Enum):
	sms = "sms"
	email = "email"
	chat = "chat"


class TextAnalyzeRequest(BaseModel):
	text: str = Field(min_length=1, max_length=5000)
	source: MessageSource


class TextAnalyzeResponse(BaseModel):
	request_id: UUID
	links_detected: int
	urgent_language: bool
	status: str


class SMSModelPredictRequest(BaseModel):
	text: str = Field(min_length=1, max_length=4096)


class SMSModelPredictResponse(BaseModel):
	prediction: dict[str, Any]


class SMSVectorSearchRequest(BaseModel):
	text: str = Field(min_length=1, max_length=4096)


class SMSVectorMatch(BaseModel):
	text: str | None = None
	similarity: float
	fraud_label: str | None = None
	label: str | None = None
	source: str | None = None
	source_file: str | None = None
	timestamp: str | None = None


class SMSVectorSearchResponse(BaseModel):
	similarity_score: float
	matched_label: str | None = None
	high_risk: bool
	threshold: float
	top_k: int
	matched_text: str | None = None
	matched_source: str | None = None
	top_k_matches: list[SMSVectorMatch]


class SMSAnalyzeRequest(BaseModel):
	text: str = Field(min_length=1, max_length=4096)
	include_llm_explanation: bool = False


class SMSAnalyzeResponse(BaseModel):
	request_id: UUID
	risk_score: float
	fraud_type: str
	confidence: float
	flags: list[str]
	explanation: str
	llm_enhanced: bool
	llm_explanation: str | None = None
	nlp_score: float
	similarity_score: float
	stylometry_score: float
	prediction: dict[str, Any]
	similarity: SMSVectorSearchResponse
	url_risk_score: float
	urgency_score: float


class SMSFraudFeedbackRequest(BaseModel):
	request_id: UUID | None = None
	text: str | None = Field(default=None, min_length=1, max_length=5000)
	fraud_label: str = Field(default="phishing", min_length=1, max_length=64)
	source: MessageSource = MessageSource.sms


class SMSFraudFeedbackResponse(BaseModel):
	feedback_id: UUID
	request_id: UUID | None = None
	vector_id: str
	status: str


class LatestEmailFetchRequest(BaseModel):
	query: str | None = Field(default=None, max_length=500)


class LatestEmailFetchResponse(BaseModel):
	message_id: str
	thread_id: str | None = None
	sender: str
	subject: str
	body: str
	preprocessing: dict[str, Any]


class LatestEmailAnalyzeRequest(BaseModel):
	query: str | None = Field(default=None, max_length=500)
	force_reauth: bool = True
	with_llm_explanation: bool = False


class EmailAnalyzeByIdRequest(BaseModel):
	thread_id: str = Field(min_length=1, max_length=256)
	message_id: str = Field(min_length=1, max_length=256)
	force_reauth: bool = False
	with_llm_explanation: bool = False


class EmailAnalyzeManualRequest(BaseModel):
	sender: str = Field(min_length=1, max_length=512)
	subject: str = Field(default="", max_length=1000)
	body: str = Field(min_length=1, max_length=100000)
	with_llm_explanation: bool = False


class LatestEmailAnalyzeResponse(BaseModel):
	message_id: str
	thread_id: str | None = None
	sender: str
	subject: str
	body: str
	risk_score: float
	nlp_score: float
	similarity_score: float
	stylometry_score: float
	confidence: float
	fraud_type: str
	nlp_prediction: dict[str, Any]
	similarity: dict[str, Any]
	llm_enhanced: bool
	llm_explanation: str | None = None
	llm_label: str | None = None
	llm_confidence: float | None = None


class URLAnalyzeRequest(BaseModel):
	url: str = Field(min_length=4, max_length=4096)
	with_llm_explanation: bool = True


class URLAnalyzeResponse(BaseModel):
	request_id: UUID | None = None
	url: str
	phishing_probability: float
	risk_score: float
	risk_level: str
	model: str
	persisted: bool = False
	pipeline_checks: dict[str, Any]
	risk_components: dict[str, float]
	llm_enhanced: bool
	llm_label: str | None = None
	llm_confidence: float | None = None
	llm_explanation: str | None = None
	llm_key_indicators: list[str] = Field(default_factory=list)
	llm_recommendations: list[str] = Field(default_factory=list)
	url_features: dict[str, Any]
	domain_features: dict[str, Any]
	tls_features: dict[str, Any]
	homoglyph_features: dict[str, Any]
	sandbox_features: dict[str, Any]
	cookie_features: dict[str, Any]
	phishing_behavior_features: dict[str, Any]
	fingerprint_beacon_features: dict[str, Any]
	fused_features: dict[str, Any]


class AttachmentEngineResult(BaseModel):
	is_flagged: bool
	hits: list[str] | None = None
	signature: str | None = None
	score: float | None = None


class AttachmentAnalyzeResponse(BaseModel):
	request_id: UUID | None = None
	analysis_id: UUID | None = None
	filename: str
	file_size: int
	s3_url: str | None = None
	status: str | None = None
	final_verdict: str
	engines: dict[str, AttachmentEngineResult]
	features: dict[str, Any]


class VoiceAnalysisResponse(BaseModel):
	request_id: UUID
	analysis_id: UUID
	status: str
	filename: str
	voice_analysis: dict[str, Any]
	transcript: str
	fraud_report: dict[str, Any]
