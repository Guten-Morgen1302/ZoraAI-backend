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
	text: str = Field(min_length=1, max_length=5000)


class SMSModelPredictResponse(BaseModel):
	prediction: dict[str, Any]


class SMSVectorSearchRequest(BaseModel):
	text: str = Field(min_length=1, max_length=5000)
	top_k: int = Field(default=5, ge=1, le=20)
	threshold: float = Field(default=0.85, ge=0.0, le=1.0)


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
	text: str = Field(min_length=1, max_length=5000)
	top_k: int = Field(default=5, ge=1, le=20)
	similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class SMSAnalyzeResponse(BaseModel):
	request_id: UUID
	risk_score: float
	fraud_type: str
	confidence: float
	flags: list[str]
	explanation: str
	nlp_score: float
	similarity_score: float
	stylometry_score: float
	prediction: dict[str, Any]
	similarity: SMSVectorSearchResponse
	url_risk_score: float
	urgency_score: float
