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
