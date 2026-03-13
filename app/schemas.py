from pydantic import BaseModel, EmailStr


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
