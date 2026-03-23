import os
from datetime import datetime, timedelta
from typing import Any, Dict

import jwt
import bcrypt
from dotenv import load_dotenv
import hashlib

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_DAYS = 3
REFRESH_TOKEN_EXPIRE_DAYS = 7


def hash_password(password: str) -> str:
    password = hashlib.sha256(password.encode()).hexdigest()
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    password = hashlib.sha256(password.encode()).hexdigest()
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def _create_token(subject: str, token_type: str, expires_delta: timedelta) -> str:
    now = datetime.utcnow()
    payload: Dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_access_token(subject: str) -> str:
    return _create_token(
        subject=subject,
        token_type="access",
        expires_delta=timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
    )


def create_refresh_token(subject: str) -> str:
    return _create_token(
        subject=subject,
        token_type="refresh",
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


def set_auth_cookies(response, access_token: str, refresh_token: str) -> None:
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_DAYS*24*60 *60,
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        samesite="lax",
        max_age=REFRESH_TOKEN_EXPIRE_DAYS*24*60*60,
    )


def clear_auth_cookies(response) -> None:
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        key="refresh_token",
        httponly=True,
        samesite="lax",
    )


def get_token_subject(token: str, expected_type: str | None = None) -> str | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        return None

    token_subject = payload.get("sub")
    token_type = payload.get("type")
    if not token_subject:
        return None
    if expected_type and token_type != expected_type:
        return None
    return token_subject
