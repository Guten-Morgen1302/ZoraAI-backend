import secrets
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.auth.security import get_token_subject
from app.database import get_db
from app.models import ApiKey, User
from app.schemas import ApiKeyCreateResponse, ApiKeyListItem, ApiKeyRevealResponse

router = APIRouter(prefix="/api-keys", tags=["api-keys"])

# API keys are long-lived for SDK integrations but still rotated periodically.
API_KEY_VALID_DAYS = 90


def _mask_api_key(value: str) -> str:
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:7]}{'*' * (len(value) - 11)}{value[-4:]}"


def _generate_api_key(db: Session) -> str:
    while True:
        candidate = f"zora_{secrets.token_urlsafe(32)}"
        exists = db.query(ApiKey).filter(ApiKey.api_key == candidate).first()
        if not exists:
            return candidate


def _get_authenticated_user(request: Request, db: Session) -> User:
    access_token = request.cookies.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user_id = get_token_subject(access_token, expected_type="access")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/request", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def request_api_key(
    request: Request,
    db: Session = Depends(get_db),
):
    user = _get_authenticated_user(request, db)

    now = datetime.utcnow()
    expires_at = now + timedelta(days=API_KEY_VALID_DAYS)
    api_key_value = _generate_api_key(db)

    key_record = ApiKey(
        user_id=user.id,
        api_key=api_key_value,
        is_active=True,
        created_at=now,
        expires_at=expires_at,
    )

    db.add(key_record)
    db.commit()
    db.refresh(key_record)

    return ApiKeyCreateResponse(
        key_id=key_record.id,
        api_key=api_key_value,
        expires_at=expires_at,
        valid_for_days=API_KEY_VALID_DAYS,
    )


@router.get("", response_model=list[ApiKeyListItem])
def list_user_api_keys(request: Request, db: Session = Depends(get_db)):
    user = _get_authenticated_user(request, db)
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user.id)
        .order_by(ApiKey.created_at.desc())
        .all()
    )
    return [
        ApiKeyListItem(
            key_id=key.id,
            masked_key=_mask_api_key(key.api_key),
            is_active=key.is_active,
            created_at=key.created_at,
            expires_at=key.expires_at,
        )
        for key in keys
    ]


@router.get("/{key_id}/reveal", response_model=ApiKeyRevealResponse)
def reveal_api_key(key_id: UUID, request: Request, db: Session = Depends(get_db)):
    user = _get_authenticated_user(request, db)
    key = (
        db.query(ApiKey)
        .filter(ApiKey.id == key_id, ApiKey.user_id == user.id)
        .first()
    )
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")

    return ApiKeyRevealResponse(
        key_id=key.id,
        api_key=key.api_key,
        expires_at=key.expires_at,
    )
