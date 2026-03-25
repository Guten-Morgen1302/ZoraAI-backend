from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.auth.security import (
    clear_auth_cookies,
    create_access_token,
    create_refresh_token,
    get_token_subject,
    hash_password,
    set_auth_cookies,
    verify_password,
)
from app.database import get_db
from app.models import User
from app.schemas import TokenResponse, UserCreate, UserLogin

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: UserCreate, response: Response, db: Session = Depends(get_db)):
    existing_email = db.query(User).filter(User.email == payload.email).first()
    if existing_email:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role or "analyst",
        organization_name=payload.organization_name,
        is_active=True,
        updated_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    set_auth_cookies(response, access_token, refresh_token)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        user.is_active = True
        db.commit()

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    set_auth_cookies(response, access_token, refresh_token)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


@router.get("/me")
def get_current_user(request: Request, db: Session = Depends(get_db)):
    access_token = request.cookies.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user_id = get_token_subject(access_token, expected_type="access")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "organization_name": user.organization_name,
    }


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    access_token = request.cookies.get("access_token")
    refresh_token = request.cookies.get("refresh_token")

    user_id = None
    if access_token:
        user_id = get_token_subject(access_token, expected_type="access")
    if not user_id and refresh_token:
        user_id = get_token_subject(refresh_token, expected_type="refresh")

    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user and user.is_active:
            user.is_active = False
            user.updated_at = datetime.utcnow()
            db.commit()

    clear_auth_cookies(response)
    return {"message": "Logged out successfully"}
