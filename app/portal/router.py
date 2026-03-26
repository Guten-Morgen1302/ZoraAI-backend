from __future__ import annotations

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import PortalChat
from app.schemas import (
    PortalChatCreateRequest,
    PortalChatDetailResponse,
    PortalChatListItem,
    PortalChatUpdateRequest,
)

router = APIRouter(prefix="/portal", tags=["portal"])


def _safe_user_uuid(request: Request) -> uuid.UUID | None:
    state_user = request.state.user_id if hasattr(request.state, "user_id") else None
    if state_user is None:
        return None

    if isinstance(state_user, uuid.UUID):
        return state_user

    try:
        return uuid.UUID(str(state_user))
    except (ValueError, TypeError):
        return None


def _parse_messages(raw_messages: str | None) -> list[dict[str, object]]:
    if not raw_messages:
        return []

    try:
        parsed = json.loads(raw_messages)
    except Exception:  # noqa: BLE001
        return []

    if not isinstance(parsed, list):
        return []

    return [m for m in parsed if isinstance(m, dict)]


def _serialize_messages(messages: list[dict[str, object]]) -> str:
    return json.dumps(messages, ensure_ascii=True)


def _build_preview(messages: list[dict[str, object]]) -> str | None:
    for message in reversed(messages):
        role = str(message.get("role") or "")
        if role != "user":
            continue

        jobs = message.get("jobs")
        if isinstance(jobs, list) and len(jobs) > 0:
            first_job = jobs[0] if isinstance(jobs[0], dict) else {}
            if isinstance(first_job, dict):
                label = str(first_job.get("label") or "").strip()
                if label:
                    return label[:80]

    return None


def _chat_owner_or_404(db: Session, user_id: uuid.UUID | None, chat_id: uuid.UUID) -> PortalChat:
    query = db.query(PortalChat).filter(PortalChat.id == chat_id)
    if user_id is not None:
        query = query.filter(PortalChat.user_id == user_id)

    chat = query.first()
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


@router.get("/chats", response_model=list[PortalChatListItem])
def list_chats(request: Request, db: Session = Depends(get_db)):
    user_id = _safe_user_uuid(request)
    query = db.query(PortalChat)
    if user_id is not None:
        query = query.filter(PortalChat.user_id == user_id)
    else:
        query = query.filter(PortalChat.user_id.is_(None))

    chats = query.order_by(PortalChat.updated_at.desc()).all()

    result: list[PortalChatListItem] = []
    for chat in chats:
        messages = _parse_messages(chat.messages)
        result.append(
            PortalChatListItem(
                id=chat.id,
                title=chat.title,
                created_at=chat.created_at,
                updated_at=chat.updated_at,
                message_count=len(messages),
                preview=_build_preview(messages),
            )
        )

    return result


@router.post("/chats", response_model=PortalChatDetailResponse, status_code=status.HTTP_201_CREATED)
def create_chat(payload: PortalChatCreateRequest, request: Request, db: Session = Depends(get_db)):
    now = datetime.utcnow()
    chat = PortalChat(
        user_id=_safe_user_uuid(request),
        title=(payload.title or "New Chat").strip()[:200] or "New Chat",
        messages=_serialize_messages(payload.messages),
        created_at=now,
        updated_at=now,
    )

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return PortalChatDetailResponse(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        messages=_parse_messages(chat.messages),
    )


@router.get("/chats/{chat_id}", response_model=PortalChatDetailResponse)
def get_chat(chat_id: uuid.UUID, request: Request, db: Session = Depends(get_db)):
    chat = _chat_owner_or_404(db=db, user_id=_safe_user_uuid(request), chat_id=chat_id)
    return PortalChatDetailResponse(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        messages=_parse_messages(chat.messages),
    )


@router.put("/chats/{chat_id}", response_model=PortalChatDetailResponse)
def update_chat(
    chat_id: uuid.UUID,
    payload: PortalChatUpdateRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    chat = _chat_owner_or_404(db=db, user_id=_safe_user_uuid(request), chat_id=chat_id)

    if payload.title is not None:
        chat.title = payload.title.strip()[:200] or "New Chat"

    if payload.messages is not None:
        chat.messages = _serialize_messages(payload.messages)

    chat.updated_at = datetime.utcnow()

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return PortalChatDetailResponse(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        messages=_parse_messages(chat.messages),
    )


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(chat_id: uuid.UUID, request: Request, db: Session = Depends(get_db)):
    chat = _chat_owner_or_404(db=db, user_id=_safe_user_uuid(request), chat_id=chat_id)
    db.delete(chat)
    db.commit()
    return None
