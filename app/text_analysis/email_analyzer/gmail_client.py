from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
logger = logging.getLogger("zora.text_analysis.gmail_client")


class GmailClientError(RuntimeError):
    """Raised for recoverable Gmail client failures."""


def _decode_b64_urlsafe(payload: str | None) -> str:
    if not payload:
        return ""
    padding = "=" * (-len(payload) % 4)
    raw = base64.urlsafe_b64decode(f"{payload}{padding}")
    return raw.decode("utf-8", errors="replace")


def _extract_body_parts(part: dict[str, Any]) -> tuple[str, str]:
    mime_type = (part.get("mimeType") or "").lower()
    body_data = ((part.get("body") or {}).get("data"))

    text_plain = ""
    text_html = ""

    if mime_type == "text/plain":
        text_plain = _decode_b64_urlsafe(body_data)
    elif mime_type == "text/html":
        text_html = _decode_b64_urlsafe(body_data)

    for child in part.get("parts") or []:
        child_plain, child_html = _extract_body_parts(child)
        if child_plain:
            text_plain += f"\n{child_plain}" if text_plain else child_plain
        if child_html:
            text_html += f"\n{child_html}" if text_html else child_html

    return text_plain.strip(), text_html.strip()


def _get_google_service(client_secrets_file: Path, token_file: Path, *, force_reauth: bool = False):
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover
        raise GmailClientError(
            "Missing Gmail dependencies. Install google-api-python-client, "
            "google-auth-httplib2, and google-auth-oauthlib"
        ) from exc

    if force_reauth and token_file.exists():
        logger.info("Force re-auth enabled; deleting existing Gmail token at %s", token_file)
        print(f"[gmail-client] Force re-auth: deleting token {token_file}")
        token_file.unlink(missing_ok=True)

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired Gmail OAuth token")
            print("[gmail-client] Refreshing expired OAuth token")
            creds.refresh(Request())
        else:
            if not client_secrets_file.exists():
                raise GmailClientError(f"Gmail client secrets file not found: {client_secrets_file}")
            logger.info("Starting Gmail OAuth local server flow")
            print("[gmail-client] Starting OAuth login flow in browser")
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_file), SCOPES)
            creds = flow.run_local_server(port=7000)

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=creds)


def _header_value(headers: list[dict[str, str]], name: str) -> str:
    for header in headers:
        if (header.get("name") or "").lower() == name.lower():
            return header.get("value") or ""
    return ""


def fetch_latest_email(
    *,
    client_secrets_path: str | Path,
    token_path: str | Path | None = None,
    query: str | None = None,
    force_reauth: bool = False,
) -> dict[str, Any]:
    """Fetches the latest Gmail message and returns sender, subject, and body."""
    secrets_file = Path(client_secrets_path)
    if token_path is None:
        token_file = secrets_file.with_name("gmail_token.json")
    else:
        token_file = Path(token_path)

    logger.info("Fetching latest email from Gmail. query=%s force_reauth=%s", query, force_reauth)
    print(f"[gmail-client] Fetching latest email. query={query!r} force_reauth={force_reauth}")
    service = _get_google_service(secrets_file, token_file, force_reauth=force_reauth)

    list_params: dict[str, Any] = {"userId": "me", "maxResults": 1}
    if query:
        list_params["q"] = query

    response = service.users().messages().list(**list_params).execute()
    messages = response.get("messages") or []
    if not messages:
        raise GmailClientError("No messages found in Gmail inbox")

    message_id = messages[0]["id"]
    message = service.users().messages().get(userId="me", id=message_id, format="full").execute()

    payload = message.get("payload") or {}
    headers = payload.get("headers") or []

    sender = _header_value(headers, "From")
    subject = _header_value(headers, "Subject")
    internal_date = message.get("internalDate")
    snippet = message.get("snippet") or ""

    text_plain, text_html = _extract_body_parts(payload)
    if not text_plain and payload.get("body"):
        text_plain = _decode_b64_urlsafe((payload.get("body") or {}).get("data"))

    body = text_plain or text_html or snippet

    logger.info("Fetched latest Gmail message id=%s subject=%s", message_id, subject[:120])
    print(f"[gmail-client] Latest message id={message_id} subject={subject[:80]!r}")

    return {
        "message_id": message_id,
        "thread_id": message.get("threadId"),
        "sender": sender,
        "subject": subject,
        "body": body,
        "body_html": text_html,
        "snippet": snippet,
        "internal_date": internal_date,
        "labels": message.get("labelIds") or [],
        "raw": {
            "sizeEstimate": message.get("sizeEstimate"),
            "historyId": message.get("historyId"),
        },
    }


def fetch_email_by_message_id(
    *,
    client_secrets_path: str | Path,
    message_id: str,
    thread_id: str | None = None,
    token_path: str | Path | None = None,
    force_reauth: bool = False,
) -> dict[str, Any]:
    """Fetches a specific Gmail message by message ID and returns sender, subject, and body."""
    if not message_id.strip():
        raise GmailClientError("message_id must not be empty")

    secrets_file = Path(client_secrets_path)
    if token_path is None:
        token_file = secrets_file.with_name("gmail_token.json")
    else:
        token_file = Path(token_path)

    logger.info(
        "Fetching Gmail message by id. message_id=%s thread_id=%s force_reauth=%s",
        message_id,
        thread_id,
        force_reauth,
    )
    print(
        "[gmail-client] Fetching message by id "
        f"message_id={message_id!r} thread_id={thread_id!r} force_reauth={force_reauth}"
    )
    service = _get_google_service(secrets_file, token_file, force_reauth=force_reauth)

    message = service.users().messages().get(userId="me", id=message_id.strip(), format="full").execute()

    payload = message.get("payload") or {}
    headers = payload.get("headers") or []

    sender = _header_value(headers, "From")
    subject = _header_value(headers, "Subject")
    internal_date = message.get("internalDate")
    snippet = message.get("snippet") or ""

    text_plain, text_html = _extract_body_parts(payload)
    if not text_plain and payload.get("body"):
        text_plain = _decode_b64_urlsafe((payload.get("body") or {}).get("data"))

    resolved_thread_id = message.get("threadId")
    if thread_id and resolved_thread_id and thread_id != resolved_thread_id:
        logger.warning(
            "Provided thread_id does not match Gmail message thread. provided=%s actual=%s",
            thread_id,
            resolved_thread_id,
        )

    body = text_plain or text_html or snippet
    return {
        "message_id": message_id,
        "thread_id": resolved_thread_id,
        "sender": sender,
        "subject": subject,
        "body": body,
        "body_html": text_html,
        "snippet": snippet,
        "internal_date": internal_date,
        "labels": message.get("labelIds") or [],
        "raw": {
            "sizeEstimate": message.get("sizeEstimate"),
            "historyId": message.get("historyId"),
        },
    }
