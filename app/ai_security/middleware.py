"""
Shadow Guard Middleware
=======================
FastAPI middleware that intercepts POST requests to LLM-backed endpoints,
extracts the user-supplied text fields from the JSON body, and runs them
through the Shadow Guard (Phi-3) classifier **before** the request reaches
the route handler.

Protected routes (JSON body endpoints):
    /text/analyze              → field: "text"
    /text/sms/analyze          → field: "text"
    /text/email/analyze/*      → field: "body"

The voice endpoint (/voice/analyse) is file-upload based, so the guard is
called inline in the voice router instead or after transcription.
"""

from __future__ import annotations

import json
import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.ai_security.guard import is_prompt_injection

logger = logging.getLogger("zora.ai_security.middleware")

# Routes where we inspect the JSON body for prompt injection
_PROTECTED_ROUTES: dict[str, list[str]] = {
    # path prefix → list of JSON fields to scan
    "/text/email/analyze": ["body", "subject"],
    "/text/sms/analyze": ["text"],
    "/text/analyze": ["text"],
}


class ShadowGuardMiddleware(BaseHTTPMiddleware):
    """Scans incoming request bodies for prompt-injection before they reach the LLM."""

    async def dispatch(self, request: Request, call_next):
        # Only intercept POST requests
        if request.method != "POST":
            return await call_next(request)

        path = request.url.path

        # Check if this route is protected
        fields_to_scan: list[str] | None = None
        for prefix, fields in _PROTECTED_ROUTES.items():
            if path.startswith(prefix):
                fields_to_scan = fields
                break

        if fields_to_scan is None:
            return await call_next(request)

        # Read and parse JSON body
        try:
            body_bytes = await request.body()
            body_json = json.loads(body_bytes)
        except Exception:
            # If we can't parse the body, let the normal handler deal with it
            return await call_next(request)

        # Scan each field for prompt injection
        for field in fields_to_scan:
            value = body_json.get(field)
            if not value or not isinstance(value, str):
                continue

            if is_prompt_injection(value):
                logger.warning(
                    "🛡️ Shadow Guard BLOCKED request  |  path=%s  |  field=%s  |  snippet=%s",
                    path,
                    field,
                    value[:80],
                )
                print(
                    f"\n🚫 REQUEST BLOCKED by Shadow Guard"
                    f"\n   Path:  {path}"
                    f"\n   Field: {field}"
                    f"\n   Text:  {value[:80]}...\n"
                )
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": "Request blocked by Shadow Guard: prompt injection detected",
                        "field": field,
                        "guard": "shadow_phi3",
                    },
                )

        return await call_next(request)
