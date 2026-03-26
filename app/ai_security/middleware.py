"""
Shadow Guard Middleware
=======================
Intercepts POST requests to verify safety before they reach the main LLM.
"""

from __future__ import annotations
import json
import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure this import matches your project structure
from app.ai_security.guard import is_prompt_injection

logger = logging.getLogger("zora.ai_security.middleware")

_PROTECTED_ROUTES: dict[str, list[str]] = {
    "/text/email/analyze": ["body", "subject"],
    "/text/sms/analyze": ["text"],
    "/text/analyze": ["text"],
}

class ShadowGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method != "POST":
            return await call_next(request)

        path = request.url.path
        fields_to_scan = None
        for prefix, fields in _PROTECTED_ROUTES.items():
            if path.startswith(prefix):
                fields_to_scan = fields
                break

        if fields_to_scan is None:
            return await call_next(request)

        # 1. Capture the body
        try:
            body_bytes = await request.body()
            if not body_bytes:
                return await call_next(request)
            body_json = json.loads(body_bytes)
        except Exception:
            return await call_next(request)

        # 2. Reset the request so the endpoint can still read the body later
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive

        # 3. Scan specified fields
        for field in fields_to_scan:
            value = body_json.get(field)
            if not value or not isinstance(value, str):
                continue

            if is_prompt_injection(value):
                logger.warning("🛡️ Shadow Guard BLOCKED path=%s | field=%s", path, field)
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": "Request blocked: Direct model hijack attempt detected.",
                        "field": field,
                        "guard": "shadow_phi3",
                    },
                )

        return await call_next(request)