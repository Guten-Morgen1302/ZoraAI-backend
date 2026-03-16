from __future__ import annotations

import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth.security import get_token_subject

logger = logging.getLogger("zora.middleware")


class AuthLoggingMiddleware(BaseHTTPMiddleware):
    EXEMPT_PATHS = {"/", "/openapi.json"}
    EXEMPT_PATH_PREFIXES = ("/docs", "/redoc", "/auth")

    async def dispatch(self, request: Request, call_next):
        logger.info("Incoming request", extra={"path": request.url.path, "method": request.method})

        path = request.url.path
        is_exempt = path in self.EXEMPT_PATHS or path.startswith(self.EXEMPT_PATH_PREFIXES)

        if not is_exempt:
            access_token = request.cookies.get("access_token")
            if not access_token:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication cookie missing"},
                )

            user_id = get_token_subject(access_token, expected_type="access")
            if not user_id:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid authentication token"},
                )

            request.state.user_id = user_id

        response = await call_next(request)
        logger.info(
            "Completed request",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
            },
        )
        return response
