from __future__ import annotations

import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth.security import get_token_subject
from app.middleware.rate_limiter import RedisTokenBucketLimiter

logger = logging.getLogger("zora.middleware")


class AuthLoggingMiddleware(BaseHTTPMiddleware):
    AUTH_EXEMPT_PATHS = {"/", "/openapi.json"}
    AUTH_EXEMPT_PATH_PREFIXES = ("/docs", "/redoc", "/auth")
    RATE_LIMIT_EXEMPT_PATHS = {"/", "/openapi.json"}
    RATE_LIMIT_EXEMPT_PATH_PREFIXES = ("/docs", "/redoc")

    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = RedisTokenBucketLimiter.from_env()

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    def _is_auth_exempt(self, path: str) -> bool:
        return path in self.AUTH_EXEMPT_PATHS or path.startswith(self.AUTH_EXEMPT_PATH_PREFIXES)

    def _is_rate_limit_exempt(self, path: str) -> bool:
        return path in self.RATE_LIMIT_EXEMPT_PATHS or path.startswith(self.RATE_LIMIT_EXEMPT_PATH_PREFIXES)

    async def dispatch(self, request: Request, call_next):
        logger.info("Incoming request", extra={"path": request.url.path, "method": request.method})

        path = request.url.path
        is_auth_exempt = self._is_auth_exempt(path)

        if not is_auth_exempt:
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

        if not self._is_rate_limit_exempt(path):
            identifier = getattr(request.state, "user_id", None)
            if identifier:
                rate_limit_key = f"user:{identifier}"
            else:
                rate_limit_key = f"ip:{self._client_ip(request)}"

            decision = self.rate_limiter.consume(identifier=rate_limit_key, requested_tokens=1)
            if not decision.allowed:
                headers = {
                    "X-RateLimit-Limit": str(decision.capacity),
                    "X-RateLimit-Remaining": str(decision.remaining_tokens),
                    "Retry-After": str(decision.retry_after_seconds),
                }
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded. Please retry later."},
                    headers=headers,
                )
        else:
            decision = None

        response = await call_next(request)
        if decision is not None and decision.enforced:
            response.headers["X-RateLimit-Limit"] = str(decision.capacity)
            response.headers["X-RateLimit-Remaining"] = str(decision.remaining_tokens)
        logger.info(
            "Completed request",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
            },
        )
        return response
